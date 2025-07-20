import logging
from sqlalchemy import create_engine, insert, update, select
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import json

# --- RAGnetic Imports ---
from app.db.models import workflows_table, workflow_runs_table, human_tasks_table
from app.schemas.workflow import Workflow, StepType, AgentCallStep, ToolCallStep, IfThenStep, LoopStep, \
    HumanInTheLoopStep, WorkflowStep
from app.agents.config_manager import load_agent_config
from app.schemas.agent import AgentConfig

# --- Tool Imports ---
from app.tools.retriever_tool import get_retriever_tool
from app.tools.sql_tool import create_sql_toolkit
from app.tools.search_engine_tool import SearchTool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.http_request_tool import HTTPRequestTool
from app.tools.slack_webhook_tool import SlackWebhookTool

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """
    The core engine for parsing and executing RAGnetic workflows.
    This version is agent-aware, allowing it to use agent-specific tools.
    """

    def __init__(self, db_engine: Any):
        self.db_engine = db_engine
        self.context: Dict[str, Any] = {}
        self.agent_config: Optional[AgentConfig] = None
        self.tools: Dict[str, Any] = {}

        self.step_handlers = {
            StepType.AGENT_CALL: self._handle_agent_call,
            StepType.TOOL_CALL: self._handle_tool_call,
            StepType.IF_THEN: self._handle_if_then,
            StepType.LOOP: self._handle_loop,
            StepType.HUMAN_IN_THE_LOOP: self._handle_human_in_the_loop,
        }

    def run_workflow(self, workflow_name: str, initial_input: Optional[Dict[str, Any]] = None,
                     resume_run_id: Optional[str] = None):
        """
        Executes a workflow from the database, making it aware of the agent's context.
        """
        logger.info(f"Starting or resuming workflow run for: {workflow_name}")

        with self.db_engine.connect() as connection:
            workflow_data = connection.execute(
                select(workflows_table).where(workflows_table.c.name == workflow_name)
            ).fetchone()

            if not workflow_data:
                logger.error(f"Workflow '{workflow_name}' not found in the database.")
                return

            try:
                # The definition from the DB is likely a JSON string
                definition_dict = workflow_data.definition
                if isinstance(definition_dict, str):
                    definition_dict = json.loads(definition_dict)

                # Add the primary key 'id' from the database row to the dictionary
                definition_dict['id'] = workflow_data.id

                # Now, validate the complete data structure
                workflow = Workflow.model_validate(definition_dict)

            except Exception as e:
                logger.error(f"Failed to load or validate workflow '{workflow_name}': {e}", exc_info=True)
                return

            if workflow.agent_name:
                try:
                    self.agent_config = load_agent_config(workflow.agent_name)
                    logger.info(f"Loaded configuration for agent: '{workflow.agent_name}'")
                except FileNotFoundError:
                    logger.error(f"Configuration for agent '{workflow.agent_name}' not found.")
                    self._update_run_status(resume_run_id or str(uuid.uuid4()), "failed",
                                            {"error": f"Agent '{workflow.agent_name}' not found."})
                    return

        run_id = resume_run_id if resume_run_id else str(uuid.uuid4())
        start_step_index = 0

        if resume_run_id:
            logger.info(f"Resuming workflow run: {run_id}")
            with self.db_engine.connect() as connection:
                run_data = connection.execute(
                    select(workflow_runs_table).where(workflow_runs_table.c.run_id == run_id)).fetchone()
                if not run_data:
                    logger.error(f"Cannot resume. Run ID '{run_id}' not found.")
                    return
                self.context = run_data.last_execution_state.get('context', {})
                start_step_index = run_data.last_execution_state.get('step_index', 0) + 1
                stmt = update(workflow_runs_table).where(workflow_runs_table.c.run_id == run_id).values(
                    status="running")
                connection.execute(stmt)
                connection.commit()
        else:
            logger.info(f"Starting new workflow run: {run_id}")
            self.context = initial_input or {}
            try:
                with self.db_engine.connect() as connection:
                    stmt = insert(workflow_runs_table).values(
                        run_id=run_id,
                        workflow_id=workflow.id,  # Use the validated workflow's ID
                        status="running",
                        initial_input=initial_input,
                        start_time=datetime.utcnow()
                    )
                    connection.execute(stmt)
                    connection.commit()
            except Exception as e:
                logger.error(f"Failed to record workflow start: {e}", exc_info=True)
                return

        try:
            self._execute_steps(workflow.steps, start_index=start_step_index, run_id=run_id)
            logger.info(f"Workflow run '{run_id}' completed successfully.")
            self._update_run_status(run_id, "completed", self.context)

        except RuntimeError as e:
            if "Workflow paused for human input" in str(e):
                logger.info(f"Workflow run '{run_id}' paused successfully. Awaiting human input.")
                # The status is already 'paused', so we don't need to do anything else.
            else:
                logger.error(f"Workflow run '{run_id}' failed due to a runtime error: {e}", exc_info=True)
                self._update_run_status(run_id, "failed", {"error": str(e)})
        except Exception as e:
            logger.error(f"Workflow run '{run_id}' failed with an unexpected error: {e}", exc_info=True)
            self._update_run_status(run_id, "failed", {"error": str(e)})

    def _execute_steps(self, steps: List[WorkflowStep], start_index: int = 0, run_id: str = ""):
        """Helper to recursively execute a list of steps."""
        for i in range(start_index, len(steps)):
            step = steps[i]
            handler = self.step_handlers.get(step.type)
            if handler:
                logger.info(f"Executing step '{step.name}' (Type: {step.type})")
                handler(step, run_id, i)
            else:
                logger.warning(f"Skipping unsupported step type: {step.type} for step '{step.name}'")

    def _get_tool(self, tool_name: str) -> Any:
        """Dynamically initializes and caches tools based on the agent's configuration."""
        if tool_name in self.tools:
            return self.tools[tool_name]

        tool = None
        if self.agent_config:
            if tool_name == "retriever":
                tool = get_retriever_tool(self.agent_config)
            elif tool_name == "sql_toolkit":
                db_source = next((s for s in self.agent_config.sources if s.type == 'db'), None)
                if db_source and db_source.db_connection:
                    tool = create_sql_toolkit(db_connection_string=db_source.db_connection,
                                              llm_model_name=self.agent_config.llm_model)
                else:
                    raise ValueError("SQL toolkit requested but no database source found in agent config.")
            elif tool_name == "search_engine":
                tool = SearchTool(agent_config=self.agent_config)
            elif tool_name == "arxiv":
                tool = get_arxiv_tool()

        if tool_name == "http_request_tool":
            tool = HTTPRequestTool()
        elif tool_name == "slack_webhook_tool":
            tool = SlackWebhookTool()

        if not tool:
            raise ValueError(f"Tool '{tool_name}' is not recognized or could not be initialized.")

        self.tools[tool_name] = tool
        return tool

    def _handle_agent_call(self, step: AgentCallStep, run_id: str, step_index: int):
        """Simulates calling an agent with the given input."""
        logger.info(f"  - Calling agent '{step.agent_name}' with input: {step.input}")
        # Placeholder for actual agent call logic
        # agent_output = await call_agent(step.agent_name, step.input)
        # self.context[step.name] = agent_output
        result = {"result": "agent call simulated", "agent_name": step.agent_name, "input": step.input}
        self.context[step.name] = result

    def _handle_tool_call(self, step: ToolCallStep, run_id: str, step_index: int):
        """Handles a tool_call step by dynamically getting the tool and running it."""
        logger.info(f"  - Preparing to call tool '{step.tool_name}' with input: {step.tool_input}")
        try:
            tool_instance = self._get_tool(step.tool_name)
            if isinstance(tool_instance, list):
                # Simple approach for toolkits: run the first tool.
                # A more advanced version might inspect input to route to the correct sub-tool.
                result = tool_instance[0].run(step.tool_input)
            else:
                result = tool_instance.run(**step.tool_input)

            self.context[step.name] = result
        except Exception as e:
            logger.error(f"Error executing tool '{step.tool_name}': {e}", exc_info=True)
            raise

    def _handle_if_then(self, step: IfThenStep, run_id: str, step_index: int):
        """Evaluates a condition and executes the appropriate branch."""
        logger.info(f"  - Evaluating condition: '{step.condition}'")
        try:
            is_true = eval(step.condition, {"__builtins__": None}, self.context)
        except Exception as e:
            raise ValueError(f"Could not evaluate condition '{step.condition}': {e}") from e

        if is_true:
            logger.info("  - Condition evaluated to true. Executing 'on_true' steps.")
            self._execute_steps(step.on_true, run_id=run_id)
        elif step.on_false:
            logger.info("  - Condition evaluated to false. Executing 'on_false' steps.")
            self._execute_steps(step.on_false, run_id=run_id)

    def _handle_loop(self, step: LoopStep, run_id: str, step_index: int):
        """Iterates over a list from the context and executes steps for each item."""
        iterable = self.context.get(step.iterable)
        if not isinstance(iterable, list):
            raise TypeError(f"Cannot loop. The context key '{step.iterable}' is not a list.")

        logger.info(f"  - Starting loop over {len(iterable)} items from '{step.iterable}'")
        for item in iterable:
            self.context[step.loop_variable] = item
            self._execute_steps(step.steps, run_id=run_id)
        logger.info("  - Loop completed.")

    def _handle_human_in_the_loop(self, step: HumanInTheLoopStep, run_id: str, step_index: int):
        """Pauses the workflow and creates a task for human intervention."""
        logger.info(f"  - Pausing workflow for human input: '{step.prompt}'")
        try:
            with self.db_engine.connect() as connection:
                run_db_id = self._get_run_db_id(connection, run_id)
                if not run_db_id:
                    raise RuntimeError(f"Could not find database ID for run_id '{run_id}'")

                stmt = insert(human_tasks_table).values(
                    run_id=run_db_id,
                    task_name=step.name,
                    status="pending",
                    payload=step.data,
                    assigned_to_user_id=None
                )
                connection.execute(stmt)

                stmt = update(workflow_runs_table).where(
                    workflow_runs_table.c.run_id == run_id
                ).values(
                    status="paused",
                    last_execution_state={"step_index": step_index, "context": self.context}
                )
                connection.execute(stmt)
                connection.commit()
        except Exception as e:
            logger.error(f"Failed to pause workflow '{run_id}': {e}", exc_info=True)
            raise
        raise RuntimeError("Workflow paused for human input.")

    def _get_run_db_id(self, connection, run_id: str) -> Optional[int]:
        """Helper to get the database primary key for a given run_id."""
        return connection.execute(
            select(workflow_runs_table.c.id).where(workflow_runs_table.c.run_id == run_id)).scalar_one_or_none()

    def _update_run_status(self, run_id: str, status: str, final_output: Dict[str, Any]):
        """Updates the status and final output of a workflow run."""
        with self.db_engine.connect() as connection:
            stmt = update(workflow_runs_table).where(
                workflow_runs_table.c.run_id == run_id
            ).values(
                status=status,
                end_time=datetime.utcnow(),
                final_output=final_output
            )
            connection.execute(stmt)
            connection.commit()
            logger.info(f"Workflow run '{run_id}' final status updated to '{status}'.")