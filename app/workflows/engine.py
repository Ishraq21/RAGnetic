import logging
import json
import os
import uuid
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from pydantic.v1 import BaseModel, Field as PydanticField
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from sqlalchemy import select, update, insert

from app.agents.config_manager import load_agent_config
from app.core.config import get_api_key, get_path_settings
from app.db.models import human_tasks_table, workflow_runs_table, workflows_table
from app.schemas.agent import AgentConfig
from app.schemas.workflow import (
    AgentCallStep,
    HumanInTheLoopStep,
    IfThenStep,
    LoopStep,
    Skill,
    StepType,
    ToolCallStep,
    Workflow,
    WorkflowStep,
)
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.http_request_tool import HTTPRequestTool
from app.tools.python_script_tool import PythonScriptTool
from app.tools.retriever_tool import get_retriever_tool
from app.tools.search_engine_tool import SearchTool
from app.tools.slack_webhook_tool import SlackWebhookTool
from app.tools.sql_tool import create_sql_toolkit

logger = logging.getLogger(__name__)
_APP_PATHS = get_path_settings()


# ---------------------------------------------------------------------------
# Structured plan object returned each ReAct iteration
# ---------------------------------------------------------------------------
class Plan(BaseModel):
    """The agent's plan, containing its thought and the next action to take."""
    thought: str = PydanticField(
        ...,
        description="The agent's reasoning and thought process for its next action.",
    )
    action: str = PydanticField(
        ...,
        description="The name of the skill or tool to execute next. Must be one of the available actions.",
    )
    action_input: Dict[str, Any] = PydanticField(
        default_factory=dict,
        description="The input parameters for the chosen action.",
    )



class WorkflowEngine:
    """
    Core engine for parsing and executing RAGnetic workflows (No-Variable Edition).
    """

    def __init__(self, db_engine: Any):
        self.db_engine = db_engine
        self.context: Dict[str, Any] = {}
        self.agent_config: Optional[AgentConfig] = None
        self.tools: Dict[str, Any] = {}
        self.skills: Dict[str, Skill] = {}
        self.llm: Optional[Any] = None

        self.step_handlers = {
            StepType.AGENT_CALL: self._handle_agent_call,
            StepType.TOOL_CALL: self._handle_tool_call,
            StepType.IF_THEN: self._handle_if_then,
            StepType.LOOP: self._handle_loop,
            StepType.HUMAN_IN_THE_LOOP: self._handle_human_in_the_loop,
        }


    def _get_llm(self):
        if self.llm:
            return self.llm
        if not self.agent_config:
            raise ValueError("Agent configuration is not loaded; cannot initialize LLM.")

        model_name = self.agent_config.llm_model
        model_kwargs = (
            self.agent_config.model_params.model_dump(exclude_unset=True)
            if self.agent_config.model_params
            else {}
        )
        logger.info(f"Initializing LLM: {model_name}")

        if model_name.startswith("ollama/"):
            ollama_model_name = model_name.split("/", 1)[1]
            self.llm = ChatOllama(model=ollama_model_name, **model_kwargs)
        else:
            provider = "openai"
            if "claude" in model_name.lower():
                provider = "anthropic"
            elif "gemini" in model_name.lower():
                provider = "google"

            api_key = get_api_key(provider)
            if not api_key:
                raise ValueError(
                    f"API key for provider '{provider}' not found. Use 'ragnetic set-api-key'."
                )

            if provider == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name, google_api_key=api_key, **model_kwargs
                )
            else:
                self.llm = init_chat_model(
                    model_name, model_provider=provider, api_key=api_key, **model_kwargs
                )
        return self.llm


    def run_workflow(
        self,
        workflow_name: str,
        initial_input: Optional[Dict[str, Any]] = None,
        resume_run_id: Optional[str] = None,
    ):
        """Executes a workflow from the database (fresh or resume)."""
        logger.info(f"Starting or resuming workflow run for: {workflow_name}")


        with self.db_engine.connect() as connection:
            row = connection.execute(
                select(workflows_table).where(workflows_table.c.name == workflow_name)
            ).fetchone()

            if not row:
                logger.error(f"Workflow '{workflow_name}' not found in DB.")
                return

            try:
                definition_dict = (
                    json.loads(row.definition)
                    if isinstance(row.definition, str)
                    else row.definition
                )
                definition_dict["id"] = row.id
                workflow = Workflow.model_validate(definition_dict)
            except Exception as e:
                logger.error(
                    f"Failed to load or validate workflow '{workflow_name}': {e}",
                    exc_info=True,
                )
                return

            # Load agent config (optional per-workflow top-level)
            if workflow.agent_name:
                try:
                    self.agent_config = load_agent_config(workflow.agent_name)
                    self._load_skills()
                    logger.info(
                        f"Loaded configuration and skills for agent: '{workflow.agent_name}'"
                    )
                except FileNotFoundError:
                    logger.error(
                        f"Configuration for agent '{workflow.agent_name}' not found."
                    )
                    self._update_run_status(
                        resume_run_id or str(uuid.uuid4()),
                        "failed",
                        {"error": f"Agent '{workflow.agent_name}' not found."},
                    )
                    return

        run_id = resume_run_id or str(uuid.uuid4())
        start_step_index = 0

        if resume_run_id:
            logger.info(f"Resuming workflow run: {run_id}")
            with self.db_engine.connect() as connection:
                run_data = connection.execute(
                    select(workflow_runs_table).where(
                        workflow_runs_table.c.run_id == run_id
                    )
                ).fetchone()
                if not run_data:
                    logger.error(f"Cannot resume. Run ID '{run_id}' not found.")
                    return
                state = run_data.last_execution_state or {}
                self.context = state.get("context", {})
                start_step_index = state.get("step_index", 0) + 1

                stmt = (
                    update(workflow_runs_table)
                    .where(workflow_runs_table.c.run_id == run_id)
                    .values(status="running")
                )
                connection.execute(stmt)
                connection.commit()
        else:
            logger.info(f"Starting new workflow run: {run_id}")
            # bootstrap trigger context
            self.context = {"trigger": initial_input or {}}
            try:
                with self.db_engine.connect() as connection:
                    stmt = insert(workflow_runs_table).values(
                        run_id=run_id,
                        workflow_id=workflow.id,
                        status="running",
                        initial_input=initial_input,
                        start_time=datetime.utcnow(),
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
                logger.info(
                    f"Workflow run '{run_id}' paused successfully. Awaiting human input."
                )
            else:
                logger.error(
                    f"Workflow run '{run_id}' failed due to runtime error: {e}",
                    exc_info=True,
                )
                self._update_run_status(run_id, "failed", {"error": str(e)})
        except Exception as e:
            logger.error(
                f"Workflow run '{run_id}' failed with unexpected error: {e}",
                exc_info=True,
            )
            self._update_run_status(run_id, "failed", {"error": str(e)})


    def _execute_steps(
        self, steps: Sequence[WorkflowStep], start_index: int = 0, run_id: str = ""
    ):
        """Execute a list of steps, updating checkpoint after each."""
        for i in range(start_index, len(steps)):
            step = steps[i]
            handler = self.step_handlers.get(step.type)
            if not handler:
                logger.warning(
                    f"Skipping unsupported step type: {step.type} for step '{step.name}'"
                )
                continue

            logger.info(f"Executing step '{step.name}' (Type: {step.type})")
            handler(step, run_id, i)
            self._checkpoint(run_id, i)  # persist progress


    def _get_tool(self, tool_name: str) -> Any:
        """Lazy-init and cache tools."""
        if tool_name in self.tools:
            return self.tools[tool_name]

        tool = None
        if self.agent_config:
            if tool_name == "retriever":
                tool = get_retriever_tool(self.agent_config)
            elif tool_name == "sql_toolkit":
                db_source = next(
                    (s for s in self.agent_config.sources if s.type == "db"), None
                )
                if db_source and db_source.db_connection:
                    tool = create_sql_toolkit(
                        db_connection_string=db_source.db_connection,
                        llm_model_name=self.agent_config.llm_model,
                    )
                else:
                    raise ValueError(
                        "SQL toolkit requested but no database source found in agent config."
                    )
            elif tool_name == "search_engine":
                tool = SearchTool(agent_config=self.agent_config)
            elif tool_name == "arxiv":
                tool = get_arxiv_tool()

        if tool_name == "http_request_tool":
            tool = HTTPRequestTool()
        elif tool_name == "slack_webhook_tool":
            tool = SlackWebhookTool()
        elif tool_name == "python_script_tool":
            tool = PythonScriptTool()

        if not tool:
            raise ValueError(
                f"Tool '{tool_name}' is not recognized or could not be initialized."
            )

        self.tools[tool_name] = tool
        return tool


    def _load_skills(self):
        """Load all skill YAML files into memory."""
        skills_dir = _APP_PATHS.get("SKILLS_DIR")
        if not skills_dir or not os.path.isdir(skills_dir):
            logger.warning(
                f"Skills directory not found at {skills_dir}. No skills will be loaded."
            )
            return

        for filename in os.listdir(skills_dir):
            if filename.endswith((".yaml", ".yml")):
                try:
                    filepath = os.path.join(skills_dir, filename)
                    with open(filepath, "r") as f:
                        skill_data = yaml.safe_load(f)
                        skill = Skill.model_validate(skill_data)
                        self.skills[skill.name] = skill
                        logger.info(f"Loaded skill: '{skill.name}'")
                except Exception as e:
                    logger.error(
                        f"Failed to load or parse skill file '{filename}': {e}",
                        exc_info=True,
                    )


    def _get_available_actions(self) -> Dict[str, str]:
        """Return a dict: action_name -> description+schema snippet."""
        actions: Dict[str, str] = {}

        # Skills
        for name, skill in self.skills.items():
            schema = self._skill_schema(skill)
            actions[name] = (
                f"[SKILL] {getattr(skill, 'description', name)}\n"
                f"Input Schema: {json.dumps(schema, indent=2)}"
            )

        # Tools
        if self.agent_config and self.agent_config.tools:
            for tool_name in self.agent_config.tools:
                try:
                    tool_instance = self._get_tool(tool_name)
                    if hasattr(tool_instance, "get_input_schema") and callable(
                        tool_instance.get_input_schema
                    ):
                        schema = tool_instance.get_input_schema()
                        actions[tool_name] = (
                            f"[TOOL] {tool_name}\nInput Schema: {json.dumps(schema, indent=2)}"
                        )
                    else:
                        actions[
                            tool_name
                        ] = f"[TOOL] {tool_name}. Takes a dictionary of arguments."
                except Exception as e:
                    logger.warning(f"Could not get schema for tool '{tool_name}': {e}")

        # Finish sentinel
        actions[
            "finish"
        ] = "Use this when you have fully completed the task and have the final answer."
        return actions

    def _skill_schema(self, skill: Skill) -> Dict[str, Any]:
        """Best-effort input schema synthesis for skills."""
        # Not all Skill schemas expose 'parameters'; adapt.
        if hasattr(skill, "parameters") and skill.parameters is not None:
            return skill.parameters  # type: ignore[attr-defined]
        if hasattr(skill, "input_variables"):
            # Represent required fields as strings
            return {
                "type": "object",
                "properties": {v: {"type": "string"} for v in skill.input_variables},
                "required": list(skill.input_variables),
            }
        return {"type": "object", "properties": {}, "additionalProperties": True}


    def _build_agent_prompt(
        self, task: str, actions: Dict[str, str], history: List[BaseMessage]
    ) -> List[BaseMessage]:
        # Truncate context to avoid massive system prompts
        ctx_json = json.dumps(self.context, indent=2)
        if len(ctx_json) > 8000:
            ctx_json = ctx_json[:8000] + "\n... [truncated] ..."

        system_prompt = f"""
You are a helpful and reliable AI agent. Complete the task by thinking step-by-step and choosing one action at a time.

You may call SKILLS (smart LLM instructions) and TOOLS (real-world actions).
Each action lists the expected JSON input schema. Follow it exactly.

--- AVAILABLE ACTIONS ---
{json.dumps(actions, indent=2)}
--- END AVAILABLE ACTIONS ---

--- WORKFLOW CONTEXT ---
{ctx_json}
--- END CONTEXT ---

Return a JSON object that matches this Pydantic schema:
{Plan.schema_json(indent=2)}
"""
        human_prompt = f'Task: "{task}"\nDecide your next step.'

        return [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=human_prompt),
        ]

    def _handle_agent_call(self, step: AgentCallStep, run_id: str, step_index: int):
        """Run a ReAct loop until finish or max_iterations."""
        logger.info(f"--- Agent Step: '{step.name}' ---")

        # derive task text: prefer step.task, else step.goal/intent to support new schema
        task = getattr(step, "task", None) or getattr(step, "goal", None) or getattr(step, "intent", None) or ""
        if not task:
            logger.warning(
                f"AgentCallStep '{step.name}' has no task/goal/intent text; defaulting to empty task."
            )

        agent_name = (
            step.agent_name
            if step.agent_name
            else (self.agent_config.name if self.agent_config else "unknown_agent")
        )
        logger.info(f"Agent: '{agent_name}' | Task: {task!r}")

        max_iterations = getattr(step, "max_iterations", 10) or 10
        history: List[BaseMessage] = []
        trace: List[Dict[str, Any]] = []
        available_actions = self._get_available_actions()
        logger.debug(f"Available actions: {list(available_actions.keys())}")

        for i in range(max_iterations):
            logger.info(f"[{step.name}] ReAct iteration {i + 1}/{max_iterations}")

            prompt = self._build_agent_prompt(task, available_actions, history)

            # Structured output attempt
            try:
                llm = self._get_llm()
                plan: Plan
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(Plan, method="function_calling")
                    plan = structured_llm.invoke(prompt)
                else:
                    # Fallback: raw call then parse
                    raw = llm.invoke(prompt)
                    plan = self._parse_plan_fallback(raw)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}", exc_info=True)
                self.context[step.name] = f"Agent failed: {e}"
                break

            logger.info(f"Thought: {plan.thought}")
            history.append(AIMessage(content=plan.json()))
            trace.append({"iteration": i + 1, "plan": json.loads(plan.json())})

            if plan.action == "finish":
                logger.info("Agent signaled finish.")
                self.context[step.name] = plan.action_input.get("final_answer", "Completed")
                break

            logger.info(f"Action: {plan.action} | Input: {plan.action_input}")

            try:
                if plan.action in self.skills:
                    result = self._execute_skill(self.skills[plan.action], plan.action_input)
                elif self.agent_config and plan.action in (self.agent_config.tools or []):
                    result = self._invoke_tool(plan.action, plan.action_input)
                else:
                    raise ValueError(
                        f"Unknown action '{plan.action}'. Valid: {list(available_actions.keys())}"
                    )

                logger.info(f"Result: {result}")
                history.append(HumanMessage(content=json.dumps({"result": result})))
                trace[-1]["result"] = result
                self.context[f"{plan.action}_result"] = result

            except Exception as e:
                err = f"Error executing action '{plan.action}': {e}"
                logger.error(err, exc_info=True)
                history.append(HumanMessage(content=json.dumps({"error": str(e)})))
                trace[-1]["error"] = str(e)

        else:
            logger.warning("Max iterations reached without finish.")
            if step.name not in self.context:
                self.context[step.name] = "Stopped (max iterations)."

        # store trace
        self.context[f"{step.name}_trace"] = trace
        logger.info(f"--- End Agent Step: '{step.name}' ---")

    def _invoke_tool(self, tool_name: str, params: Dict[str, Any]):
        tool_instance = self._get_tool(tool_name)
        # Support toolkits that are lists-of-tools
        if isinstance(tool_instance, list):
            # naive: call first; refine later
            return tool_instance[0].run(**params)
        # most tools implement run(...)
        if hasattr(tool_instance, "run"):
            return tool_instance.run(**params)
        raise RuntimeError(f"Tool '{tool_name}' has no runnable interface.")


    def _execute_skill(self, skill: Skill, skill_input: Dict[str, Any]) -> Any:
        """Executes a skill via focused LLM call."""
        logger.info(f"--- Executing Skill: {skill.name} ---")

        # pick instructions field
        instructions = getattr(skill, "instructions", None) or getattr(skill, "prompt_template", "")
        skill_prompt = f"""
You are an expert in '{skill.name}'.
Follow these instructions precisely:
{instructions}

Here is the input data:
{json.dumps(skill_input, indent=2)}
"""
        try:
            llm = self._get_llm()
            logger.info(f"Invoking LLM to execute skill '{skill.name}'...")
            resp = llm.invoke(skill_prompt)
            # unify message types
            if hasattr(resp, "content"):
                return resp.content
            return resp
        except Exception as e:
            logger.error(f"LLM invocation for skill '{skill.name}' failed: {e}", exc_info=True)
            return f"Error executing skill: {e}"


    def _handle_tool_call(self, step: ToolCallStep, run_id: str, step_index: int):
        logger.info(
            f"  - Tool step '{step.name}' calling '{step.tool_name}' with input: {step.tool_input}"
        )
        try:
            tool_instance = self._get_tool(step.tool_name)
            if isinstance(tool_instance, list):
                logger.warning("Executing the first tool in toolkit list.")
                result = tool_instance[0].run(**step.tool_input)
            else:
                result = tool_instance.run(**step.tool_input)
            self.context[step.name] = result
        except Exception as e:
            logger.error(f"Error executing tool '{step.tool_name}': {e}", exc_info=True)
            raise


    def _handle_if_then(self, step: IfThenStep, run_id: str, step_index: int):
        logger.info(f"  - Evaluating condition: '{step.condition}'")
        try:
            is_true = eval(step.condition, {"__builtins__": None}, self.context)
        except Exception as e:
            raise ValueError(
                f"Could not evaluate condition '{step.condition}': {e}"
            ) from e

        if is_true:
            logger.info("  - Condition True → running 'on_true' branch.")
            self._execute_steps(step.on_true, run_id=run_id)
        elif step.on_false:
            logger.info("  - Condition False → running 'on_false' branch.")
            self._execute_steps(step.on_false, run_id=run_id)


    def _handle_loop(self, step: LoopStep, run_id: str, step_index: int):
        iterable = self.context.get(step.iterable)
        if not isinstance(iterable, list):
            raise TypeError(
                f"Cannot loop. Context key '{step.iterable}' is not a list (got {type(iterable)})."
            )

        logger.info(f"  - Looping over {len(iterable)} items from '{step.iterable}'")
        for item in iterable:
            self.context[step.loop_variable] = item
            self._execute_steps(step.steps, run_id=run_id)

        if step.loop_variable in self.context:
            del self.context[step.loop_variable]
        logger.info("  - Loop completed.")


    def _handle_human_in_the_loop(
        self, step: HumanInTheLoopStep, run_id: str, step_index: int
    ):
        logger.info(f"  - Pausing workflow for human input: '{step.prompt}'")
        try:
            with self.db_engine.connect() as connection:
                run_db_id = self._get_run_db_id(connection, run_id)
                if not run_db_id:
                    raise RuntimeError(
                        f"Could not find database ID for run_id '{run_id}'"
                    )

                stmt = insert(human_tasks_table).values(
                    run_id=run_db_id,
                    task_name=step.name,
                    status="pending",
                    payload={"prompt": step.prompt, "data": step.data},
                    assigned_to_user_id=None,
                )
                connection.execute(stmt)

                stmt = (
                    update(workflow_runs_table)
                    .where(workflow_runs_table.c.run_id == run_id)
                    .values(
                        status="paused",
                        last_execution_state={
                            "step_index": step_index,
                            "context": self.context,
                        },
                    )
                )
                connection.execute(stmt)
                connection.commit()
        except Exception as e:
            logger.error(f"Failed to pause workflow '{run_id}': {e}", exc_info=True)
            raise
        raise RuntimeError("Workflow paused for human input.")


    def _parse_plan_fallback(self, raw_response: Any) -> Plan:
        """Attempt to coerce a raw LLM response into a Plan."""
        text = getattr(raw_response, "content", None) or str(raw_response)
        try:
            data = json.loads(text)
        except Exception:
            # naive parse
            data = {"thought": text, "action": "finish", "action_input": {"final_answer": text}}
        return Plan(**data)

    def _checkpoint(self, run_id: str, step_index: int):
        """Persist current context & step index for resumability."""
        try:
            with self.db_engine.connect() as connection:
                stmt = (
                    update(workflow_runs_table)
                    .where(workflow_runs_table.c.run_id == run_id)
                    .values(
                        last_execution_state={
                            "step_index": step_index,
                            "context": self.context,
                        }
                    )
                )
                connection.execute(stmt)
                connection.commit()
        except Exception as e:
            logger.warning(f"Checkpoint update failed for run '{run_id}': {e}")

    def _get_run_db_id(self, connection, run_id: str) -> Optional[int]:
        return connection.execute(
            select(workflow_runs_table.c.id).where(
                workflow_runs_table.c.run_id == run_id
            )
        ).scalar_one_or_none()

    def _update_run_status(
        self, run_id: str, status: str, final_output: Dict[str, Any]
    ):
        with self.db_engine.connect() as connection:
            stmt = (
                update(workflow_runs_table)
                .where(workflow_runs_table.c.run_id == run_id)
                .values(
                    status=status,
                    end_time=datetime.utcnow(),
                    final_output=final_output,
                )
            )
            connection.execute(stmt)
            connection.commit()
            logger.info(f"Workflow run '{run_id}' final status updated to '{status}'.")
