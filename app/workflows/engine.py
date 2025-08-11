# app/workflows/engine.py

# --- IMPORTANT NOTE ON SKILLS ---
# While this engine contains underlying code for managing and executing 'Skills'
# (formal capabilities defined in separate YAML files), these are currently
# considered **BETA features** and are not the primary recommended way for end-users
# to extend agent capabilities.
#
# The design currently prioritizes the agent's intrinsic LLM intelligence
# (driven by its robust system prompt and universal policies) to perform
# diverse tasks and process data directly within 'agent_call' workflow steps.
#
# Developers may choose to leverage explicit 'Skills' for highly specialized,
# deterministic, or complex integrations that go beyond basic LLM reasoning,
# but the default expectation is that capabilities are derived from the LLM's
# understanding of tasks and its available 'Tools'.
# Users are generally encouraged to rely on defining clear tasks within workflows,
# rather than creating separate skill definitions. Skills are still work in progress.
# -------------------------------

import logging
import json
import os
import uuid
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
import asyncio
import re
from app.tools.email_tool import EmailTool
from types import SimpleNamespace
import ast

from pydantic import ValidationError, PrivateAttr
import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools.base import ToolException

from sqlalchemy import select, update, insert  # Ensure 'insert' is imported
from langchain_core.exceptions import OutputParserException
from app.schemas.orchestrator import OrchestratorConfig
from langchain_core.tools import BaseTool

import httpx
import openai

from app.agents.config_manager import load_agent_config
from app.core.config import get_path_settings, get_llm_model
from app.db.models import human_tasks_table, workflow_runs_table, \
    workflows_table  # Ensure workflow_runs_table is imported
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
    Plan,
    HierarchicalPlan,
    StepPlan
)
from app.tools.parsers.code_parser_tool import CodeParserTool
from app.tools.parsers.notebook_parser_tool import NotebookParserTool
from app.tools.parsers.sql_parser_tool import SQLParserTool
from app.tools.parsers.yaml_parser_tool import YAMLParserTool
from app.tools.retriever_tool import get_retriever_tool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.python_script_tool import PythonScriptTool
from app.tools.search_engine_tool import SearchTool
from app.tools.sql_tool import create_sql_toolkit
from app.tools.parsers.terraform_parser_tool import TerraformParserTool

logger = logging.getLogger(__name__)
_APP_PATHS = get_path_settings()



class AgentTool(BaseTool):
    name: str
    description: str
    sub_agent_name: str

    # private (non‐model) attribute must be sundered
    _engine: Any = PrivateAttr()

    def __init__(self, *, engine: Any, **data):
        super().__init__(**data)
        # stash the engine privately
        object.__setattr__(self, "_engine", engine)

    def _run(self, **kwargs) -> Any:
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> Any:
        # This method is correct and already uses 'await'
        orchestrator_context = self._engine.context
        sub_run_input = {"task": kwargs.get("task", ""), **orchestrator_context}
        parent_run_id = kwargs.pop("parent_run_id", None)
        # Here, we 'await' the call to the now-async run_workflow
        return await self._engine.run_workflow(
            workflow_name   = self.sub_agent_name,
            initial_input   = sub_run_input,
            is_sub_run      = True,
            parent_run_id   = parent_run_id,
            user_id=self._engine.user_id
        )




def _to_namespace(obj):
    """
    Recursively convert dicts/lists into SimpleNamespace (allowing dot access),
    and leave other types alone.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _safe_eval_condition(condition: str, context: Dict[str, Any]) -> bool:
    """
    Safely evaluate simple boolean expressions like "foo.is_urgent == True".
    Automatically parses JSON strings and wraps dicts in namespaces.
    """
    # only allow alphanum, dot, quotes, operators, spaces, parens
    if not re.match(r"^[ a-zA-Z0-9_.'\"<>=!&|()]+$", condition):
        raise ValueError(f"Condition contains invalid characters: {condition}")

    # extract variable names
    names = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', condition)
    safe_locals: Dict[str, Any] = {}

    for name in names:
        raw = context.get(name, None)

        # if it's a string that looks like JSON, try parsing
        if isinstance(raw, str):
            s = raw.strip()
            if s.startswith('{') and s.endswith('}'):
                try:
                    raw = json.loads(s)
                except json.JSONDecodeError:
                    pass

        # wrap dict/list into namespace for attribute access
        if isinstance(raw, (dict, list)):
            raw = _to_namespace(raw)

        safe_locals[name] = raw

    try:
        return bool(eval(condition, {"__builtins__": {}}, safe_locals))
    except Exception as e:
        raise ValueError(f"Could not safely evaluate condition '{condition}': {e}") from e


def _serialize_for_db(data: Any) -> Any:
    if isinstance(data, Document):
        return {"page_content": data.page_content, "metadata": data.metadata}
    if isinstance(data, list):
        return [_serialize_for_db(item) for item in data]
    if isinstance(data, dict):
        return {key: _serialize_for_db(value) for key, value in data.items()}
    return data


class WorkflowEngine:
    MAX_REACT_ITERATIONS = 10
    BASE_BACKOFF_SECONDS = 1.0  # Initial delay for exponential backoff
    MAX_BACKOFF_SECONDS = 60.0  # Maximum delay for exponential backoff

    # Universal Tool Policy to be injected into the system prompt
    UNIVERSAL_TOOL_POLICY = """
    --- UNIVERSAL TOOL POLICY ---
    1. Your plan must be a valid JSON object matching the `HierarchicalPlan` schema.
    2. Pay close attention to the `plan` field, which is a list of sequential steps.
    3. Each step within the `plan` is a JSON object with `action` and `action_input`.
        • The `action` is the name of a tool, skill, or sub-agent.
        • The `action_input` MUST be a JSON object with keys that exactly match the parameters of the chosen action.
        • All **required** parameters for a given action must be present.
    4. When a task is fully and completely met by the results of your planned steps, the final step **MUST** use the `"finish"` action.
        • The `action_input` for the `"finish"` action must contain a concise and direct `final_answer`. Remove any API metadata, thoughts, or extra commentary.
    5. Continue planning only if further actions are strictly necessary to achieve the user's objective. Do not generate extraneous steps.
    6. If you encounter an error or cannot fulfill the objective, explain clearly to the user why and suggest what they might do next.
    --- END UNIVERSAL TOOL POLICY ---
    """

    def __init__(self, db_engine: Any, user_id: Optional[int] = None, run_id: Optional[str] = None):
        self.db_engine = db_engine
        self.context: Dict[str, Any] = {}
        self.agent_config: Optional[AgentConfig] = None
        self.tools: Dict[str, Any] = {}
        self.skills: Dict[str, Skill] = {}
        self.llm: Optional[Any] = None
        self.agent_tools: Dict[str, AgentTool] = {}
        self.step_handlers = {
            StepType.AGENT_CALL: self._handle_agent_call,
            StepType.TOOL_CALL: self._handle_tool_call,
            StepType.IF_THEN: self._handle_if_then,
            StepType.LOOP: self._handle_loop,
            StepType.HUMAN_IN_THE_LOOP: self._handle_human_in_the_loop,
        }

        self.user_id = user_id if user_id is not None else 0
        self.run_id = run_id

    def _get_llm(self) -> Any:
        if self.llm:
            return self.llm
        if not self.agent_config:
            raise ValueError("Agent configuration is not loaded.")
        model_name = self.agent_config.llm_model
        model_params = self.agent_config.model_params
        self.llm = get_llm_model(
            model_name=model_name,
            model_params=model_params,
            retries=model_params.llm_retries if model_params and model_params.llm_retries is not None else 0,
            timeout=model_params.llm_timeout if model_params and model_params.llm_timeout is not None else 60,
        )
        return self.llm

    def _parse_hierarchical_plan_fallback(self, raw_response: Any) -> HierarchicalPlan:
        """Fallback to parse LLM output into a HierarchicalPlan."""
        text = getattr(raw_response, "content", "") or str(raw_response)
        logger.debug(f"Attempting fallback parse of raw LLM response: {text[:200]}...")

        try:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})', text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.group(1) else match.group(2)
                data = json.loads(json_str)
                logger.debug("Successfully extracted JSON block for fallback plan parsing.")
                return HierarchicalPlan(**data)
            else:
                data = json.loads(text)
                logger.debug("Successfully parsed raw text as JSON for fallback plan parsing.")
                return HierarchicalPlan(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Fallback JSON parsing failed: {e}. Raw output: {text[:200]}...", exc_info=True)
            return HierarchicalPlan(thought=f"Could not parse LLM output: {e}. Raw output: {text[:100]}...", plan=[
                StepPlan(action="finish", action_input={"final_answer": "Error: Could not understand agent's plan."})])
        except Exception as e:
            logger.error(f"Unexpected error during fallback plan parsing: {e}", exc_info=True)
            return HierarchicalPlan(thought=f"Unexpected error parsing LLM output: {e}", plan=[
                StepPlan(action="finish", action_input={"final_answer": "Error: Agent internal parsing failed."})])

    async def run_workflow(
            self,
            workflow_name: str,
            initial_input: Optional[Dict[str, Any]] = None,
            resume_run_id: Optional[str] = None,
            user_id: Optional[int] = None,
            is_sub_run: bool = False,
            parent_run_id: Optional[str] = None
    ):
        if is_sub_run and workflow_name == self.agent_config.name:
            raise ValueError("Recursive call detected: An orchestrator cannot call itself as a sub-agent.")
        if is_sub_run and initial_input:
            self.context.update(initial_input)

        logger.info(f"Starting or resuming workflow run for: {workflow_name} (User ID: {user_id or 'N/A'})")
        with self.db_engine.connect() as connection:
            row = connection.execute(
                select(workflows_table).where(workflows_table.c.name == workflow_name)
            ).fetchone()
            if not row:
                logger.error(f"Workflow '{workflow_name}' not found in DB.")
                return
            try:
                definition = (
                    json.loads(row.definition)
                    if isinstance(row.definition, str)
                    else row.definition
                )
                definition["id"] = row.id
                workflow = Workflow.model_validate(definition)
            except Exception as e:
                logger.error(
                    f"Failed to load or validate workflow '{workflow_name}': {e}",
                    exc_info=True,
                )
                return

            if workflow.agent_name:
                try:
                    config_path = os.path.join(_APP_PATHS["AGENTS_DIR"], f"{workflow.agent_name}.yaml")
                    with open(config_path, 'r') as f:
                        yaml_content = yaml.safe_load(f)
                    try:
                        self.agent_config = OrchestratorConfig.model_validate(yaml_content)
                        logger.info(f"Loaded OrchestratorConfig for agent: '{workflow.agent_name}'")
                    except ValidationError:
                        self.agent_config = AgentConfig.model_validate(yaml_content)
                        logger.info(f"Loaded generic AgentConfig for agent: '{workflow.agent_name}'")

                    self._load_skills()
                    logger.info(
                        f"Loaded config and skills for agent: '{workflow.agent_name}'"
                    )
                except FileNotFoundError:
                    logger.error(f"Config for agent '{workflow.agent_name}' not found.")
                    self._update_run_status(
                        resume_run_id or str(uuid.uuid4()),
                        "failed",
                        {"error": f"Agent '{workflow.agent_name}' not found."},
                    )
                    return

        run_id = resume_run_id or str(uuid.uuid4())
        self.run_id = run_id
        self.user_id = user_id
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
                connection.execute(
                    update(workflow_runs_table)
                    .where(workflow_runs_table.c.run_id == run_id)
                    .values(status="running")
                )
                connection.commit()
        elif not is_sub_run:
            logger.info(f"Starting new workflow run: {run_id}")
            self.context = {"trigger": initial_input or {}}
            try:
                with self.db_engine.connect() as connection:
                    # CORRECTED LINE: Removed the named argument 'insert_stmt'
                    connection.execute(
                        insert(workflow_runs_table).values(
                            run_id=run_id,
                            workflow_id=workflow.id,
                            status="running",
                            initial_input=initial_input,
                            start_time=datetime.utcnow(),
                            user_id=user_id,
                            parent_run_id=parent_run_id,
                        )
                    )
                    connection.commit()
            except Exception as e:
                logger.error(f"Failed to record workflow start: {e}", exc_info=True)
                return

        try:
            await self._execute_steps(
                workflow.steps, start_index=start_step_index, run_id=run_id
            )
            logger.info(f"Workflow run '{run_id}' completed successfully.")
            self._update_run_status(run_id, "completed", self.context)
            return self.context
        except Exception as e:
            logger.error(
                f"Workflow run '{run_id}' failed with unexpected error: {e}",
                exc_info=True,
            )
            self._update_run_status(run_id, "failed", {"error": str(e)})
            return {"error": str(e)}

    async def _execute_steps(
            self, steps: Sequence[WorkflowStep], start_index: int = 0, run_id: str = ""
    ):
        for i in range(start_index, len(steps)):
            step = steps[i]
            handler = self.step_handlers.get(step.type)
            if handler:
                logger.info(f"Executing step '{step.name}' (Type: {step.type})")
                await handler(step, run_id, i)
                self._checkpoint(run_id, i)
            else:
                logger.warning(
                    f"Skipping unsupported step type: {step.type} for step '{step.name}'"
                )

    async def _get_tool(self, tool_name: str) -> Any:
        """Lazy-init and cache tools."""
        if tool_name in self.tools:
            return self.tools[tool_name]

        tool = None
        if self.agent_config:
            if tool_name == "retriever":
                tool = await get_retriever_tool(
                    agent_config=self.agent_config,
                    user_id=self.user_id,
                    thread_id=self.run_id
                )
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

        elif tool_name == "python_script_tool":
            tool = PythonScriptTool()
        elif tool_name == "terraform_parser_tool":
            tool = TerraformParserTool()
        elif tool_name == "sql_parser_tool":
            tool = SQLParserTool()
        elif tool_name == "code_parser_tool":
            tool = CodeParserTool()
        elif tool_name == "yaml_parser_tool":
            tool = YAMLParserTool()
        elif tool_name == "email_tool":
            tool = EmailTool()
        elif tool_name == "notebook_parser_tool":
            tool = NotebookParserTool()

        if not tool:
            raise ValueError(
                f"Tool '{tool_name}' is not recognized or could not be initialized."
            )

        self.tools[tool_name] = tool
        return tool

    def _load_skills(self):
        skills_dir = _APP_PATHS.get("SKILLS_DIR")
        if not skills_dir or not os.path.isdir(skills_dir):
            logger.debug(
                f"Skills directory not found at {skills_dir}. No skills loaded."
            )
            return
        for filename in os.listdir(skills_dir):
            if filename.endswith((".yaml", ".yml")):
                try:
                    with open(os.path.join(skills_dir, filename), "r") as f:
                        skill = Skill.model_validate(yaml.safe_load(f))
                        self.skills[skill.name] = skill
                        logger.info(f"Loaded skill: '{skill.name}'")
                except Exception as e:
                    logger.error(
                        f"Failed to load skill '{filename}': {e}", exc_info=True
                    )

    async def _get_available_actions(self) -> Dict[str, Dict[str, Any]]:

        actions: Dict[str, Dict[str, Any]] = {}

        for name, skill in self.skills.items():
            actions[name] = {
                "type": "skill",
                "description": skill.description,
                "parameters": skill.parameters
            }

        if isinstance(self.agent_config, OrchestratorConfig) and self.agent_config.roster:
            for sub_agent_name in self.agent_config.roster:
                try:
                    workflow_path = _APP_PATHS["WORKFLOWS_DIR"] / f"{sub_agent_name}.yaml"
                    with open(workflow_path, 'r') as f:
                        workflow_content = yaml.safe_load(f)

                    sub_agent_config = load_agent_config(workflow_content.get("agent_name"))

                    actions[sub_agent_name] = {
                        "type": "sub_agent",
                        "description": sub_agent_config.description or f"Executes the '{sub_agent_name}' workflow.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string",
                                         "description": "The task for the sub-agent's workflow to perform."}},
                            "required": ["task"]
                        }
                    }
                except Exception as e:
                    logger.warning(f"Could not load sub-agent '{sub_agent_name}' for orchestrator tool: {e}")

        if self.agent_config and self.agent_config.tools:
            for tool_name in self.agent_config.tools:
                try:
                    tool = await self._get_tool(tool_name)
                    tool_params = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
                    actions[tool_name] = {
                        "type": "tool",
                        "description": tool.description,
                        "parameters": tool_params
                    }
                except Exception as e:
                    logger.warning(
                        f"Could not get detailed description for tool '{tool_name}': {e}"
                    )
                    actions[tool_name] = {
                        "type": "tool",
                        "description": f"A tool named {tool_name}.",
                        "parameters": {}
                    }
        actions["finish"] = {
            "type": "special",
            "description": "Use this action when the task is fully complete. Provide a 'final_answer' in action_input.",
            "parameters": {"type": "object", "properties": {"final_answer": {"type": "string"}},
                           "required": ["final_answer"]}
        }
        return actions

    def _build_agent_prompt(
            self, task: str, actions: Dict[str, Dict[str, Any]], history: List[BaseMessage]
    ) -> List[BaseMessage]:
        ctx_json = json.dumps(self.context, indent=2, default=str)
        if len(ctx_json) > 8000:
            ctx_json = ctx_json[:8000] + "..."

        action_details_str = ""
        for action_name, details in actions.items():
            action_details_str += f"\n--- {details['type'].upper()}: {action_name} ---\n"
            action_details_str += f"Description: {details['description']}\n"
            if details.get('parameters'):
                action_details_str += f"Input Parameters (JSON Schema): {json.dumps(details['parameters'], indent=2)}\n"
                if 'required' in details['parameters'] and details['parameters']['required']:
                    action_details_str += f"NOTE: For this action, you MUST provide these required parameters: {', '.join(details['parameters']['required'])}.\n"

        system_prompt = f"""You are a helpful AI agent. Your task is to complete the objective by creating a detailed, multi-step plan.
        {self.UNIVERSAL_TOOL_POLICY}

        --- AVAILABLE ACTIONS AND THEIR PARAMETERS ---
        {action_details_str}
        --- END AVAILABLE ACTIONS ---

        --- WORKFLOW CONTEXT (Results from previous steps) ---
        {ctx_json}
        --- END CONTEXT ---

        **Execution Guidelines:**
        - Your response MUST be a JSON object with a `thought` field and a `plan` field.
        - The `plan` field MUST be an ordered list of steps. Each step can either be a single action (a JSON object) or a list of actions (a list of JSON objects) to be run in parallel.
        - Use a list of actions `[action1, action2, ...]` to indicate that actions can be performed at the same time. This is a powerful feature and should be used whenever a single task requires multiple, independent actions.
        - Your final planned step MUST be the "finish" action, with a `final_answer` in its `action_input`.

        Return a JSON object matching this schema: {HierarchicalPlan.schema_json(indent=2)}"""

        return [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=f'Objective: "{task}"'),
        ]

    async def _handle_agent_call(self, step: AgentCallStep, run_id: str, step_index: int):
        logger.info(f"--- Agent Step: '{step.name}' ---")

        task_template = getattr(step, "task", "")
        context_map = {k: _to_namespace(v) for k, v in self.context.items()}
        try:
            task = task_template.format(**context_map)
        except KeyError as e:
            logger.warning(
                f"Could not format task for step '{step.name}'. Missing key: {e}. Using raw task."
            )
            task = task_template

        agent_name = step.agent_name or (
            self.agent_config.name if self.agent_config else "unknown_agent"
        )
        logger.info(f"Agent: '{agent_name}' | Task: {task!r}")

        max_iterations = getattr(step, "max_iterations", self.MAX_REACT_ITERATIONS)
        history: List[BaseMessage] = []
        trace: List[Dict[str, Any]] = []
        available_actions = await self._get_available_actions()

        hierarchical_plan: Optional[HierarchicalPlan] = None

        for i in range(max_iterations):
            logger.debug(f"[{step.name}] ReAct iteration {i + 1}/{max_iterations}")

            # We need to rebuild the prompt before every call to ensure the history is included
            prompt = self._build_agent_prompt(task, available_actions, history)

            llm_max_retries = self.agent_config.model_params.llm_retries if self.agent_config.model_params and self.agent_config.model_params.llm_retries is not None else 3
            for api_retry_count in range(llm_max_retries + 1):
                try:
                    llm = self._get_llm()
                    if hasattr(llm, "with_structured_output"):
                        structured_llm = llm.with_structured_output(HierarchicalPlan, method="function_calling")
                        try:
                            hierarchical_plan = await structured_llm.ainvoke(prompt)
                            if not isinstance(hierarchical_plan.plan, list):
                                raise ValidationError(
                                    [{"loc": ("plan",), "msg": "value is not a valid list", "type": "type_error.list"}],
                                    HierarchicalPlan)
                            logger.info(f"Thought: {hierarchical_plan.thought}")
                            break
                        except (httpx.RequestError, openai.APIConnectionError) as e:
                            logger.debug(
                                f"Connection error during structured LLM call (ReAct iter {i + 1}, api_retry {api_retry_count + 1}/{llm_max_retries}): {e}. Retrying...")
                            if api_retry_count < llm_max_retries:
                                await asyncio.sleep(
                                    min(self.BASE_BACKOFF_SECONDS * (2 ** api_retry_count), self.MAX_BACKOFF_SECONDS))
                            continue
                        except (ValidationError, OutputParserException) as e:
                            logger.error(
                                f"LLM structured output validation/parsing failed (ReAct iter {i + 1}, api_retry {api_retry_count}): {e}")
                            try:
                                raw_llm_output = await llm.ainvoke(prompt)
                                hierarchical_plan = self._parse_hierarchical_plan_fallback(raw_llm_output)
                                if not isinstance(hierarchical_plan.plan, list):
                                    raise ValidationError([{"loc": ("plan",), "msg": "value is not a valid list",
                                                            "type": "type_error.list"}], HierarchicalPlan)
                                logger.info(f"Fallback successful. Thought: {hierarchical_plan.thought}")
                                break
                            except Exception as fallback_e:
                                logger.error(
                                    f"Fallback parsing also failed (ReAct iter {i + 1}, api_retry {api_retry_count}): {fallback_e}",
                                    exc_info=True)
                                break
                    else:
                        try:
                            raw = await llm.ainvoke(prompt)
                            hierarchical_plan = self._parse_hierarchical_plan_fallback(raw)
                            if not isinstance(hierarchical_plan.plan, list):
                                raise ValidationError(
                                    [{"loc": ("plan",), "msg": "value is not a valid list", "type": "type_error.list"}],
                                    HierarchicalPlan)
                            logger.info(f"Fallback successful. Thought: {hierarchical_plan.thought}")
                            break
                        except (httpx.RequestError, openai.APIConnectionError) as conn_err:
                            logger.debug(
                                f"Connection error during LLM call (ReAct iter {i + 1}, api_retry {api_retry_count + 1}/{llm_max_retries}): {conn_err}. Retrying...")
                            if api_retry_count < llm_max_retries:
                                await asyncio.sleep(
                                    min(self.BASE_BACKOFF_SECONDS * (2 ** api_retry_count), self.MAX_BACKOFF_SECONDS))
                            continue
                        except Exception as e:
                            logger.error(
                                f"An unexpected error occurred during agent call without structured output (ReAct iter {i + 1}, api_retry {api_retry_count}): {e}",
                                exc_info=True)
                            break
                except Exception as e:
                    logger.error(
                        f"An unexpected non-connection/non-parsing error occurred during agent call (ReAct iter {i + 1}, api_retry {api_retry_count}): {e}",
                        exc_info=True)
                    break
            if hierarchical_plan:
                break
            else:
                logger.warning(
                    f"No valid plan obtained from LLM after all attempts in ReAct iteration {i + 1}. Moving to next iteration (if allowed).")

        if hierarchical_plan is None:
            raise RuntimeError(
                f"Failed to get a valid plan after {max_iterations} ReAct iterations, possibly due to persistent issues with the LLM API or its output format.")

        # Process each step in the plan
        for planned_step in hierarchical_plan.plan:
            action_type = available_actions.get(planned_step.action, {}).get("type")

            if planned_step.action == "finish":
                logger.info("Agent signaled finish.")
                ai_input = planned_step.action_input
                final = ai_input.get("final_answer", "")
                parsed = None
                if isinstance(final, dict):
                    parsed = final
                elif isinstance(final, str):
                    try:
                        parsed = json.loads(final)
                    except json.JSONDecodeError:
                        try:
                            tmp = ast.literal_eval(final)
                            if isinstance(tmp, dict):
                                parsed = tmp
                        except Exception:
                            parsed = None
                self.context[step.name] = parsed if isinstance(parsed, dict) else final
                # A 'finish' action stops the entire plan execution.
                return

            elif action_type == "sub_agent":
                logger.info(f"Orchestrator calling sub-agent: '{planned_step.action}'")
                sub_agent_tool = AgentTool(
                    name=planned_step.action,
                    description=available_actions[planned_step.action]["description"],
                    engine=self,
                    sub_agent_name=planned_step.action
                )
                sub_agent_input = planned_step.action_input.get("task", planned_step.action_input)
                try:
                    # Execute the sub-agent and get its final context.
                    sub_agent_final_context = await sub_agent_tool._arun(
                        task=sub_agent_input,
                        parent_run_id=run_id
                    )
                    self.context.update(sub_agent_final_context)

                    result = sub_agent_final_context.get(planned_step.action, {}).get('final_answer') or json.dumps(
                        sub_agent_final_context)

                    serialized_result = _serialize_for_db(result)
                    logger.debug(f"Sub-agent result merged into context: {str(sub_agent_final_context)[:500]}")
                    history.append(
                        ToolMessage(content=json.dumps({"result": serialized_result}), tool_name=planned_step.action,
                                    tool_call_id=str(uuid.uuid4()))
                    )
                    trace.append({"iteration": i + 1, "plan": planned_step.model_dump(), "result": serialized_result})
                    self.context[f"{step.name}_{planned_step.action}_result"] = serialized_result
                except Exception as e:
                    err = f"Error executing sub-agent '{planned_step.action}': {e}"
                    logger.error(err, exc_info=True)
                    history.append(HumanMessage(content=json.dumps({"error": str(e)})))
                    trace.append({"iteration": i + 1, "plan": planned_step.model_dump(), "error": str(e)})

            elif planned_step.action in self.skills:
                try:
                    result = await self._execute_skill(
                        self.skills[planned_step.action], planned_step.action_input
                    )
                    serialized_result = _serialize_for_db(result)
                    logger.debug(f"Result: {str(serialized_result)[:500]}")
                    history.append(
                        ToolMessage(content=json.dumps({"result": serialized_result}), tool_name=planned_step.action,
                                    tool_call_id=str(uuid.uuid4()))
                    )
                    trace.append({"iteration": i + 1, "plan": planned_step.model_dump(), "result": serialized_result})
                    self.context[f"{step.name}_{planned_step.action}_result"] = serialized_result
                except Exception as e:
                    err = f"Error executing action '{planned_step.action}': {e}"
                    logger.error(err, exc_info=True)
                    history.append(HumanMessage(content=json.dumps(
                        {"error": str(e)})))
                    trace.append({"iteration": i + 1, "plan": planned_step.model_dump(), "error": str(e)})
            elif self.agent_config and planned_step.action in (self.agent_config.tools or []):
                try:
                    result = await self._invoke_tool(planned_step.action, planned_step.action_input)
                    serialized_result = _serialize_for_db(result)
                    logger.debug(f"Result: {str(serialized_result)[:500]}")
                    history.append(
                        ToolMessage(content=json.dumps({"result": serialized_result}), tool_name=planned_step.action,
                                    tool_call_id=str(uuid.uuid4()))
                    )
                    trace.append({"iteration": i + 1, "plan": planned_step.model_dump(), "result": serialized_result})
                    self.context[f"{step.name}_{planned_step.action}_result"] = serialized_result
                except Exception as e:
                    err = f"Error executing action '{planned_step.action}': {e}"
                    logger.error(err, exc_info=True)
                    history.append(HumanMessage(content=json.dumps(
                        {"error": str(e)})))
                    trace.append({"iteration": i + 1, "plan": planned_step.model_dump(), "error": str(e)})
            else:
                raise ValueError(
                    f"Unknown action '{planned_step.action}'. Valid actions are: {list(available_actions.keys())}"
                )

        self.context[f"{step.name}_trace"] = trace
        logger.info(f"--- End Agent Step: '{step.name}' ---")

    async def _invoke_tool(self, tool_name: str, params: Dict[str, Any]):
        tool = await self._get_tool(tool_name)
        # Check if tool is the fallback error tool
        if hasattr(tool, 'name') and tool.name == 'retriever_error':
            error_message = tool.run(
                params)  # This was the old error. Tool.from_function creates a func(input) not func(**kwargs)
            # But even better, this fallback path should be removed entirely
            # by fixing the user_id problem.
            raise ValueError(error_message)

        # REVISED LOGIC: Prioritize the async method if it's a coroutine.
        if hasattr(tool, '_arun') and asyncio.iscoroutinefunction(tool._arun):
            try:
                return await tool._arun(**params)
            except (TypeError, ToolException) as e:
                logger.warning(f"Async tool '{tool_name}' failed with **kwargs. Retrying with single arg. Error: {e}")
                return await tool._arun(params)
        elif hasattr(tool, '_run') and not asyncio.iscoroutinefunction(tool._run):
            try:
                # For synchronous tools, we can run them in a separate thread.
                return await asyncio.to_thread(tool._run, **params)
            except (TypeError, ToolException) as e:
                logger.warning(f"Sync tool '{tool_name}' failed with **kwargs. Retrying with single arg. Error: {e}")
                return await asyncio.to_thread(tool._run, params)
        else:
            # Fallback to the original run method (if neither _run nor _arun exist)
            try:
                return await asyncio.to_thread(tool.run, **params)
            except (TypeError, ToolException) as e:
                return await asyncio.to_thread(tool.run, params)
                return tool.run(params)

    async def _execute_skill(self, skill: Skill, skill_input: Dict[str, Any]) -> Any:
        logger.info(f"--- Executing Skill: {skill.name} ---")
        prompt = f"Instructions for '{skill.name}':\n{skill.instructions}\nInput data:\n{json.dumps(skill_input, indent=2)}"
        return (await self._get_llm().ainvoke(prompt)).content

    async def _handle_tool_call(self, step: ToolCallStep, run_id: str, step_index: int):
        logger.info(
            f"  - Tool step '{step.name}' calling '{step.tool_name}' with original input: {step.tool_input}"
        )
        tool_instance = await self._get_tool(step.tool_name)

        format_eval_locals = {}
        for key, value in self.context.items():
            format_eval_locals[key] = _to_namespace(value)

        formatted_tool_input = {}
        for key, value in step.tool_input.items():
            if isinstance(value, str):
                try:
                    formatted_tool_input[key] = eval(f"f'''{value}'''", {"__builtins__": {}}, format_eval_locals)
                except Exception as e:
                    logger.warning(
                        f"Failed to format tool input '{key}' using f-string evaluation: {e}. Using raw string.")
                    formatted_tool_input[key] = value
            else:
                formatted_tool_input[key] = value

        logger.info(f"  - Tool step '{step.name}' sending formatted input: {formatted_tool_input}")

        try:
            if asyncio.iscoroutinefunction(tool_instance.run):
                try:
                    result = await tool_instance.run(**formatted_tool_input)
                except TypeError:
                    result = await tool_instance.run(formatted_tool_input)
            else:
                try:
                    result = tool_instance.run(**formatted_tool_input)
                except TypeError:
                    result = tool_instance.run(formatted_tool_input)

            if isinstance(result, list) and result and isinstance(result[0], Document):
                logger.debug(
                    f"Retriever returned {len(result)} documents. Consolidating into a single text block."
                )
                result = "\n\n".join(doc.page_content for doc in result)

            self.context[step.name] = _serialize_for_db(result)

        except Exception as e:
            logger.error(f"Error executing tool '{step.tool_name}': {e}", exc_info=True)
            raise

    async def _handle_if_then(self, step: IfThenStep, run_id: str, step_index: int):
        try:
            is_true = _safe_eval_condition(step.condition, self.context)
        except Exception as e:
            raise ValueError(f"Could not evaluate condition '{step.condition}': {e}") from e
        branch_to_run = step.on_true if is_true else step.on_false
        if branch_to_run:
            logger.info(f"Condition was {is_true}, running branch.")
            await self._execute_steps(branch_to_run, run_id=run_id)

    async def _handle_loop(self, step: LoopStep, run_id: str, step_index: int):
        iterable = self.context.get(step.iterable)
        if not isinstance(iterable, list):
            raise TypeError(f"'{step.iterable}' is not a list.")
        logger.info(f"Looping over {len(iterable)} items from '{step.iterable}'")
        for item in iterable:
            self.context[step.loop_variable] = item
            await self._execute_steps(step.steps, run_id=run_id)

        if step.loop_variable in self.context:
            del self.context[step.loop_variable]

    async def _handle_human_in_the_loop(
            self, step: HumanInTheLoopStep, run_id: str, step_index: int
    ):
        logger.info(f"Pausing for human input: '{step.prompt}'")
        try:
            with self.db_engine.connect() as connection:
                run_db_id = self._get_run_db_id(connection, run_id)
                if not run_db_id:
                    raise RuntimeError(f"Could not find run ID '{run_id}'")
                connection.execute(
                    insert(human_tasks_table).values(
                        run_id=run_db_id,
                        task_name=step.name,
                        status="pending",
                        payload={"prompt": step.prompt, "data": step.data},
                    )
                )
                connection.execute(
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
                connection.commit()
        except Exception as e:
            logger.error(f"Failed to pause workflow '{run_id}': {e}", exc_info=True)
            raise
        raise RuntimeError("Workflow paused for human input.")

    def _parse_plan_fallback(self, raw_response: Any) -> Plan:
        text = getattr(raw_response, "content", "") or str(raw_response)
        logger.debug(f"Attempting fallback parse of raw LLM response: {text[:200]}...")

        try:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})', text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.group(1) else match.group(2)
                data = json.loads(json_str)
                logger.debug(f"Successfully extracted JSON block for fallback plan parsing.")
                return Plan(**data)
            else:
                data = json.loads(text)
                logger.debug(f"Successfully parsed raw text as JSON for fallback plan parsing.")
                return Plan(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Fallback JSON parsing failed: {e}. Raw output: {text[:200]}...", exc_info=True)
            return Plan(thought=f"Could not parse LLM output: {e}. Raw output: {text[:100]}...", action="finish",
                        action_input={"final_answer": "Error: Could not understand agent's plan."})
        except Exception as e:
            logger.error(f"Unexpected error during fallback plan parsing: {e}", exc_info=True)
            return Plan(thought=f"Unexpected error parsing LLM output: {e}", action="finish",
                        action_input={"final_answer": "Error: Agent internal parsing failed."})

    def _checkpoint(self, run_id: str, step_index: int):
        try:
            with self.db_engine.connect() as connection:
                connection.execute(
                    update(workflow_runs_table)
                    .where(workflow_runs_table.c.run_id == run_id)
                    .values(
                        last_execution_state={
                            "step_index": step_index,
                            "context": _serialize_for_db(self.context),
                        }
                    )
                )
                connection.commit()
        except Exception as e:
            logger.warning(f"Checkpoint failed for run '{run_id}': {e}")

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
            connection.execute(
                update(workflow_runs_table)
                .where(workflow_runs_table.c.run_id == run_id)
                .values(
                    status=status,
                    end_time=datetime.utcnow(),
                    final_output=_serialize_for_db(final_output),
                )
            )
            connection.commit()
            logger.info(f"Workflow run '{run_id}' status updated to '{status}'.")