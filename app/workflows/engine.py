# app/workflows/engine.py

import logging
import json
import os
import uuid
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import asyncio
import re

from pydantic.v1 import BaseModel, Field as PydanticField, ValidationError
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from sqlalchemy import select, update, insert
from langchain_core.runnables import Runnable
from langchain_core.exceptions import OutputParserException

# ADDED IMPORTS FOR GENERIC RETRY LOGIC
import httpx
import openai

from app.agents.config_manager import load_agent_config
from app.core.config import get_api_key, get_path_settings, get_llm_model
from app.db.models import human_tasks_table, workflow_runs_table, workflows_table
from app.schemas.agent import AgentConfig
from app.schemas.workflow import (
    AgentCallStep,
    HumanInTheLoopStep,  # Corrected from HumanIn_the_LoopStep
    IfThenStep,
    LoopStep,
    Skill,
    StepType,
    ToolCallStep,
    Workflow,
    WorkflowStep,
    Plan
)
from app.tools.parsers.code_parser_tool import CodeParserTool
from app.tools.parsers.notebook_parser_tool import NotebookParserTool
from app.tools.parsers.sql_parser_tool import SQLParserTool
from app.tools.parsers.yaml_parser_tool import YAMLParserTool
from app.tools.retriever_tool import get_retriever_tool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.http_request_tool import HTTPRequestTool
from app.tools.python_script_tool import PythonScriptTool
from app.tools.search_engine_tool import SearchTool
from app.tools.slack_webhook_tool import SlackWebhookTool
from app.tools.sql_tool import create_sql_toolkit
from app.tools.parsers.terraform_parser_tool import TerraformParserTool

logger = logging.getLogger(__name__)
_APP_PATHS = get_path_settings()


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
    # Added constants for backoff strategy
    BASE_BACKOFF_SECONDS = 1.0  # Initial delay for exponential backoff
    MAX_BACKOFF_SECONDS = 60.0  # Maximum delay for exponential backoff

    # Universal Tool Policy to be injected into the system prompt
    UNIVERSAL_TOOL_POLICY = """
    --- UNIVERSAL TOOL POLICY ---
    1. Your action must be a valid JSON object matching the `Plan` schema.
    2. Pay close attention to the `action_input` field:
        • It MUST be a JSON object.
        • Keys must exactly match the parameter names in the “Input Parameters” for your chosen action.
        • All **required** parameters must be present.
    3. When using a [TOOL] action:
        • Select the tool that is most relevant and efficient for the current sub-objective.
        • Fill in all required parameters. If a parameter can be safely defaulted or inferred, inject it.
        • ALWAYS process the `result` from a tool. Do NOT assume the tool's return is the final answer without processing.
    4. As soon as the user's objective is **fully and completely met** by processed information or a tool's result, you **MUST** use the `"finish"` action.
        • If the user's objective was to retrieve specific data (like a fact, a piece of information, or a calculation result), the `action_input.final_answer` MUST contain **ONLY** that requested data, concise and direct. Remove any API metadata, thoughts, or extra commentary.
        • If the objective was to perform an action (e.g., send a message, update a record), the `action_input.final_answer` should be a brief confirmation of success to the user.
    5. Continue iterating only if further actions are strictly necessary to fully achieve the user's objective. Do not generate extraneous steps.
    6. If you encounter an error or cannot fulfill the objective, explain clearly to the user why and suggest what they might do next.
    --- END UNIVERSAL TOOL POLICY ---
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

    def _get_llm(self) -> Any:
        if self.llm:
            return self.llm
        if not self.agent_config:
            raise ValueError("Agent configuration is not loaded.")
        model_name = self.agent_config.llm_model
        model_params = self.agent_config.model_params  # Get the model_params object

        self.llm = get_llm_model(
            model_name=model_name,
            model_params=model_params,
            # Ensuring max_retries and timeout from model_params are explicitly passed
            retries=model_params.max_retries if model_params and model_params.max_retries is not None else 0,
            timeout=model_params.timeout if model_params and model_params.timeout is not None else 60,
        )
        return self.llm

    def run_workflow(
            self,
            workflow_name: str,
            initial_input: Optional[Dict[str, Any]] = None,
            resume_run_id: Optional[str] = None,
    ):
        logger.info(f"Starting or resuming workflow run for: {workflow_name}")
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
                    self.agent_config = load_agent_config(workflow.agent_name)
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
        else:
            logger.info(f"Starting new workflow run: {run_id}")
            self.context = {"trigger": initial_input or {}}
            try:
                with self.db_engine.connect() as connection:
                    connection.execute(
                        insert(workflow_runs_table).values(
                            run_id=run_id,
                            workflow_id=workflow.id,
                            status="running",
                            initial_input=initial_input,
                            start_time=datetime.utcnow(),
                        )
                    )
                    connection.commit()
            except Exception as e:
                logger.error(f"Failed to record workflow start: {e}", exc_info=True)
                return

        try:
            asyncio.run(self._execute_steps(
                workflow.steps, start_index=start_step_index, run_id=run_id
            ))
            logger.info(f"Workflow run '{run_id}' completed successfully.")
            self._update_run_status(run_id, "completed", self.context)
        except Exception as e:
            logger.error(
                f"Workflow run '{run_id}' failed with unexpected error: {e}",
                exc_info=True,
            )
            self._update_run_status(run_id, "failed", {"error": str(e)})

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
        elif tool_name == "terraform_parser_tool":
            tool = TerraformParserTool()
        elif tool_name == "sql_parser_tool":
            tool = SQLParserTool()
        elif tool_name == "code_parser_tool":
            tool = CodeParserTool()
        elif tool_name == "yaml_parser_tool":
            tool = YAMLParserTool()
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
            logger.warning(
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

    def _get_available_actions(self) -> Dict[str, Dict[str, Any]]:
        actions: Dict[str, Dict[str, Any]] = {}

        for name, skill in self.skills.items():
            actions[name] = {
                "type": "skill",
                "description": skill.description,
                "parameters": skill.parameters
            }

        if self.agent_config and self.agent_config.tools:
            for tool_name in self.agent_config.tools:
                try:
                    tool = self._get_tool(tool_name)
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

        system_prompt = f"""You are a helpful AI agent. Complete the objective by thinking step-by-step and choosing one action at a time.
{self.UNIVERSAL_TOOL_POLICY}

--- AVAILABLE ACTIONS AND THEIR PARAMETERS ---
{action_details_str}
--- END AVAILABLE ACTIONS ---

--- WORKFLOW CONTEXT (Results from previous steps) ---
{ctx_json}
--- END CONTEXT ---

**Execution Guidelines:**
- For data retrieval tasks (e.g., "get X", "find Y", "fetch Z"), you are expected to:
    1. Select and use the appropriate tool to get the raw data.
    2. Immediately process the tool's `result` to extract the specific information requested.
    3. Conclude the turn by using the `finish` action, with the extracted, refined answer in `action_input.final_answer`.
- If the objective is ambiguous or requires more information from the user, you may ask clarifying questions.
- Always provide direct answers based on your capabilities and tools.

Return a JSON object matching this schema: {Plan.schema_json(indent=2)}"""
        return [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=f'Objective: "{task}"'),
        ]

    async def _handle_agent_call(self, step: AgentCallStep, run_id: str, step_index: int):
        logger.info(f"--- Agent Step: '{step.name}' ---")

        task_template = getattr(step, "task", "")
        try:
            task = task_template.format(**self.context)
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
        # Re-added the actions list acquisition
        available_actions = self._get_available_actions()
        plan = None  # Initialize plan as None

        for i in range(max_iterations):
            logger.debug(f"[{step.name}] ReAct iteration {i + 1}/{max_iterations}")
            prompt = self._build_agent_prompt(task, available_actions, history)

            # Retrieve retry parameters from agent_config.model_params
            llm_max_retries = self.agent_config.model_params.max_retries if self.agent_config.model_params and self.agent_config.model_params.max_retries is not None else 3  # Default to 3 retries

            # Use fixed base for backoff, capped at MAX_BACKOFF_SECONDS
            # initial_backoff_delay is BASE_BACKOFF_SECONDS for the first retry, then exponential

            for api_retry_count in range(
                    llm_max_retries + 1):  # Loop from 0 to llm_max_retries (total attempts = llm_max_retries + 1)
                try:
                    llm = self._get_llm()

                    # Attempt structured output first
                    if hasattr(llm, "with_structured_output"):
                        structured_llm = llm.with_structured_output(Plan, method="function_calling")
                        try:
                            plan = await structured_llm.ainvoke(prompt)
                            if not isinstance(plan.action_input, dict):
                                raise ValidationError([{"loc": ("action_input",), "msg": "value is not a valid dict",
                                                        "type": "type_error.dict"}], Plan)
                            logger.info(f"Thought: {plan.thought}")
                            break  # Exit inner api_retry loop on successful structured parse
                        except (httpx.RequestError, openai.APIConnectionError) as e:
                            logger.debug(
                                f"Connection error during structured LLM call (ReAct iter {i + 1}, api_retry {api_retry_count + 1}/{llm_max_retries}): {e}. Retrying...")
                            if api_retry_count < llm_max_retries:  # Only sleep if more retries are available
                                await asyncio.sleep(
                                    min(self.BASE_BACKOFF_SECONDS * (2 ** api_retry_count), self.MAX_BACKOFF_SECONDS))
                            continue  # Continue to next api_retry
                        except (ValidationError, OutputParserException) as e:
                            # Parsing/Validation error, not a network error. Try fallback.
                            logger.error(
                                f"LLM structured output validation/parsing failed (ReAct iter {i + 1}, api_retry {api_retry_count}): {e}")
                            # Fallback to raw invoke and custom parsing
                            try:
                                raw_llm_output = await llm.ainvoke(prompt)
                                plan = self._parse_plan_fallback(raw_llm_output)
                                if not isinstance(plan.action_input, dict):
                                    raise ValidationError(
                                        [{"loc": ("action_input",), "msg": "value is not a valid dict",
                                          "type": "type_error.dict"}], Plan)
                                logger.info(f"Fallback successful. Thought: {plan.thought}")
                                break  # Exit inner api_retry loop on successful fallback parse
                            except (httpx.RequestError, openai.APIConnectionError) as conn_err_fallback:
                                logger.debug(
                                    f"Connection error during fallback LLM call (ReAct iter {i + 1}, api_retry {api_retry_count + 1}/{llm_max_retries}): {conn_err_fallback}. Retrying...")
                                if api_retry_count < llm_max_retries:  # Only sleep if more retries are available
                                    await asyncio.sleep(min(self.BASE_BACKOFF_SECONDS * (2 ** api_retry_count),
                                                            self.MAX_BACKOFF_SECONDS))
                                continue  # Continue to next api_retry
                            except Exception as fallback_e:
                                logger.error(
                                    f"Fallback parsing also failed (ReAct iter {i + 1}, api_retry {api_retry_count}): {fallback_e}",
                                    exc_info=True)
                                # If all retries are exhausted, or this is a non-retriable parsing error after all attempts, break and let outer loop handle.
                                break  # Break inner retry loop, plan remains None or invalid if not set.
                    else:  # For LLMs without structured_output built-in
                        try:
                            raw = await llm.ainvoke(prompt)
                            plan = self._parse_plan_fallback(raw)
                            if not isinstance(plan.action_input, dict):
                                raise ValidationError(
                                    [{"loc": ("action_input",), "msg": "value is not a valid dict",
                                      "type": "type_error.dict"}],
                                    Plan)
                            logger.info(f"Thought: {plan.thought}")
                            break  # Exit inner api_retry loop on success
                        except (httpx.RequestError, openai.APIConnectionError) as conn_err:
                            logger.debug(
                                f"Connection error during LLM call (ReAct iter {i + 1}, api_retry {api_retry_count + 1}/{llm_max_retries}): {conn_err}. Retrying...")
                            if api_retry_count < llm_max_retries:  # Only sleep if more retries are available
                                await asyncio.sleep(
                                    min(self.BASE_BACKOFF_SECONDS * (2 ** api_retry_count), self.MAX_BACKOFF_SECONDS))
                            continue  # Continue to next api_retry
                        except Exception as e:
                            logger.error(
                                f"An unexpected error occurred during agent call without structured output (ReAct iter {i + 1}, api_retry {api_retry_count}): {e}",
                                exc_info=True)
                            break  # Break inner retry loop, plan remains None or invalid.

                except (httpx.RequestError, openai.APIConnectionError) as e:
                    # This catches any network errors if they occur directly outside the specific invoke attempts
                    logger.debug(
                        f"Connection error during agent call (ReAct iter {i + 1}, api_retry {api_retry_count + 1}/{llm_max_retries}): {e}. Retrying...")
                    if api_retry_count < llm_max_retries:  # Only sleep if more retries are available
                        await asyncio.sleep(min(self.BASE_BACKOFF_SECONDS * (2 ** api_retry_count),
                                                self.MAX_BACKOFF_SECONDS))  # Exponential backoff
                    continue  # Continue to next api_retry
                except Exception as e:
                    # Catch any other unexpected errors that are not connection or parsing related
                    logger.error(
                        f"An unexpected non-connection/non-parsing error occurred during agent call (ReAct iter {i + 1}, api_retry {api_retry_count}): {e}",
                        exc_info=True)
                    # No retry for these types of errors in this loop, so we break out of the retry loop.
                    break

                    # After the inner api_retry loop, check if a valid plan was obtained
            if plan:  # If plan was set successfully in the inner loop
                break  # Exit the outer ReAct iteration loop, as we have a valid plan
            else:
                # If plan is still None here, it means the LLM did not provide a parseable plan for this ReAct iteration,
                # even after retries for connection issues. Log a warning and the outer `for i in range(max_iterations)` loop will handle proceeding to the next ReAct iteration.
                logger.warning(
                    f"No valid plan obtained from LLM after all attempts in ReAct iteration {i + 1}. Moving to next iteration (if allowed).")

        # Final check and processing of the obtained plan
        # This block is executed AFTER all MAX_REACT_ITERATIONS attempts.
        if plan is None:
            raise RuntimeError(
                f"Failed to get a valid plan after {max_iterations} ReAct iterations, possibly due to persistent issues with the LLM API or its output format.")

        # Always capture trace and log step end here, after plan is confirmed not None
        # and before any early exit on "finish" or further action execution.
        self.context[f"{step.name}_trace"] = trace
        logger.debug(f"Action: {plan.action} | Input: {plan.action_input}")  # Moved here, now guarded by `if plan`
        history.append(AIMessage(content=plan.json()))  # Ensure history is updated after plan is confirmed
        trace.append(  # Ensure trace is updated after plan is confirmed
            {"iteration": i + 1, "plan": json.loads(plan.json())})

        # This log now runs after the trace is fully captured for the step
        logger.info(f"--- End Agent Step: '{step.name}' ---")

        if plan.action == "finish":
            logger.info("Agent signaled finish.")
            self.context[step.name] = plan.action_input.get(
                "final_answer", "Completed"
            )
            return  # Exit the async function for this step if agent finished.
        else:
            try:
                if plan.action in self.skills:
                    result = await self._execute_skill(
                        self.skills[plan.action], plan.action_input
                    )
                elif self.agent_config and plan.action in (
                        self.agent_config.tools or []
                ):
                    result = await self._invoke_tool(plan.action, plan.action_input)
                else:
                    raise ValueError(
                        f"Unknown action '{plan.action}'. Valid actions are: {list(available_actions.keys())}"
                    )

                serialized_result = _serialize_for_db(result)
                logger.debug(f"Result: {str(serialized_result)[:500]}")
                history.append(
                    ToolMessage(content=json.dumps({"result": serialized_result}), tool_name=plan.action,
                                tool_call_id=str(uuid.uuid4()))
                )
                trace[-1]["result"] = serialized_result
                self.context[f"{step.name}_{plan.action}_result"] = serialized_result
            except Exception as e:
                err = f"Error executing action '{plan.action}': {e}"
                logger.error(err, exc_info=True)
                history.append(HumanMessage(content=json.dumps(
                    {"error": str(e)})))
                trace[-1]["error"] = str(e)

        # Removed redundant trace assignment from here, it's now handled before the 'if plan.action == "finish"' block.
        # Removed redundant end logging from here.

    async def _invoke_tool(self, tool_name: str, params: Dict[str, Any]):
        tool = self._get_tool(tool_name)
        if asyncio.iscoroutinefunction(tool.run):
            try:
                result = await tool.run(**params)
            except TypeError:
                result = await tool.run(params)
        else:
            try:
                result = tool.run(**params)
            except TypeError:
                result = tool.run(params)
        return result

    async def _execute_skill(self, skill: Skill, skill_input: Dict[str, Any]) -> Any:
        logger.info(f"--- Executing Skill: {skill.name} ---")
        prompt = f"Instructions for '{skill.name}':\n{skill.instructions}\nInput data:\n{json.dumps(skill_input, indent=2)}"
        return (await self._get_llm().ainvoke(prompt)).content

    async def _handle_tool_call(self, step: ToolCallStep, run_id: str, step_index: int):
        logger.info(
            f"  - Tool step '{step.name}' calling '{step.tool_name}' with input: {step.tool_input}"
        )
        tool_instance = self._get_tool(step.tool_name)
        try:
            if asyncio.iscoroutinefunction(tool_instance.run):
                try:
                    result = await tool_instance.run(**step.tool_input)
                except TypeError:
                    result = await tool_instance.run(step.tool_input)
            else:
                try:
                    result = tool_instance.run(**step.tool_input)
                except TypeError:
                    result = tool_instance.run(step.tool_input)

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
            is_true = eval(step.condition, {"__builtins__": None}, self.context)
        except Exception as e:
            raise ValueError(
                f"Could not evaluate condition '{step.condition}': {e}"
            ) from e
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