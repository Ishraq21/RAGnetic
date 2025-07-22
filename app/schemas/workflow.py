# app/schemas/workflow.py

from enum import Enum
from typing import Dict, List, Literal, Optional, Any, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict

# --- Step Type Enum ---
class StepType(str, Enum):
    AGENT_CALL = "agent_call"
    TOOL_CALL = "tool_call"
    IF_THEN = "if_then"
    LOOP = "loop"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"

# --- Base Step Model ---
class BaseStep(BaseModel):
    name: str = Field(..., description="A unique name for the step within the workflow.")
    description: Optional[str] = None

# --- Specific Step Models ---
# This is the model for our new agent-driven steps
class AgentCallStep(BaseStep):
    type: Literal[StepType.AGENT_CALL] = StepType.AGENT_CALL
    task: str = Field(..., description="The high-level task for the agent to complete.")
    # This is optional because it can be inherited from the top-level workflow
    agent_name: Optional[str] = None

class ToolCallStep(BaseStep):
    type: Literal[StepType.TOOL_CALL] = StepType.TOOL_CALL
    tool_name: str
    tool_input: Dict[str, Any] = Field(..., description="The input arguments for the tool.")

class IfThenStep(BaseStep):
    type: Literal[StepType.IF_THEN] = StepType.IF_THEN
    condition: str = Field(..., description="A condition to evaluate.")
    on_true: List['WorkflowStep'] = Field(..., description="A list of steps to execute if the condition is true.")
    on_false: Optional[List['WorkflowStep']] = Field(None, description="A list of steps to execute if the condition is false.")

class LoopStep(BaseStep):
    type: Literal[StepType.LOOP] = StepType.LOOP
    iterable: str = Field(..., description="The context variable to iterate over.")
    loop_variable: str = Field(..., description="The name of the variable to hold the current item in the loop.")
    steps: List['WorkflowStep'] = Field(..., description="A list of steps to execute in each iteration.")

class HumanInTheLoopStep(BaseStep):
    type: Literal[StepType.HUMAN_IN_THE_LOOP] = StepType.HUMAN_IN_THE_LOOP
    prompt: str = Field(..., description="The prompt or question to present to the human user.")
    user_id: Optional[str] = Field(None, description="The ID of the user to assign the task to. Optional.")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data to present to the user. Optional.")

# This tells Pydantic to use the 'type' field to decide which model to use for validation.
WorkflowStep = Annotated[
    Union[AgentCallStep, ToolCallStep, IfThenStep, LoopStep, HumanInTheLoopStep],
    Field(discriminator="type")
]

# --- Main Workflow Schemas ---
class WorkflowBase(BaseModel):
    name: str = Field(..., description="The unique name of the workflow.")
    agent_name: Optional[str] = Field(None, description="The default agent to use for this workflow. Optional.")
    description: Optional[str] = Field(None, description="A description of what the workflow does.")
    steps: List[WorkflowStep] = Field(..., description="An ordered list of steps to execute.")
    trigger: Optional[Dict[str, Any]] = Field(default=None, description="The trigger that starts the workflow, e.g., a schedule.")


class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    agent_name: Optional[str] = None
    trigger: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]]
class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    agent_name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[WorkflowStep]] = None

class Workflow(WorkflowBase):
    id: int
    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
    )

# --- Skill Schema ---
class Skill(BaseModel):
    name: str = Field(..., description="The unique name of the skill.")
    description: str = Field(..., description="A short, clear description of what the skill does.")
    instructions: str = Field(..., description="The detailed, step-by-step instructions for the LLM to follow when executing this skill.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="A dictionary defining the expected parameters for the skill.")

# Pydantic's forward reference handling
IfThenStep.model_rebuild()
LoopStep.model_rebuild()