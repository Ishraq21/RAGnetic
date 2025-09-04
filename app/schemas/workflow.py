"""
RAGnetic Workflow System

This module defines the core workflow system for RAGnetic, enabling:
1. Multi-step AI agent workflows with conditional logic and loops
2. Robust error handling and retry mechanisms
3. Integration with external tools and services
4. Human-in-the-loop capabilities
5. State management and execution tracking

Workflow Architecture:
- Workflows are defined declaratively in YAML
- Each workflow consists of typed steps (agent_call, tool_call, if_then, loop, human_in_the_loop)
- Steps can reference outputs from previous steps
- Complex conditional logic is supported with boolean expressions
- Comprehensive error handling with retries and recovery strategies
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Any, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

# --- Step Type Enum ---
class StepType(str, Enum):
    AGENT_CALL = "agent_call"
    TOOL_CALL = "tool_call"
    IF_THEN = "if_then"
    LOOP = "loop"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"

# --- Error Handling Models ---
class RetryConfig(BaseModel):
    """Configuration for step retry behavior"""
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum number of retry attempts")
    delay_seconds: float = Field(default=1.0, ge=0.1, le=300, description="Initial delay between retries in seconds")
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0, description="Multiplier for exponential backoff")
    retry_on_errors: List[str] = Field(default=["timeout", "connection_error", "rate_limit"], description="Error types that trigger retries")

class ErrorHandling(BaseModel):
    """Configuration for error handling behavior"""
    on_failure: Literal["stop", "continue", "skip_remaining"] = Field(default="stop", description="What to do when step fails after all retries")
    failure_message: Optional[str] = Field(None, description="Custom error message to display on failure")
    continue_on_errors: List[str] = Field(default=[], description="Specific error types to ignore and continue")

# --- Base Step Model ---
class BaseStep(BaseModel):
    name: str = Field(..., description="A unique name for the step within the workflow.")
    description: Optional[str] = None
    retry: Optional[RetryConfig] = Field(default=None, description="Retry configuration for this step")
    error_handling: Optional[ErrorHandling] = Field(default=None, description="Error handling configuration")
    timeout_seconds: Optional[int] = Field(default=None, ge=1, le=3600, description="Maximum execution time in seconds")
    depends_on: Optional[List[str]] = Field(default=[], description="List of step names this step depends on")
    tags: Optional[List[str]] = Field(default=[], description="Tags for categorizing and filtering steps")
    
    @field_validator('name')
    @classmethod
    def validate_name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Step name cannot be empty or whitespace only")
        return v

# --- Specific Step Models ---
# This is the model for our new agent-driven steps
class AgentCallStep(BaseStep):
    type: Literal[StepType.AGENT_CALL] = StepType.AGENT_CALL
    task: str = Field(..., description="The high-level task for the agent to complete.")
    # This is optional because it can be inherited from the top-level workflow
    agent_name: Optional[str] = None
    
    @field_validator('task')
    @classmethod
    def validate_task_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Agent task cannot be empty or whitespace only")
        return v

class ToolCallStep(BaseStep):
    type: Literal[StepType.TOOL_CALL] = StepType.TOOL_CALL
    tool_name: str
    tool_input: Dict[str, Any] = Field(..., description="The input arguments for the tool.")
    
    @field_validator('tool_name')
    @classmethod
    def validate_tool_name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty or whitespace only")
        return v

class Condition(BaseModel):
    """Enhanced condition model supporting complex boolean expressions"""
    expression: str = Field(..., description="Boolean expression to evaluate (e.g., 'step1.score > 0.8 AND user.tier == \"premium\"')")
    description: Optional[str] = Field(None, description="Human readable description of what this condition checks")
    
class IfThenStep(BaseStep):
    type: Literal[StepType.IF_THEN] = StepType.IF_THEN
    condition: Union[str, Condition] = Field(..., description="A condition to evaluate. Can be a simple string or complex Condition object.")
    on_true: List['WorkflowStep'] = Field(..., description="A list of steps to execute if the condition is true.")
    on_false: Optional[List['WorkflowStep']] = Field(None, description="A list of steps to execute if the condition is false.")
    else_if: Optional[List[Dict[str, Any]]] = Field(None, description="Additional elif conditions with their steps")

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

class WorkflowTrigger(BaseModel):
    type: str = Field(..., description="The type of trigger (e.g., 'api_webhook', 'webhook', 'schedule').")
    path: Optional[str] = Field(None, description="The API path for 'api_webhook' triggers (e.g., '/webhooks/v1/my-hook'). Not used for 'webhook' (generic dispatcher) or 'schedule' triggers.")
    # schedule: Optional[Dict[str, Any]] = Field(None, description="Schedule configuration for 'schedule' triggers.")


# This tells Pydantic to use the 'type' field to decide which model to use for validation.
WorkflowStep = Annotated[
    Union[AgentCallStep, ToolCallStep, IfThenStep, LoopStep, HumanInTheLoopStep],
    Field(discriminator="type")
]

# --- Workflow Execution State ---
class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class StepExecution(BaseModel):
    """Runtime execution state of a workflow step"""
    step_name: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempts: int = 0
    error_message: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    execution_time_seconds: Optional[float] = None

class WorkflowExecution(BaseModel):
    """Runtime execution state of an entire workflow"""
    workflow_name: str
    run_id: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: Optional[str] = None
    step_executions: Dict[str, StepExecution] = {}
    global_context: Dict[str, Any] = {}
    error_message: Optional[str] = None

# --- Main Workflow Schemas ---
class WorkflowBase(BaseModel):
    name: str = Field(..., description="The unique name of the workflow.")
    agent_name: Optional[str] = Field(None, description="The default agent to use for this workflow. Optional.")
    description: Optional[str] = Field(None, description="A description of what the workflow does.")
    steps: List[WorkflowStep] = Field(..., description="An ordered list of steps to execute.")
    trigger: Optional[WorkflowTrigger] = Field(default=None, description="The trigger that starts the workflow.")
    global_retry: Optional[RetryConfig] = Field(default=None, description="Default retry configuration for all steps")
    global_error_handling: Optional[ErrorHandling] = Field(default=None, description="Default error handling for all steps")
    variables: Optional[Dict[str, Any]] = Field(default={}, description="Global workflow variables and constants")
    version: str = Field(default="1.0", description="Workflow version for tracking changes")
    tags: Optional[List[str]] = Field(default=[], description="Tags for categorizing workflows")


class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    agent_name: Optional[str] = None
    trigger: Optional[WorkflowTrigger] = None
    steps: List[Dict[str, Any]]

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    agent_name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[WorkflowStep]] = None
    trigger: Optional[WorkflowTrigger] = None

# --- Workflow Validation ---
class ValidationError(BaseModel):
    """Represents a validation error in workflow configuration"""
    error_type: str
    message: str
    step_name: Optional[str] = None
    severity: Literal["error", "warning", "info"] = "error"

class WorkflowValidationResult(BaseModel):
    """Result of workflow validation"""
    is_valid: bool
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    suggestions: List[str] = []
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

# --- Workflow Testing and Debugging ---
class WorkflowTestConfig(BaseModel):
    """Configuration for testing workflows"""
    dry_run: bool = True
    mock_external_calls: bool = True
    test_data: Dict[str, Any] = {}
    stop_on_error: bool = True
    validate_outputs: bool = True

class WorkflowDebugInfo(BaseModel):
    """Debug information for workflow execution"""
    step_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time_ms: float
    memory_usage_mb: Optional[float] = None
    debug_logs: List[str] = []

class Workflow(WorkflowBase):
    id: Optional[int] = Field(default=None, description="Unique identifier for the workflow (auto-generated if not provided)")
    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
    )
    
    @model_validator(mode='after')
    def validate_no_circular_dependencies(self):
        """Validate that there are no circular dependencies in the workflow steps"""
        if not self.steps:
            return self
            
        # Build dependency graph
        step_names = {step.name for step in self.steps}
        dependencies = {}
        
        for step in self.steps:
            depends_on = getattr(step, 'depends_on', []) or []
            dependencies[step.name] = depends_on
            
            # Check for self-referencing dependencies
            if step.name in depends_on:
                raise ValueError(f"Step '{step.name}' cannot depend on itself (circular dependency)")
            
            # Check that all dependencies reference valid steps
            for dep in depends_on:
                if dep not in step_names:
                    raise ValueError(f"Step '{step.name}' depends on non-existent step '{dep}'")
        
        # Detect circular dependencies using depth-first search
        def has_cycle(graph, start, visited, rec_stack):
            visited[start] = True
            rec_stack[start] = True
            
            for neighbor in graph.get(start, []):
                if not visited.get(neighbor, False):
                    if has_cycle(graph, neighbor, visited, rec_stack):
                        return True
                elif rec_stack.get(neighbor, False):
                    return True
            
            rec_stack[start] = False
            return False
        
        visited = {}
        rec_stack = {}
        
        for step_name in step_names:
            if not visited.get(step_name, False):
                if has_cycle(dependencies, step_name, visited, rec_stack):
                    raise ValueError(f"Circular dependency detected involving step '{step_name}'")
        
        return self

# --- Skill Schema ---
class Skill(BaseModel):
    name: str = Field(..., description="The unique name of the skill.")
    description: str = Field(..., description="A short, clear description of what the skill does.")
    instructions: str = Field(..., description="The detailed, step-by-step instructions for the LLM to follow when executing this skill.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="A dictionary defining the expected parameters for the skill.")

class Plan(BaseModel):
    thought: str = Field(
        ..., description="The agent's reasoning for its next action."
    )
    action: str = Field(
        ..., description="The name of the skill or tool to execute next."
    )
    action_input: Dict[str, Any] = Field(
        default_factory=dict, description="The input parameters for the chosen action."
    )

class StepPlan(BaseModel):
    """A single step within a larger plan."""
    action: str = Field(
        ..., description="The name of the skill, tool, or sub-agent to execute."
    )
    action_input: Dict[str, Any] = Field(
        default_factory=dict, description="The input parameters for the chosen action."
    )

class HierarchicalPlan(BaseModel):
    """
    The orchestrator's multi-step plan for accomplishing the objective.
    """
    thought: str = Field(
        ..., description="The orchestrator's reasoning for the overall plan."
    )
    # This field now supports a list of steps or a list of lists of steps (for parallel execution)
    plan: List[Union[StepPlan, List[StepPlan]]] = Field(
        ..., description="An ordered list of steps to execute. A list of steps within this list indicates parallel execution."
    )

# Pydantic's forward reference handling
IfThenStep.model_rebuild()
LoopStep.model_rebuild()