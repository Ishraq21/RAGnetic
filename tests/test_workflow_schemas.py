"""
Unit tests for enhanced workflow schemas and validation
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.schemas.workflow import (
    RetryConfig,
    ErrorHandling,
    BaseStep,
    AgentCallStep,
    ToolCallStep,
    IfThenStep,
    Condition,
    LoopStep,
    HumanInTheLoopStep,
    WorkflowBase,
    StepStatus,
    StepExecution,
    WorkflowExecution,
    WorkflowValidationResult,
    ValidationError as WorkflowValidationError,
    WorkflowTestConfig,
    WorkflowDebugInfo,
    StepType
)


class TestRetryConfig:
    """Test retry configuration validation and defaults"""
    
    def test_default_retry_config(self):
        """Test default retry configuration values"""
        retry = RetryConfig()
        assert retry.max_attempts == 3
        assert retry.delay_seconds == 1.0
        assert retry.backoff_multiplier == 2.0
        assert retry.retry_on_errors == ["timeout", "connection_error", "rate_limit"]
    
    def test_custom_retry_config(self):
        """Test custom retry configuration"""
        retry = RetryConfig(
            max_attempts=5,
            delay_seconds=0.5,
            backoff_multiplier=1.5,
            retry_on_errors=["network_error", "timeout"]
        )
        assert retry.max_attempts == 5
        assert retry.delay_seconds == 0.5
        assert retry.backoff_multiplier == 1.5
        assert retry.retry_on_errors == ["network_error", "timeout"]
    
    def test_retry_config_validation(self):
        """Test retry configuration validation limits"""
        # Test max_attempts bounds
        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=0)  # Below minimum
        
        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=15)  # Above maximum
        
        # Test delay_seconds bounds
        with pytest.raises(ValidationError):
            RetryConfig(delay_seconds=0.05)  # Below minimum
        
        with pytest.raises(ValidationError):
            RetryConfig(delay_seconds=500)  # Above maximum
        
        # Test backoff_multiplier bounds
        with pytest.raises(ValidationError):
            RetryConfig(backoff_multiplier=0.5)  # Below minimum
        
        with pytest.raises(ValidationError):
            RetryConfig(backoff_multiplier=15)  # Above maximum


class TestErrorHandling:
    """Test error handling configuration"""
    
    def test_default_error_handling(self):
        """Test default error handling values"""
        error_handling = ErrorHandling()
        assert error_handling.on_failure == "stop"
        assert error_handling.failure_message is None
        assert error_handling.continue_on_errors == []
    
    def test_custom_error_handling(self):
        """Test custom error handling configuration"""
        error_handling = ErrorHandling(
            on_failure="continue",
            failure_message="Custom failure message",
            continue_on_errors=["minor_error", "warning"]
        )
        assert error_handling.on_failure == "continue"
        assert error_handling.failure_message == "Custom failure message"
        assert error_handling.continue_on_errors == ["minor_error", "warning"]
    
    def test_error_handling_validation(self):
        """Test error handling validation"""
        # Test invalid on_failure values
        with pytest.raises(ValidationError):
            ErrorHandling(on_failure="invalid_option")


class TestBaseStep:
    """Test base step functionality"""
    
    def test_minimal_base_step(self):
        """Test minimal base step creation"""
        # Since BaseStep is abstract, we'll test through AgentCallStep
        step = AgentCallStep(name="test_step", task="Test task")
        assert step.name == "test_step"
        assert step.description is None
        assert step.retry is None
        assert step.error_handling is None
        assert step.timeout_seconds is None
        assert step.depends_on == []
        assert step.tags == []
    
    def test_full_base_step(self):
        """Test base step with all optional fields"""
        retry = RetryConfig(max_attempts=5)
        error_handling = ErrorHandling(on_failure="continue")
        
        step = AgentCallStep(
            name="test_step",
            task="Test task",
            description="Test description",
            retry=retry,
            error_handling=error_handling,
            timeout_seconds=300,
            depends_on=["step1", "step2"],
            tags=["test", "critical"]
        )
        
        assert step.name == "test_step"
        assert step.description == "Test description"
        assert step.retry == retry
        assert step.error_handling == error_handling
        assert step.timeout_seconds == 300
        assert step.depends_on == ["step1", "step2"]
        assert step.tags == ["test", "critical"]
    
    def test_timeout_validation(self):
        """Test timeout validation"""
        # Valid timeout
        step = AgentCallStep(name="test", task="test", timeout_seconds=60)
        assert step.timeout_seconds == 60
        
        # Invalid timeout (too small)
        with pytest.raises(ValidationError):
            AgentCallStep(name="test", task="test", timeout_seconds=0)
        
        # Invalid timeout (too large)
        with pytest.raises(ValidationError):
            AgentCallStep(name="test", task="test", timeout_seconds=4000)


class TestCondition:
    """Test enhanced condition functionality"""
    
    def test_simple_condition_string(self):
        """Test simple string condition (backward compatibility)"""
        step = IfThenStep(
            name="test_condition",
            condition="step1.result == True",
            on_true=[AgentCallStep(name="true_step", task="True task")]
        )
        assert step.condition == "step1.result == True"
    
    def test_complex_condition_object(self):
        """Test complex condition with Condition object"""
        condition = Condition(
            expression="step1.score > 0.8 AND user.tier == 'premium'",
            description="High score premium user condition"
        )
        
        step = IfThenStep(
            name="test_condition",
            condition=condition,
            on_true=[AgentCallStep(name="true_step", task="True task")]
        )
        
        assert isinstance(step.condition, Condition)
        assert step.condition.expression == "step1.score > 0.8 AND user.tier == 'premium'"
        assert step.condition.description == "High score premium user condition"
    
    def test_if_then_with_else_if(self):
        """Test if-then step with else-if branches"""
        step = IfThenStep(
            name="complex_condition",
            condition="priority == 'high'",
            on_true=[AgentCallStep(name="high_priority", task="Handle high priority")],
            on_false=[AgentCallStep(name="low_priority", task="Handle low priority")],
            else_if=[
                {
                    "condition": "priority == 'medium'",
                    "steps": [{"name": "medium_priority", "type": "agent_call", "task": "Handle medium priority"}]
                }
            ]
        )
        
        assert step.else_if is not None
        assert len(step.else_if) == 1
        assert step.else_if[0]["condition"] == "priority == 'medium'"


class TestWorkflowExecution:
    """Test workflow execution state tracking"""
    
    def test_step_execution_creation(self):
        """Test step execution state creation"""
        execution = StepExecution(step_name="test_step")
        
        assert execution.step_name == "test_step"
        assert execution.status == StepStatus.PENDING
        assert execution.start_time is None
        assert execution.end_time is None
        assert execution.attempts == 0
        assert execution.error_message is None
        assert execution.output is None
        assert execution.execution_time_seconds is None
    
    def test_step_execution_with_data(self):
        """Test step execution with full data"""
        now = datetime.now()
        execution = StepExecution(
            step_name="test_step",
            status=StepStatus.COMPLETED,
            start_time=now,
            end_time=now,
            attempts=2,
            error_message="Temporary error on first attempt",
            output={"result": "success", "score": 0.95},
            execution_time_seconds=1.5
        )
        
        assert execution.step_name == "test_step"
        assert execution.status == StepStatus.COMPLETED
        assert execution.start_time == now
        assert execution.end_time == now
        assert execution.attempts == 2
        assert execution.error_message == "Temporary error on first attempt"
        assert execution.output == {"result": "success", "score": 0.95}
        assert execution.execution_time_seconds == 1.5
    
    def test_workflow_execution_creation(self):
        """Test workflow execution state creation"""
        execution = WorkflowExecution(
            workflow_name="test_workflow",
            run_id="run_12345"
        )
        
        assert execution.workflow_name == "test_workflow"
        assert execution.run_id == "run_12345"
        assert execution.status == StepStatus.PENDING
        assert execution.start_time is None
        assert execution.end_time is None
        assert execution.current_step is None
        assert execution.step_executions == {}
        assert execution.global_context == {}
        assert execution.error_message is None


class TestWorkflowValidation:
    """Test workflow validation functionality"""
    
    def test_validation_error_creation(self):
        """Test validation error creation"""
        error = WorkflowValidationError(
            error_type="missing_field",
            message="Required field 'name' is missing",
            step_name="step1",
            severity="error"
        )
        
        assert error.error_type == "missing_field"
        assert error.message == "Required field 'name' is missing"
        assert error.step_name == "step1"
        assert error.severity == "error"
    
    def test_validation_result_properties(self):
        """Test validation result properties"""
        errors = [
            WorkflowValidationError(error_type="syntax", message="Invalid syntax", severity="error")
        ]
        warnings = [
            WorkflowValidationError(error_type="optimization", message="Could be optimized", severity="warning")
        ]
        
        result = WorkflowValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            suggestions=["Consider using retry configuration"]
        )
        
        assert result.is_valid is False
        assert result.has_errors is True
        assert result.has_warnings is True
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.suggestions) == 1
    
    def test_validation_result_no_issues(self):
        """Test validation result with no issues"""
        result = WorkflowValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.has_errors is False
        assert result.has_warnings is False
        assert len(result.errors) == 0
        assert len(result.warnings) == 0


class TestWorkflowTestConfig:
    """Test workflow testing configuration"""
    
    def test_default_test_config(self):
        """Test default test configuration"""
        config = WorkflowTestConfig()
        
        assert config.dry_run is True
        assert config.mock_external_calls is True
        assert config.test_data == {}
        assert config.stop_on_error is True
        assert config.validate_outputs is True
    
    def test_custom_test_config(self):
        """Test custom test configuration"""
        test_data = {
            "customer_email": "test@example.com",
            "priority": "high"
        }
        
        config = WorkflowTestConfig(
            dry_run=False,
            mock_external_calls=False,
            test_data=test_data,
            stop_on_error=False,
            validate_outputs=False
        )
        
        assert config.dry_run is False
        assert config.mock_external_calls is False
        assert config.test_data == test_data
        assert config.stop_on_error is False
        assert config.validate_outputs is False


class TestWorkflowDebugInfo:
    """Test workflow debugging information"""
    
    def test_debug_info_creation(self):
        """Test debug info creation"""
        debug_info = WorkflowDebugInfo(
            step_name="test_step",
            inputs={"query": "test query"},
            outputs={"result": "success"},
            execution_time_ms=150.5,
            memory_usage_mb=25.6,
            debug_logs=["Step started", "Processing query", "Step completed"]
        )
        
        assert debug_info.step_name == "test_step"
        assert debug_info.inputs == {"query": "test query"}
        assert debug_info.outputs == {"result": "success"}
        assert debug_info.execution_time_ms == 150.5
        assert debug_info.memory_usage_mb == 25.6
        assert len(debug_info.debug_logs) == 3


class TestWorkflowBase:
    """Test enhanced workflow base functionality"""
    
    def test_minimal_workflow(self):
        """Test minimal workflow creation"""
        workflow = WorkflowBase(
            name="test_workflow",
            steps=[AgentCallStep(name="step1", task="Test task")]
        )
        
        assert workflow.name == "test_workflow"
        assert workflow.agent_name is None
        assert workflow.description is None
        assert len(workflow.steps) == 1
        assert workflow.trigger is None
        assert workflow.global_retry is None
        assert workflow.global_error_handling is None
        assert workflow.variables == {}
        assert workflow.version == "1.0"
        assert workflow.tags == []
    
    def test_full_workflow(self):
        """Test workflow with all optional fields"""
        retry = RetryConfig(max_attempts=5)
        error_handling = ErrorHandling(on_failure="continue")
        variables = {"threshold": 0.8, "max_items": 100}
        
        workflow = WorkflowBase(
            name="comprehensive_workflow",
            agent_name="test_agent",
            description="A comprehensive test workflow",
            steps=[
                AgentCallStep(name="step1", task="First task"),
                ToolCallStep(name="step2", tool_name="test_tool", tool_input={"param": "value"})
            ],
            global_retry=retry,
            global_error_handling=error_handling,
            variables=variables,
            version="2.1",
            tags=["test", "comprehensive", "production"]
        )
        
        assert workflow.name == "comprehensive_workflow"
        assert workflow.agent_name == "test_agent"
        assert workflow.description == "A comprehensive test workflow"
        assert len(workflow.steps) == 2
        assert workflow.global_retry == retry
        assert workflow.global_error_handling == error_handling
        assert workflow.variables == variables
        assert workflow.version == "2.1"
        assert workflow.tags == ["test", "comprehensive", "production"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])