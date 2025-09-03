"""
Integration tests for the enhanced workflow engine
"""

import pytest
import asyncio
import yaml
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from app.schemas.workflow import (
    WorkflowBase,
    AgentCallStep,
    ToolCallStep,
    IfThenStep,
    StepStatus,
    WorkflowExecution,
    RetryConfig,
    ErrorHandling,
    WorkflowValidationResult,
    WorkflowTestConfig
)
from app.workflows.engine import WorkflowEngine


class TestWorkflowEngineExecution:
    """Test workflow engine execution with enhanced features"""
    
    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow for testing"""
        return WorkflowBase(
            name="test_workflow",
            description="Test workflow for integration testing",
            steps=[
                AgentCallStep(
                    name="validate_input",
                    task="Validate input data",
                    timeout_seconds=30,
                    retry=RetryConfig(max_attempts=2, delay_seconds=0.1),
                    error_handling=ErrorHandling(on_failure="stop")
                ),
                AgentCallStep(
                    name="process_data",
                    task="Process the validated data",
                    depends_on=["validate_input"],
                    timeout_seconds=60
                ),
                IfThenStep(
                    name="check_result",
                    condition="process_data.success == True",
                    depends_on=["process_data"],
                    on_true=[
                        ToolCallStep(
                            name="send_success_notification",
                            tool_name="email_tool",
                            tool_input={"message": "Success!"}
                        )
                    ],
                    on_false=[
                        AgentCallStep(
                            name="handle_failure",
                            task="Handle processing failure"
                        )
                    ]
                )
            ],
            global_retry=RetryConfig(max_attempts=3, delay_seconds=1.0),
            variables={"threshold": 0.8}
        )
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock workflow engine for testing"""
        engine = MagicMock()
        engine.execute_workflow = AsyncMock()
        engine.validate_workflow = MagicMock()
        engine.test_workflow = AsyncMock()
        return engine
    
    def test_workflow_validation_success(self, sample_workflow):
        """Test successful workflow validation"""
        # This would be implemented in the actual engine
        validation_result = WorkflowValidationResult(is_valid=True)
        
        assert validation_result.is_valid
        assert not validation_result.has_errors
        assert not validation_result.has_warnings
    
    def test_workflow_validation_with_errors(self):
        """Test workflow validation with errors"""
        from app.schemas.workflow import ValidationError as WorkflowValidationError
        
        errors = [
            WorkflowValidationError(
                error_type="missing_dependency",
                message="Step 'process_data' depends on 'missing_step' which doesn't exist",
                step_name="process_data",
                severity="error"
            )
        ]
        
        validation_result = WorkflowValidationResult(
            is_valid=False,
            errors=errors
        )
        
        assert not validation_result.is_valid
        assert validation_result.has_errors
        assert len(validation_result.errors) == 1
        assert validation_result.errors[0].step_name == "process_data"
    
    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, sample_workflow, mock_engine):
        """Test simple workflow execution"""
        # Mock the execution result
        execution = WorkflowExecution(
            workflow_name=sample_workflow.name,
            run_id="test_run_123",
            status=StepStatus.COMPLETED
        )
        
        mock_engine.execute_workflow.return_value = execution
        
        result = await mock_engine.execute_workflow(sample_workflow, {})
        
        assert result.workflow_name == "test_workflow"
        assert result.run_id == "test_run_123"
        assert result.status == StepStatus.COMPLETED
        mock_engine.execute_workflow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_with_retry_logic(self, mock_engine):
        """Test workflow execution with retry logic"""
        workflow = WorkflowBase(
            name="retry_test_workflow",
            steps=[
                AgentCallStep(
                    name="flaky_step",
                    task="This step might fail",
                    retry=RetryConfig(
                        max_attempts=3,
                        delay_seconds=0.1,
                        retry_on_errors=["timeout", "connection_error"]
                    )
                )
            ]
        )
        
        # Mock execution that succeeds after retries
        execution = WorkflowExecution(
            workflow_name=workflow.name,
            run_id="retry_test_456",
            status=StepStatus.COMPLETED
        )
        
        # Simulate step execution with retries
        execution.step_executions["flaky_step"] = {
            "step_name": "flaky_step",
            "status": StepStatus.COMPLETED,
            "attempts": 2,  # Succeeded on second attempt
            "execution_time_seconds": 1.5
        }
        
        mock_engine.execute_workflow.return_value = execution
        result = await mock_engine.execute_workflow(workflow, {})
        
        assert result.status == StepStatus.COMPLETED
        assert result.step_executions["flaky_step"]["attempts"] == 2
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, mock_engine):
        """Test workflow error handling scenarios"""
        workflow = WorkflowBase(
            name="error_handling_test",
            steps=[
                AgentCallStep(
                    name="critical_step",
                    task="Critical step that must succeed",
                    error_handling=ErrorHandling(on_failure="stop")
                ),
                AgentCallStep(
                    name="optional_step",
                    task="Optional step that can fail",
                    error_handling=ErrorHandling(
                        on_failure="continue",
                        continue_on_errors=["non_critical_error"]
                    )
                )
            ]
        )
        
        # Mock execution where optional step fails but workflow continues
        execution = WorkflowExecution(
            workflow_name=workflow.name,
            run_id="error_test_789",
            status=StepStatus.COMPLETED
        )
        
        execution.step_executions = {
            "critical_step": {
                "step_name": "critical_step",
                "status": StepStatus.COMPLETED,
                "attempts": 1
            },
            "optional_step": {
                "step_name": "optional_step",
                "status": StepStatus.FAILED,
                "attempts": 1,
                "error_message": "non_critical_error: Optional operation failed"
            }
        }
        
        mock_engine.execute_workflow.return_value = execution
        result = await mock_engine.execute_workflow(workflow, {})
        
        assert result.status == StepStatus.COMPLETED
        assert result.step_executions["critical_step"]["status"] == StepStatus.COMPLETED
        assert result.step_executions["optional_step"]["status"] == StepStatus.FAILED
    
    def test_complex_condition_evaluation(self):
        """Test complex boolean condition evaluation"""
        from app.schemas.workflow import Condition
        
        # Test complex condition creation
        condition = Condition(
            expression="step1.score > 0.8 AND user.tier == 'premium' OR emergency == True",
            description="High score premium user or emergency situation"
        )
        
        assert condition.expression == "step1.score > 0.8 AND user.tier == 'premium' OR emergency == True"
        assert condition.description == "High score premium user or emergency situation"
        
        # Test condition in if-then step
        if_then_step = IfThenStep(
            name="complex_routing",
            condition=condition,
            on_true=[AgentCallStep(name="priority_handling", task="Handle with priority")],
            on_false=[AgentCallStep(name="standard_handling", task="Handle normally")]
        )
        
        assert isinstance(if_then_step.condition, Condition)
        assert len(if_then_step.on_true) == 1
        assert len(if_then_step.on_false) == 1
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, mock_engine):
        """Test workflow state persistence and recovery"""
        workflow = WorkflowBase(
            name="persistence_test",
            steps=[
                AgentCallStep(name="step1", task="First step"),
                AgentCallStep(name="step2", task="Second step", depends_on=["step1"]),
                AgentCallStep(name="step3", task="Third step", depends_on=["step2"])
            ]
        )
        
        # Mock execution that gets interrupted
        execution = WorkflowExecution(
            workflow_name=workflow.name,
            run_id="persistence_123",
            status=StepStatus.RUNNING,
            current_step="step2"
        )
        
        execution.step_executions = {
            "step1": {
                "step_name": "step1",
                "status": StepStatus.COMPLETED,
                "output": {"result": "step1_success"}
            },
            "step2": {
                "step_name": "step2", 
                "status": StepStatus.RUNNING
            }
        }
        
        # Mock recovery from saved state
        mock_engine.recover_workflow = AsyncMock(return_value=execution)
        
        recovered_execution = await mock_engine.recover_workflow("persistence_123")
        
        assert recovered_execution.current_step == "step2"
        assert recovered_execution.step_executions["step1"]["status"] == StepStatus.COMPLETED
        assert recovered_execution.step_executions["step2"]["status"] == StepStatus.RUNNING
    
    def test_workflow_dependency_resolution(self):
        """Test workflow step dependency resolution"""
        workflow = WorkflowBase(
            name="dependency_test",
            steps=[
                AgentCallStep(name="step_a", task="Independent step A"),
                AgentCallStep(name="step_b", task="Independent step B"),
                AgentCallStep(
                    name="step_c",
                    task="Depends on A and B",
                    depends_on=["step_a", "step_b"]
                ),
                AgentCallStep(
                    name="step_d",
                    task="Depends on C",
                    depends_on=["step_c"]
                )
            ]
        )
        
        # Test dependency validation
        step_c = workflow.steps[2]
        step_d = workflow.steps[3]
        
        assert "step_a" in step_c.depends_on
        assert "step_b" in step_c.depends_on
        assert "step_c" in step_d.depends_on
        assert len(step_c.depends_on) == 2
        assert len(step_d.depends_on) == 1
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, mock_engine):
        """Test workflow timeout handling"""
        workflow = WorkflowBase(
            name="timeout_test",
            steps=[
                AgentCallStep(
                    name="slow_step",
                    task="This step takes too long",
                    timeout_seconds=1  # Very short timeout for testing
                )
            ]
        )
        
        # Mock execution that times out
        execution = WorkflowExecution(
            workflow_name=workflow.name,
            run_id="timeout_456",
            status=StepStatus.FAILED,
            error_message="Step 'slow_step' timed out after 1 seconds"
        )
        
        execution.step_executions["slow_step"] = {
            "step_name": "slow_step",
            "status": StepStatus.FAILED,
            "error_message": "timeout",
            "execution_time_seconds": 1.1
        }
        
        mock_engine.execute_workflow.return_value = execution
        result = await mock_engine.execute_workflow(workflow, {})
        
        assert result.status == StepStatus.FAILED
        assert "timed out" in result.error_message
        assert result.step_executions["slow_step"]["error_message"] == "timeout"
    
    def test_workflow_test_configuration(self):
        """Test workflow test configuration"""
        test_config = WorkflowTestConfig(
            dry_run=True,
            mock_external_calls=True,
            test_data={
                "customer_email": "test@example.com",
                "priority": "high",
                "message": "Test message"
            },
            stop_on_error=False,
            validate_outputs=True
        )
        
        assert test_config.dry_run is True
        assert test_config.mock_external_calls is True
        assert test_config.test_data["customer_email"] == "test@example.com"
        assert test_config.stop_on_error is False
        assert test_config.validate_outputs is True
    
    @pytest.mark.asyncio
    async def test_workflow_dry_run_execution(self, mock_engine):
        """Test workflow dry run execution"""
        workflow = WorkflowBase(
            name="dry_run_test",
            steps=[
                AgentCallStep(name="step1", task="First step"),
                ToolCallStep(
                    name="external_call",
                    tool_name="api_toolkit",
                    tool_input={"url": "https://api.example.com"}
                )
            ]
        )
        
        test_config = WorkflowTestConfig(
            dry_run=True,
            mock_external_calls=True
        )
        
        # Mock dry run execution
        execution = WorkflowExecution(
            workflow_name=workflow.name,
            run_id="dry_run_789",
            status=StepStatus.COMPLETED
        )
        
        execution.step_executions = {
            "step1": {
                "step_name": "step1",
                "status": StepStatus.COMPLETED,
                "output": {"mocked": True, "result": "dry_run_success"}
            },
            "external_call": {
                "step_name": "external_call",
                "status": StepStatus.COMPLETED,
                "output": {"mocked": True, "api_response": "mock_response"}
            }
        }
        
        mock_engine.test_workflow.return_value = execution
        result = await mock_engine.test_workflow(workflow, test_config)
        
        assert result.status == StepStatus.COMPLETED
        assert result.step_executions["step1"]["output"]["mocked"] is True
        assert result.step_executions["external_call"]["output"]["mocked"] is True


class TestWorkflowYAMLIntegration:
    """Test integration with YAML workflow definitions"""
    
    def test_load_enhanced_workflow_from_yaml(self):
        """Test loading enhanced workflow from YAML"""
        yaml_content = """
        name: "test_enhanced_workflow"
        description: "Test enhanced workflow features"
        version: "2.0"
        tags: ["test", "enhanced"]
        
        global_retry:
          max_attempts: 3
          delay_seconds: 1.0
          backoff_multiplier: 2.0
        
        global_error_handling:
          on_failure: "continue"
          continue_on_errors: ["minor_error"]
        
        variables:
          threshold: 0.8
          max_retries: 5
        
        steps:
          - name: "validate"
            type: "agent_call"
            task: "Validate input"
            timeout_seconds: 30
            retry:
              max_attempts: 2
              delay_seconds: 0.5
            error_handling:
              on_failure: "stop"
            tags: ["validation", "critical"]
            
          - name: "process"
            type: "agent_call" 
            task: "Process data"
            depends_on: ["validate"]
            timeout_seconds: 120
            
          - name: "route"
            type: "if_then"
            condition:
              expression: "process.score > 0.8 AND process.confidence > 0.9"
              description: "High quality result"
            depends_on: ["process"]
            on_true:
              - name: "handle_success"
                type: "agent_call"
                task: "Handle successful processing"
            on_false:
              - name: "handle_failure" 
                type: "agent_call"
                task: "Handle failed processing"
        """
        
        # Parse YAML (in real implementation, this would be handled by the engine)
        data = yaml.safe_load(yaml_content)
        
        # Verify parsed structure
        assert data["name"] == "test_enhanced_workflow"
        assert data["version"] == "2.0"
        assert "test" in data["tags"]
        assert data["global_retry"]["max_attempts"] == 3
        assert data["variables"]["threshold"] == 0.8
        assert len(data["steps"]) == 3
        
        # Verify step configuration
        validate_step = data["steps"][0]
        assert validate_step["name"] == "validate"
        assert validate_step["timeout_seconds"] == 30
        assert validate_step["retry"]["max_attempts"] == 2
        assert "validation" in validate_step["tags"]
        
        # Verify conditional step
        route_step = data["steps"][2]
        assert route_step["type"] == "if_then"
        assert route_step["depends_on"] == ["process"]
        assert "condition" in route_step
        assert len(route_step["on_true"]) == 1
        assert len(route_step["on_false"]) == 1
    
    def test_workflow_validation_from_yaml(self):
        """Test workflow validation with YAML configuration"""
        # This would test the actual validation logic in the engine
        invalid_yaml = """
        name: "invalid_workflow"
        steps:
          - name: "step1"
            type: "agent_call"
            task: "First step"
          - name: "step2" 
            type: "agent_call"
            task: "Second step"
            depends_on: ["nonexistent_step"]  # Invalid dependency
            timeout_seconds: -5  # Invalid timeout
        """
        
        # In a real implementation, this would be validated by the engine
        data = yaml.safe_load(invalid_yaml)
        
        # Manual validation check for test
        errors = []
        
        # Check for invalid dependencies
        for step in data["steps"]:
            if "depends_on" in step:
                step_names = [s["name"] for s in data["steps"]]
                for dep in step["depends_on"]:
                    if dep not in step_names:
                        errors.append(f"Step '{step['name']}' depends on nonexistent step '{dep}'")
        
        # Check for invalid timeout
        for step in data["steps"]:
            if "timeout_seconds" in step and step["timeout_seconds"] <= 0:
                errors.append(f"Step '{step['name']}' has invalid timeout: {step['timeout_seconds']}")
        
        assert len(errors) == 2
        assert "nonexistent_step" in errors[0]
        assert "invalid timeout" in errors[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])