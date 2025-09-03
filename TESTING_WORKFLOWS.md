# Workflow Testing Framework

This document describes the comprehensive testing framework for RAGnetic workflows, including unit tests, integration tests, performance tests, and load testing capabilities.

## Overview

The workflow testing framework provides multiple layers of testing to ensure workflow reliability, performance, and scalability:

1. **Unit Tests** - Test individual workflow components and schemas
2. **Integration Tests** - Test complete workflow execution flows
3. **Performance Tests** - Measure execution speed and resource usage
4. **Load Tests** - Test system behavior under concurrent load
5. **Stress Tests** - Test system limits and failure modes

## Test Structure

### Unit Tests (`tests/test_workflow_schemas.py`)

Tests individual workflow schema components:
- RetryConfig validation and defaults
- ErrorHandling configuration
- Step creation and validation
- Complex condition objects
- Workflow execution state tracking
- Validation error handling

```bash
# Run unit tests
python -m pytest tests/test_workflow_schemas.py -v
```

### Integration Tests (`tests/test_workflow_engine.py`)

Tests complete workflow execution scenarios:
- Simple linear workflows
- Complex nested workflows with loops and conditions  
- Error handling and recovery
- Retry mechanisms
- Step dependencies
- Variable interpolation
- Workflow state persistence

```bash
# Run integration tests
python -m pytest tests/test_workflow_engine.py -v
```

### Performance Tests (`tests/test_workflow_performance.py`)

Measures workflow execution performance:
- Simple workflow execution speed
- Complex workflow with loops and conditions
- Concurrent workflow execution
- Memory leak detection
- Retry performance impact
- Large workflow scalability
- Deep nested workflow handling

```bash
# Run performance tests
python -m pytest tests/test_workflow_performance.py -v -s
```

### Load Tests (`tests/test_workflow_load.py`)

Tests system behavior under load:
- Light load scenarios (5 concurrent users)
- Moderate load scenarios (20 concurrent users)
- Heavy load scenarios (50+ concurrent users)  
- Stress testing with increasing load
- Endurance testing for sustained load
- Performance degradation analysis

```bash
# Run load tests
python -m pytest tests/test_workflow_load.py -v -s
```

## CLI Testing Commands

The CLI provides several commands for testing workflows:

### 1. Validate Workflow

Validates workflow syntax and configuration:

```bash
ragnetic validate-workflow my-workflow.yaml
```

Options:
- `--strict`: Enable strict validation mode
- `--fix`: Automatically fix common issues

### 2. Test Workflow

Runs a workflow in test mode with mocked dependencies:

```bash
ragnetic test-workflow my-workflow.yaml --iterations 5 --mock-external
```

Options:
- `--iterations`: Number of test iterations
- `--mock-external`: Mock external API calls
- `--test-data`: JSON test data file
- `--export`: Export results to file

### 3. Debug Workflow

Provides detailed execution debugging:

```bash
ragnetic workflow-debug my-workflow.yaml
```

Features:
- Step-by-step execution tracing
- Variable value inspection
- Performance timing
- Error analysis
- Memory usage tracking

### 4. Benchmark Workflow

Measures workflow performance:

```bash
ragnetic benchmark-workflow my-workflow.yaml --iterations 10 --concurrent 2
```

Options:
- `--iterations`: Number of benchmark runs
- `--concurrent`: Concurrent executions
- `--test-data`: Test data for consistent benchmarking
- `--mock-external`: Mock external calls

### 5. Load Test Workflow

Comprehensive load testing:

```bash
ragnetic load-test-workflow my-workflow.yaml --users 20 --per-user 5 --ramp-up 10
```

Options:
- `--users`: Number of concurrent users
- `--per-user`: Workflows per user
- `--ramp-up`: Ramp-up time in seconds
- `--failure-rate`: Simulated failure rate
- `--export`: Export results to JSON

### 6. Stress Test Workflow

Stress testing with increasing load:

```bash
ragnetic stress-test-workflow my-workflow.yaml --max-users 100 --step 10
```

Options:
- `--max-users`: Maximum concurrent users
- `--step`: User increment step size
- `--duration`: Duration per load step
- `--export`: Export results to JSON

### 7. Performance Test Suite

Run comprehensive performance test suite:

```bash
ragnetic performance-test --suite all --export results.json
```

Suites:
- `all`: Run all test suites
- `unit`: Unit tests only
- `integration`: Integration tests only
- `performance`: Performance tests only
- `load`: Load tests only

## Test Workflows

### Edge Cases Test Workflow (`workflows/test-edge-cases-workflow.yaml`)

Tests various edge cases and boundary conditions:
- Maximum timeout values
- Complex multi-condition logic
- Nested loop structures
- Error handling with retry exhaustion
- Human-in-the-loop timeouts
- Complex tool calls with large payloads

### Error Scenarios Test Workflow (`workflows/test-error-scenarios-workflow.yaml`)

Tests error handling and recovery:
- Immediate failure scenarios
- Timeout failures
- Retry exhaustion
- Conditional error handling
- Tool call failures
- Human interaction timeouts
- Loop error handling
- Cascade failure prevention

### Benchmark Workflow (`workflows/benchmark-workflow.yaml`)

Performance benchmarking workflow:
- Quick response tests
- CPU intensive operations
- Memory usage testing
- Concurrent processing
- I/O performance testing
- Complex conditional logic
- Stress testing with errors
- Comprehensive reporting

## Performance Metrics

The testing framework tracks comprehensive performance metrics:

### Response Time Metrics
- Average response time
- 95th percentile (P95) response time
- 99th percentile (P99) response time
- Minimum/maximum response times

### Throughput Metrics
- Workflows per second
- Operations per second
- Peak concurrent workflows
- Sustained throughput

### Reliability Metrics
- Success rate percentage
- Error rate percentage
- Retry success rate
- Failure recovery time

### Resource Usage Metrics
- Memory usage (peak and average)
- CPU utilization
- Execution time per step
- Resource leak detection

## Performance Thresholds

The testing framework uses these performance thresholds:

### Response Times
- Excellent: < 2s average, < 5s P95
- Good: 2-5s average, 5-10s P95  
- Poor: > 5s average, > 10s P95

### Throughput
- High: > 10 workflows/sec
- Moderate: 5-10 workflows/sec
- Low: < 5 workflows/sec

### Reliability
- Excellent: < 5% error rate
- Good: 5-15% error rate
- Poor: > 15% error rate

### Memory Usage
- Efficient: < 100MB growth over time
- Acceptable: 100-200MB growth
- Concerning: > 200MB growth or leaks

## Running Tests

### Quick Test Run

```bash
# Run basic validation and unit tests
ragnetic performance-test --suite unit

# Test a specific workflow
ragnetic test-workflow benchmark-workflow.yaml --iterations 3
```

### Comprehensive Testing

```bash
# Full performance test suite
ragnetic performance-test --suite all --verbose --export full-results.json

# Load test with realistic parameters
ragnetic load-test-workflow benchmark-workflow.yaml \
  --users 25 --per-user 10 --ramp-up 15 --export load-results.json

# Stress test to find limits
ragnetic stress-test-workflow benchmark-workflow.yaml \
  --max-users 200 --step 25 --export stress-results.json
```

### Continuous Integration

For CI/CD pipelines:

```bash
# Fast test suite (< 2 minutes)
ragnetic performance-test --suite unit,integration

# Performance regression test
ragnetic benchmark-workflow benchmark-workflow.yaml \
  --iterations 5 --concurrent 3 --export ci-benchmark.json
```

## Test Data Management

### Test Data Files

Create JSON files with test data:

```json
{
  "customer_email": "test@example.com",
  "priority": "high",
  "items": ["item1", "item2", "item3"],
  "threshold": 0.8,
  "max_retries": 3
}
```

Use with tests:

```bash
ragnetic test-workflow my-workflow.yaml --test-data test-data.json
```

### Mock Configuration

Tests automatically mock external dependencies:
- Agent API calls
- Tool API calls  
- Database connections
- External services

This ensures:
- Consistent test results
- Fast test execution
- No external dependencies
- Repeatable test scenarios

## Troubleshooting

### Common Issues

**Tests are slow:**
- Use `--mock-external` to mock API calls
- Reduce `--iterations` for faster feedback
- Run specific test suites instead of `all`

**High memory usage:**
- Check for memory leaks in workflow logic
- Monitor long-running loops
- Review large data structure handling

**Low throughput:**
- Optimize step timeout values
- Reduce retry delays
- Check for unnecessary sequential dependencies

**High error rates:**
- Review retry configuration
- Check error handling logic
- Validate input data formats

### Debug Commands

```bash
# Detailed workflow debugging
ragnetic workflow-debug my-workflow.yaml

# Verbose test output
ragnetic performance-test --suite all --verbose

# Export detailed metrics
ragnetic load-test-workflow my-workflow.yaml --export debug-results.json
```

## Best Practices

### Test Development
1. Start with unit tests for new workflow features
2. Add integration tests for complete workflows  
3. Use performance tests to catch regressions
4. Run load tests before production deployment

### Performance Optimization
1. Set appropriate timeout values
2. Use parallel execution for independent steps
3. Implement efficient retry strategies
4. Monitor resource usage over time

### Reliability Testing
1. Test failure scenarios explicitly
2. Validate error recovery mechanisms
3. Test edge cases and boundary conditions
4. Ensure graceful degradation under load

### Continuous Monitoring
1. Run performance tests in CI/CD
2. Track performance trends over time
3. Set up alerts for performance regressions
4. Monitor production metrics

## Contributing

When adding new workflow features:

1. Add unit tests for new schemas/components
2. Add integration tests for new workflow types
3. Update performance tests if adding resource-intensive features
4. Add load tests for features affecting concurrency
5. Update this documentation

### Test Naming Conventions

- Unit tests: `test_component_feature`
- Integration tests: `test_workflow_scenario`
- Performance tests: `test_feature_performance`
- Load tests: `test_feature_load_scenario`