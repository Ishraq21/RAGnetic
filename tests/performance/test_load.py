# Load and performance tests for RAGnetic
import pytest
import asyncio
import time
import psutil
import gc
from httpx import AsyncClient
from concurrent.futures import ThreadPoolExecutor
from tests.fixtures.sample_data import get_sample_training_job, SAMPLE_AGENTS, LOAD_TEST_SCENARIOS


class TestAPIPerformance:
    """Test API endpoint performance under various loads."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_query_performance(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test agent query performance under load."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Performance test configuration
        num_queries = 100
        concurrent_sessions = 10
        
        query_data = {
            "message": "What is a contract?",
            "session_id": "perf_test_session"
        }
        
        async def single_query():
            start_time = time.perf_counter()
            response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            end_time = time.perf_counter()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        # Run queries concurrently
        start_time = time.perf_counter()
        
        tasks = [single_query() for _ in range(num_queries)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful_queries = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results if r["success"]]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            # Performance assertions
            assert successful_queries >= num_queries * 0.95  # 95% success rate
            assert avg_response_time < 5.0  # Average under 5 seconds
            assert max_response_time < 15.0  # Max under 15 seconds
            
            # Calculate throughput
            throughput = successful_queries / total_time
            assert throughput > 1.0  # At least 1 query per second
            
            print(f"Query Performance Results:")
            print(f"  Total queries: {num_queries}")
            print(f"  Successful: {successful_queries} ({successful_queries/num_queries*100:.1f}%)")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Min/Max response time: {min_response_time:.3f}s / {max_response_time:.3f}s")
            print(f"  Throughput: {throughput:.2f} queries/second")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_training_job_creation_performance(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test training job creation performance under load."""
        num_jobs = 50
        
        async def create_job(job_index):
            job_data = get_sample_training_job("basic_lora")
            job_data["job_name"] = f"perf_test_job_{job_index}"
            job_data["project_id"] = test_project["id"]
            
            start_time = time.perf_counter()
            response = await client.post("/api/v1/training/jobs", json=job_data)
            end_time = time.perf_counter()
            
            return {
                "job_index": job_index,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 201
            }
        
        # Create jobs concurrently
        start_time = time.perf_counter()
        
        tasks = [create_job(i) for i in range(num_jobs)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful_jobs = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results if r["success"]]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            
            # Performance assertions
            assert successful_jobs >= num_jobs * 0.90  # 90% success rate
            assert avg_response_time < 2.0  # Average under 2 seconds
            
            throughput = successful_jobs / total_time
            assert throughput > 5.0  # At least 5 jobs per second
            
            print(f"Training Job Creation Performance:")
            print(f"  Total jobs: {num_jobs}")
            print(f"  Successful: {successful_jobs}")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Throughput: {throughput:.2f} jobs/second")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self, client: AsyncClient, user_factory, project_factory, mock_all_providers):
        """Test system performance with multiple concurrent users."""
        num_users = 20
        operations_per_user = 10
        
        async def simulate_user(user_id):
            """Simulate a user performing various operations."""
            operations_completed = 0
            start_time = time.perf_counter()
            
            try:
                # Create project
                project_data = project_factory()
                project_data["user_id"] = user_id
                project_response = await client.post("/api/v1/projects", json=project_data)
                
                if project_response.status_code != 201:
                    return {"user_id": user_id, "operations_completed": 0, "success": False}
                
                project_id = project_response.json()["project_id"]
                operations_completed += 1
                
                # Create agent
                agent_config = SAMPLE_AGENTS["legal_agent"].copy()
                agent_config["name"] = f"user_{user_id}_agent"
                agent_config["project_id"] = project_id
                
                agent_response = await client.post("/api/v1/agents", json=agent_config)
                if agent_response.status_code == 201:
                    operations_completed += 1
                    agent_id = agent_response.json()["agent_id"]
                    
                    # Perform queries
                    for i in range(operations_per_user - 2):
                        query_data = {
                            "message": f"Test query {i} from user {user_id}",
                            "session_id": f"user_{user_id}_session"
                        }
                        
                        query_response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
                        if query_response.status_code == 200:
                            operations_completed += 1
                
                end_time = time.perf_counter()
                return {
                    "user_id": user_id,
                    "operations_completed": operations_completed,
                    "total_time": end_time - start_time,
                    "success": operations_completed >= operations_per_user * 0.8
                }
                
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "user_id": user_id,
                    "operations_completed": operations_completed,
                    "total_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Simulate concurrent users
        start_time = time.perf_counter()
        
        tasks = [simulate_user(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful_users = sum(1 for r in results if r["success"])
        total_operations = sum(r["operations_completed"] for r in results)
        
        # Performance assertions
        assert successful_users >= num_users * 0.70  # 70% of users successful
        assert total_operations >= num_users * operations_per_user * 0.60  # 60% of operations successful
        
        print(f"Concurrent User Simulation:")
        print(f"  Users: {num_users}")
        print(f"  Successful users: {successful_users}")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Operations per second: {total_operations/total_time:.2f}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test memory usage under sustained load."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Perform sustained operations
        num_operations = 200
        memory_samples = []
        
        for i in range(num_operations):
            # Perform operation
            query_data = {
                "message": f"Memory test query {i}",
                "session_id": f"memory_test_session_{i % 10}"  # Reuse 10 sessions
            }
            
            await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            
            # Sample memory every 10 operations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
        
        # Force garbage collection
        gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage analysis
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        
        print(f"Memory Usage Analysis:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Peak memory: {max_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        
        # Memory assertions
        assert memory_growth < 100  # Less than 100MB growth
        assert max_memory < initial_memory + 200  # Peak less than 200MB above initial
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_database_performance(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test database performance under load."""
        # Create many entities to test database performance
        num_agents = 50
        num_jobs = 30
        
        # Create agents
        agent_creation_times = []
        for i in range(num_agents):
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = f"db_perf_agent_{i}"
            agent_config["project_id"] = test_project["id"]
            
            start_time = time.perf_counter()
            response = await client.post("/api/v1/agents", json=agent_config)
            end_time = time.perf_counter()
            
            if response.status_code == 201:
                agent_creation_times.append(end_time - start_time)
        
        # Create training jobs
        job_creation_times = []
        for i in range(num_jobs):
            job_data = get_sample_training_job("basic_lora")
            job_data["job_name"] = f"db_perf_job_{i}"
            job_data["project_id"] = test_project["id"]
            
            start_time = time.perf_counter()
            response = await client.post("/api/v1/training/jobs", json=job_data)
            end_time = time.perf_counter()
            
            if response.status_code == 201:
                job_creation_times.append(end_time - start_time)
        
        # Test listing performance
        start_time = time.perf_counter()
        agents_response = await client.get("/api/v1/agents")
        agents_list_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        jobs_response = await client.get("/api/v1/training/jobs")
        jobs_list_time = time.perf_counter() - start_time
        
        # Performance assertions
        if agent_creation_times:
            avg_agent_creation = sum(agent_creation_times) / len(agent_creation_times)
            assert avg_agent_creation < 1.0  # Less than 1 second per agent
        
        if job_creation_times:
            avg_job_creation = sum(job_creation_times) / len(job_creation_times)
            assert avg_job_creation < 1.0  # Less than 1 second per job
        
        assert agents_list_time < 2.0  # List agents in under 2 seconds
        assert jobs_list_time < 2.0  # List jobs in under 2 seconds
        
        print(f"Database Performance:")
        print(f"  Average agent creation: {avg_agent_creation:.3f}s")
        print(f"  Average job creation: {avg_job_creation:.3f}s")
        print(f"  Agents list time: {agents_list_time:.3f}s")
        print(f"  Jobs list time: {jobs_list_time:.3f}s")


class TestScalabilityLimits:
    """Test system scalability limits."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_maximum_concurrent_connections(self, client: AsyncClient, test_user):
        """Test maximum number of concurrent connections."""
        max_connections = 100
        
        async def make_request(connection_id):
            try:
                response = await client.get("/api/v1/dashboard/overview")
                return {
                    "connection_id": connection_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "connection_id": connection_id,
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }
        
        # Create many concurrent connections
        start_time = time.perf_counter()
        
        tasks = [make_request(i) for i in range(max_connections)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        successful_connections = sum(1 for r in results if r["success"])
        
        # Should handle most connections successfully
        assert successful_connections >= max_connections * 0.80  # 80% success rate
        
        print(f"Connection Scalability Test:")
        print(f"  Attempted connections: {max_connections}")
        print(f"  Successful connections: {successful_connections}")
        print(f"  Success rate: {successful_connections/max_connections*100:.1f}%")
        print(f"  Total time: {total_time:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_data_handling(self, client: AsyncClient, test_user, test_project):
        """Test handling of large data payloads."""
        # Test with increasingly large payloads
        payload_sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
        
        for size in payload_sizes:
            large_description = "A" * size
            
            agent_config = {
                "name": f"large_data_test_{size}",
                "display_name": "Large Data Test",
                "description": large_description,
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "project_id": test_project["id"],
                "persona_prompt": "You are a helpful assistant.",
                "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
            }
            
            start_time = time.perf_counter()
            response = await client.post("/api/v1/agents", json=agent_config)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            
            if response.status_code == 201:
                print(f"  {size} bytes: {processing_time:.3f}s - SUCCESS")
                assert processing_time < 10.0  # Should process within 10 seconds
            elif response.status_code == 413:  # Payload too large
                print(f"  {size} bytes: REJECTED (too large)")
                # This is acceptable for very large payloads
                break
            else:
                print(f"  {size} bytes: ERROR {response.status_code}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_session_scalability(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test scalability with many concurrent sessions."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Test with many concurrent sessions
        num_sessions = 100
        queries_per_session = 5
        
        async def session_queries(session_id):
            successful_queries = 0
            
            for i in range(queries_per_session):
                query_data = {
                    "message": f"Query {i} from session {session_id}",
                    "session_id": f"scale_test_session_{session_id}"
                }
                
                try:
                    response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
                    if response.status_code == 200:
                        successful_queries += 1
                except Exception:
                    pass
            
            return successful_queries
        
        start_time = time.perf_counter()
        
        tasks = [session_queries(i) for i in range(num_sessions)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        total_successful_queries = sum(results)
        expected_queries = num_sessions * queries_per_session
        
        # Should handle most queries successfully
        success_rate = total_successful_queries / expected_queries
        assert success_rate >= 0.70  # 70% success rate
        
        print(f"Session Scalability Test:")
        print(f"  Sessions: {num_sessions}")
        print(f"  Queries per session: {queries_per_session}")
        print(f"  Total successful queries: {total_successful_queries}/{expected_queries}")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Total time: {total_time:.2f}s")


class TestResourceUtilization:
    """Test resource utilization under various loads."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cpu_utilization(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test CPU utilization under load."""
        # Monitor CPU usage during sustained operations
        process = psutil.Process()
        
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Baseline CPU usage
        baseline_cpu = process.cpu_percent(interval=1)
        
        # Perform intensive operations
        num_operations = 50
        cpu_samples = []
        
        async def cpu_intensive_operation(op_id):
            query_data = {
                "message": f"Complex query requiring processing {op_id}",
                "session_id": f"cpu_test_session_{op_id % 5}"
            }
            
            return await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        start_time = time.perf_counter()
        
        # Run operations in batches to monitor CPU
        batch_size = 10
        for i in range(0, num_operations, batch_size):
            batch_tasks = [
                cpu_intensive_operation(j) 
                for j in range(i, min(i + batch_size, num_operations))
            ]
            
            await asyncio.gather(*batch_tasks)
            
            # Sample CPU usage
            cpu_usage = process.cpu_percent(interval=0.1)
            cpu_samples.append(cpu_usage)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # CPU analysis
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0
        
        print(f"CPU Utilization Test:")
        print(f"  Baseline CPU: {baseline_cpu:.1f}%")
        print(f"  Average CPU under load: {avg_cpu:.1f}%")
        print(f"  Peak CPU: {max_cpu:.1f}%")
        print(f"  Operations: {num_operations}")
        print(f"  Total time: {total_time:.2f}s")
        
        # CPU should not exceed reasonable limits
        assert max_cpu < 90  # Peak CPU under 90%
        assert avg_cpu < 70  # Average CPU under 70%
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_disk_io_performance(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test disk I/O performance during file operations."""
        # Test file upload performance
        file_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        
        for size in file_sizes:
            file_content = "A" * size
            files = {"file": (f"perf_test_{size}.txt", file_content, "text/plain")}
            
            start_time = time.perf_counter()
            response = await client.post("/api/v1/data/upload", files=files)
            end_time = time.perf_counter()
            
            upload_time = end_time - start_time
            
            if response.status_code == 200:
                # Calculate throughput
                throughput = size / upload_time / 1024  # KB/s
                
                print(f"File Upload Performance ({size} bytes):")
                print(f"  Upload time: {upload_time:.3f}s")
                print(f"  Throughput: {throughput:.2f} KB/s")
                
                # Reasonable performance expectations
                assert upload_time < 5.0  # Under 5 seconds
                assert throughput > 1.0  # At least 1 KB/s
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_network_throughput(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test network throughput with large responses."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Test with queries that generate large responses
        large_query = "Please provide a very detailed explanation of contract law, including all major principles, examples, and case studies. Make it as comprehensive as possible."
        
        query_data = {
            "message": large_query,
            "session_id": "network_throughput_test"
        }
        
        start_time = time.perf_counter()
        response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            response_time = end_time - start_time
            response_size = len(response.content)
            throughput = response_size / response_time / 1024  # KB/s
            
            print(f"Network Throughput Test:")
            print(f"  Response size: {response_size} bytes")
            print(f"  Response time: {response_time:.3f}s")
            print(f"  Throughput: {throughput:.2f} KB/s")
            
            # Performance expectations
            assert response_time < 30.0  # Under 30 seconds
            assert throughput > 1.0  # At least 1 KB/s


@pytest.mark.stress
class TestStressTesting:
    """Stress tests to find system breaking points."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test system behavior under extreme concurrent load."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Extreme load test
        num_concurrent_requests = 200
        
        async def stress_request(request_id):
            try:
                query_data = {
                    "message": f"Stress test query {request_id}",
                    "session_id": f"stress_session_{request_id % 20}"
                }
                
                start_time = time.perf_counter()
                response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
                end_time = time.perf_counter()
                
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "status_code": None,
                    "response_time": None,
                    "success": False,
                    "error": str(e)
                }
        
        print(f"Starting extreme load test with {num_concurrent_requests} concurrent requests...")
        
        start_time = time.perf_counter()
        
        # Launch all requests concurrently
        tasks = [stress_request(i) for i in range(num_concurrent_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = num_concurrent_requests - successful_requests
        
        successful_response_times = [
            r["response_time"] for r in results 
            if r["success"] and r["response_time"] is not None
        ]
        
        if successful_response_times:
            avg_response_time = sum(successful_response_times) / len(successful_response_times)
            max_response_time = max(successful_response_times)
        else:
            avg_response_time = 0
            max_response_time = 0
        
        throughput = successful_requests / total_time
        
        print(f"Extreme Load Test Results:")
        print(f"  Total requests: {num_concurrent_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {failed_requests}")
        print(f"  Success rate: {successful_requests/num_concurrent_requests*100:.1f}%")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Max response time: {max_response_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} requests/second")
        print(f"  Total time: {total_time:.2f}s")
        
        # System should maintain some level of functionality under stress
        assert successful_requests > 0  # At least some requests should succeed
        
        # If success rate is too low, system might need optimization
        if successful_requests / num_concurrent_requests < 0.50:
            print("⚠️  Warning: Success rate under extreme load is below 50%")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test system behavior under memory pressure."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Monitor memory during sustained operations
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory pressure with many concurrent sessions
        num_sessions = 50
        queries_per_session = 20
        
        async def memory_intensive_session(session_id):
            session_name = f"memory_pressure_session_{session_id}"
            successful_queries = 0
            
            for i in range(queries_per_session):
                # Create queries with large context to increase memory usage
                large_query = f"Query {i}: " + "Please analyze this complex scenario. " * 100
                
                query_data = {
                    "message": large_query,
                    "session_id": session_name
                }
                
                try:
                    response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
                    if response.status_code == 200:
                        successful_queries += 1
                except Exception:
                    break  # Stop if we hit memory limits
            
            return successful_queries
        
        print(f"Starting memory pressure test...")
        
        start_time = time.perf_counter()
        
        tasks = [memory_intensive_session(i) for i in range(num_sessions)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Memory analysis
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        total_successful_queries = sum(results)
        expected_queries = num_sessions * queries_per_session
        
        print(f"Memory Pressure Test Results:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(f"  Successful queries: {total_successful_queries}/{expected_queries}")
        print(f"  Success rate: {total_successful_queries/expected_queries*100:.1f}%")
        print(f"  Total time: {total_time:.2f}s")
        
        # System should handle memory pressure gracefully
        assert total_successful_queries > 0  # Should complete some queries
        assert memory_growth < 500  # Memory growth should be reasonable (< 500MB)
        
        # Force garbage collection and check for memory leaks
        gc.collect()
        time.sleep(1)  # Allow cleanup
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_after_cleanup = cleanup_memory - initial_memory
        
        print(f"  Memory after cleanup: {cleanup_memory:.1f} MB")
        print(f"  Residual growth: {memory_after_cleanup:.1f} MB")
        
        # Should release most memory after cleanup
        assert memory_after_cleanup < memory_growth * 0.5  # At least 50% cleanup
