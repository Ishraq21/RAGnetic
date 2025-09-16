# Integration tests for agents API endpoints
import pytest
import json
from httpx import AsyncClient
from tests.fixtures.sample_data import SAMPLE_AGENTS, SAMPLE_TEST_SETS


class TestAgentsAPI:
    """Test agents API endpoints integration."""
    
    @pytest.mark.asyncio
    async def test_create_agent_success(self, client: AsyncClient, test_user, test_project):
        """Test successful agent creation."""
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        response = await client.post("/api/v1/agents", json=agent_config)
        
        assert response.status_code == 201
        result = response.json()
        assert result["name"] == agent_config["name"]
        assert result["project_id"] == test_project["id"]
        assert "agent_id" in result
    
    @pytest.mark.asyncio
    async def test_create_agent_invalid_config(self, client: AsyncClient, test_user, test_project):
        """Test agent creation with invalid configuration."""
        invalid_config = {
            "name": "",  # Empty name
            "llm_model": "nonexistent-model",
            "project_id": test_project["id"]
        }
        
        response = await client.post("/api/v1/agents", json=invalid_config)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_get_agent_success(self, client: AsyncClient, test_user, test_project):
        """Test getting agent details."""
        # Create agent first
        agent_config = SAMPLE_AGENTS["code_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Get agent
        response = await client.get(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["agent_id"] == agent_id
        assert result["name"] == agent_config["name"]
        assert result["llm_model"] == agent_config["llm_model"]
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, client: AsyncClient, test_user):
        """Test getting non-existent agent."""
        response = await client.get("/api/v1/agents/nonexistent-agent-id")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_list_agents(self, client: AsyncClient, test_user, test_project):
        """Test listing agents."""
        # Create multiple agents
        agent_names = ["agent1", "agent2", "agent3"]
        for name in agent_names:
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = name
            agent_config["project_id"] = test_project["id"]
            await client.post("/api/v1/agents", json=agent_config)
        
        # List agents
        response = await client.get("/api/v1/agents")
        
        assert response.status_code == 200
        result = response.json()
        assert "agents" in result
        assert len(result["agents"]) >= 3
        
        # Check agent names
        returned_names = [agent["name"] for agent in result["agents"]]
        for name in agent_names:
            assert name in returned_names
    
    @pytest.mark.asyncio
    async def test_list_agents_with_filters(self, client: AsyncClient, test_user, test_project):
        """Test listing agents with filters."""
        # Create agents with different models
        gpt4_agent = SAMPLE_AGENTS["legal_agent"].copy()
        gpt4_agent["name"] = "gpt4_agent"
        gpt4_agent["llm_model"] = "gpt-4o"
        gpt4_agent["project_id"] = test_project["id"]
        await client.post("/api/v1/agents", json=gpt4_agent)
        
        gpt35_agent = SAMPLE_AGENTS["customer_support"].copy()
        gpt35_agent["name"] = "gpt35_agent"
        gpt35_agent["llm_model"] = "gpt-3.5-turbo"
        gpt35_agent["project_id"] = test_project["id"]
        await client.post("/api/v1/agents", json=gpt35_agent)
        
        # Filter by model
        response = await client.get("/api/v1/agents?llm_model=gpt-4o")
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["agents"]) >= 1
        
        # All returned agents should use gpt-4o
        for agent in result["agents"]:
            assert agent["llm_model"] == "gpt-4o"
    
    @pytest.mark.asyncio
    async def test_update_agent(self, client: AsyncClient, test_user, test_project):
        """Test updating agent configuration."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Update agent
        update_data = {
            "description": "Updated description",
            "llm_model": "gpt-4o-mini",
            "persona_prompt": "Updated persona prompt"
        }
        
        response = await client.patch(f"/api/v1/agents/{agent_id}", json=update_data)
        
        assert response.status_code == 200
        result = response.json()
        assert result["description"] == update_data["description"]
        assert result["llm_model"] == update_data["llm_model"]
        assert result["persona_prompt"] == update_data["persona_prompt"]
    
    @pytest.mark.asyncio
    async def test_delete_agent(self, client: AsyncClient, test_user, test_project):
        """Test deleting agent."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Delete agent
        response = await client.delete(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 204
        
        # Verify agent is deleted
        get_response = await client.get(f"/api/v1/agents/{agent_id}")
        assert get_response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_agent_query_success(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test querying an agent."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Query agent
        query_data = {
            "message": "What is a contract?",
            "session_id": "test-session-123"
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "response" in result
        assert "session_id" in result
        assert result["session_id"] == query_data["session_id"]
        assert len(result["response"]) > 0
    
    @pytest.mark.asyncio
    async def test_agent_query_streaming(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test streaming agent query."""
        # Create agent
        agent_config = SAMPLE_AGENTS["code_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Query with streaming
        query_data = {
            "message": "Explain Python functions",
            "stream": True
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        # Streaming should return 200 with server-sent events
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
    
    @pytest.mark.asyncio
    async def test_agent_query_with_context(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test agent query with conversation context."""
        # Create agent
        agent_config = SAMPLE_AGENTS["customer_support"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        session_id = "context-test-session"
        
        # First query
        query1_data = {
            "message": "Hello, I need help with my account",
            "session_id": session_id
        }
        
        response1 = await client.post(f"/api/v1/agents/{agent_id}/query", json=query1_data)
        assert response1.status_code == 200
        
        # Follow-up query (should have context)
        query2_data = {
            "message": "What are my options?",
            "session_id": session_id
        }
        
        response2 = await client.post(f"/api/v1/agents/{agent_id}/query", json=query2_data)
        assert response2.status_code == 200
        
        result2 = response2.json()
        assert "response" in result2
        # Response should be contextually relevant
    
    @pytest.mark.asyncio
    async def test_agent_query_invalid_input(self, client: AsyncClient, test_user, test_project):
        """Test agent query with invalid input."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Query with empty message
        query_data = {
            "message": "",  # Empty message
            "session_id": "test-session"
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_agent_session_management(self, client: AsyncClient, test_user, test_project):
        """Test agent session management."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        session_id = "session-mgmt-test"
        
        # Get session (should create if not exists)
        response = await client.get(f"/api/v1/agents/{agent_id}/sessions/{session_id}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["session_id"] == session_id
        assert "messages" in result
        assert isinstance(result["messages"], list)
    
    @pytest.mark.asyncio
    async def test_clear_agent_session(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test clearing agent session."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        session_id = "clear-session-test"
        
        # Have a conversation first
        query_data = {
            "message": "Hello",
            "session_id": session_id
        }
        await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        # Clear session
        response = await client.delete(f"/api/v1/agents/{agent_id}/sessions/{session_id}")
        
        assert response.status_code == 204
        
        # Verify session is cleared
        session_response = await client.get(f"/api/v1/agents/{agent_id}/sessions/{session_id}")
        assert session_response.status_code == 200
        result = session_response.json()
        assert len(result["messages"]) == 0


class TestAgentTools:
    """Test agent tools integration."""
    
    @pytest.mark.asyncio
    async def test_agent_with_retriever_tool(self, client: AsyncClient, test_user, test_project, test_data_dir):
        """Test agent with retriever tool."""
        # Create agent with retriever tool
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        agent_config["tools"] = ["retriever"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Upload some documents for retrieval
        document_content = "A contract is a legally binding agreement between two or more parties."
        files = {"file": ("legal_doc.txt", document_content, "text/plain")}
        
        upload_response = await client.post(
            f"/api/v1/agents/{agent_id}/documents",
            files=files
        )
        assert upload_response.status_code == 200
        
        # Query agent (should use retriever)
        query_data = {
            "message": "What is a contract?",
            "session_id": "retriever-test"
        }
        
        with patch('app.tools.retriever.search_documents') as mock_search:
            mock_search.return_value = [{"content": document_content, "score": 0.9}]
            
            response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            
            assert response.status_code == 200
            result = response.json()
            assert "contract" in result["response"].lower()
    
    @pytest.mark.asyncio
    async def test_agent_with_lambda_tool(self, client: AsyncClient, test_user, test_project):
        """Test agent with Lambda (code execution) tool."""
        # Create agent with lambda tool
        agent_config = SAMPLE_AGENTS["code_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        agent_config["tools"] = ["lambda"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Query agent to execute code
        query_data = {
            "message": "Calculate 2 + 2 using Python",
            "session_id": "lambda-test"
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        # Should contain the calculation result
        assert "4" in result["response"] or "four" in result["response"].lower()
    
    @pytest.mark.asyncio
    async def test_agent_with_search_tool(self, client: AsyncClient, test_user, test_project):
        """Test agent with web search tool."""
        # Create agent with search tool
        agent_config = SAMPLE_AGENTS["customer_support"].copy()
        agent_config["project_id"] = test_project["id"]
        agent_config["tools"] = ["search"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Query agent to search for information
        query_data = {
            "message": "What's the latest news about AI?",
            "session_id": "search-test"
        }
        
        with patch('app.tools.search.web_search') as mock_search:
            mock_search.return_value = [
                {"title": "AI News", "content": "Latest AI developments...", "url": "https://example.com"}
            ]
            
            response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            
            assert response.status_code == 200
            result = response.json()
            assert len(result["response"]) > 0


class TestAgentDocuments:
    """Test agent document management."""
    
    @pytest.mark.asyncio
    async def test_upload_agent_document(self, client: AsyncClient, test_user, test_project):
        """Test uploading document to agent."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Upload document
        document_content = "Sample legal document content for testing."
        files = {"file": ("sample_doc.txt", document_content, "text/plain")}
        
        response = await client.post(
            f"/api/v1/agents/{agent_id}/documents",
            files=files
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "document_id" in result
        assert result["filename"] == "sample_doc.txt"
        assert result["status"] == "uploaded"
    
    @pytest.mark.asyncio
    async def test_list_agent_documents(self, client: AsyncClient, test_user, test_project):
        """Test listing agent documents."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Upload multiple documents
        for i in range(3):
            files = {"file": (f"doc_{i}.txt", f"Document {i} content", "text/plain")}
            await client.post(f"/api/v1/agents/{agent_id}/documents", files=files)
        
        # List documents
        response = await client.get(f"/api/v1/agents/{agent_id}/documents")
        
        assert response.status_code == 200
        result = response.json()
        assert "documents" in result
        assert len(result["documents"]) >= 3
    
    @pytest.mark.asyncio
    async def test_delete_agent_document(self, client: AsyncClient, test_user, test_project):
        """Test deleting agent document."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Upload document
        files = {"file": ("to_delete.txt", "Document to delete", "text/plain")}
        upload_response = await client.post(f"/api/v1/agents/{agent_id}/documents", files=files)
        document_id = upload_response.json()["document_id"]
        
        # Delete document
        response = await client.delete(f"/api/v1/agents/{agent_id}/documents/{document_id}")
        
        assert response.status_code == 204
        
        # Verify document is deleted
        list_response = await client.get(f"/api/v1/agents/{agent_id}/documents")
        documents = list_response.json()["documents"]
        document_ids = [doc["document_id"] for doc in documents]
        assert document_id not in document_ids


class TestAgentEvaluation:
    """Test agent evaluation endpoints."""
    
    @pytest.mark.asyncio
    async def test_evaluate_agent(self, client: AsyncClient, test_user, test_project):
        """Test evaluating agent performance."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Create evaluation
        eval_data = {
            "name": "Legal Agent Evaluation",
            "test_set": SAMPLE_TEST_SETS["domain_qa"],
            "metrics": ["accuracy", "response_time", "cost"]
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/evaluate", json=eval_data)
        
        assert response.status_code == 202  # Accepted for processing
        result = response.json()
        assert "evaluation_id" in result
        assert result["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_get_evaluation_results(self, client: AsyncClient, test_user, test_project):
        """Test getting evaluation results."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Start evaluation
        eval_data = {
            "name": "Test Evaluation",
            "test_set": SAMPLE_TEST_SETS["basic_qa"],
            "metrics": ["accuracy"]
        }
        
        eval_response = await client.post(f"/api/v1/agents/{agent_id}/evaluate", json=eval_data)
        evaluation_id = eval_response.json()["evaluation_id"]
        
        # Get evaluation results
        response = await client.get(f"/api/v1/evaluations/{evaluation_id}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["evaluation_id"] == evaluation_id
        assert "status" in result
        # Results might be empty if evaluation hasn't completed
        assert "results" in result
    
    @pytest.mark.asyncio
    async def test_list_agent_evaluations(self, client: AsyncClient, test_user, test_project):
        """Test listing agent evaluations."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Create multiple evaluations
        for i in range(3):
            eval_data = {
                "name": f"Evaluation {i}",
                "test_set": SAMPLE_TEST_SETS["basic_qa"],
                "metrics": ["accuracy"]
            }
            await client.post(f"/api/v1/agents/{agent_id}/evaluate", json=eval_data)
        
        # List evaluations
        response = await client.get(f"/api/v1/agents/{agent_id}/evaluations")
        
        assert response.status_code == 200
        result = response.json()
        assert "evaluations" in result
        assert len(result["evaluations"]) >= 3


@pytest.mark.performance
class TestAgentsAPIPerformance:
    """Performance tests for agents API."""
    
    @pytest.mark.asyncio
    async def test_agent_query_performance(self, client: AsyncClient, test_user, test_project, mock_all_providers, benchmark):
        """Benchmark agent query performance."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        query_data = {
            "message": "What is a contract?",
            "session_id": "perf-test"
        }
        
        async def query_agent():
            response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            return response.status_code
        
        result = await benchmark(query_agent)
        assert result == 200
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_queries(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test concurrent queries to same agent."""
        import asyncio
        
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        async def query_agent(session_id):
            query_data = {
                "message": f"Test query from session {session_id}",
                "session_id": session_id
            }
            return await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        # Run 10 concurrent queries
        tasks = [query_agent(f"session_{i}") for i in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
