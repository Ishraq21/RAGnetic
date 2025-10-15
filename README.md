
<img width="20388" height="5692" alt="RAGnetic Logo" src="https://github.com/user-attachments/assets/92e1f139-9acb-43f3-9072-02d7f5336663" />


RAGnetic is an open-source, production-ready AI framework for building and deploying intelligent multi-agent systems with advanced RAG capabilities. It provides a complete ecosystem for creating specialized AI agents that can collaborate, execute code, search the web, analyze documents, and interact with databases. RAGnetic offers real-time streaming, sandboxed code execution, comprehensive API access, and enterprise-grade security with YAML-based configuration and a streamlined chat interface.

One of RAGnetic’s core philosophies is to give developers and organizations complete ownership of their on-premise AI while eliminating infrastructure overhead and reducing development time and cost. 

>  **Work in Progress:** This documentation is under active development. More tutorials, guides, and official documentation website are coming soon!
> 
---

## Table of Contents


- [Why Use RAGnetic?](#why-use-ragnetic)
- [Features](#features)
- [Multi-Agent Systems](#multi-agent-systems)
- [Custom Model Fine-Tuning](#custom-model-fine-tuning)
- [RAGnetic API](#ragnetic-api)
- [Enterprise & Government Readiness](#enterprise--government-readiness)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage & Commands](#usage--commands)
- [Hello World](#hello-world)
- [YAML Configuration Examples](#yaml-configuration-examples)
- [Citation Tracking & Explainable Reasoning](#citation-tracking--explainable-reasoning)
- [Troubleshooting](#troubleshooting)
- [Development & Contributing](#development--contributing)
- [License](#license)

---

## Why Use RAGnetic?

RAGnetic moves beyond simple RAG libraries by providing a full-stack platform that addresses the entire lifecycle of an AI agent, from data ingestion to evaluation and enterprise-grade security.


* **Multi-Agent Intelligence**: At its core, RAGnetic enables sophisticated multi-agent systems built on LangGraph. Agents can collaborate to reason, plan, and execute complex tasks across various tools and data sources, with each agent specialized for specific domains or capabilities.


* **Simplified Model Fine-Tuning**: RAGnetic provides streamlined LoRA fine-tuning for open-source models on your private data, with sensible defaults that work out of the box.


* **Production-Ready by Design:**  The framework is built with features crucial for enterprise deployment, including robust data ingestion, user and role management, and core performance validation.


* **Built-in Observability & Analytics:** Don't just deploy, understand. The platform provides detailed logging, metrics for LLM usage and costs, and a powerful CLI for auditing agent runs and performance in real-time.

---


## Features

An agent's intelligence is defined not just by its core LLM, but by the ecosystem of data, tools, and processes that surround it. RAGnetic is built on **five core pillars**:

### 1. Ingest: The Agent’s Knowledge Pipeline
- **Multi-Source Ingestion**: Load data from PDFs, DOCX, CSV, Parquet, web pages, code repos, and live SQL databases into a unified knowledge base.
- **Data Policies**: Enforce security/compliance during ingestion with built-in PII redaction and keyword filtering.
- **Hybrid Retrieval**: Combine semantic vector search with keyword-based retrieval (BM25) for precise, comprehensive results.

### 2. Adapt: The Agent's Specialized Intelligence
- **Custom & Hosted Models**: Use hosted LLMs (OpenAI, Anthropic, Google Gemini) or local open-source models via Ollama or Hugging Face.
- **Simplified Fine-Tuning**: Streamlined LoRA fine-tuning with sensible defaults for Hugging Face models, focusing on core training parameters.

### 3. Collaborate: Multi-Agent Intelligence
- **Stateful Conversations**: Durable, conversation-level memory for iterative dialogue without repeating context.
- **Agent Collaboration**: Primary agents can delegate to specialized sub-agents for domain-specific tasks.
- **Declarative YAML**: Configure agent behavior, tool access, and collaboration patterns in readable YAML format.
- **Dynamic Interaction**: Trigger agents via API calls, chat interface, or programmatic integration.

### 4. Interact: The Agent's Toolkit
- **Document Retrieval**: Advanced document retrieval with vector search, BM25 keyword matching, and hybrid retrieval strategies.
- **Code Execution**: Sandboxed Python code execution with automatic file staging and path rewriting.
- **Database Integration**: SQL toolkit for querying databases with natural language.
- **Web Search**: Real-time web search via Brave Search API for up-to-date information.
- **API Integration**: HTTP requests to external APIs with security controls and rate limiting.
- **Research Tools**: ArXiv integration for academic paper search and analysis.
- **Specialized Parsers**: Code analysis, SQL parsing, YAML/JSON processing, Terraform analysis, and Jupyter notebook parsing.

### 5. Evaluate: The Agent's Feedback Loop
- **Core Performance Validation**: Simple benchmarking to measure retrieval accuracy and response quality with pass/fail indicators.
- **Essential Analytics**: Track LLM usage, costs, and key metrics across agent runs and conversations.


## Multi-Agent Systems

RAGnetic's core power lies in its ability to create sophisticated multi-agent systems that can handle complex tasks requiring diverse expertise. The platform enables agents to collaborate seamlessly, with each agent specialized for specific domains or capabilities.

- **Agent Specialization**: Create agents with specific expertise - research, analysis, code generation, data processing, or domain knowledge.

- **Intelligent Collaboration**: Primary agents can delegate tasks to specialized sub-agents, combining their outputs for comprehensive solutions.

- **How to Use It**: Use the command <code>ragnetic deploy</code> to deploy agents with defined collaboration patterns, creating intelligent systems that can handle complex multi-step reasoning.

## Simplified Model Fine-Tuning

RAGnetic provides streamlined LoRA fine-tuning with sensible defaults, making it easy to train open-source models on your data.

**Key Features:**
- **LoRA Fine-Tuning**: Parameter-efficient training that works on consumer hardware
- **Sensible Defaults**: Minimal configuration required - just specify your data and model
- **Auto-Detection**: Automatic device detection and optimization

**Simple Workflow:**

1. Prepare your data with the <code>ragnetic dataset prepare</code> command.  
2. Create a simple YAML config with just the essentials.  
3. Run <code>ragnetic training apply -f your-config.yaml</code>  
4. Use the trained model in your agent configuration. 

## RAGnetic API

RAGnetic's entire functionality is exposed through a robust RESTful API, allowing you to programmatically interact with the framework from any application. This enables integration into larger enterprise ecosystems and custom application development.

### Core API Endpoints

- **Agent Management** (`/api/v1/agents`): Create, configure, and manage AI agents
- **Stateless Invocation** (`/api/v1/invoke/{deployment_id}`): Execute agents via API keys with rate limiting
- **Lambda Execution** (`/api/v1/lambda`): Run sandboxed Python code with file staging
- **Document Management** (`/api/v1/documents`): Upload and manage documents for agents
- **Temporary Documents** (`/api/v1/chat/upload-temp-document`): Quick file uploads for chat sessions

### Analytics & Monitoring

- **Usage Analytics** (`/api/v1/analytics`): LLM usage, costs, and performance metrics
- **System Monitoring** (`/api/v1/monitoring`): Resource usage, security metrics, and health checks
- **Audit Trails** (`/api/v1/audit`): Complete audit logs of agent runs and user actions
- **Metrics API** (`/api/v1/metrics`): Detailed performance and cost metrics

### Advanced Features

- **Fine-tuning Management** (`/api/v1/training`): Submit and monitor model fine-tuning jobs
- **Citation Management** (`/api/v1/citations`): Track and manage document citations
- **Deployment Management** (`/api/v1/deployments`): Manage API deployments and access keys
- **User Security** (`/security/me`): User authentication and permission management

### Real-time Features

- **WebSocket Chat** (`/ws`): Streaming conversations with real-time token delivery
- **File Upload Integration**: Seamless file handling in chat sessions
- **Cancellation Support**: Interrupt long-running agent operations
- **Citation Streaming**: Real-time citation tracking and display  

---

## Enterprise & Government Readiness

RAGnetic is built with security, compliance, and scalability in mind, making it ready for enterprise and government contracts.

- **Identity & Access Management (IAM):** Built-in user and role management with Role-Based Access Control (RBAC) and assignable permissions for granular security.  
- **Operational Logging & Auditing:** Tamper-resistant audit trails of all queries, agent executions, and document access. Logs can be configured to write to console, files, or a database for comprehensive monitoring.  
- **Deployment Flexibility:** Fully self-hosted or air-gapped deployments with first-class support for Docker and Kubernetes, ensuring data residency and compliance in regulated environments.  
- **Advanced Features:** Inline document citation, explainable reasoning, and an AI Governance Engine to monitor policy compliance and ensure traceability.  


## Getting Started

### Quick Installation

```bash
# Complete installation - everything included
pip install ragnetic
```

** For detailed installation options, see [INSTALL.md](INSTALL.md)**

### 5-Minute Setup

```bash
# 1. Install RAGnetic (includes all AI features)
pip install ragnetic

# 2. Initialize project
ragnetic init

# 3. Set API keys (interactive)
ragnetic set-api-key

# 4. Create admin user
ragnetic user create admin --superuser

# 5. Start server
ragnetic start-server
```

Visit `http://localhost:8000` to access the chat interface!

### Prerequisites

- **Python 3.9+** 
- **Redis** (for task queue)
- **API Keys** for AI providers (OpenAI, Anthropic, etc.)

### Installation Options

| Install Command | Use Case | What's Included |
|---|---|---|
| `pip install ragnetic` | Complete functionality | All AI providers, vector stores, document processing |
| `pip install ragnetic[training]` | + Model fine-tuning | + PyTorch, PEFT, LoRA adapters |
| `pip install ragnetic[gpu]` | + GPU acceleration | + CUDA support for training |

### System Requirements

| Component | Requirement |
|---|---|
| **Python** | 3.9, 3.10, 3.11, 3.12 |
| **Redis** | Any recent version |
| **Database** | SQLite (default) or PostgreSQL |
| **Memory** | 4GB+ recommended |
| **GPU** | Optional (NVIDIA/Apple Silicon) |

<img width="3432" height="1343" alt="RAGnetic New Chat Screen" src="https://github.com/user-attachments/assets/7077454d-44a0-4eac-a8f0-5d7cb09a8cbe" />

---

## Hello World

Start with a basic "Hello World Agent" 
```yaml
# agents/hello_world_agent.yaml
name: hello_world_agent
display_name: Hello World Agent
description: >
  A simple agent that always returns a “Hello, World!” message.
persona_prompt: >
  You are a friendly assistant. Your only task is to return the text "Hello, World!" 
  in a JSON object with the key `message`.

# (no external data needed)
sources: []

# Available tools for the agent
tools:
  - retriever       # Document retrieval with vector search
  - lambda_tool     # Sandboxed Python code execution
  - sql_toolkit     # Database queries (requires db source)
  - search_engine   # Web search via Brave Search API
  - api_toolkit     # HTTP requests to external APIs
  - arxiv           # ArXiv research paper search

# disable chunking
chunking:
  mode: none
  chunk_size: 0
  chunk_overlap: 0

# Vector store configuration
vector_store:
  type: faiss  # Options: faiss, chroma, qdrant, pinecone, mongodb_atlas
  bm25_k: 0
  semantic_k: 0
  retrieval_strategy: hybrid  # Options: hybrid, enhanced

```
You can deploy your agent using:
```bash
ragnetic deploy hello_world_agent
```

## Usage & Commands 

### Project & System Management

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic init` | Initializes a new project, creating a project structure and setting up a default database. | `ragnetic init` |
| `ragnetic configure` | Interactive wizard to configure system settings, databases, and secrets. | `ragnetic configure` |
| `ragnetic start-server` | Starts the RAGnetic server, worker, and scheduler. | `ragnetic start-server --reload` |
| `ragnetic set-server-key` | Generates and sets a secret key for the server API. | `ragnetic set-server-key` |
| `ragnetic set-api-key` | Interactive wizard to set API keys for external services. | `ragnetic set-api-key` |
| `ragnetic reset-db` | **DANGEROUS:** Drops all tables from the database to create a clean slate. | `ragnetic reset-db --force` |
| `ragnetic show-config` | Displays the current system configurations. | `ragnetic show-config` |
| `ragnetic check-system-db` | Verifies connections and migration status of configured databases. | `ragnetic check-system-db` |
| `ragnetic auth gdrive` | Authenticates with Google Drive for data ingestion. | `ragnetic auth gdrive` |
| `ragnetic test` | Runs the entire test suite using `pytest`. | `ragnetic test` |
| `ragnetic makemigrations` | Autogenerates a new database migration script based on model changes. | `ragnetic makemigrations -m "Added new field"` |
| `ragnetic migrate` | Applies database migrations to update the schema. | `ragnetic migrate head` |
| `ragnetic sync` | Manually stamps the database with a migration revision without running any SQL. | `ragnetic sync head` |
|| `ragnetic set-debug` | Enable or disable debug mode for detailed logging. | `ragnetic set-debug --enable` |

### User & Role Management

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic user create` | Creates a new user account with a password. | `ragnetic user create my_user --superuser` |
| `ragnetic user update` | Updates an existing user account by ID. | `ragnetic user update 1 --first-name "John"` |
| `ragnetic user delete` | Deletes a user account by ID. | `ragnetic user delete 1` |
| `ragnetic user list` | Lists all user accounts in the database. | `ragnetic user list` |
| `ragnetic user generate-key` | Generates a new API key for a user. | `ragnetic user generate-key 1` |
| `ragnetic user revoke-key` | Revokes a user's API key. | `ragnetic user revoke-key <key_string>` |
| `ragnetic login` | Logs in a user and saves their API key for the CLI. | `ragnetic login my_user` |
| `ragnetic logout` | Clears the active CLI login session. | `ragnetic logout` |
| `ragnetic whoami` | Displays the currently active user and their permissions. | `ragnetic whoami` |
| `ragnetic role create` | Creates a new role with an optional description. | `ragnetic role create admin` |
| `ragnetic role list` | Lists all roles and their permissions. | `ragnetic role list` |
| `ragnetic role delete` | Deletes a role by ID. | `ragnetic role delete 1` |
| `ragnetic role assign-permission` | Assigns a permission string to a role. | `ragnetic role assign-permission 1 agent:create` |
| `ragnetic role remove-permission` | Removes a permission from a role. | `ragnetic role remove-permission 1 agent:create` |

### Agent Management

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic list-agents` | Lists all configured agents. | `ragnetic list-agents` |
| `ragnetic deploy` | Deploys an agent by processing its data sources and building its vector store. | `ragnetic deploy my_research_agent` |
| `ragnetic inspect-agent` | Displays an agent's configuration and can check connections or document metadata. | `ragnetic inspect-agent my_research_agent --check-connections` |
| `ragnetic reset-agent` | Resets an agent by deleting its vector store and memory files. | `ragnetic reset-agent my_agent` |
| `ragnetic delete-agent` | Permanently deletes an agent's configuration and all its data. | `ragnetic delete-agent my_old_agent` |

### Training & Evaluation

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic dataset prepare` | Prepares a raw dataset for fine-tuning using a YAML configuration. | `ragnetic dataset prepare -f data_prep_configs/my_prep.yaml` |
| `ragnetic generate-test` | Generates a test set from an agent’s data sources. | `ragnetic generate-test my_agent -o test_set.json` |
| `ragnetic benchmark` | Runs core performance validation on an agent. | `ragnetic benchmark my_agent -t test_set.json` |
| `ragnetic training apply` | Submits a simplified fine-tuning job. | `ragnetic training apply -f configs/my_ft_job.yaml` |
| `ragnetic training status` | Checks the status of a fine-tuning job. | `ragnetic training status <adapter_id>` |
| `ragnetic training list-models` | Lists available fine-tuned models. | `ragnetic training list-models` |

### Analytics & Auditing

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic analytics usage` | Displays aggregated LLM usage and cost metrics. | `ragnetic analytics usage --agent my_agent` |
| `ragnetic analytics benchmarks` | Displays core performance validation results. | `ragnetic analytics benchmarks --agent my_agent` |
| `ragnetic analytics agent-runs` | Displays aggregated agent run metrics. | `ragnetic analytics agent-runs --agent my_agent` |
| `ragnetic analytics agent-steps` | Displays aggregated agent step metrics. | `ragnetic analytics agent-steps --agent my_agent` |
| `ragnetic inspect-run` | Inspects a specific agent run and its steps. | `ragnetic inspect-run <run_id>` |

## YAML Configuration Examples

RAGnetic uses a declarative YAML-based approach for configuring agents and training jobs.

### 1. Agent Configuration

This example defines an agent that uses an embedding model, a specific vector store, and a CSV data sources. (You can assign multiple data sources as well.)
```yaml
# agents/support_summarizer_agent.yaml
name: support_summarizer_agent
display_name: Support Summarizer Agent
description: >
  An agent specialized in summarizing raw support-ticket data into concise,
  human-readable summaries.
persona_prompt: >
  You are a helpful support summarizer. Given raw JSON input from a support
  ticket, generate a clear, single-sentence summary of the issue.

# Embedding + LLM + optional evaluation model
embedding_model: "text-embedding-3-small"
llm_model:       "gpt-4o-mini"
evaluation_llm_model: "ollama/llama3"

# Data sources for the agent
sources: 
  - type: local
    path: ./data/customers.csv
  # Other supported source types: url, code_repository, db, gdoc, web_crawler, api, notebook, parquet, pdf, txt, docx
  # Example configurations:
  # - type: url
  #   url: "https://example.com/api/data"
  #   headers: {"Authorization": "Bearer token"}
  # - type: db  
  #   db_connection: "postgresql://user:pass@localhost/db"
  # - type: code_repository
  #   url: "https://github.com/user/repo"
  #   max_depth: 3

# Available tools for the agent
tools:
  - retriever       # Document retrieval with vector search
  - lambda_tool     # Sandboxed Python code execution
  - sql_toolkit     # Database queries (requires db source)
  - search_engine   # Web search via Brave Search API
  - api_toolkit     # HTTP requests to external APIs
  - arxiv           # ArXiv research paper search

# Disable chunking since payloads are small
chunking:
  mode: none
  chunk_size: 0
  chunk_overlap: 0

# Vector store configuration
vector_store:
  type: faiss  # Options: faiss, chroma, qdrant, pinecone, mongodb_atlas
  bm25_k: 0
  semantic_k: 0
  retrieval_strategy: none  # Options: hybrid, enhanced, none

# (Optional) you can still enforce data policies on the incoming JSON
data_policies:
  - type: pii_redaction
    pii_config:
      types:
        - email
        - phone
      redaction_placeholder: "[REDACTED]"


```


### 2. Fine-Tuning Job Configuration
This example shows how to configure a simplified fine-tuning job.
```yaml
# training_configs/my_agent_finetune.yaml
job_name: my_agent_finetune
base_model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset_path: data/prepared_datasets/my_training_data.jsonl
output_base_dir: models/fine_tuned

hyperparameters:
  learning_rate: 0.0002
  epochs: 3
  batch_size: 4
  lora_rank: 8
  lora_alpha: 16
  device: "auto"

```

### 3. Data Preparation Configuration

```yaml
# data_prep_configs/jsonl_qa_prep.yaml
prep_name: jsonl_qa_prep
format_type: jsonl-instruction
input_file: data_raw/raw_qa_data.jsonl
output_file: data_prepared/prepared_qa_data.jsonl
```

## Lambda Tool: Sandboxed Code Execution

RAGnetic includes a powerful Lambda Tool that allows agents to execute Python code in a secure, sandboxed environment. This enables dynamic computation, data analysis, and integration with external libraries.

### Features

- **Secure Sandbox**: Isolated Python execution environment with restricted access
- **File Integration**: Automatic staging of uploaded documents for code processing
- **Path Rewriting**: Intelligent file path resolution for uploaded documents
- **Code Normalization**: Automatic markdown fence stripping and escape sequence handling
- **Resource Management**: Configurable timeouts and resource limits

### Example Usage

```python
# Lambda Tool automatically stages uploaded files
import pandas as pd

# Read uploaded CSV file (path automatically resolved)
df = pd.read_csv('/work/uploaded_data.csv')
result = df.describe()

# Process and return results
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
return result.to_dict()
```

### Configuration

The Lambda Tool is enabled by adding `lambda_tool` to your agent's tools list:

```yaml
tools:
  - lambda_tool
```

## Temporary Documents

RAGnetic supports temporary document uploads for chat sessions, allowing users to quickly share files with agents without permanent storage.

### Features

- **Session-based Storage**: Documents are automatically cleaned up when sessions end
- **Multiple Formats**: Support for PDF, CSV, TXT, DOCX, and more
- **Automatic Processing**: Documents are immediately available for retrieval and Lambda Tool processing
- **Security**: Documents are isolated per user and session

### API Usage

```bash
# Upload a temporary document
curl -X POST "http://localhost:8000/api/v1/chat/upload-temp-document" \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F "thread_id=your-thread-id"
```

## WebSocket Real-time Communication

RAGnetic provides a WebSocket interface for real-time, streaming interactions with agents. This enables responsive user experiences with live token streaming and interactive features.

### Features

- **Token Streaming**: Real-time delivery of agent responses as they are generated
- **Cancellation Support**: Interrupt long-running operations with `interrupt` message type
- **File Upload Integration**: Seamless file handling during conversations
- **Citation Streaming**: Real-time citation tracking and display
- **Session Management**: Persistent conversation state across WebSocket connections

### WebSocket Protocol

Connect to `/ws` endpoint with authentication:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws', [], {
  headers: {
    'X-API-Key': 'your-api-key'
  }
});

// Send query with files
ws.send(JSON.stringify({
  type: 'query',
  payload: {
    agent: 'my_agent',
    thread_id: 'optional-thread-id',
    query: 'Your question here',
    files: [
      {
        file_name: 'document.pdf',
        temp_doc_id: 'uploaded-file-id'
      }
    ]
  }
}));

// Handle streaming responses
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.token) {
    // Stream token to UI
    appendToken(data.token);
  }
  
  if (data.done) {
    // Response complete
    finalizeResponse(data.citations);
  }
};

// Cancel generation
ws.send(JSON.stringify({ type: 'interrupt' }));
```

### Message Types

- **Query**: `{ type: 'query', payload: { agent, thread_id?, query, files? } }`
- **Interrupt**: `{ type: 'interrupt' }` - Cancel current generation
- **Token**: `{ token: string }` - Streaming response token
- **Done**: `{ done: true, citations: [...] }` - Response complete

## Deployment API

RAGnetic provides a stateless deployment API for programmatic agent invocation with API key authentication and rate limiting.

### Features

- **API Key Authentication**: Secure access with user-specific API keys
- **Rate Limiting**: Configurable rate limits based on user scope and permissions
- **Credit System**: Usage tracking and credit management for enterprise deployments
- **Stateless Execution**: No session management required for simple queries

### Usage

```bash
# Create a deployment
curl -X POST "http://localhost:8000/api/v1/deployments" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-deployment",
    "agent_name": "research_agent",
    "description": "Research assistant deployment"
  }'

# Invoke agent via deployment ID
curl -X POST "http://localhost:8000/api/v1/invoke/1" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "files": [
      {
        "file_name": "research_paper.pdf",
        "temp_doc_id": "uploaded-file-id"
      }
    ]
  }'
```

### Rate Limiting

Rate limits are applied based on API key scope:
- **Viewer**: 10 requests/minute
- **User**: 60 requests/minute  
- **Admin**: 300 requests/minute
- **Superuser**: 1000 requests/minute

## Citation Tracking & Explainable Reasoning

RAGnetic's agents are designed to produce traceable and auditable outputs, a key requirement for regulated industries.

- **Inline Document Citation**  
  The agent generates inline citations (`[1]`, `[2]`) in its response, which link back to the specific chunk of a source document used.

- **Viewing Citations**  
  In the web UI, these citations are rendered as clickable links. Via the API and CLI, you can inspect a run’s output to see the full citation data, including `doc_name`, `page_number`, and `chunk_content`.

- **How it Works**  
  During retrieval, metadata (`source`, `page`) is attached to each chunk. When the LLM generates a response, the system parses the output for citation markers and uses the stored metadata to reconstruct the full citation trail.

---

## Troubleshooting

### Common Issues & Solutions

- **Server Fails to Start**  
  Check the logs for port conflicts or missing dependencies. The <code>--reload</code> flag can provide more verbose error messages.

- **ConnectionRefusedError**  
  Ensure the RAGnetic server is running with <code>ragnetic start-server</code> and that no firewall is blocking access to the configured host and port.

- **FileNotFoundError**  
  Verify that all file paths in your agent YAML files are correct and that the necessary files exist.

- **Database Migration Errors**  
  If you encounter errors during <code>ragnetic migrate</code>, it could be due to a malformed connection string or an incompatible database schema. Try running <code>ragnetic check-system-db</code> to diagnose the          connection.

- **Ollama or Hugging Face Model Issues**  
  Ensure that local models are correctly installed and running, and that the <code>base_model_name</code> in your YAML is a valid identifier from the Hugging Face Hub.

---

## Development & Contributing

### Contributing

We welcome contributions! Our **CONTRIBUTING.md** is currently being drafted—check back soon for guidelines on submitting bug reports, feature requests, and pull requests.

### Running Tests

To run the full test suite:

<code>ragnetic test</code>

## Community & Support

### Community

- **Discord Server:** Invite link coming soon!
- **X (formerly Twitter):** Follow `@RAGneticAI` for updates—account launching soon.

### Support

- **GitHub Discussions:** Coming soon!
- **GitHub Issues:** If you encounter a bug or reproducible error, please open an issue on our `https://github.com/your-repo/ragnetic/issues` tracker and include your OS, RAGnetic version, and               reproduction steps.

### Code of Conduct

Our community `CODE_OF_CONDUCT.md` is being finalized—check back soon for the full details.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

## Created by Mirza

**Twitter:** [@mirzaishraq](https://twitter.com/mirzaishraq)
