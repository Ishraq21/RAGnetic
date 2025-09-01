
<img width="20388" height="5692" alt="RAGnetic Logo" src="https://github.com/user-attachments/assets/92e1f139-9acb-43f3-9072-02d7f5336663" />


RAGnetic is an open-source, highly configurable, compliance-first AI framework for building and deploying production-ready agents and multi-step workflows. It offers full control at every layer, from data ingestion and vector embeddings to retrieval pipelines, benchmarking, and deployment with no infrastrucutre overhead. RAGnetic provides YAML for developers who need infra-as-code and automation, and a GUI for business users who need visibility and management without coding. Both share the same backend, enabling seamless collaboration.

One of RAGnetic’s core philosophies is to give organizations complete ownership of their on-premise AI while eliminating infrastructure overhead and reducing development time and cost. 

> 🚧 **Work in Progress:** This documentation is under active development. More tutorials, guides, and official documentation website are coming soon!
> 
---

## Table of Contents


- [Why Use RAGnetic?](#why-use-ragnetic)
- [Features](#features)
- [Multi-Agent Workflows & Orchestration](#multi-agent-workflows--orchestration)
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


* **Agentic Orchestration**: At its core, RAGnetic uses a multi-agent, multi-step system built on the powerful LangGraph. This allows you to define complex, stateful workflows where multiple agents can collaborate to reason, plan, and execute tasks across various tools and data sources.


* **Custom Model Fine-Tuning**: RAGnetic empowers you to fine-tune open-source Hugging Face models on your private data, making them specialized for your specific domain and use cases. This is crucial for improving accuracy and controlling an agent's behavior.


* **Production-Ready by Design:**  The framework is built with features crucial for enterprise deployment, including robust data ingestion, user and role management, performance benchmarking, and detailed analytics.


* **Built-in Observability & Analytics:** Don't just deploy, understand. The platform provides detailed logging, metrics for LLM usage and costs, and a powerful CLI for auditing agent and workflow runs in real-time.

---


## Features

An agent's intelligence is defined not just by its core LLM, but by the ecosystem of data, tools, and processes that surround it. RAGnetic is built on **five core pillars**:

### 1. Ingest: The Agent’s Knowledge Pipeline
- **Multi-Source Ingestion**: Load data from PDFs, DOCX, CSV, Parquet, web pages, code repos, and live SQL databases into a unified knowledge base.
- **Data Policies**: Enforce security/compliance during ingestion with built-in PII redaction and keyword filtering.
- **Hybrid Retrieval**: Combine semantic vector search with keyword-based retrieval (BM25) for precise, comprehensive results.

### 2. Adapt: The Agent’s Specialized Intelligence
- **Custom & Hosted Models**: Use hosted LLMs (OpenAI, Anthropic, Google Gemini) or local open-source models via Ollama or Hugging Face.
- **Fine-Tuning**: Full fine-tuning pipeline for Hugging Face models using LoRA (PEFT) or full-model training, to control behavior, tone, and accuracy.

### 3. Orchestrate: The Agent’s Strategic Workflow
- **Stateful Conversations**: Durable, conversation-level memory for iterative dialogue without repeating context.
- **Multi-Agent Workflows**: Primary orchestrator agent can call on specialized sub-agents to complete complex tasks.
- **Declarative YAML**: Define conditional logic, loops, and human-in-the-loop steps in a readable YAML format.
- **Dynamic Triggers**: Kick off workflows via API webhooks, schedules, or manual triggers.

### 4. Interact: The Agent’s Toolkit
- **Core Toolkit**: Pre-built tools for real-world tasks: SQL queries, HTTP requests, sandboxed Python scripts, email notifications.
- **Specialized Parsers**: Structured analysis on SQL, YAML, Terraform, and more.

### 5. Evaluate: The Agent’s Feedback Loop
- **Automated Benchmarking**: Generate test sets and run benchmarks to measure accuracy, faithfulness, and relevance.
- **Detailed Analytics**: Track LLM usage, costs, and step-by-step metrics across agent and workflow runs.


## Multi-Agent Workflows & Orchestration

RAGnetic’s core power lies in its ability to orchestrate complex tasks that go beyond a single chat prompt. The platform’s multi-agent workflow engine allows you to chain together multiple steps, where each step can be a call to a specialized tool or a call to another agent.

- **Define Complex Logic**: Use declarative YAML files to define workflows with conditional logic, loops, and even human-in-the-loop steps.

- **Orchestrate Multiple Agents**: Define an orchestrator agent that calls on a roster of specialized sub-agents to complete different parts of a complex task.

- **How to Use It**: Use the command <code>ragnetic deploy-orchestrator</code> to deploy a primary orchestrator and all its constituent sub-agents, creating a complete, interconnected system.

## Custom Model Fine-Tuning

RAGnetic provides a full-featured fine-tuning pipeline, allowing you to train open-source Hugging Face models on your proprietary data to create highly specialized agents.

**Parameter-Efficient Fine-Tuning (PEFT):** The system uses LoRA (Low-Rank Adaptation) to fine-tune models, drastically reducing the number of trainable parameters. This makes it possible to fine-tune large models on consumer-grade GPUs or even Apple Silicon (MPS).

**Supported Data Formats:**  

- **jsonl-instruction:** For training models on Q&A or instruction-following tasks.  
- **conversational-jsonl:** For training models on multi-turn chat dialogues.

**How to Use It:**

1. Prepare your data with the <code>ragnetic dataset prepare</code> command.  
2. Define your training parameters in a YAML file.  
3. Submit the job with ragnetic <code>training apply -f `your-training-config.yaml`.</code>  
4. Use the <code>fine_tuned_model_id</code> in your agent’s configuration to deploy it. 

## RAGnetic API

RAGnetic’s entire functionality is exposed through a robust RESTful API, allowing you to programmatically interact with the framework from any application. This is how RAGnetic integrates into larger enterprise ecosystems, enabling you to build custom frontends, connect to external services, and automate workflows.

- **API-Driven Workflows:** Trigger any RAGnetic workflow from an external service via a simple HTTP POST request to a dedicated webhook URL.  
- **Programmatic Management:** Use API endpoints to manage agents, users, roles, and fine-tuning jobs.  
- **Real-time Interaction:** A WebSocket-based chat interface allows for streaming, real-time interactions with your agents.  
- **Data & Analytics Access:** Pull detailed performance metrics, cost data, and audit logs directly from the API for integration with dashboards and reporting tools.  

---

## Enterprise & Government Readiness

RAGnetic is built with security, compliance, and scalability in mind, making it ready for enterprise and government contracts.

- **Identity & Access Management (IAM):** Built-in user and role management with Role-Based Access Control (RBAC) and assignable permissions for granular security.  
- **Operational Logging & Auditing:** Tamper-resistant audit trails of all queries, agent executions, and document access. Logs can be configured to write to console, files, or a database for comprehensive monitoring.  
- **Deployment Flexibility:** Fully self-hosted or air-gapped deployments with first-class support for Docker and Kubernetes, ensuring data residency and compliance in regulated environments.  
- **Advanced Features:** Inline document citation, explainable reasoning, and an AI Governance Engine to monitor policy compliance and ensure traceability.  


## Getting Started

### Prerequisites

- Python 3.9 or higher
- <code>git</code>
- <code>redis</code>
- Operating System: RAGnetic supports Linux, macOS, and Windows. Specific
  hardware limitations (e.g., <code>bitsandbytes</code> requires NVIDIA CUDA)



1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-repo/ragnetic.git
    cd ragnetic
    ```

2.  **Install Dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate    # macOS/Linux
    .venv\Scripts\activate       # Windows
    pip install . -e
    
    ```
    To enable GPU acceleration for fine-tuning or embeddings, install the optional gpu or mps
    dependencies that match your hardware.

    For NVIDIA GPUs: <code>pip install ".[gpu]"</code>

    For Apple Silicon (M-series) GPUs: <code>pip install ".[mps]"</code>

3.  **Initialize the Project:**
    This command creates the necessary project structure and sets up a default database. RAGnetic ships with a pre-migrated SQLite database, so no further database setup is required for the default experience.
    ```bash
    ragnetic init
    ```
4. **Set your API Keys:**
   The framework needs API keys to use external services. Use the interactive <code>set-api-key</code> command to set a master administrative key, which is used for initial setup and emergency access.
   ```bash
    ragnetic set-api-key
   ```
5. **Create an Admin User:**
   Create an Admin User:
   Before you can log in, you need to create a user account. Use the master key you just set to create a user with superuser privileges (the equivalent of an admin user). You'll be prompted to set a password.
   ```bash
   ragnetic user create my_admin_user --superuser
   ```
   The <code>ragnetic set-server-key</code> command (we saw on the previous step) creates a master API key which is a global key with full access. It should be reserved for administrative and emergency situations.

6. **Start the Server**:
   Start the RAGnetic server, Celery worker, and scheduler in a single command.
   ```bash
    ragnetic start-server
   ```
You can use the <code>--reload</code> flag for development. Remove it for production.

7. **Access the Web UI:**
   With the server running, navigate to http://127.0.0.1:8000 in your browser. You will be redirected to the login page.
   Enter the username and password for the <code>my_admin_user</code> you created earlier to log in. You now have access to the full administrative web interface.

**Note**:
If you prefer to use a different database, such as PostgreSQL, you can use the <code>ragnetic configure</code> command to set it up. This will overwrite the default SQLite database file. After configuration, run the <code>ragnetic migrate</code> command to apply the database schema to your new database.

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

# no tools required
tools: []

# disable chunking
chunking:
  mode: none
  chunk_size: 0
  chunk_overlap: 0

# no vector store
vector_store:
  type: faiss
  bm25_k: 0
  semantic_k: 0
  retrieval_strategy: hybrid

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

### Agent & Workflow Management

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic list-agents` | Lists all configured agents. | `ragnetic list-agents` |
| `ragnetic deploy` | Deploys an agent by processing its data sources and building its vector store. | `ragnetic deploy my_research_agent` |
| `ragnetic deploy-orchestrator` | Deploys an orchestrator and all its sub-agents from a roster. | `ragnetic deploy-orchestrator my_team_orchestrator` |
| `ragnetic inspect-agent` | Displays an agent's configuration and can check connections or document metadata. | `ragnetic inspect-agent my_research_agent --check-connections` |
| `ragnetic reset-agent` | Resets an agent by deleting its vector store and memory files. | `ragnetic reset-agent my_agent` |
| `ragnetic delete-agent` | Permanently deletes an agent's configuration and all its data. | `ragnetic delete-agent my_old_agent` |
| `ragnetic list-workflows` | Lists recent workflow runs. | `ragnetic list-workflows` |
| `ragnetic trigger-workflow` | Triggers a workflow to run via the API. | `ragnetic trigger-workflow my_report_gen --input '{"topic":"Q3 results"}'` |
| `ragnetic inspect-orchestration` | Inspects a full orchestration, showing all sub-runs in a tree view. | `ragnetic inspect-orchestration <run_id>` |
| `ragnetic delete-workflow` | Permanently deletes a workflow definition and its YAML file. | `ragnetic delete-workflow my_old_workflow` |

### Training & Evaluation

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic dataset prepare` | Prepares a raw dataset for fine-tuning using a YAML configuration. | `ragnetic dataset prepare -f data_prep_configs/my_prep.yaml` |
| `ragnetic generate-test` | Generates a test set from an agent’s data sources. | `ragnetic generate-test my_agent -o test_set.json` |
| `ragnetic benchmark` | Runs a retrieval quality benchmark on an agent. | `ragnetic benchmark my_agent -t test_set.json` |
| `ragnetic training apply` | Submits a fine-tuning job via a YAML config file. | `ragnetic training apply -f configs/my_ft_job.yaml` |
| `ragnetic training status` | Checks the status of a fine-tuning job. | `ragnetic training status <adapter_id>` |
| `ragnetic training list-models` | Lists all available fine-tuned models (completed jobs). | `ragnetic training list-models` |

### Analytics & Auditing

| Command | Description | Example |
| :--- | :--- | :--- |
| `ragnetic analytics usage` | Displays aggregated LLM usage and cost metrics. | `ragnetic analytics usage --agent my_agent` |
| `ragnetic analytics benchmarks` | Displays summaries of past benchmark runs. | `ragnetic analytics benchmarks --agent my_agent` |
| `ragnetic analytics agent-runs` | Displays aggregated agent run metrics. | `ragnetic analytics agent-runs --agent my_agent` |
| `ragnetic analytics agent-steps` | Displays aggregated agent step metrics. | `ragnetic analytics agent-steps --agent my_agent` |
| `ragnetic analytics workflow-runs` | Displays aggregated workflow run metrics. | `ragnetic analytics workflow-runs --workflow my_workflow` |
| `ragnetic inspect-run` | Inspects a specific agent run and its steps. | `ragnetic inspect-run <run_id>` |
| `ragnetic inspect-workflow` | Inspects a specific workflow run and its I/O. | `ragnetic inspect-workflow <run_id>` |
| `ragnetic inspect-orchestration` | Inspects a full orchestration, showing all sub-runs in a tree view. | `ragnetic inspect-orchestration <run_id>` |

## YAML Configuration Examples

RAGnetic uses a declarative YAML-based approach for configuring agents, workflows, and training jobs.

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

# No external data ingestion—this agent only works on incoming webhook payloads
sources: ./data/customers.csv

# No tools required—pure LLM reasoning on the provided JSON
tools: []

# Disable chunking since payloads are small
chunking:
  mode: none
  chunk_size: 0
  chunk_overlap: 0

# No vector store needed
vector_store:
  type: faiss
  bm25_k: 0
  semantic_k: 0
  retrieval_strategy: none

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
This example shows how to configure a fine-tuning job for a model.
```yaml
# training_configs/policy_fine_tune.yaml
job_name: policy_fine_tune
base_model_name: microsoft/phi-2
dataset_file: data_prepared/policy_qa_dataset.jsonl
output_dir: fine_tuned_models/phi-2-policy

hyperparameters:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  device: "mps" # Use Apple Silicon GPU

```

### 3. Workflow Configuration
Workflows chain together agents and tools to accomplish multi-step, complex tasks.
```yaml
# workflows_data/support_ticket_analyzer.yaml
name: support_ticket_analyzer
description: An automated workflow to analyze and route support tickets.
trigger:
  type: api_webhook
  path: /webhooks/v1/new-support-ticket

steps:
  - name: retrieve_ticket_info
    type: tool_call
    tool_name: http_request_tool
    tool_input:
      method: "GET"
      url: "https://api.internal-crm.com/ticket/{{trigger.body.ticket_id}}"

  - name: summarize_ticket
    type: agent_call
    agent_name: support_summarizer_agent
    task: |
      **WHAT is the Goal?**
      Summarize the support ticket information.

      **IMPLICIT FROM WHERE?**
      The raw JSON from the "retrieve_ticket_info" step.

      **HOW should the Output Look?**
      Return a single sentence summary.

  - name: route_to_team
    type: agent_call
    agent_name: ticket_router_agent
    task: |
      **WHAT is the Goal?**
      Categorize this ticket summary and determine which team should handle it.

      **IMPLICIT FROM WHERE?**
      The summary from the "summarize_ticket" step.

      **HOW should the Output Look?**
      Return a JSON object with a single key, `team`, which contains the team name
      (e.g., "sales", "engineering").

  - name: send_notification
    type: tool_call
    tool_name: email_tool
    tool_input:
      to_email: "support_manager@company.com"
      subject: "New Ticket for {{route_to_team.team}} Team"
      body: |
        A new support ticket has been assigned to your team.
        Summary: "{{summarize_ticket.summary}}"
        Ticket ID: {{trigger.body.ticket_id}}
```

### 4. Data Preparation Configuration

```yaml
# data_prep_configs/jsonl_qa_prep.yaml
prep_name: jsonl_qa_prep
format_type: jsonl-instruction
input_file: data_raw/raw_qa_data.jsonl
output_file: data_prepared/prepared_qa_data.jsonl
```

## Citation Tracking & Explainable Reasoning

RAGnetic’s agents are designed to produce traceable and auditable outputs, a key requirement for regulated industries.

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
  Verify that all file paths in your agent or workflow YAML files are correct and that the necessary files exist.

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
