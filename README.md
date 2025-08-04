
<img width="20388" height="5692" alt="RAGnetic Logo" src="https://github.com/user-attachments/assets/92e1f139-9acb-43f3-9072-02d7f5336663" />


RAGnetic is an open-source framework for building and deploying production-ready AI agents and complex, multi-step workflows. It provides a plug-and-play solution for creating agents that leverage your organization’s internal knowledge base through Retrieval Augmented Generation (RAG), LangGraph pipelines, and seamlessly integrate custom fine-tuned models via LoRA (PEFT) or full-model training.

RAGnetic’s core philosophy is to enable organizations to embed, analyze and interact with their own code, infrastructure, data, and their docs. It’s a local-first, vendor-agnostic platform that lets you deploy private AI agents trained on everything inside your company, without lock-in. With built-in support for LoRA adapters and Hugging Face fine-tuning pipelines, you can quickly specialize models on proprietary data to meet your unique compliance, accuracy, and performance needs.

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
- alembic (for database migrations)
- Operating System: RAGnetic supports Linux, macOS, and Windows. Specific
  hardware limitations (e.g., <code>bitsandbytes</code> requires NVIDIA CUDA) are noted in the fine-tuning
  documentation.



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
    This command creates the necessary project structure and a default configuration file.
    ```bash
    ragnetic init
    ```
    
4. **Configure the Database**:
   Use the interactive configure command to set up your system databases for logging and
   memory storage.
   ```bash
    ragnetic configure
    ```
5. **Run Database Migrations:**
Apply the initial database schema. This is mandatory for using most features.
   ```bash
    ragnetic migrate
   ```
6. **Set your API Keys:**
   The framework needs API keys to use external services. Use the interactive <code>set-api-key</code> command to set a master administrative key, which is used for initial setup and emergency access.
   ```bash
    ragnetic set-api-key
   ```
7. **Start the Server**:
   Start the RAGnetic server, Celery worker, and scheduler in a single command.
   ```bash
    ragnetic start-server
   ```
You can use the <code>--reload</code> flag for development. Remove it for production.

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
  retrieval_strategy: none

```
You can deploy your agent using:
```bash
ragnetic deploy hello_world_agent
```

## Usage & Commands 

### Agent & Workflow Management

<table style="width:100%; table-layout:fixed;">
  <colgroup>
    <col style="width:60%;">
    <col style="width:25%;">
    <col style="width:15%;">
  </colgroup>
  <thead>
    <tr>
      <th>Command</th>
      <th>Description</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic list-agents</code></td>
      <td>Lists all configured agents.</td>
      <td style="white-space:nowrap;"><code>ragnetic list-agents</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic deploy</code></td>
      <td>Deploys an agent by processing its data sources.</td>
      <td style="white-space:nowrap;"><code>ragnetic deploy my_research_agent</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic deploy-orchestrator</code></td>
      <td>Deploys an orchestrator and its sub-agents.</td>
      <td style="white-space:nowrap;"><code>ragnetic deploy-orchestrator my_team_orchestrator</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic inspect-agent</code></td>
      <td>Displays an agent’s config and checks connections.</td>
      <td style="white-space:nowrap;"><code>ragnetic inspect-agent my_research_agent --check-connections</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic list-workflows</code></td>
      <td>Lists recent workflow runs.</td>
      <td style="white-space:nowrap;"><code>ragnetic list-workflows</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic trigger-workflow</code></td>
      <td>Triggers a workflow via the API.</td>
      <td style="white-space:nowrap;"><code>ragnetic trigger-workflow my_report_gen --input '{"topic":"Q3 results"}'</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic inspect-orchestration</code></td>
      <td>Inspects a full run, showing a tree of nested sub-runs.</td>
      <td style="white-space:nowrap;"><code>ragnetic inspect-orchestration &lt;run_id&gt;</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic delete-agent</code></td>
      <td>Permanently deletes an agent and all its data.</td>
      <td style="white-space:nowrap;"><code>ragnetic delete-agent my_old_agent</code></td>
    </tr>
  </tbody>
</table>

### Training & Evaluation

<table style="width:100%; table-layout:fixed;">
  <colgroup>
    <col style="width:60%;">
    <col style="width:25%;">
    <col style="width:15%;">
  </colgroup>
  <thead>
    <tr>
      <th>Command</th>
      <th>Description</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic generate-test</code></td>
      <td>Generates a test set from an agent’s data sources.</td>
      <td style="white-space:nowrap;"><code>ragnetic generate-test my_agent -o test_set.json</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic benchmark</code></td>
      <td>Runs a retrieval quality benchmark on an agent.</td>
      <td style="white-space:nowrap;"><code>ragnetic benchmark my_agent -t test_set.json</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic training apply</code></td>
      <td>Submits a fine-tuning job via a YAML config file.</td>
      <td style="white-space:nowrap;"><code>ragnetic training apply -f configs/my_ft_job.yaml</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic training status</code></td>
      <td>Checks the status of a fine-tuning job.</td>
      <td style="white-space:nowrap;"><code>ragnetic training status &lt;adapter_id&gt;</code></td>
    </tr>
  </tbody>
</table>

### Analytics & Auditing

<table style="width:100%; table-layout:fixed;">
  <colgroup>
    <col style="width:60%;">
    <col style="width:25%;">
    <col style="width:15%;">
  </colgroup>
  <thead>
    <tr>
      <th>Command</th>
      <th>Description</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic analytics usage</code></td>
      <td>Displays aggregated LLM usage and cost metrics.</td>
      <td style="white-space:nowrap;"><code>ragnetic analytics usage --agent my_agent</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic analytics benchmarks</code></td>
      <td>Displays summaries of past benchmark runs.</td>
      <td style="white-space:nowrap;"><code>ragnetic analytics benchmarks --agent my_agent</code></td>
    </tr>
    <tr>
      <td style="white-space:nowrap;"><code>ragnetic inspect-run</code></td>
      <td>Inspects a specific agent run and its steps.</td>
      <td style="white-space:nowrap;"><code>ragnetic inspect-run &lt;run_id&gt;</code></td>
    </tr>
  </tbody>
</table>

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
