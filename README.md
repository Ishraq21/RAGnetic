# RAGnetic

RAGnetic is an open-source, plug-and-play framework for deploying on-premise ChatGPT-style agents powered by your internal enterprise knowledge. It enables organizations to create sophisticated AI assistants that securely leverage their private data, ensuring that sensitive information never leaves their control. The core philosophy is to bridge the gap between powerful Large Language Models (LLMs) and proprietary data, allowing you to build agents that are not only intelligent but also deeply context-aware, auditable, and secure.

## Features

* **Multi-Source Ingestion:** Go beyond simple text files. RAGnetic can create a unified knowledge base from a wide array of sources simultaneously. It includes dedicated loaders for local files (`.pdf`, `.docx`, `.csv`), public websites, entire code repositories (respecting `.gitignore`), and can even connect to live SQL databases to ingest their schema. This creates a single, queryable source of truth for your agent, allowing it to reason over disparate data types.


* **Stateful Conversations:** Agents possess durable, conversation-level memory, allowing for natural, iterative dialogue. A user can ask a broad question, then follow up with "can you elaborate on the second point?" or "summarize that in three bullets" without needing to repeat the initial context. This stateful nature is crucial for complex research and analysis tasks.


* **Tool-Using Hybrid Agents:** This is the core of RAGnetic's intelligence. You can equip agents with a versatile set of tools, such as a document `retriever` for semantic search and a `sql_toolkit` for live database querying. The underlying language model is prompted to be a reasoning agent. It analyzes the user's query and the conversation history to intelligently choose the best tool for the job, seamlessly switching between searching documents and executing SQL queries to find the most accurate answer.


* **Easy Deployment:** RAGnetic is designed for maximum accessibility. It supports two distinct workflows: a native Python CLI for developers who want to contribute, and a one-command Docker deployment that provides a reliable, isolated, and conflict-free environment for end-users.


* **Simple YAML Configuration:** No code changes are required to create, modify, or manage agents. All agent behavior—from its core persona and system prompt to its data sources, models, and tools—is defined in clean, human-readable YAML files. This empowers both technical and non-technical users to build and maintain their own custom AI assistants.


* **Automated API Key Management:** To enhance both security and user experience, RAGnetic features an interactive API key setup. On the first run, if a required key (like `OPENAI_API_KEY`) is not found, the command-line will prompt the user to enter it once, then securely save it to a local `.env` file for all future sessions.

---

## Workflow 1: Docker (Recommended for End-Users)

This is the simplest and most reliable method for running RAGnetic. It provides a completely isolated environment using Docker containers, guaranteeing that the application and its dependencies will work correctly regardless of your local machine's configuration. This approach eliminates "it works on my machine" problems.

### Installation & Setup

**Prerequisites:** Docker and Docker Compose must be installed on your system.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd ragnetic
    ```

2.  **Run the Application:** This single command reads the `docker-compose.yaml` file, which orchestrates the entire setup. It builds the Docker image from the `Dockerfile`, defining a perfect, repeatable Linux environment with the correct Python version and dependencies installed. It then starts the application server.
    ```bash
    docker-compose up --build
    ```

3.  **First-Time API Key Setup:** On the very first run, the application, running inside the container, will detect that no `.env` file exists. The terminal logs will display a friendly, interactive prompt asking for any required API keys. Enter your key once, and RAGnetic will automatically create the `.env` file in your project directory. This file is linked via a Docker volume, so it persists locally and is used for all future runs. **You do not need to create or edit the `.env` file manually.**

The application will now be available at `http://localhost:8000`.

### Usage & Agent Management (Docker)

All management commands are run from your **local terminal** using the `docker-compose run --rm` syntax. This executes a command inside a *new, temporary container* based on your application's image, ensuring the environment is identical. The `--rm` flag automatically cleans up the container after the command completes.

* **Creating an Agent:**
    1.  Place your source files (PDFs, database files, etc.) into the `./data/` directory on your local machine.
    2.  Create a new `.yaml` file inside the `agents_data/` directory.

    **Example `finance_agent.yaml`:**
    ```yaml
    name: finance-agent
    description: An agent for answering questions about financial reports and sales data.
    persona_prompt: You are a helpful financial analyst who provides clear and concise answers based on the provided documents and database.
    sources:
      # This agent learns from all files in a local directory...
      - type: local
        path: ./data/reports/
      # ...and also knows the schema of a database.
      - type: db
        db_connection: "sqlite:///./data/finance_database.db"
    tools:
      - retriever
      - sql_toolkit
    llm_model: "gpt-4-turbo"
    ```

* **Deploying an Agent:** This command processes an agent's data sources and creates its knowledge base. It must be run for each new agent or whenever you update an agent's source files.
    ```bash
    docker-compose run --rm ragnetic-app ragnetic deploy-agent --config ./agents_data/finance_agent.yaml
    ```

* **Starting/Stopping the Server:**
    ```bash
    # Start the server in the background (detached mode)
    docker-compose up -d
    
    # Stop the server and remove the container
    docker-compose down
    ```

* **Listing & Inspecting Agents:**
    ```bash
    # List all available agents found in the agents_data directory
    docker-compose run --rm ragnetic-app ragnetic list-agents
    
    # View the full, parsed configuration of a specific agent
    docker-compose run --rm ragnetic-app ragnetic inspect-agent finance-agent
    ```

* **Validating an Agent:** Run a health check to ensure an agent's configuration is valid and its associated files (like the vector store) exist. This is a great debugging step.
    ```bash
    docker-compose run --rm ragnetic-app ragnetic validate-agent finance-agent
    ```

* **Deleting Agent Data:** Safely removes the vector store and all conversation memory for a specific agent. This is useful for a clean re-deployment. The agent's YAML configuration file is not affected.
    ```bash
    docker-compose run --rm ragnetic-app ragnetic delete-agent finance-agent
    ```

---

## Workflow 2: Native Python (For Developers)

This method is for developers who wish to contribute to the RAGnetic codebase, debug its internal functions, or integrate it as a library into a larger Python project. It offers more flexibility at the cost of requiring manual environment management.

### Installation & Setup

**Prerequisites:** Python 3.9+ and a tool for creating virtual environments (e.g., `venv`).

1.  **Clone and Setup Environment:**
    ```bash
    git clone <your-repo-url>
    cd ragnetic
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

2.  **Install the Project:** This command installs all dependencies from `pyproject.toml` and makes the `ragnetic` CLI tool available in your terminal. The `-e` (editable) flag is highly recommended, as it allows your code changes to take effect immediately without needing to reinstall.
    ```bash
    pip install -e .
    ```

3.  **Initialize Folders:** The `init` command creates the necessary project directories (`agents_data`, `data`, etc.) and a helpful `.env.example` template file.
    ```bash
    ragnetic init
    ```

### Usage & Agent Management (Native Python)

All commands are run directly from your terminal (with your virtual environment activated).

* **Creating an Agent:**
    1.  Place your source files into the `./data/` directory.
    2.  Create a new `.yaml` file inside the `agents_data/` directory.

* **Deploying an Agent:**
    ```bash
    ragnetic deploy-agent --config agents_data/finance_agent.yaml
    ```
    *On first use, this command will interactively prompt for any required API keys and save them to a new `.env` file.*

* **Starting the Web Server:** For development, the `--reload` flag is invaluable as it automatically restarts the server when you save changes to a Python file.
    ```bash
    # With auto-reload for easy development
    ragnetic start-server --reload
    ```

* **Listing & Inspecting Agents:**
    ```bash
    ragnetic list-agents
    ragnetic inspect-agent finance-agent
    ```

* **Validating an Agent:**
    ```bash
    ragnetic validate-agent finance-agent
    ```

* **Deleting Agent Data:**
    ```bash
    ragnetic delete-agent finance-agent
    ```
