# RAGnetic

RAGnetic is an open-source, plug-and-play framework for deploying on-premise ChatGPT-style agents powered by your internal enterprise knowledge. It enables organizations to create sophisticated AI assistants that securely leverage their private data, ensuring that sensitive information never leaves their control. The core philosophy is to bridge the gap between powerful Large Language Models (LLMs) and proprietary data, allowing you to build agents that are not only intelligent but also deeply context-aware, auditable, and secure.

## Features

* **Multi-Source Ingestion:** Go beyond simple text files. RAGnetic can create a unified knowledge base from a wide array of sources simultaneously. It includes dedicated loaders for local files (`.pdf`, `.docx`, `.csv`), public websites, entire code repositories (respecting `.gitignore`), and can even connect to live SQL databases to ingest their schema. This creates a single, queryable source of truth for your agent, allowing it to reason over disparate data types.


* **Stateful Conversations:** Agents possess durable, conversation-level memory, allowing for natural, iterative dialogue. A user can ask a broad question, then follow up with "can you elaborate on the second point?" or "summarize that in three bullets" without needing to repeat the initial context. This stateful nature is crucial for complex research and analysis tasks.


* **Tool-Using Hybrid Agents:** This is the core of RAGnetic's intelligence. You can equip agents with a versatile set of tools, such as a document `retriever` for semantic search and a `sql_toolkit` for live database querying. The underlying language model is prompted to be a reasoning agent.


* **Easy Deployment:** RAGnetic is designed for maximum accessibility. It supports two distinct workflows: a native Python CLI for developers and a one-command Docker deployment that provides a reliable, isolated environment for end-users.


* **Simple YAML Configuration:** No code changes are required to create, modify, or manage agents. All agent behavior—from its core persona and system prompt to its data sources, models, and tools—is defined in clean, human-readable YAML files. This empowers both technical and non-technical users to build and maintain their own custom AI assistants.


* **Guided API Key Setup:** RAGnetic features a dedicated and secure command-line interface for setting up your API keys. This guided process ensures your keys are stored correctly in a local `.env` file for the application to use.

---

## Getting Started: The 3-Step Setup

Whether you use Docker or native Python, the initial setup is the same and only needs to be done once.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd ragnetic
    ```

2.  **Initialize the Project:**
    This command creates the necessary project directories (`agents_data`, `data`, etc.). Note that you may need to install the project dependencies first (see the Native Python workflow) or build the Docker image to make the `ragnetic` command available.
    ```bash
    # For Native Python (after installing dependencies)
    ragnetic init

    # For Docker (after building the image)
    docker-compose run --rm ragnetic-app ragnetic init
    ```

3.  **Set Your API Keys:**
    This interactive command will guide you through securely saving your API keys (e.g., for OpenAI, Anthropic, or Google) to a local `.env` file.
    ```bash
    # For Native Python
    ragnetic set-api

    # For Docker
    docker-compose run --rm ragnetic-app ragnetic set-api
    ```
    You will be presented with a menu to choose the provider and prompted to enter your key.

After these three steps, you are ready to follow either the Docker or Native Python workflow.

---

## Workflow 1: Docker (Recommended for End-Users)

This is the simplest and most reliable method for running RAGnetic. It provides a completely isolated environment using Docker containers, guaranteeing that the application and its dependencies will work correctly regardless of your local machine's configuration.

### Installation & Setup

**Prerequisites:** Docker and Docker Compose must be installed on your system.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd ragnetic
    ```

2.  **Build, Initialize, and Configure:**
    Run these one-off commands to build the Docker image and set up your project.
    ```bash
    # First, build the image so the CLI is available
    docker-compose build

    # Run the init command
    docker-compose run --rm ragnetic-app ragnetic init

    # Run the interactive set-api command
    docker-compose run --rm ragnetic-app ragnetic set-api
    ```

3.  **Run the Application:**
    This command starts the server. Because you created the `.env` file in the previous step, Docker Compose will automatically load it into the container.
    ```bash
    docker-compose up
    ```
    The application will now be available at `http://localhost:8000`.

### Usage & Agent Management (Docker)

All management commands are run from your **local terminal** using the `docker-compose run --rm` syntax. This executes a command inside a *new, temporary container*.

* **Creating an Agent:**
    1.  Place your source files (PDFs, database files, etc.) into the `./data/` directory.
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

* **Deploying an Agent:**
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
    docker-compose run --rm ragnetic-app ragnetic list-agents
    docker-compose run --rm ragnetic-app ragnetic inspect-agent finance-agent
    ```

* **Validating an Agent:**
    ```bash
    docker-compose run --rm ragnetic-app ragnetic validate-agent finance-agent
    ```

* **Deleting Agent Data:**
    ```bash
    docker-compose run --rm ragnetic-app ragnetic delete-agent finance-agent
    ```

---

## Workflow 2: Native Python (For Developers)

This method is for developers who wish to contribute to the RAGnetic codebase or integrate it as a library.

### Installation & Setup

**Prerequisites:** Python 3.9+ and a tool for creating virtual environments (e.g., `venv`).

1.  **Clone and Setup Environment:**
    ```bash
    git clone <your-repo-url>
    cd ragnetic
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

2.  **Install the Project:**
    This command installs all dependencies from `pyproject.toml` and makes the `ragnetic` CLI tool available in your terminal.
    ```bash
    pip install -e .
    ```

3.  **Initialize and Configure:**
    Complete the one-time setup steps:
    ```bash
    # Create project folders
    ragnetic init

    # Set your API keys
    ragnetic set-api
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

* **Starting the Web Server:**
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

---

## Community and Support

We welcome you to the RAGnetic community! Here’s how you can get involved and find help.

### Community

* **Join our Discord Server:** For general chat, sharing what you've built, and getting to know other users, join our [Community Discord](https://discord.gg/your-invite-link).
* **Follow us on X (formerly Twitter):** Stay up-to-date with the latest announcements, features, and tips by following [@RAGneticAI](https://twitter.com/RAGneticAI).

### Support

* **GitHub Discussions:** For questions, feature requests, and showing off your projects, please use our [GitHub Discussions](https://github.com/your-repo/ragnetic/discussions) page. This is the best place for non-urgent support.
* **GitHub Issues:** If you've found a bug or are experiencing a reproducible error, please open an issue on our [GitHub Issues](https://github.com/your-repo/ragnetic/issues) tracker. Be sure to include as much detail as possible, including your OS, RAGnetic version, and steps to reproduce the problem.

### Contributing

RAGnetic is an open-source project, and we welcome contributions of all kinds! Whether you're a developer, a writer, or a designer, there are many ways to get involved.

* **Contribution Guide:** Before you start, please read our [CONTRIBUTING.md](https://github.com/your-repo/ragnetic/blob/main/CONTRIBUTING.md) file for a detailed guide on our development process, coding standards, and how to submit pull requests.
* **Code of Conduct:** We are committed to fostering a welcoming and inclusive community. All participants are expected to adhere to our [Code of Conduct](https://github.com/your-repo/ragnetic/blob/main/CODE_OF_CONDUCT.md).

