import typer
import uvicorn
import os
import shutil
import yaml
import glob

from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file, AGENTS_DIR
from app.pipelines.embed import embed_agent_data
from app.core.config import get_api_key

app = typer.Typer(
    name="ragnetic",
    help="RAGnetic: Your on-premise, plug-and-play AI agent framework.",
    add_completion=False,
)


@app.command(help="Initialize a new RAGnetic project in the current directory.")
def init():
    """
    Creates the necessary folders (agents_data, data) and a default .env.example file.
    """
    print("Initializing new RAGnetic project...")
    folders_to_create = ["agents_data", "data", "vectorstore", "memory"]
    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"  - Created directory: ./{folder}")

    env_example_path = ".env.example"
    if not os.path.exists(env_example_path):
        with open(env_example_path, "w") as f:
            f.write("# Rename this file to .env and fill in your secrets\n")
            f.write("OPENAI_API_KEY=\"sk-...\"\n")
            f.write("ANTHROPIC_API_KEY=\"sk-ant-...\"\n")
            f.write("GOOGLE_API_KEY=\"...\"\n")
        print(f"  - Created .env.example file.")

    print(
        "\nProject initialized successfully. Place your agent YAML configs in 'agents_data' and your source files in 'data'.")


@app.command(help="Lists all deployed agents.")
def list_agents():
    """Scans the agents_data directory and lists all configured agents."""
    if not os.path.exists(AGENTS_DIR):
        print("Error: Directory 'agents_data' not found. Have you run 'ragnetic init'?")
        raise typer.Exit(code=1)

    agents = [f.split(".")[0] for f in os.listdir(AGENTS_DIR) if f.endswith((".yaml", ".yml"))]

    if not agents:
        print("No agents found in the 'agents_data' directory.")
        return

    print("Available Agents:")
    for agent_name in agents:
        print(f"  - {agent_name}")


@app.command(help="Displays the configuration of a specific agent.")
def inspect_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to inspect.")
):
    """Loads and prints the specified agent's configuration."""
    try:
        print(f"Inspecting configuration for agent: '{agent_name}'")
        config = load_agent_config(agent_name)
        # Pretty print the config dictionary
        print(yaml.dump(config.model_dump(), indent=2, sort_keys=False))
    except FileNotFoundError:
        print(f"Error: Agent '{agent_name}' not found.")
        raise typer.Exit(code=1)


@app.command(help="Deploys an agent by processing and embedding its data sources.")
def deploy_agent(
        config: str = typer.Option(..., "--config", "-c", help="Path to the agent's YAML configuration file."),
):
    """Loads an agent config and creates a vector store from its data sources."""
    try:
        print(f"Loading agent configuration from: {config}")
        agent_config = load_agent_from_yaml_file(config)

        api_key = get_api_key(agent_config.embedding_model)

        print(f"Deploying agent '{agent_config.name}' using '{agent_config.embedding_model}'...")

        embed_agent_data(agent_config, openai_api_key=api_key)

        print("\nAgent deployment successful!")
        print(f"  - Vector store created at: vectorstore/{agent_config.name}")
        print("  - You can now start the server and interact with your agent.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"An unexpected error occurred during deployment: {e}")
        raise typer.Exit(code=1)


@app.command(help="Starts the RAGnetic FastAPI server.")
def start_server(
        host: str = typer.Option("127.0.0.1", help="The host to bind the server to."),
        port: int = typer.Option(8000, help="The port to run the server on."),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
):
    """Starts the Uvicorn server."""
    print(f"Starting RAGnetic server on http://{host}:{port}")
    try:
        get_api_key("openai")
    except ValueError as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


# --- RENAMED: from reset-agent to delete-agent ---
@app.command(help="Deletes all data associated with an agent (vector store and memory).")
def delete_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent whose data will be deleted."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    """
    Deletes an agent's vector store and conversation memory.
    Does not delete the agent's YAML config file.
    """
    vectorstore_path = f"vectorstore/{agent_name}"
    memory_pattern = f"memory/{agent_name}_*.db"

    print(f"Warning: This will permanently delete the following for agent '{agent_name}':")
    print(f"  - Vector store directory: {vectorstore_path}")
    print(f"  - All memory files matching: {memory_pattern}")

    if not force:
        typer.confirm("Are you sure you want to proceed?", abort=True)

    try:
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
            print(f"  - Deleted vector store: {vectorstore_path}")
        else:
            print(f"  - No vector store found to delete at {vectorstore_path}")

        memory_files = glob.glob(memory_pattern)
        if memory_files:
            for f in memory_files:
                os.remove(f)
            print(f"  - Deleted {len(memory_files)} memory file(s).")
        else:
            print("  - No memory files found to delete.")

        print(f"\nAgent '{agent_name}' data has been deleted.")
    except Exception as e:
        print(f"An error occurred during deletion: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
