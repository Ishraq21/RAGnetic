# RAGnetic Installation Guide

## Quick Start

### 1. Basic Installation
```bash
pip install ragnetic
```

This installs RAGnetic with minimal dependencies (31 core packages) for basic functionality.

### 2. Recommended Installation (with AI features)
```bash
pip install ragnetic[ai,vectorstores]
```

This includes AI providers (OpenAI, Anthropic, etc.) and vector store support.

### 3. Full Installation
```bash
pip install ragnetic[all]
```

Includes all optional features: AI, vector stores, data processing, and development tools.

## Installation Options

### Core Features Only
```bash
pip install ragnetic
# 31 dependencies - FastAPI, SQLAlchemy, basic CLI
```

### AI & Machine Learning
```bash
pip install ragnetic[ai]
# Adds: langchain, openai, anthropic, transformers
```

### Vector Stores
```bash
pip install ragnetic[vectorstores]  
# Adds: chromadb, pinecone, qdrant, faiss
```

### Data Processing
```bash
pip install ragnetic[data]
# Adds: pdf parsing, document processing, web scraping
```

### Model Training
```bash
pip install ragnetic[training,gpu]
# Adds: torch, peft, accelerate, bitsandbytes
```

### Development
```bash
pip install ragnetic[dev]
# Adds: pytest, black, mypy, ruff
```

## Hardware-Specific

### NVIDIA GPUs
```bash
pip install ragnetic[gpu]
# Includes CUDA support for training
```

### Apple Silicon (M1/M2/M3)
```bash
pip install ragnetic[mps]
# Optimized for Apple Silicon
```

### CPU Only
```bash
pip install ragnetic[cpu]
# Lighter installation, no GPU dependencies
```

## System Requirements

- Python 3.9 or higher
- Redis (for task queue)
- PostgreSQL or SQLite (database)

## Quick Setup
```bash
# Install with AI features
pip install ragnetic[ai,vectorstores]

# Initialize project
ragnetic init

# Set API keys
ragnetic set-api-key

# Create admin user  
ragnetic user create admin --superuser

# Start server
ragnetic start-server
```

Visit `http://localhost:8000` to access the web interface.

## Troubleshooting

### Common Issues

**ImportError: No module named 'torch'**
```bash
pip install ragnetic[training]  # or [gpu] for CUDA
```

**Redis connection error**
```bash
# Install Redis
# macOS: brew install redis
# Ubuntu: sudo apt install redis-server
```

**Database errors**
```bash
ragnetic migrate  # Apply database migrations
```

## Upgrading
```bash
pip install --upgrade ragnetic
ragnetic migrate  # Update database schema
```