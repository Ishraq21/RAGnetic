# RAGnetic Installation Guide

## Quick Start

### 1. Complete Installation (Recommended)
```bash
pip install ragnetic
```

This installs RAGnetic with **everything included**:
- ✅ All AI providers (OpenAI, Anthropic, Google, Ollama)
- ✅ All vector stores (ChromaDB, Pinecone, Qdrant, FAISS)
- ✅ Document processing (PDF, DOCX, web scraping)
- ✅ Web UI, API, CLI, workflows
- ✅ Ready to use immediately

### 2. With Model Training
```bash
pip install ragnetic[training]
```

Adds fine-tuning capabilities (PyTorch, PEFT, LoRA).

### 3. With GPU Support
```bash
pip install ragnetic[gpu]
```

For CUDA acceleration during training.

## Installation Options

### Complete Functionality (Default)
```bash
pip install ragnetic
# Everything included - AI providers, vector stores, document processing
```

### Model Training
```bash
pip install ragnetic[training]
# Adds: torch, peft, accelerate, tensorboard
```

### GPU Acceleration
```bash
pip install ragnetic[gpu]
# Adds: CUDA support, bitsandbytes for training
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