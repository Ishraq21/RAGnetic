# Sample data for testing
# This module provides realistic test data for all RAGnetic components

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Sample agent configurations
SAMPLE_AGENTS = {
    "legal_agent": {
        "name": "legal_agent",
        "display_name": "Legal Document Analyzer",
        "description": "Analyzes legal documents and provides summaries",
        "persona_prompt": "You are a legal expert who analyzes contracts and legal documents.",
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "sources": [],
        "tools": ["retriever"],
        "vector_store": {
            "type": "faiss",
            "bm25_k": 5,
            "semantic_k": 5
        }
    },
    
    "code_agent": {
        "name": "code_agent",
        "display_name": "Code Analysis Agent",
        "description": "Reviews and analyzes code for best practices",
        "persona_prompt": "You are a senior software engineer who reviews code.",
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o",
        "sources": [],
        "tools": ["retriever", "lambda"],
        "vector_store": {
            "type": "chroma",
            "bm25_k": 3,
            "semantic_k": 7
        }
    },
    
    "customer_support": {
        "name": "customer_support",
        "display_name": "Customer Support Bot",
        "description": "Handles customer inquiries and support tickets",
        "persona_prompt": "You are a helpful customer support representative.",
        "embedding_model": "text-embedding-ada-002",
        "llm_model": "gpt-3.5-turbo",
        "sources": [],
        "tools": ["retriever", "search"],
        "vector_store": {
            "type": "pinecone",
            "bm25_k": 5,
            "semantic_k": 5
        }
    }
}

# Sample training datasets
SAMPLE_TRAINING_DATASETS = {
    "instruction_following": [
        {"instruction": "What is machine learning?", "output": "Machine learning is a subset of AI that enables computers to learn from data."},
        {"instruction": "Explain neural networks", "output": "Neural networks are computing systems inspired by biological neural networks."},
        {"instruction": "What is deep learning?", "output": "Deep learning uses neural networks with multiple layers to learn complex patterns."},
        {"instruction": "Define artificial intelligence", "output": "AI is the simulation of human intelligence in machines."},
        {"instruction": "What is supervised learning?", "output": "Supervised learning uses labeled data to train models."}
    ],
    
    "conversational": [
        {
            "messages": [
                {"role": "user", "content": "Hello, can you help me?"},
                {"role": "assistant", "content": "Hello! I'd be happy to help you. What do you need assistance with?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "I need help with Python programming"},
                {"role": "assistant", "content": "I'd be glad to help with Python! What specific topic or problem are you working on?"}
            ]
        }
    ],
    
    "domain_specific": [
        {"instruction": "What is a contract?", "output": "A contract is a legally binding agreement between two or more parties."},
        {"instruction": "Define tort law", "output": "Tort law deals with civil wrongs and provides remedies for damages."},
        {"instruction": "What is due process?", "output": "Due process ensures fair treatment through the judicial system."}
    ]
}

# Sample test sets for evaluation
SAMPLE_TEST_SETS = {
    "basic_qa": [
        {"question": "What is the capital of France?", "expected_answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "expected_answer": "William Shakespeare"},
        {"question": "What is 2 + 2?", "expected_answer": "4"}
    ],
    
    "domain_qa": [
        {"question": "What is a breach of contract?", "expected_answer": "A breach of contract occurs when one party fails to fulfill their obligations under the agreement."},
        {"question": "What is negligence in tort law?", "expected_answer": "Negligence is the failure to exercise reasonable care, resulting in harm to another person."}
    ],
    
    "reasoning": [
        {"question": "If all roses are flowers and all flowers need water, do roses need water?", "expected_answer": "Yes, roses need water."},
        {"question": "A train leaves at 2 PM and travels for 3 hours. What time does it arrive?", "expected_answer": "5 PM"}
    ]
}

# Sample fine-tuning job configurations
SAMPLE_TRAINING_JOBS = {
    "basic_lora": {
        "job_name": "test_lora_training",
        "base_model_name": "microsoft/DialoGPT-small",
        "dataset_path": "data/training_datasets/sample.jsonl",
        "output_base_dir": "models/fine_tuned",
        "hyperparameters": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "save_steps": 50
        }
    },
    
    "gpu_training": {
        "job_name": "gpu_test_training",
        "base_model_name": "meta-llama/Llama-2-7b-hf",
        "dataset_path": "data/training_datasets/sample.jsonl",
        "output_base_dir": "models/fine_tuned",
        "use_gpu": True,
        "gpu_type": "A100",
        "gpu_provider": "runpod",
        "max_hours": 2.0,
        "hyperparameters": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "mixed_precision_dtype": "bf16"
        }
    }
}

# Sample GPU provider configurations
SAMPLE_GPU_PROVIDERS = [
    {
        "name": "runpod",
        "gpu_type": "RTX4090",
        "cost_per_hour": 0.50,
        "availability": True
    },
    {
        "name": "runpod",
        "gpu_type": "A100",
        "cost_per_hour": 2.00,
        "availability": True
    },
    {
        "name": "coreweave",
        "gpu_type": "A100",
        "cost_per_hour": 1.80,
        "availability": True
    },
    {
        "name": "vast",
        "gpu_type": "RTX3090",
        "cost_per_hour": 0.30,
        "availability": False
    }
]

# Sample API usage data
SAMPLE_API_USAGE = [
    {
        "endpoint": "/api/v1/agents/test_agent/query",
        "method": "POST",
        "status_code": 200,
        "response_time_ms": 1250,
        "tokens_used": 150,
        "cost_usd": 0.003
    },
    {
        "endpoint": "/api/v1/training/jobs",
        "method": "POST", 
        "status_code": 202,
        "response_time_ms": 500,
        "tokens_used": 0,
        "cost_usd": 0.0
    }
]

# Sample benchmark results
SAMPLE_BENCHMARK_RESULTS = {
    "run_id": "bench_abc123",
    "agent_name": "test_agent",
    "test_set_size": 10,
    "accuracy": 0.85,
    "avg_response_time": 1.2,
    "total_cost": 0.15,
    "results": [
        {"question": "Test question 1", "expected": "Answer 1", "actual": "Answer 1", "score": 1.0},
        {"question": "Test question 2", "expected": "Answer 2", "actual": "Close answer", "score": 0.8}
    ]
}

# Sample user and project data
SAMPLE_USERS = [
    {
        "username": "admin_user",
        "email": "admin@ragnetic.ai",
        "first_name": "Admin",
        "last_name": "User",
        "is_active": True,
        "is_superuser": True,
        "roles": ["admin"]
    },
    {
        "username": "dev_user",
        "email": "dev@company.com",
        "first_name": "Developer",
        "last_name": "User",
        "is_active": True,
        "is_superuser": False,
        "roles": ["developer"]
    },
    {
        "username": "viewer_user",
        "email": "viewer@company.com",
        "first_name": "Viewer",
        "last_name": "User",
        "is_active": True,
        "is_superuser": False,
        "roles": ["viewer"]
    }
]

SAMPLE_PROJECTS = [
    {
        "name": "Legal AI Assistant",
        "description": "AI assistant for legal document analysis and contract review",
        "agents": ["legal_agent"]
    },
    {
        "name": "Code Review Bot",
        "description": "Automated code review and analysis system",
        "agents": ["code_agent"]
    },
    {
        "name": "Customer Support System",
        "description": "Intelligent customer support and ticket handling",
        "agents": ["customer_support"]
    }
]

# Utility functions to create sample data files
def create_sample_agent_file(agent_name: str, output_dir: Path) -> Path:
    """Create a sample agent YAML file."""
    import yaml
    
    agent_config = SAMPLE_AGENTS.get(agent_name, SAMPLE_AGENTS["legal_agent"])
    agent_file = output_dir / f"{agent_name}.yaml"
    
    with open(agent_file, 'w') as f:
        yaml.dump(agent_config, f, default_flow_style=False)
    
    return agent_file

def create_sample_training_dataset(dataset_type: str, output_file: Path) -> Path:
    """Create a sample training dataset file."""
    dataset = SAMPLE_TRAINING_DATASETS.get(dataset_type, SAMPLE_TRAINING_DATASETS["instruction_following"])
    
    with open(output_file, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    return output_file

def create_sample_test_set(test_type: str, output_file: Path) -> Path:
    """Create a sample test set file."""
    test_set = SAMPLE_TEST_SETS.get(test_type, SAMPLE_TEST_SETS["basic_qa"])
    
    with open(output_file, 'w') as f:
        json.dump(test_set, f, indent=2)
    
    return output_file

def get_sample_training_job(job_type: str = "basic_lora") -> Dict[str, Any]:
    """Get sample training job configuration."""
    return SAMPLE_TRAINING_JOBS.get(job_type, SAMPLE_TRAINING_JOBS["basic_lora"]).copy()

def get_sample_user(user_type: str = "dev_user") -> Dict[str, Any]:
    """Get sample user data."""
    users_by_type = {user["username"]: user for user in SAMPLE_USERS}
    return users_by_type.get(user_type, SAMPLE_USERS[1]).copy()

def get_sample_project(project_name: str = None) -> Dict[str, Any]:
    """Get sample project data."""
    if project_name:
        for project in SAMPLE_PROJECTS:
            if project["name"] == project_name:
                return project.copy()
    return SAMPLE_PROJECTS[0].copy()

# Attack payloads for security testing
SECURITY_TEST_PAYLOADS = {
    "sql_injection": [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM users--",
        "'; INSERT INTO users (username) VALUES ('hacker'); --"
    ],
    
    "xss": [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "<svg onload=alert('xss')>",
        "';alert('xss');//"
    ],
    
    "path_traversal": [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%252f..%252f..%252fetc%252fpasswd"
    ],
    
    "command_injection": [
        "; ls -la",
        "| whoami",
        "&& cat /etc/passwd",
        "`whoami`",
        "$(whoami)"
    ],
    
    "ldap_injection": [
        "*)(uid=*))(|(uid=*",
        "*)(|(password=*))",
        "admin)(&(password=*))"
    ],
    
    "xml_injection": [
        "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
        "<![CDATA[<script>alert('xss')</script>]]>"
    ]
}

# Load testing scenarios
LOAD_TEST_SCENARIOS = {
    "light_load": {
        "users": 10,
        "spawn_rate": 2,
        "duration": "30s"
    },
    
    "medium_load": {
        "users": 50,
        "spawn_rate": 5,
        "duration": "2m"
    },
    
    "heavy_load": {
        "users": 200,
        "spawn_rate": 10,
        "duration": "5m"
    },
    
    "spike_test": {
        "users": 500,
        "spawn_rate": 50,
        "duration": "1m"
    }
}
