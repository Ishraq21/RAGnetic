import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConversationalJsonlLoader:
    """
    Loads conversational data from a JSONL file.
    Expected format: Each line is a JSON object with a 'messages' key
    containing a list of dictionaries, where each dictionary has 'role' and 'content' keys.
    Example:
    {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads and validates conversational data from the specified JSONL file.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")

        loaded_data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    # Basic validation for conversational format
                    if "messages" not in record or not isinstance(record["messages"], list):
                        raise ValueError("Record must contain a 'messages' key with a list.")
                    for message in record["messages"]:
                        if not isinstance(message, dict) or "role" not in message or "content" not in message:
                            raise ValueError("Each message must be a dictionary with 'role' and 'content'.")
                    loaded_data.append(record)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON on line {line_num} in {self.file_path}: {e}")
                    raise ValueError(f"Invalid JSON in dataset file at line {line_num}.")
                except ValueError as e:
                    logger.error(f"Validation error on line {line_num} in {self.file_path}: {e}")
                    raise ValueError(f"Invalid record format in dataset file at line {line_num}: {e}")

        logger.info(f"Successfully loaded {len(loaded_data)} conversational samples from {self.file_path}")
        return loaded_data