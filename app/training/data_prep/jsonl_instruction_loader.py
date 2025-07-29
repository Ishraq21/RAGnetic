
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class JsonlInstructionLoader:
    """
    Loads instruction-tuning data from a .jsonl file.
    Each line in the .jsonl file should be a JSON object
    conforming to the {"instruction": "...", "input": "...", "output": "..."} format.
    The 'input' field is optional.
    """

    def __init__(self, file_path: str):
        """
        Initializes the loader with the path to the .jsonl file.

        Args:
            file_path (str): The path to the .jsonl dataset file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            logger.error(f"Dataset file not found: {self.file_path}")
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads and parses the .jsonl file, validating each record.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                   represents a validated instruction-tuning record.

        Raises:
            ValueError: If any line in the file is not valid JSON or does not
                        conform to the expected schema (missing 'instruction' or 'output').
        """
        data = []
        logger.info(f"Loading instruction-tuning data from: {self.file_path}")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())

                    # Validate required keys
                    if "instruction" not in record:
                        raise ValueError("Missing 'instruction' key in record.")
                    if "output" not in record:
                        raise ValueError("Missing 'output' key in record.")

                    # Ensure 'input' field exists, even if empty, for consistency
                    if "input" not in record:
                        record["input"] = ""

                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON on line {line_num} in {self.file_path}: {e}")
                    raise ValueError(f"Invalid JSON format in file: {self.file_path} on line {line_num}. Error: {e}")
                except ValueError as e:
                    logger.error(f"Validation error on line {line_num} in {self.file_path}: {e}")
                    raise ValueError(f"Invalid data format in file: {self.file_path} on line {line_num}. Error: {e}")

        logger.info(f"Successfully loaded {len(data)} records from {self.file_path}.")
        return data

