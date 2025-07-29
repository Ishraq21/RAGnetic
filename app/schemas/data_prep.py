# app/schemas/data_prep.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Pydantic model representing the structure of a dataset preparation YAML configuration file
class DatasetPreparationConfig(BaseModel):
    prep_name: str = Field(..., description="A user-defined name for this dataset preparation configuration.")
    input_file: str = Field(..., description="Path to the raw input data file (e.g., CSV, JSONL).")
    format_type: str = Field(..., description="Target output format for fine-tuning: 'jsonl-instruction', 'conversational-jsonl'.")
    output_file: str = Field(..., description="Path to save the prepared dataset file.")
    # Optional format-specific options for various input types (e.g., for CSV parsing)
    csv_delimiter: Optional[str] = Field(",", description="Delimiter character for CSV input files (if format_type is CSV-based).")
    instruction_column: Optional[str] = Field(None, description="Column name for instructions in CSV input (if applicable).")
    output_column: Optional[str] = Field(None, description="Column name for outputs in CSV input (if applicable).")
