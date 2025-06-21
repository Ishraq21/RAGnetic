import pandas as pd
from langchain_core.documents import Document
from typing import List
import os


def load(file_path: str) -> List[Document]:
    """
    Loads a CSV file and creates a well-formatted Document for each row.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    try:
        df = pd.read_csv(file_path)
        docs = []
        for index, row in df.iterrows():
            # Instead of a dense string, create a more readable, structured format.
            # This helps the LLM understand the data for each row individually.

            # Use the first column's value as a title, e.g., the customer's name or ID
            title_col = df.columns[0]
            title_val = row[title_col]

            # Format each column-value pair on a new line
            row_details = "\n".join([f"- {str(col).replace('_', ' ').strip()}: {val}" for col, val in row.items()])

            page_content = f"Record for {title_col} '{title_val}':\n{row_details}"

            doc = Document(
                page_content=page_content,
                metadata={
                    "source": os.path.abspath(file_path),
                    "source_type": "csv",
                    "row_number": index + 1
                }
            )
            docs.append(doc)

        print(f"Loaded {len(docs)} rows from {os.path.basename(file_path)}")
        return docs
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return []