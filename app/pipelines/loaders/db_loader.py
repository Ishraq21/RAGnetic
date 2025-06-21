from sqlalchemy import create_engine, inspect
from langchain_core.documents import Document
from typing import List


def load(connection_string: str) -> List[Document]:
    """
    Connects to a SQL database, inspects its schema, and creates a
    Document for each table with its schema and sample rows.
    """
    if not connection_string:
        print("Error: Database connection string is required.")
        return []

    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        docs = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)

            # Get table schema as a CREATE TABLE statement

            schema_info = f"Table Name: {table_name}\nColumns:\n"
            for column in columns:
                schema_info += f"- {column['name']} ({column['type']})\n"

            # Get a few sample rows
            try:
                with engine.connect() as connection:
                    result = connection.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows = "Sample Rows:\n"
                    for row in result:
                        sample_rows += str(dict(row)) + "\n"
            except Exception as e:
                sample_rows = f"Could not fetch sample rows: {e}\n"

            page_content = schema_info + "\n" + sample_rows

            doc = Document(
                page_content=page_content,
                metadata={
                    "source": connection_string,
                    "source_type": "database",
                    "table_name": table_name
                }
            )
            docs.append(doc)

        print(f"Loaded schema for {len(docs)} tables from the database.")
        return docs

    except Exception as e:
        print(f"Error connecting to database or inspecting schema: {e}")
        return []