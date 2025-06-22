import requests
import json
import jsonpointer
from typing import List, Dict, Any
from langchain_core.documents import Document


def load(
        url: str,
        method: str = 'GET',
        headers: Dict = None,
        params: Dict = None,
        payload: Dict = None,
        json_pointer: str = None
) -> List[Document]:
    """
    Fetches data from a REST API endpoint using GET or POST and creates a Document for each record.
    """
    if not url:
        print("Error: An API URL is required.")
        return []

    try:
        response = None
        if method.upper() == 'POST':
            print(f"Making POST request to {url}")
            response = requests.post(url, headers=headers, json=payload)
        else:  # Default to GET
            print(f"Making GET request to {url}")
            response = requests.get(url, headers=headers, params=params)

        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        data = response.json()

        # If a json_pointer is provided, navigate to the specific part of the JSON response
        if json_pointer:
            records = jsonpointer.resolve_pointer(data, json_pointer)
        else:
            records = data

        # Ensure we're working with a list, even if the API returns a single object
        if not isinstance(records, list):
            records = [records]

        docs = []
        for record in records:
            # Convert each JSON object into a formatted string for the document content
            content = json.dumps(record, indent=2)
            doc = Document(
                page_content=content,
                metadata={"source": url, "source_type": "api"}
            )
            docs.append(doc)

        print(f"Loaded {len(docs)} records from API endpoint: {url}")
        return docs

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API {url}: {e}")
    except jsonpointer.JsonPointerException as e:
        print(f"Error resolving JSON pointer '{json_pointer}': {e}")
    except Exception as e:
        print(f"An error occurred in the API loader: {e}")

    return []