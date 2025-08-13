import base64
import os
from typing import List, Dict
from app.db.dao import list_temp_documents_for_user_thread
from pathlib import Path
from app.core.config import get_path_settings

async def get_temp_files_as_payload(db, user_id: int, thread_id: str) -> List[Dict]:
    """
    Fetch all temp docs for this user/thread and return as LambdaTool-ready files payload.
    """
    temp_docs = await list_temp_documents_for_user_thread(db, user_id, thread_id)
    payload = []

    for doc in temp_docs:
        # The stored file path should match TemporaryDocumentService save location
        file_path = os.path.join(
            os.environ.get("TEMP_CLONES_DIR", "/tmp"),  # Or use get_path_settings()
            "chat_uploads",
            str(user_id),
            thread_id,
            f"{doc['temp_doc_id']}_{doc['original_name']}"
        )

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            payload.append({
                "name": doc["original_name"],
                "data_b64": b64_data
            })

    return payload
