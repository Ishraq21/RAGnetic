import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.schemas.agent import AgentConfig
from app.pipelines.loaders import directory_loader, url_loader, pdf_loader, docx_loader, csv_loader, code_loader, db_loader, gdoc_loader, web_crawler_loader, api_loader

def load_documents_from_source(source: dict) -> list[Document]:
    """
    Dispatcher function that calls the appropriate loader based on source type.
    """
    source_type = source.get("type")
    path = source.get("path")

    if source_type == "local":
        if not path or not os.path.exists(path):
            print(f"Warning: Local path not found: {path}. Skipping.")
            return []

        if os.path.isfile(path):
            if path.lower().endswith('.pdf'):
                return pdf_loader.load(path)
            elif path.lower().endswith('.docx'):
                return docx_loader.load(path)
            elif path.lower().endswith('.csv'):
                return csv_loader.load(path)
            else:
                print(f"Warning: Unsupported local file type: {os.path.basename(path)}. Skipping.")
                return []

        elif os.path.isdir(path):
            return directory_loader.load(path)

        else:
            print(f"Warning: Local path '{path}' is not a valid file or directory. Skipping.")
            return []

    elif source_type == "url":
        return url_loader.load(source["url"])

    elif source_type == "code_repository":
        return code_loader.load(source["path"])

    elif source_type == "db":
        return db_loader.load(source["db_connection"])

    elif source_type == "gdoc":
        return gdoc_loader.load(
            folder_id=source.get("folder_id"),
            document_ids=source.get("document_ids"),
            file_types=source.get("file_types")
        )

    elif source_type == "web_crawler":
        return web_crawler_loader.load(
            url=source.get("url"),
            max_depth=source.get("max_depth", 2)
        )

    elif source_type == "api":
        return api_loader.load(
            url=source.get("url"),
            method=source.get("method", "GET"),
            headers=source.get("headers"),
            params=source.get("params"),
            payload=source.get("payload"),
            json_pointer=source.get("json_pointer")
        )


    else:
        print(f"Warning: Unknown source type: {source_type}. Skipping.")
        return []


def embed_agent_data(config: AgentConfig, openai_api_key: str = None):
    all_docs = []
    for source in config.sources:
        loaded_docs = load_documents_from_source(source.dict())
        all_docs.extend(loaded_docs)

    if not all_docs:
        raise ValueError("No valid documents found to embed from any source.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(f"vectorstore/{config.name}")