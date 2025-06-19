import os
from pathlib import Path
import trafilatura

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.schemas.agent import AgentConfig


def load_webpage(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) if downloaded else ""


def load_local_files(folder: str) -> list[str]:
    texts = []
    for file in Path(folder).rglob("*.*"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return texts


def embed_agent_data(config: AgentConfig, openai_api_key: str = None):
    texts = []

    for source in config.sources:
        if source.type == "local" and source.path:
            texts.extend(load_local_files(source.path))
        elif source.type == "url" and source.url:
            text = load_webpage(source.url)
            if text:
                texts.append(text)

    if not texts:
        raise ValueError("No valid text found to embed.")

    docs = [Document(page_content=t.strip()) for t in texts if t.strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(f"vectorstore/{config.name}")
