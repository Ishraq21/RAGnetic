from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool

def get_retriever_tool(agent_name: str, description: str = None) -> Tool:
    vectordb = FAISS.load_local(
        f"vectorstore/{agent_name}",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever()

    def tool_fn(input_dict: dict):
        if "input" not in input_dict:
            raise ValueError(f"Tool input must include 'input' key. Got: {input_dict}")
        query = input_dict["input"]
        return retriever.invoke(query)

    return Tool(
        name=f"{agent_name}_retriever",
        description=description or f"Search the embedded knowledge base for {agent_name}.",
        func=tool_fn,
    )

def get_retriever(agent_name: str):
    vectordb = FAISS.load_local(
        f"vectorstore/{agent_name}",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    return vectordb.as_retriever()