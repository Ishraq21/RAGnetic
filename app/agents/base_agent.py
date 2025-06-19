from app.agents.loader import load_agent_config
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_agent(name: str) -> RetrievalQA:
    config = load_agent_config(name)
    vectordb = FAISS.load_local(
        f"vectorstore/{name}",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
{persona}

Use the following context to answer the question:
{context}

Question: {question}
Answer:""".strip()
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt.partial(persona=config.persona_prompt)
        }
    )
