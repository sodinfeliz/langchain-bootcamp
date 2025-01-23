import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini")

    query = "What is Pinecone in machine learning?"

    # Directly querying the LLM
    chain = PromptTemplate.from_template(template=query).pipe(llm)
    result = chain.invoke({})
    print(result)

    # RAG with chains
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke({"input": query})
    print(result)

    # RAG with LCEL
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "Thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer: """

    custom_rag_prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    rag_chain: Runnable = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )
    res = rag_chain.invoke(query)
    print(res)
