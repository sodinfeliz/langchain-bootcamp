import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Loading data...")
    loader = TextLoader("./intro-to-vector-dbs/mediumblog1.txt")
    documents = loader.load()

    print("Splitting data...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Ingesting data...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )
    print("Data ingested successfully")
