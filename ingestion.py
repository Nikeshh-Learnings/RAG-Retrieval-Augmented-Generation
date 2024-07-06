import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    # Load the blog
    loader = TextLoader(
        "/Users/nikeshhvijayabaskaran/Desktop/Projects/RAG-Retrieval-Augmented-Generation/medium-blog.txt"
    )
    document = loader.load()

    # Split the document
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Create the embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
