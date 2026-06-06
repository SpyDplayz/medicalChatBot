from dotenv import load_dotenv
import os

from src.helper import load_pdf_files, filter_to_minimal_docs, split_text, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

extracted_data = load_pdf_files("data/")
filtered_docs = filter_to_minimal_docs(extracted_data)
text_chunks = split_text(filtered_docs)

embeddings = download_embeddings()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

#----------------------------------------------------------
#Creating vector store from documents and embedding
#----------------------------------------------------------

# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     embedding=embeddings,
#     index_name=index_name
# )

#----------------------------------------------------------
#load existing index
#----------------------------------------------------------

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
    )
    