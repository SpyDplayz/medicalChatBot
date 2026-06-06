# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from typing import List
# from langchain.schema import Document


# #Extract text from pdf files
# def load_pdf_files(data):
#     loader = DirectoryLoader(data, 
#                              glob="*.pdf", show_progress=True,
#                              loader_cls=PyPDFLoader)
    
#     documents = loader.load()
#     return documents

# extracted_data = load_pdf_files("data")

# def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
#     minimal_docs : List[Document] = []
#     for doc in docs:
#         src = doc.metadata.get("source")
#         minimal_docs.append(
#             Document(
#                 page_content=doc.page_content,
#                 metadata={"source": src}
#             )
#         )
#     return minimal_docs

# minimal_docs = filter_to_minimal_docs(extracted_data)


# def split_text(docs : List[Document]) -> List[Document]:
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                    chunk_overlap=20,
#                                                    length_function=len)
#     texts_chunk = text_splitter.split_documents(minimal_docs)
#     return texts_chunk

# text_chunks = split_text(minimal_docs)
# print("Number of chunks :",len(text_chunks))

# #hugging face embedding model
# from langchain.embeddings import HuggingFaceEmbeddings


# def download_embeddings():
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     embeddings = HuggingFaceEmbeddings(model_name=model_name)
#     return embeddings

# embedding = download_embeddings()

# vector = embedding.embed_query("hello world")
# print("Vector Length :",len(vector))

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader
    )
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs


def split_text(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    return splitter.split_documents(docs)


def download_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )