from flask import Flask, request, jsonify, render_template
from src.helper import load_pdf_files, filter_to_minimal_docs, split_text, download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_ollama import ChatOllama
import os



app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Initialize Ollama LLM
chatModel = ChatOllama(model="llama3.2:1b",temperature=0.7)


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create QA Chain
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

# Create RAG Chain
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response :", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)