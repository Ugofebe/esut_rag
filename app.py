# app.py

import warnings
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be .* leaked semaphore objects"
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
from jac_functions import answer_research_question
# from langchain_openai import ChatOpenAI
import uvicorn
# openai_api_key = os.getenv("OPENAI_API_KEY")
# Load environment variables
load_dotenv()
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize FastAPI
app = FastAPI(title="RAG Chioma API", version="1.0")


# llm_gpt = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)




# Pydantic model for input
class QueryRequest(BaseModel):
    question: str

# @app.on_event("startup")
# def startup_event():
#     global client, collection, embeddings, llm_gpt
#     client = chromadb.PersistentClient(path="./research_db")
#     collection = client.get_or_create_collection(
#         name="ml_publications",
#         metadata={"hnsw:space": "cosine"}
#     )
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     llm_gpt = ChatOpenAI(
#             model_name='gpt-4o-mini',
#             temperature=0.7
#         )
@app.on_event("startup")
def startup_event():
    global client, collection
    client = chromadb.PersistentClient(path="./research_db")
    collection = client.get_or_create_collection(
        name="ml_publications",
        metadata={"hnsw:space": "cosine"}
    )


# Endpoint: Ask a research question
@app.post("/ask")
def ask_question(payload: QueryRequest):
    """
    Run a research query against the RAG system.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm_gpt = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7)
    try:
        answer, sources = answer_research_question(
            payload.question, collection, embeddings, llm_gpt
        )
        return {
            "question": payload.question,
            "answer": answer,
            "sources": sources[0]["title"],
            "sources": sources #this can be removed it shows all the sources
        }
    except Exception as e:
        return {"error": str(e)}
# @app.on_event("shutdown")
# def shutdown_event():
#     """
#     Gracefully close ChromaDB client to avoid resource leaks.
#     This does NOT delete any data stored in the persistent database.
#     """
#     try:
#         if client is not None:
#             client.reset()  # or client.close()
#     except Exception as e:
#         print(f"Chroma shutdown warning: {e}")

# Health check endpoint
@app.get("/")
def home():
    return {"status": "ok", "message": "RAG API is running "}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "ESUT RAG API is running "}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT env var
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)