"""
Funtions for Building Jacmates AI Model
Author: Ugochukwu Febechukwu

"""
#Imported libraries
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import TextLoader
import torch
from langchain_core.prompts import PromptTemplate

# from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the embedding model (same one used during ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
    )
# 2. Connect to your persistent ChromaDB folder
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)
# 3. Wrap the existing collection in a LangChain Chroma vectorstore
# vectorstore = Chroma(
#     client=client,
#     collection_name="ml_publications",
#     embedding_function=embeddings
# )

# 4. Get retriever for RAG
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# # Set up our embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
#     )

# Chunk files
def chunk_research_paper(paper_content, title):
    """Break a research paper into searchable chunks"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~200 words per chunk
        chunk_overlap=200,        # Overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(paper_content)
    
    # Add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    
    return chunk_data

# Embed chunked files


def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings

# Insert the chunked and embedded files into chromaDB
def insert_publications(collection: chromadb.Collection, publications: list[str], title ="doctest"):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        publications (list[str]): The documents to insert

    Returns:
        None
    """
    next_id = collection.count()

    for publication in publications:
        chunked_publication = chunk_research_paper(publication,title)

        # Step 2: extract only the text content for embeddings
        chunk_texts = [chunk["content"] for chunk in chunked_publication]
        
        embeddings = embed_documents(chunk_texts)
        ids = list(range(next_id, next_id + len(chunked_publication)))
        ids = [f"document_{id}" for id in ids]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunk_texts,
            metadatas= chunked_publication
        )
        next_id += len(chunked_publication)
    return next_id

#This functions searches the chromadb database for the top three similar chunks
def search_research_db(query, collection, embeddings, top_k=5):
    """Find the most relevant research chunks for a query"""
    
    # Convert question to vector
    query_vector = embeddings.embed_query(query)
    
    # Search for similar content
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        relevant_chunks.append({
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    
    return relevant_chunks




def answer_research_question(query, collection, embeddings, llm):
    """Generate an answer based on retrieved research"""
    
    # Get relevant research chunks
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)
    
    # Build context from research
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}" 
        for chunk in relevant_chunks
    ])
    
    # Create research-focused prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Based on the following research findings, answer the students's question:

        Research Context:
        {context}

        Researcher's Question: {question}

        Answer: Provide a comprehensive answer based on the findings above.
        """
    )
    
    # Generate answer
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    
    return response.content, relevant_chunks


"""
PGVECTOR
from langchain_postgres import PGVector
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

connection_string = "postgresql+psycopg2://user:password@localhost/research_db"
vectorstore = PGVector(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name="research_docs",
)
"""
