import chromadb
# from langchain_huggingface import HuggingFaceEmbeddings
from inserting_file import load_pdf_to_strings, load_txt_to_strings
from ocr import ocr_pdfs_to_text
from jac_functions import chunk_research_paper, search_research_db, insert_publications
# from langchain_community.vectorstores import Chroma

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./research_db")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)
# Set up our embedding model
# vectorstore = Chroma(
#     client=client,
#     collection_name="ml_publications",
#     embedding_function=embeddings
# )
# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
#     )

# publication =ocr_pdfs_to_text(
#         input_folder=input_dir,
#         output_folder=output_dir,
#         dpi=300,  # Good balance of quality and speed
#         max_workers=1 # Process 4 PDFs in parallel
#         )
# publication = load_pdf_to_strings("data/400 Level/1st Semester")
# db = insert_publications(collection, publication, title="400 level")


if __name__ == "__main__":
    # Example 1: Basic usage
    input_dir = "data/ADULT AND CONTINUING EDUCATION"
    output_dir = "ocr_data_output"
    
    #Process all PDFs with OCR
    stats =ocr_pdfs_to_text(
        input_folder=input_dir,
        output_folder=output_dir,
        dpi=300,  # Good balance of quality and speed
        max_workers=1 # Process 4 PDFs in parallel
        )
    print(f"Processing complete! Success: {stats['successful']}/{stats['total']}")

    publication = load_txt_to_strings("ocr_data_output")
    db = insert_publications(collection, publication)
    