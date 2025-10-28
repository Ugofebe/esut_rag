import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

# def load_txt_to_strings(documents_path):
#     """Load research publications from .txt files and return as list of strings"""
    
#     # List to store all documents
#     documents = []
    
#     # Load each .txt file in the documents folder
#     for file in os.listdir(documents_path):
#         if file.endswith(".txt"):
#             file_path = os.path.join(documents_path, file)
#             try:
#                 loader = TextLoader(file_path)
#                 loaded_docs = loader.load()
#                 documents.extend(loaded_docs)
#                 print(f"Successfully loaded: {file}")
#             except Exception as e:
#                 print(f"Error loading {file}: {str(e)}")
    
#     print(f"\nTotal documents loaded: {len(documents)}")
    
#     # Extract content as strings and return
#     publications = []
#     for doc in documents:
#         publications.append(doc.page_content)
    
#     return publications

def load_txt_to_strings(documents_path):
    """Load research publications from .txt files and return as list of dictionaries with content and title"""
    
    # List to store all documents
    documents = []
    
    # Load each .txt file in the documents folder
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                
                # Extract title from filename (remove .txt extension)
                title = os.path.splitext(file)[0]
                
                # Create dictionary with content and title for each document
                for doc in loaded_docs:
                    documents.append({
                        "content": doc.page_content,
                        "title": title
                    })
                
                print(f"Successfully loaded: {file} as '{title}'")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    return documents

def load_pdf_to_strings(documents_path):
    """Load research publications from PDF files (including subfolders) and return as list of strings"""
    
    # List to store all documents
    documents = []
    
    # Walk through all subfolders
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.lower().endswith(".pdf"):  # check for PDFs
                file_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    print(f"✅ Successfully loaded: {file}")
                except Exception as e:
                    print(f"❌ Error loading {file}: {str(e)}")
    
    print(f"\n📂 Total documents loaded: {len(documents)}")
    
    # # Extract content as strings and return
    # publications = [doc.page_content for doc in documents]
    # Extract content as strings and return
    publications = [doc.page_content for doc in documents if doc.page_content.strip()]

    print(f"\n📂 Total documents after stripping: {len(publications)}")

    return publications

def main():
    publication_pdfs = load_pdf_to_strings("data/400 Level")
    print(len(publication_pdfs))

    
if __name__ == "__main__":
    main()
