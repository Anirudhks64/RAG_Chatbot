import os 
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pathlib import Path

### Read all the pdf's inside the directory
### Returns a list of langchain document data type
def process_all_pdfs(pdf_directory):
    
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    # Find all PDF files recursively
    # Stores the paths of these documents in a list as type Path
    
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load() # Loads document as type Langchain Document

            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            # Appends all pages to all_documents which is a list 
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

# Process all PDFs in the data directory
all_pdf_documents = process_all_pdfs("data")
