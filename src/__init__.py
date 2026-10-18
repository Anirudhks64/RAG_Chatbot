from chunker import split_documents, load_pdf_files
from embedding import EmbeddingManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
from gemini_LLM_integration import build_context, generate_answer
import time
from pathlib import Path

if __name__ == "__main__": 

    start = time.perf_counter()
    
    question = "how can I contact Azuria?"

    print("\n")
    print("Loading PDF Files . . .")
    print("\n")

    # Load PDF Files as Path Objects in a list    
    pdf_files = list(Path("data").glob("**/*.pdf"))

    print(f"Chunking {len(pdf_files)} files ...")
    print("\n")

    # Load the PDF Files into chunks 
    # Stores Prose and Table content in different chunks for easy retrieval
    chunks = load_pdf_files(pdf_files, chunk_size=500, chunk_overlap=100)
    
    # Initialise object of class Embedding Manager
    embedding_manager = EmbeddingManager()
    
    # Initialise Chroma DB vector store 
    vector_store=VectorStore()
    
    # Extract content from chunks as elements of a string
    texts = [doc.page_content for doc in chunks]

    ## Prints all Chunks for debugging
    # debug.print_all_chunks()

    # Embed the strings using BAAI/bge-small-en-v1.5 model from Hugging Face
    embeddings = embedding_manager.generate_embeddings(texts)

    # Store the embeddings into Chroma DB vector database 
    vector_store.add_documents(chunks, embeddings)

    # Initialise Object of class RAG Retriever
    rag_retriever=RAGRetriever(vector_store,embedding_manager)      

    print("\n")
    print("Retrieving Relevant Chunks . . .")
    print("\n")

    # Retrieves relevant chunks from vector database 
    retrieved_chunks = rag_retriever.retrieve(question)

    ## Prints retrieved chunks for debugging
    # debug.print_retrieved_chunks(retrieved_chunks)

    # Extract information from Langchain Document Data Type to String  
    context = build_context(retrieved_chunks)

    print("\n")
    print("CONNECTING TO GEMINI API . . .")
    print("\n")

    # Query Gemini API providing it with the question and extracted context 
    generate_answer(question, context)

    print(f"took {time.perf_counter() - start:.2f}s")