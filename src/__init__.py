from data_loader import process_all_pdfs
from chunker import split_documents
from embedding import EmbeddingManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
from gemini_LLM_integration import build_context, generate_answer
import time



if __name__ == "__main__": 

    start = time.perf_counter()
    
    question = "how does the payment plan work?"
    
    # Load all PDF Documents from Data folder as Langchain documents
    # Returns List of Langchain Documents
    all_pdf_documents = process_all_pdfs("data") 

    # Split PDF documents into chunks
    # Returns List of Langchain Documents
    chunks = split_documents(all_pdf_documents, chunk_size=500, chunk_overlap=100)

    # Initialise object of class Embedding Manager
    embedding_manager = EmbeddingManager()
    
    # Initialise Chroma DB vector store 
    vector_store=VectorStore()
    
    # Extract content from chunks as elements of a string
    texts = [doc.page_content for doc in chunks]
    
    print("Chunks are printed here")
    for (i, val) in enumerate(texts, start=1): 
        print("\n")
        print("\n")
        print(i)
        print(val)
        print("\n")

    
    # Embed the strings using all-MiniLM-L6-v2 model from Hugging Face
    embeddings = embedding_manager.generate_embeddings(texts)

    # Store the embeddings into Chroma DB vector database 
    vector_store.add_documents(chunks, embeddings)

    # Initialise Object of class RAG Retriever
    rag_retriever=RAGRetriever(vector_store,embedding_manager)      

    # Retrieves relevant chunks from vector database 
    retrieved_chunks = rag_retriever.retrieve(question)
    
    # Print the Retrieved Chunks content
    print("\n")
    print("\n")
    print("YEEHAW")
    print("\n")
    print("\n")
    for (i, chunk) in enumerate(retrieved_chunks, start=1): 
        print("\n")
        print(i)
        print(chunk["content"])
        print("\n")

    
    # Extract information from Langchain Document Data Type to String  
    context = build_context(retrieved_chunks)
    print("Extracted Context")
    # Query Gemini API providing it with the question and extracted context 
    
    generate_answer(question, context)

    print(f"took {time.perf_counter() - start:.2f}s")