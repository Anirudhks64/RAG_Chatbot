from chunker import load_pdf_files
from embedding import EmbeddingManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
from gemini_LLM_integration import build_context, generate_answer
import time
from pathlib import Path
import debug
import chainlit as cl
import asyncio

def get_answer(question: str) -> str:        
    start = time.perf_counter()
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

    # # Prints retrieved chunks for debugging
    # debug.print_retrieved_chunks(retrieved_chunks)

    # Extract information from Langchain Document Data Type to String  
    context = build_context(retrieved_chunks)

    print("\n")
    print("CONNECTING TO GEMINI API . . .")
    print("\n")

    # Query Gemini API providing it with the question and extracted context 
    answer = generate_answer(question, context)
    print(f"took {time.perf_counter() - start:.2f}s")
    
    return answer



THINKING_LINES = [
    "Flipping through your PDFs . . .",
    "Hunting for the best chunks . . .",
    "Waking up the smart robot in the cloud . . .",
    "Connecting the dots . . .",
    "Reading the fine print so you don't have to . . .",
    "Polishing the answer . . .",
    "Almost there . . .",
]
 
TRACK_LENGTH = 10  # how many steps the robot takes before it reaches the books
 
 
async def animate_thinking(reply: cl.Message):
    """Updates the 'Thinking' bubble every moment until it is cancelled."""
    frame = 0
    while True:
        step = frame % (TRACK_LENGTH + 1)
        track = "  ".join(["·"] * step + ["🤖"] + ["·"] * (TRACK_LENGTH - step)) + "  📚"
        line = THINKING_LINES[(frame // 4) % len(THINKING_LINES)]
 
        reply.content = f"{line}\n\n{track}"
        await reply.update()
 
        frame += 1
        await asyncio.sleep(0.7)
 
 
@cl.on_chat_start
async def start():
    await cl.Message(content="Hi! Ask me a question about your documents.").send()
 
 
@cl.on_message
async def main(message: cl.Message):
    reply = cl.Message(content="Thinking . . .")
    await reply.send()
 
    # Start the animation in the background
    animation = asyncio.create_task(animate_thinking(reply))
 
    try:
        # Runs your function without freezing the chat page
        answer = await cl.make_async(get_answer)(message.content)
        if not answer:
            answer = (
                "The bot did not return any text. "
                "Make sure generate_answer() uses return, not just print()."
            )
    except Exception as error:
        answer = f"Something went wrong: {error}"
    finally:
        # Stop the animation as soon as the answer is ready
        animation.cancel()
        try:
            await animation
        except asyncio.CancelledError:
            pass
 
    # Replace the animation with the real answer
    reply.content = str(answer)
    await reply.update()
 