"""
Everything except the chat screen: loading the PDFs and models at startup,
and the per-question steps (prepare question, cache, search, answer, memory).
The chat screen is in chat_ui.py.
"""
from chunker import load_pdf_files
from embedding import EmbeddingManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
import time
from pathlib import Path
import debug
import chainlit as cl
import asyncio
from ollama_LLM_integration import build_context, generate_answer, stream_answer
#from gemini_LLM_integration import build_context, generate_answer, stream_answer
from ollama_LLM_integration import prepare_question, same_question, checked_translation, TranslationFailed
from answer_cache import AnswerCache

print("\nLoading PDF Files . . .\n")

# Find all PDF files inside the data folder
pdf_files = list(Path("data").glob("**/*.pdf"))

print(f"Chunking {len(pdf_files)} files ...\n")

# Split the PDFs into chunks (prose and tables are stored separately)
chunks = load_pdf_files(pdf_files, chunk_size=500, chunk_overlap=100)

# Load the embedding model 
embedding_manager = EmbeddingManager()

# Create the Chroma DB vector store
vector_store = VectorStore()

# Get the plain text out of each chunk
texts = [doc.page_content for doc in chunks]

# Turn every chunk into an embedding (also slow, so only once)
embeddings = embedding_manager.generate_embeddings(texts)

# Save the chunks and embeddings into Chroma
vector_store.add_documents(chunks, embeddings)

# Create the retriever that searches the vector store
rag_retriever = RAGRetriever(vector_store, embedding_manager)

# Saved answers, matched by meaning and across languages (empty after a restart)
answer_cache = AnswerCache(embedding_manager, same_question)

print("\nSetup done. Ready for questions.\n")


# ======================================================================
# PER-QUESTION WORK
# This runs for every question, and only does the fast steps.
# ======================================================================

def answer_one(question: str) -> str:
    start = time.perf_counter()

    context, _ = get_context(question)

    print("\nCONNECTING TO MODEL . . .\n")

    answer = generate_answer(question, context)

    print(f"took {time.perf_counter() - start:.2f}s")

    return answer

MAX_HISTORY = 3  # past question/answer pairs remembered per chat


def remember(question: str, answer: str):
    """Saves this exchange in this chat's memory (each browser tab has its own)."""
    if not answer:
        return
    history = cl.user_session.get("history") or []
    history.append((question, answer))
    cl.user_session.set("history", history[-MAX_HISTORY:])


def retrieve_context(search_question: str) -> str:
    print("\nRetrieving Relevant Chunks . . .\n")
    retrieved_chunks = rag_retriever.retrieve(search_question)
    debug.print_retrieved_chunks(retrieved_chunks)
    return build_context(retrieved_chunks)


def get_context(question: str, history=None, last_search=None):
    """Returns (context, search_question).
    Search in English (the PDFs are English), answer in the user's language.
    Follow-ups like "what about it?" are filled in from the history first,
    and "say that in English" reuses last_search."""
    search_question = prepare_question(question, history, last_search)["search"]
    return retrieve_context(search_question), search_question


# Answers that are error messages must never be saved in the cache
ERROR_MARKERS = ("I could not reach the local model", "Gemini is busy or not responding")


def is_error(answer: str) -> bool:
    return any(marker in answer for marker in ERROR_MARKERS)


async def stream_into(msg: cl.Message, question: str, prefix: str = "", on_first_piece=None) -> str:
    """Streams the answer into msg as it arrives, then saves it to this chat's
    memory and to the answer cache. Returns the full answer.

    Three ways to answer:
      1. cache hit, same language   -> saved answer, instantly
      2. cache hit, other language  -> saved answer translated (no PDF search)
      3. no cache hit               -> full search + answer, then saved
    """
    start = time.perf_counter()
    history = cl.user_session.get("history") or []
    last_search = cl.user_session.get("last_search")

    prep = await cl.make_async(prepare_question)(question, history, last_search)
    search_question, language = prep["search"], prep["language"]
    cl.user_session.set("last_search", search_question)

    # "Say that in English" / "make it shorter" depend on the previous answer,
    # so they never use or fill the cache. Same if the language is unknown.
    use_cache = not prep["redo"] and bool(language)
    cached = await cl.make_async(answer_cache.lookup)(search_question) if use_cache else None

    if cached and language in cached["answers"]:
        print(f"Cache hit ({language}): instant answer")
        pieces = iter([cached["answers"][language]])
    elif cached:
        # Translate from the English version if there is one (most reliable source)
        source_language = "english, latin" if "english, latin" in cached["answers"] else next(iter(cached["answers"]))
        print(f"Cache hit in {source_language}: translating to {language}, no PDF search")
        pieces = checked_translation(cached["answers"][source_language], language)
    else:
        pieces = None

    # Was this language already saved before this question? (then nothing new to save)
    already_saved = cached is not None and language in cached["answers"]
    answer = ""

    while True:
        if pieces is None:
            # Normal path: search the PDFs and write a fresh answer
            context = await cl.make_async(retrieve_context)(search_question)
            print("\nCONNECTING TO MODEL (streaming) . . .\n")
            pieces = stream_answer(question, context, history)
        try:
            # The model's stream is blocking, so read each piece in a thread
            piece = await asyncio.to_thread(next, pieces, None)
        except TranslationFailed:
            # Nothing was shown yet: answer normally instead. The bad translation is
            # not saved; the fresh answer is saved as this language's version.
            print("Translation came back untranslated, answering normally instead")
            pieces = None
            continue
        if piece is None:
            break
        if not answer and on_first_piece:
            await on_first_piece()
        answer += piece
        msg.content = prefix + answer
        await msg.update()

    print(f"took {time.perf_counter() - start:.2f}s")
    remember(question, answer)

    if use_cache and answer and not is_error(answer) and not already_saved:
        # New question -> new entry; translated answer -> new language for that entry
        await cl.make_async(answer_cache.add)(search_question, language, answer, cached)
    return answer


def split_questions(text: str) -> list:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if line.endswith("?")]
