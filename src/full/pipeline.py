"""
Everything except the chat screen: loading the PDFs and models at startup,
and the per-question steps (prepare question, cache, search, answer, memory).
The chat screen is in chat_ui.py.
"""
import sys
from pathlib import Path
# The code both versions use (website loader, Groq, prompts, answer cache) is in src/shared
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

from chunker import load_pdf_files
from web_loader import load_websites
from embedding import EmbeddingManager
from vector_store import VectorStore
from rag_retriever import RAGRetriever
import time
from pathlib import Path
import debug
import chainlit as cl
import asyncio
# Which model answers. Groq is on; to go back to the local Ollama model,
# comment out the 3 Groq lines and remove the # from the 3 Ollama lines.
from groq_LLM_integration import build_context, generate_answer, stream_answer
from groq_LLM_integration import prepare_question, same_question, checked_translation, TranslationFailed
from groq_LLM_integration import contact_details, describe_language
#from ollama_LLM_integration import build_context, generate_answer, stream_answer
#from ollama_LLM_integration import prepare_question, same_question, checked_translation, TranslationFailed
#from ollama_LLM_integration import contact_details, describe_language
#from gemini_LLM_integration import build_context, generate_answer, stream_answer
from answer_cache import AnswerCache

# Where the bot gets its text from. Both can be on at the same time.
# TRANSACCELORS ONLY for now: the 2 PDFs are Azuria and Wayfarra, so PDFs are off.
# TransAccelors comes from its website (data/websites.txt). Put back True to read the PDFs.
# USE_PDFS = True       # every PDF inside data/
USE_PDFS = False
USE_WEBSITES = True   # every link listed in data/websites.txt
WEBSITES_FILE = "data/websites.txt"

chunks = []

if USE_PDFS:
    print("\nLoading PDF Files . . .\n")

    # Find all PDF files inside the data folder
    pdf_files = list(Path("data").glob("**/*.pdf"))

    print(f"Chunking {len(pdf_files)} files ...\n")

    # Split the PDFs into chunks (prose and tables are stored separately)
    chunks += load_pdf_files(pdf_files, chunk_size=500, chunk_overlap=100)

if USE_WEBSITES:
    # Download the listed pages and chunk them the same way
    chunks += load_websites(WEBSITES_FILE, chunk_size=500, chunk_overlap=100)

if not chunks:
    raise RuntimeError("Nothing to read: put PDFs in data/ or links in data/websites.txt "
                       "(and check USE_PDFS / USE_WEBSITES in pipeline.py)")

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


def remember(session: dict, question: str, answer: str):
    """Saves this exchange in this chat's memory (each chat has its own session dict:
    one per browser tab in Chainlit, one per session_id in the API)."""
    if not answer:
        return
    history = session.get("history") or []
    history.append((question, answer))
    session["history"] = history[-MAX_HISTORY:]


MAX_CHUNKS = 7    # most chunks sent to the model when a question has several parts
PART_CHUNKS = 2   # best chunks kept for each part
PART_RERANK = 6   # only the 6 best matches of a part are reranked (keeps it fast)


def take_turns(results: list, limit: int) -> list:
    """Best chunk of every search first, then the second best of every search, ...
    so each part of the question gets its own chunks. No duplicates."""
    chosen, seen = [], set()
    for rank in range(max(len(found) for found in results)):
        for found in results:
            if rank < len(found) and found[rank]["id"] not in seen:
                seen.add(found[rank]["id"])
                chosen.append(found[rank])
                if len(chosen) == limit:
                    return chosen
    return chosen


def retrieve_context(prep: dict, question: str = "") -> str:
    """Searches the PDFs for a prepared question (see prepare_question).
    When the company is known, only its own chunks are used (unless the label looks wrong).
    A question with several parts gets one extra search per part, so no part is left out.
    Ends with the contact details of that company only."""
    print("\nRetrieving Relevant Chunks . . .\n")
    company = prep.get("company")
    query = prep.get("query") or prep["search"]
    # The keyword search also gets the other wording: the rewrite fixes typos
    # ("Azura"), the user's own words keep anything the rewrite left out.
    # Not for "say that in English", whose words are not about the topic.
    if prep.get("redo"):
        extra_words = ""
    else:
        extra_words = prep["search"] if query != prep["search"] else question
    results = [rag_retriever.retrieve(query, company=company, extra_words=extra_words)]
    # If the company label was wrong, the best chunk comes from another PDF:
    # then search everything for the topics too, and show all contact details
    top = results[0][0]["metadata"].get("source_file", "").lower() if results[0] else ""
    if company and top and not top.startswith(company):
        company = None
    for part in prep.get("parts", []):
        results.append(rag_retriever.retrieve(part, top_k=PART_CHUNKS, candidates=10,
                                              company=company, rerank_limit=PART_RERANK))
    retrieved_chunks = take_turns(results, MAX_CHUNKS) if len(results) > 1 else results[0]
    debug.print_retrieved_chunks(retrieved_chunks)
    return build_context(retrieved_chunks) + "\n\n" + contact_details(company)


def get_context(question: str, history=None, last=None):
    """Returns (context, search_question).
    Search in English (the PDFs are English), answer in the user's language.
    Follow-ups like "what about it?" are filled in from the history first,
    and "say that in English" reuses the previous search (last)."""
    prep = prepare_question(question, history, last)
    return retrieve_context(prep, question), prep["search"]


# Answers that are error messages must never be saved in the cache
ERROR_MARKERS = ("I could not reach the local model", "Gemini is busy or not responding",
                 "I could not reach the Groq API")


def is_error(answer: str) -> bool:
    return any(marker in answer for marker in ERROR_MARKERS)


# "Not found" answers are often a search miss: don't save them in the cache,
# so the next person asking gets a fresh search instead of the same miss
NOT_FOUND_MARKERS = ("could not find", "couldn't find", "couldn’t find")


def is_not_found(answer: str) -> bool:
    return any(marker in answer.lower() for marker in NOT_FOUND_MARKERS)


async def stream_into(msg: cl.Message, question: str, prefix: str = "", on_first_piece=None) -> str:
    """Chainlit chat page: streams the answer into msg as it arrives. Returns the full answer.
    The work is done by answer_pieces (shared with the API in api.py)."""
    # This browser tab's memory, kept in Chainlit's session
    session = {"history": cl.user_session.get("history") or [],
               "last_prep": cl.user_session.get("last_prep")}
    answer = ""
    try:
        async for piece in answer_pieces(question, session):
            if not answer and on_first_piece:
                await on_first_piece()
            answer += piece
            msg.content = prefix + answer
            await msg.update()
    finally:
        cl.user_session.set("history", session["history"])
        cl.user_session.set("last_prep", session["last_prep"])
    return answer


async def answer_pieces(question: str, session: dict):
    """Yields the answer piece by piece as it is written, then saves it to this
    chat's memory (session) and to the answer cache. Used by the Chainlit page
    and by the API. session: {"history": [...], "last_prep": ...}, updated here.

    Three ways to answer:
      1. cache hit, same language   -> saved answer, instantly
      2. cache hit, other language  -> saved answer translated (no PDF search)
      3. no cache hit               -> full search + answer, then saved
    """
    start = time.perf_counter()
    history = session.get("history") or []
    last_prep = session.get("last_prep")

    prep = await asyncio.to_thread(prepare_question, question, history, last_prep)
    language = prep["language"]
    # Cache key: what the user typed (first English message), else the English rewrite.
    # The rewrite words the same message differently each time and can drop details.
    search_question = prep.get("query") or prep["search"]
    session["last_prep"] = prep  # reused by "say that in English"

    # "Say that in English" / "make it shorter" depend on the previous answer,
    # so they never use or fill the cache. Same if the language is unknown.
    use_cache = not prep["redo"] and bool(language)
    cached = await asyncio.to_thread(answer_cache.lookup, search_question, prep.get("company")) if use_cache else None

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
            context = await asyncio.to_thread(retrieve_context, prep, question)
            if language:
                # Tell the model the reply language: left to guess, it once
                # answered an English question in Marathi
                context += f"\n\nReply language: {describe_language(language)}"
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
        answer += piece
        yield piece

    print(f"took {time.perf_counter() - start:.2f}s")
    remember(session, question, answer)

    if use_cache and answer and not is_error(answer) and not is_not_found(answer) and not already_saved:
        # New question -> new entry; translated answer -> new language for that entry
        await asyncio.to_thread(answer_cache.add, search_question, language, answer, cached,
                                prep.get("company"))


def split_questions(text: str) -> list:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if line.endswith("?")]
