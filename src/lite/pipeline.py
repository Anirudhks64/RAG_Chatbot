"""
LITE version of pipeline.py for a 512 MB server: the same steps (prepare question, cache,
search, answer with Groq, memory), but with the small models from search.py and
without Chainlit, Chroma or PyTorch. Used by api.py in this folder.

The full version (src/full/pipeline.py) is used by the Chainlit page and src/full/api.py.
"""
import asyncio
import hashlib
import threading
import time
import sys
from pathlib import Path
# The code both versions use (website loader, Groq, prompts, answer cache) is in src/shared
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))


from web_loader import load_websites
from search import LiteEmbedder, LiteRetriever
from groq_LLM_integration import build_context, stream_answer
from groq_LLM_integration import prepare_question, same_question, checked_translation, TranslationFailed
from groq_LLM_integration import contact_details, describe_language
from answer_cache import AnswerCache

# Same sources as pipeline.py: TransAccelors website only for now
USE_PDFS = False
USE_WEBSITES = True
WEBSITES_FILE = "data/websites.txt"

chunks = []
if USE_PDFS:
    from chunker import load_pdf_files
    chunks += load_pdf_files(list(Path("data").glob("**/*.pdf")), chunk_size=500, chunk_overlap=100)
if USE_WEBSITES:
    chunks += load_websites(WEBSITES_FILE, chunk_size=500, chunk_overlap=100)
if not chunks:
    raise RuntimeError("Nothing to read: put links in data/websites.txt")

embedder = LiteEmbedder()
rag_retriever = LiteRetriever(chunks, embedder)
answer_cache = AnswerCache(embedder, same_question)
del chunks  # the retriever keeps its own copy of the text

print("\nSetup done (lite). Ready for questions.\n")


# ======================================================================
# KEEPING UP WITH THE WEBSITE
# Every READ_HOURS all pages are downloaded again. If their text changed, the search
# index is rebuilt (only new text is embedded), swapped in, and the saved answers are
# cleared. Visitors keep getting answers the whole time. To update sooner, open the
# /refresh link (api.py).
# ======================================================================

READ_HOURS = 6
refresh_lock = threading.Lock()  # one re-read at a time (timer and /refresh link)


def _text_fingerprint(texts) -> str:
    return hashlib.sha1("\n\x00".join(texts).encode()).hexdigest()


def refresh_from_website(reason: str) -> str:
    """Reads every page again; rebuilds the search if the text changed. Returns what happened."""
    global rag_retriever
    if not USE_WEBSITES:
        return "websites are switched off"
    if not refresh_lock.acquire(blocking=False):
        return "already reading the website"
    try:
        start = time.perf_counter()
        print(f"Website re-read ({reason}) ...")
        new_chunks = load_websites(WEBSITES_FILE, chunk_size=500, chunk_overlap=100, fresh=True)
        if USE_PDFS:
            new_chunks = load_pdf_files(list(Path("data").glob("**/*.pdf")), chunk_size=500,
                                        chunk_overlap=100) + new_chunks
        if not new_chunks:
            return "nothing could be read, kept the current version"
        if _text_fingerprint(c.page_content for c in new_chunks) == _text_fingerprint(rag_retriever.texts):
            result = "no change"
        else:
            new_retriever = LiteRetriever(new_chunks, embedder, old=rag_retriever)
            rag_retriever = new_retriever  # one step: questions use either the old or the new index
            answer_cache.clear()
            result = f"website changed: search rebuilt ({len(new_retriever.texts)} chunks)"
        return result + f", took {time.perf_counter() - start:.1f}s"
    finally:
        refresh_lock.release()


def _watch_website():
    while True:
        time.sleep(READ_HOURS * 3600)
        try:
            print(refresh_from_website(f"every {READ_HOURS} hours"))
        except Exception as error:  # never let the watcher die
            print(f"Website watcher error: {error}")


def start_website_watcher():
    if USE_WEBSITES:
        threading.Thread(target=_watch_website, daemon=True, name="website-watcher").start()
        print(f"Website watcher on: reads the website again every {READ_HOURS} hours")


# ======================================================================
# PER-QUESTION WORK (same as pipeline.py)
# ======================================================================

MAX_HISTORY = 3
MAX_CHUNKS = 7
PART_CHUNKS = 2
PART_RERANK = 6


def remember(session: dict, question: str, answer: str):
    if not answer:
        return
    history = session.get("history") or []
    history.append((question, answer))
    session["history"] = history[-MAX_HISTORY:]


def take_turns(results: list, limit: int) -> list:
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
    company = prep.get("company")
    query = prep.get("query") or prep["search"]
    if prep.get("redo"):
        extra_words = ""
    else:
        extra_words = prep["search"] if query != prep["search"] else question
    results = [rag_retriever.retrieve(query, company=company, extra_words=extra_words)]
    top = results[0][0]["metadata"].get("source_file", "").lower() if results[0] else ""
    if company and top and not top.startswith(company):
        company = None
    for part in prep.get("parts", []):
        results.append(rag_retriever.retrieve(part, top_k=PART_CHUNKS, candidates=10,
                                              company=company, rerank_limit=PART_RERANK))
    retrieved_chunks = take_turns(results, MAX_CHUNKS) if len(results) > 1 else results[0]
    print("Chunks used: " + ", ".join(f"{c['rerank_score']:.2f}" for c in retrieved_chunks))
    return build_context(retrieved_chunks) + "\n\n" + contact_details(company)


ERROR_MARKERS = ("I could not reach the local model", "Gemini is busy or not responding",
                 "I could not reach the Groq API",
                 "I'm getting a lot of questions right now")  # the Groq busy message
NOT_FOUND_MARKERS = ("could not find", "couldn't find", "couldn’t find")


def is_error(answer: str) -> bool:
    return any(marker in answer for marker in ERROR_MARKERS)


def is_not_found(answer: str) -> bool:
    return any(marker in answer.lower() for marker in NOT_FOUND_MARKERS)


async def answer_pieces(question: str, session: dict):
    """Same as answer_pieces in pipeline.py: yields the answer piece by piece,
    then saves it to the session memory and the answer cache."""
    start = time.perf_counter()
    history = session.get("history") or []
    last_prep = session.get("last_prep")

    prep = await asyncio.to_thread(prepare_question, question, history, last_prep)
    language = prep["language"]
    search_question = prep.get("query") or prep["search"]
    session["last_prep"] = prep

    use_cache = not prep["redo"] and bool(language)
    cached = await asyncio.to_thread(answer_cache.lookup, search_question, prep.get("company")) if use_cache else None

    if cached and language in cached["answers"]:
        print(f"Cache hit ({language}): instant answer")
        pieces = iter([cached["answers"][language]])
    elif cached:
        source_language = "english, latin" if "english, latin" in cached["answers"] else next(iter(cached["answers"]))
        pieces = checked_translation(cached["answers"][source_language], language)
    else:
        pieces = None

    already_saved = cached is not None and language in cached["answers"]
    answer = ""

    while True:
        if pieces is None:
            context = await asyncio.to_thread(retrieve_context, prep, question)
            if language:
                context += f"\n\nReply language: {describe_language(language)}"
            pieces = stream_answer(question, context, history)
        try:
            piece = await asyncio.to_thread(next, pieces, None)
        except TranslationFailed:
            pieces = None
            continue
        if piece is None:
            break
        answer += piece
        yield piece

    print(f"took {time.perf_counter() - start:.2f}s")
    remember(session, question, answer)

    if use_cache and answer and not is_error(answer) and not is_not_found(answer) and not already_saved:
        await asyncio.to_thread(answer_cache.add, search_question, language, answer, cached,
                                prep.get("company"))
