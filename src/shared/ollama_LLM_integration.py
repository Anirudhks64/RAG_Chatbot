import re
import time
import unicodedata
import ollama
from typing import List, Dict, Any


# The prompts, company info and text helpers live in prompts.py (shared with the Groq version).
# They are imported here, so everything that imported them from this file still works.
from prompts import (  # noqa: F401
    build_context, SYSTEM_PROMPT, COMPANIES, ABOUT,
    contact_details, company_key, HISTORY_TURNS, HISTORY_ANSWER_CHARS,
    SAME_TOPIC, TRANSLATE_PROMPT, SAME_QUESTION_PROMPT, format_history,
    words, detect_script, MAX_PARTS, LABELED_LINE,
    parse_prepared, TRANSLATE_ANSWER_PROMPT, describe_language, TranslationFailed,
    first_line,
)


# The model you downloaded with: ollama pull gpt-oss:20b
MODEL = "gpt-oss:20b"

# Context window for EVERY Ollama call. Keep it the same everywhere,
# otherwise Ollama reloads the model each time it changes.
NUM_CTX = 8192
 

def same_question(a: str, b: str) -> bool:
    """Quick yes/no check used by the answer cache before reusing an answer."""
    try:
        response = ollama.chat(
            model=MODEL,
            think="low",
            messages=[
                {"role": "system", "content": SAME_QUESTION_PROMPT},
                {"role": "user", "content": f"Question 1: {a}\nQuestion 2: {b}"},
            ],
            options={"num_ctx": NUM_CTX, "num_predict": 200},
        )
        return response["message"]["content"].strip().upper().startswith("YES")
    except Exception as error:
        print(f"Same-question check failed, not using the cache: {error}")
        return False


def prepare_question(question: str, history=None, last=None) -> dict:
    """One fast Ollama call that works out, for the user's message:
      search:   a standalone English question for searching the English PDFs
                (follow-ups are filled in from the recent conversation)
      query:    (only for the first message of a chat in English) the message itself,
                searched instead of the rewrite
      language: the language the reply should be in, e.g. 'hindi, latin'
      company:  which business it is about ('azuria', 'wayfarra'), None if both or unsure
      parts:    short search phrases, one per topic, when the message asks several things, else []
      redo:     True if the user only asks to redo the previous answer
                (translate it, shorten it, ...); then the previous search is reused
    last: what this returned for the previous message (needed for redo).
    The answer itself is still written from the ORIGINAL message."""
    start = time.perf_counter()
    result = {"search": question, "language": None, "redo": False, "company": None, "parts": []}
    history = (history or [])[-HISTORY_TURNS:]
    if history:
        user_text = (
            f"Earlier conversation:\n{format_history(history, HISTORY_ANSWER_CHARS)}\n\n"
            f"Latest question: {question}"
        )
    else:
        user_text = question
    try:
        response = ollama.chat(
            model=MODEL,
            think="low",  # short reasoning, so this step stays fast
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT},
                {"role": "user", "content": user_text},
            ],
            # num_ctx MUST match the answer step: if it differs, Ollama
            # reloads the whole model on every switch (about 4s each time)
            # 500: room for the longer reply (a limit only, it never waits for it)
            options={"num_ctx": NUM_CTX, "num_predict": 500},
        )
        reply = response["message"]["content"].strip()
    except Exception as error:
        print(f"Translation failed, searching with the original question: {error}")
        return result

    parsed = parse_prepared(reply)
    language, english = parsed["language"], parsed["question"]
    # The model names the language; the alphabet comes from the message itself.
    # (English is always Latin, e.g. when the user asks "in English" in another script.)
    name = (language or "").split(",")[0].strip()
    # The model sometimes labels English as Hinglish. Reliable check: if the
    # English version is the same as the message, the message was English.
    if english and name and words(english) == words(question):
        name = "english"
    if name:
        script = "latin" if name == "english" else detect_script(question)
        result["language"] = f"{name}, {script}"

    if SAME_TOPIC in reply:
        if history and last:
            last = last if isinstance(last, dict) else {"search": last}
            print(f"Redo of the previous answer, reusing search: '{last['search']}'")
            result["search"] = last["search"]
            result["query"] = last.get("query", last["search"])
            result["company"] = last.get("company")
            result["parts"] = last.get("parts", [])
            result["redo"] = True
        return result  # nothing to redo yet: search with the message itself
    if english:
        result["search"] = english
    result["company"], result["parts"] = parsed["company"], parsed["parts"]
    # The first message of a chat, in English, needs no translating or filling in:
    # search with exactly what the user typed, so the rewrite cannot drop a detail.
    # (The rewrite is still used for the keyword search and the answer cache.)
    if name == "english" and not history:
        result["query"] = question
    print(f"Search question: '{result['search']}' | company: {result['company'] or 'both'} "
          f"| reply language: {result['language']} ({time.perf_counter() - start:.2f}s)")
    if result["parts"]:
        print(f"Search topics: {result['parts']}")
    return result


def translate_for_search(question: str, history=None, last=None) -> str:
    """Only the English search question (see prepare_question)."""
    return prepare_question(question, history, last)["search"]


def checked_translation(answer: str, language: str):
    """Like stream_translation, but holds back the first line and checks it.
    If it is identical to the original, the translation failed: raises
    TranslationFailed BEFORE anything is shown, so the caller can fall back
    to a normal answer."""
    source_first = words(first_line(answer))
    pieces = stream_translation(answer, language)
    buffer = ""
    checked = False
    try:
        for piece in pieces:
            if checked:
                yield piece
                continue
            buffer += piece
            # Check once the first line is complete
            if "\n" in buffer.lstrip():
                if words(first_line(buffer)) == source_first:
                    raise TranslationFailed()
                checked = True
                yield buffer
        if not checked:  # one-line answer: check all of it
            if words(buffer) == words(answer):
                raise TranslationFailed()
            yield buffer
    finally:
        pieces.close()  # stop the model if we gave up early


def stream_translation(answer: str, language: str):
    """Yields a cached answer translated into another language, piece by piece.
    No document search: the facts come from the saved answer."""
    start = time.perf_counter()
    got_any = False
    target = describe_language(language)
    try:
        for chunk in ollama.chat(
            model=MODEL,
            think="low",  # translating needs little reasoning, so it starts faster
            messages=[
                {"role": "system", "content": TRANSLATE_ANSWER_PROMPT.format(target=target)},
                {"role": "user", "content": answer},
            ],
            options={"num_ctx": NUM_CTX, "num_predict": 2048},
            stream=True,
        ):
            piece = chunk["message"]["content"]
            if piece:
                if not got_any:
                    print(f"Translation first words after {time.perf_counter() - start:.2f}s")
                got_any = True
                yield piece
    except Exception as error:
        yield (
            "I could not reach the local model. Is the Ollama app open, "
            f"and did you run 'ollama pull {MODEL}'? Details: {error}"
        )


def stream_answer(question: str, context: str, history=None):
    """Yields the answer piece by piece as the local model writes it.
    history: recent (question, answer) pairs, so follow-ups make sense."""
    start = time.perf_counter()
    got_any = False

    # Past turns go in as normal chat messages (without their old context),
    # then the new question with the freshly retrieved context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past_q, past_a in (history or [])[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": past_q})
        messages.append({"role": "assistant", "content": past_a})
    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            # Earlier turns may be in another language; the reply must follow THIS question
            "(If this question asks for a specific language, reply in that language. "
            "Otherwise reply in the same language and script as this question, "
            "even if earlier messages were in a different language. "
            "If it asks to repeat, translate or reformat an earlier answer, do that "
            "using the context above, and do not add facts that are not in the context.)"
        ),
    })

    try:
        for chunk in ollama.chat(
            model=MODEL,
            messages=messages,
            options={"num_ctx": NUM_CTX, "num_predict": 2048},
            stream=True,
        ):
            piece = chunk["message"]["content"]
            if piece:
                if not got_any:
                    print(f"Ollama first words after {time.perf_counter() - start:.2f}s")
                got_any = True
                yield piece
    except Exception as error:
        yield (
            "I could not reach the local model. Is the Ollama app open, "
            f"and did you run 'ollama pull {MODEL}'? Details: {error}"
        )


def generate_answer(question: str, context: str) -> str:
    """Same name and inputs as the Gemini version, but runs on your own Mac."""
    start = time.perf_counter()
 
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        options={
        "num_ctx": NUM_CTX,   # context window: how much text the model can hold at once
        "num_predict": 2048,  # max length of the answer
    },
        )
    except Exception as error:
        return (
            "I could not reach the local model. Is the Ollama app open, "
            f"and did you run 'ollama pull {MODEL}'? Details: {error}"
        )
 
    answer = response["message"]["content"]
    print(f"Ollama took {time.perf_counter() - start:.2f}s")
    return answer
 