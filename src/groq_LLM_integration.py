"""
Groq version of ollama_LLM_integration.py: same functions, same inputs and outputs,
but every model call goes to the Groq API instead of the local Ollama app.

Switch between them in pipeline.py (the import lines at the top).
The prompts, the contact details and the small helper functions are imported
from ollama_LLM_integration.py, so both versions always behave the same way.

Needs GROQ_API_KEY in the .env file (free key: https://console.groq.com/keys).
"""
import os
import time

from dotenv import load_dotenv
from groq import Groq

from ollama_LLM_integration import (
    # prompts
    SYSTEM_PROMPT, TRANSLATE_PROMPT, SAME_QUESTION_PROMPT, TRANSLATE_ANSWER_PROMPT, SAME_TOPIC,
    # helpers that do not call a model (re-exported for pipeline.py)
    build_context, contact_details, describe_language, TranslationFailed,
    format_history, words, detect_script, parse_prepared, first_line,
    HISTORY_TURNS, HISTORY_ANSWER_CHARS,
)

load_dotenv(dotenv_path=".env")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

# The bigger gpt-oss 120b model, for better answers.
# To go back to the same model as the local Ollama one (gpt-oss 20b), swap the # between these 2 lines.
# MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Give up after 30s and retry twice, so a busy server shows an error instead of hanging
client = Groq(api_key=api_key, timeout=30, max_retries=2)

# Pipeline.py checks for this text, so error replies are never saved in the answer cache
ERROR_TEXT = "I could not reach the Groq API"


def _error(error) -> str:
    return f"{ERROR_TEXT}. Check your internet and GROQ_API_KEY. Details: {error}"


def _chat(messages, effort="medium", max_tokens=2048, stream=False):
    """One Groq call. effort: how long the model thinks before answering
    ('low' for the quick helper steps, like think='low' in the Ollama version)."""
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort=effort,
        include_reasoning=False,  # only the answer text comes back, not the model's thinking
        max_completion_tokens=max_tokens,
        stream=stream,
    )


def _stream(messages, label: str, effort="medium"):
    """Yields the reply piece by piece as Groq writes it."""
    start = time.perf_counter()
    got_any = False
    try:
        for chunk in _chat(messages, effort=effort, stream=True):
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                if not got_any:
                    print(f"{label} first words after {time.perf_counter() - start:.2f}s")
                got_any = True
                yield piece
    except Exception as error:
        yield _error(error)


# ======================================================================
# THE ANSWER: context + question -> Groq -> answer
# ======================================================================

def stream_answer(question: str, context: str, history=None):
    """Yields the answer piece by piece as Groq writes it.
    history: recent (question, answer) pairs, so follow-ups make sense."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past_q, past_a in (history or [])[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": past_q})
        messages.append({"role": "assistant", "content": past_a})
    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "(If this question asks for a specific language, reply in that language. "
            "Otherwise reply in the same language and script as this question, "
            "even if earlier messages were in a different language. "
            "If it asks to repeat, translate or reformat an earlier answer, do that "
            "using the context above, and do not add facts that are not in the context.)"
        ),
    })
    yield from _stream(messages, "Groq")


def generate_answer(question: str, context: str) -> str:
    """The whole answer at once (no streaming)."""
    start = time.perf_counter()
    try:
        response = _chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ])
    except Exception as error:
        return _error(error)
    print(f"Groq took {time.perf_counter() - start:.2f}s")
    return response.choices[0].message.content or ""


# ======================================================================
# HELPER STEPS (same as the Ollama version, but calling Groq)
# ======================================================================

def same_question(a: str, b: str) -> bool:
    """Quick yes/no check used by the answer cache before reusing an answer."""
    try:
        response = _chat([
            {"role": "system", "content": SAME_QUESTION_PROMPT},
            {"role": "user", "content": f"Question 1: {a}\nQuestion 2: {b}"},
        ], effort="low", max_tokens=200)
        return (response.choices[0].message.content or "").strip().upper().startswith("YES")
    except Exception as error:
        print(f"Same-question check failed, not using the cache: {error}")
        return False


def prepare_question(question: str, history=None, last=None) -> dict:
    """Works out the English search question, reply language, company and topics.
    Exactly the same as prepare_question in ollama_LLM_integration.py (see there)."""
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
        response = _chat([
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user", "content": user_text},
        ], effort="low", max_tokens=500)
        reply = (response.choices[0].message.content or "").strip()
    except Exception as error:
        print(f"Translation failed, searching with the original question: {error}")
        return result

    parsed = parse_prepared(reply)
    language, english = parsed["language"], parsed["question"]
    name = (language or "").split(",")[0].strip()
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
        return result
    if english:
        result["search"] = english
    result["company"], result["parts"] = parsed["company"], parsed["parts"]
    if name == "english" and not history:
        result["query"] = question
    print(f"Search question: '{result['search']}' | company: {result['company'] or 'both'} "
          f"| reply language: {result['language']} ({time.perf_counter() - start:.2f}s)")
    if result["parts"]:
        print(f"Search topics: {result['parts']}")
    return result


def stream_translation(answer: str, language: str):
    """Yields a cached answer translated into another language, piece by piece."""
    target = describe_language(language)
    yield from _stream([
        {"role": "system", "content": TRANSLATE_ANSWER_PROMPT.format(target=target)},
        {"role": "user", "content": answer},
    ], "Translation", effort="low")


def checked_translation(answer: str, language: str):
    """Like stream_translation, but raises TranslationFailed before showing anything
    if the first line comes back untranslated (see the Ollama version)."""
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
            if "\n" in buffer.lstrip():
                if words(first_line(buffer)) == source_first:
                    raise TranslationFailed()
                checked = True
                yield buffer
        if not checked:
            if words(buffer) == words(answer):
                raise TranslationFailed()
            yield buffer
    finally:
        pieces.close()
