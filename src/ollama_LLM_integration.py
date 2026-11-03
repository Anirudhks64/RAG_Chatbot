import re
import time
import unicodedata
import ollama
from typing import List, Dict, Any

# Extracts information from Langchain Document Data Type to a String
def build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"""
            --- Retrieved Chunk {i} ---

            Content:
            {doc['content']}

            Metadata:
            {doc['metadata']}

            Similarity Score:
            {doc['similarity_score']}

            """
        )

    return "\n".join(context_parts)


# The model you downloaded with: ollama pull gpt-oss:20b
MODEL = "gpt-oss:20b"

# Context window for EVERY Ollama call. Keep it the same everywhere,
# otherwise Ollama reloads the model each time it changes.
NUM_CTX = 8192
 

SYSTEM_PROMPT = (

    "You are a friendly assistant for a business. "
    "You answer questions using ONLY the context provided by the user. "
    "Keep answers short and clear.\n\n"

    "Rules:\n"
    "1. If the user greets you or makes small talk (hi, hello, wassup, how are you), "
    "reply in a warm, short, natural way and ask how you can help with their questions about the business. "
    "Do not use the context for this. Do not print the contact details for greetings.\n"
    "2. If the answer is not in the context, say you could not find it in the documents, "
    "3. If you can only answer part of the question, or you are not fully sure, "
    "give what you do know, say the answer may be incomplete, "
    "4. If you cannot answer in any way, for any reason, ALWAYS print the contact details below.\n"
    "5. When in doubt, print the contact details. It is better to show them too often than too rarely.\n"
    "6. If the user asks for a specific language (for example 'in English', 'Kannada mein', 'in Arabic please'), "
    "reply in that language. Otherwise, always reply in the same language AND the same script the user wrote in. "
    "If the question is in Arabic, answer fully in Arabic. "
    "If the user writes Hindi in English letters (for example 'kya haal hain bhai' or "
    "'booking ke liye kaunse documents chahiye'), answer in Hindi written in English letters, "
    "never in Devanagari script. "
    "Translate the fixed sentences from the rules above into the user's language and script as well. "
    "Keep names, numbers and contact details unchanged.\n\n"
    "Formatting: write in plain text and simple Markdown only. "
    "Never use LaTeX or math formatting (no square brackets or parentheses around formulas, no backslashes). "
    "Show calculations on normal lines, for example: 2 x 6,499 = 12,998.\n"
    "Tables: answer with a Markdown table whenever the answer has rows of related details, for example:\n"
    "- the answer comes from a table in the context (lines like '| A | B |'), for any topic: "
    "prices, charges, fees, plans, packages, projects, schedules, contacts, refund methods, visas, and so on. "
    "Keep the same columns and include every row that answers the question.\n"
    "- the user asks to compare two or more things, or asks for a list of options where each one has "
    "details like a price, time, duration or condition, even if the context is not a table.\n"
    "Do not use a table for greetings, yes/no answers, or a single fact (one number, one date, one step). "
    "Table format: header row first, then a '| --- | --- |' line, then one row per line. "
    "Never turn table rows into bullet points. Translate the table text into the user's language if needed. "
    "You may add one short sentence before or after the table.\n\n"



    "Contact details: Azuria - 800 298743 or care@azuriaproperties.ae OR Wayfarra - 800 929 3277 or care@wayfarraholidays.ae"
)

 
 
HISTORY_TURNS = 3          # how many past question/answer pairs to remember
HISTORY_ANSWER_CHARS = 600  # past answers are shortened to this length for the rewrite step

SAME_TOPIC = "SAME_TOPIC"

TRANSLATE_PROMPT = (
    "You prepare the user's latest message for a document search.\n"
    "Reply in exactly this format, two lines, nothing else:\n"
    "LANGUAGE: <language>, <script>\n"
    "QUESTION: <question>\n\n"
    "LANGUAGE is the language the reply must be written in. If the message asks for a specific "
    "language ('in English', 'Kannada mein', 'in Arabic please'), use that one. Otherwise use the "
    "language and script the message is written in. Always write it like these examples: "
    "'English, Latin' / 'Hindi, Latin' (Hindi written in English letters) / 'Hindi, Devanagari' / "
    "'Kannada, Kannada' / 'Arabic, Arabic'.\n\n"
    "QUESTION:\n"
    "- FIRST check: does the latest message only ask to redo the previous answer, "
    "for example translate it, repeat it, shorten it, expand it, explain it or show it as a table "
    "('write the same output in English', 'now in Kannada', 'say that again in Hindi', "
    f"'make it shorter', 'show it as a table')? If yes, write exactly {SAME_TOPIC}.\n"
    "- Otherwise, rewrite the latest message as ONE standalone question in English. "
    "The message may be in any language, including Hindi written in English letters "
    "(for example 'mein kutta rakh sakta hoon kya?' means 'Can I keep a dog?').\n"
    "- If it is a follow-up (uses words like 'it', 'that', 'this one', 'and for kids?', 'uska price'), "
    "use the earlier conversation to fill in what it refers to, "
    "for example 'What is the cancellation policy for it?' becomes "
    "'What is the cancellation policy for the Maldives Island Escape package?'.\n"
    "- If it is a new topic, do NOT add anything from the earlier conversation.\n"
    "- If it is already a clear English question, keep it unchanged.\n"
    "- The QUESTION line must ALWAYS be in English, whatever language the message is in.\n\n"
    "Examples:\n"
    "Message: mein kutta rakh saktha hoon kya?\n"
    "LANGUAGE: Hindi, Latin\n"
    "QUESTION: Can I keep a dog?\n\n"
    "Message: holiday package cancel karne ka charge kitna hai?\n"
    "LANGUAGE: Hindi, Latin\n"
    "QUESTION: What are the cancellation charges for a holiday package?\n\n"
    "Message: आप कौन से हॉलिडे पैकेज देते हैं?\n"
    "LANGUAGE: Hindi, Devanagari\n"
    "QUESTION: Which holiday packages do you offer?\n\n"
    "Message: ನಾನು ನಾಯಿಯನ್ನು ಸಾಕಬಹುದೇ?\n"
    "LANGUAGE: Kannada, Kannada\n"
    "QUESTION: Can I keep a dog?\n\n"
    "Message: How much are the service charges?\n"
    "LANGUAGE: English, Latin\n"
    "QUESTION: How much are the service charges?"
)


SAME_QUESTION_PROMPT = (
    "Do these two questions ask for exactly the same information, so that the same answer "
    "fully answers both? Questions about different things (for example a dog vs a cat, "
    "hotels vs holiday packages, one community vs all communities, one package vs another) "
    "are NOT the same. Reply with only YES or NO."
)


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


def format_history(history, max_answer_chars=None) -> str:
    lines = []
    for past_q, past_a in history:
        if max_answer_chars and len(past_a) > max_answer_chars:
            past_a = past_a[:max_answer_chars] + " ..."
        lines.append(f"User: {past_q}\nAssistant: {past_a}")
    return "\n\n".join(lines)


def words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def detect_script(text: str) -> str:
    """Which alphabet the message is written in ('latin', 'devanagari', 'kannada', 'arabic', ...).
    Read from the characters themselves, so it is always right, unlike asking the model."""
    counts = {}
    for ch in text:
        if ch.isalpha():
            try:
                script = unicodedata.name(ch).split()[0].lower()
            except ValueError:
                continue
            counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else "latin"


def parse_prepared(text: str):
    """Reads the 'LANGUAGE: ... / QUESTION: ...' reply. Returns (language, question)."""
    language, question = None, None
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("LANGUAGE:"):
            language = line.split(":", 1)[1].strip().strip("'\"").lower() or None
        elif line.upper().startswith("QUESTION:"):
            question = line.split(":", 1)[1].strip().strip("'\"") or None
    if question is None and language is None:
        question = text.strip() or None  # model ignored the format: use the whole reply
    return language, question


def prepare_question(question: str, history=None, last_search=None) -> dict:
    """One fast Ollama call that works out, for the user's message:
      search:   a standalone English question for searching the English PDFs
                (follow-ups are filled in from the recent conversation)
      language: the language the reply should be in, e.g. 'hindi, latin'
      redo:     True if the user only asks to redo the previous answer
                (translate it, shorten it, ...); then the previous search is reused
    The answer itself is still written from the ORIGINAL message."""
    start = time.perf_counter()
    result = {"search": question, "language": None, "redo": False}
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
            options={"num_ctx": NUM_CTX, "num_predict": 300},
        )
        reply = response["message"]["content"].strip()
    except Exception as error:
        print(f"Translation failed, searching with the original question: {error}")
        return result

    language, english = parse_prepared(reply)
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
        if history and last_search:
            print(f"Redo of the previous answer, reusing search: '{last_search}'")
            result["search"] = last_search
            result["redo"] = True
        return result  # nothing to redo yet: search with the message itself
    if english:
        result["search"] = english
    print(f"Search question: '{result['search']}' | reply language: {result['language']} "
          f"({time.perf_counter() - start:.2f}s)")
    return result


def translate_for_search(question: str, history=None, last_search=None) -> str:
    """Only the English search question (see prepare_question)."""
    return prepare_question(question, history, last_search)["search"]


TRANSLATE_ANSWER_PROMPT = (
    "You are a translator. Translate the text the user sends into {target}.\n"
    "Rules:\n"
    "- Translate ALL the words, including the table headers and the text inside table cells.\n"
    "- Keep the same structure: Markdown tables stay Markdown tables with the same rows and "
    "columns, bullet points stay bullet points, bold stays bold.\n"
    "- Keep numbers, prices, dates, percentages, phone numbers and emails exactly as they are.\n"
    "- NEVER translate names. Copy every company, community, project, package, plan and "
    "place name exactly as written, in English letters, even inside text in another alphabet. "
    "For example 'Azuria Creek', 'Pearl Residences', 'Maldives Island Escape', "
    "'Baku Getaway, Azerbaijan', 'TravelSafe Plus', 'Wayfarra', 'WayCash' and 'Dubai (DXB)' stay unchanged.\n"
    "- Translate every sentence, including any sentence before or after a table. Do not drop any.\n"
    "- Do NOT add any heading, label, note or explanation (never write things like "
    "'Hindi (Latin script):'). Do not add or remove information.\n"
    "- Reply with only the translated text."
)


def describe_language(language: str) -> str:
    """'hindi, latin' -> a plain description the model understands."""
    name, _, script = (part.strip() for part in language.partition(","))
    name = name.capitalize()
    if script == "latin" and name.lower() != "english":
        return (f"{name} written in English letters (romanized {name}, the way people type it "
                f"in chat messages). Use {name} words, not English words, "
                f"and never use the {name} alphabet")
    if script and script != "latin":
        return f"{name}, written in the {script.capitalize()} alphabet"
    return name


class TranslationFailed(Exception):
    """The model returned the text untranslated (happens sometimes with tables)."""


def first_line(text: str) -> str:
    return next((line for line in text.splitlines() if line.strip()), "")


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
 