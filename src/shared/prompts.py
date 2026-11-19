"""
The chatbot's rules and shared text helpers, used by every model integration
(groq_LLM_integration.py, ollama_LLM_integration.py). To change how the chatbot answers,
edit this file: the prompts, the company list, what each business does, and the
small helpers that read the model's replies.

Moved here unchanged from ollama_LLM_integration.py, so the Groq version no longer
has to import (and install) the Ollama library just to use these.
"""
import re
import unicodedata
from typing import List, Dict, Any


# Turns the retrieved chunks into the context text the model reads.
# Only the chunk text (it already starts with the PDF name): the metadata and
# score lines are not used by the model and made the context ~40% longer (slower).
def build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"--- Chunk {i} ---\n{doc['content']}"
        for i, doc in enumerate(retrieved_docs, start=1)
    )


SYSTEM_PROMPT = (

    "You are a friendly assistant for a business. "
    "You answer questions using ONLY the context provided by the user. "
    "Keep answers short and clear.\n\n"

    "Rules:\n"
    "1. If the user greets you or makes small talk (hi, hello, wassup, how are you), "
    "reply in a warm, short, natural way and ask how you can help with their questions about the business. "
    "Do not use the context for this. Do not print the contact details for greetings.\n"
    # OLD rules 2-5: they made the model print the contact details under almost every answer
    # "2. If the answer is not in the context, say you could not find it in the documents, "
    # "3. If you can only answer part of the question, or you are not fully sure, "
    # "give what you do know, say the answer may be incomplete, "
    # "4. If you cannot answer in any way, for any reason, ALWAYS print the contact details "
    # "(they are given at the end of the context).\n"
    # "5. When in doubt, print the contact details. It is better to show them too often than too rarely.\n"
    # OLD rule 2: for "are you a catering service?" it said "could not find", as if they might be
    # "2. If the answer is not in the context, say you could not find it in the documents "
    # "and then print the contact details.\n"
    # OLD rule 2 start (used the 'About the business' summary, now removed):
    # "2. If the answer is not in the context: only if the user asks whether the business offers "
    # "something from a COMPLETELY different industry (see 'About the business' at the end of the "
    # "context; for an AI company: food, catering, selling physical products, building houses, travel), "
    # "say clearly that it does not offer that and say in one sentence what it does instead. "
    # "'About the business' is NOT a complete list: something missing from it may still be offered. "
    "2. If the answer is not in the context: only if the user asks whether the business offers "
    "something from a COMPLETELY different industry than the one the context shows "
    "(for an AI company: food, catering, selling physical products, building houses, travel), "
    "say clearly that it does not offer that and say in one sentence what it does instead, "
    "based on the context. The context does not show everything: something missing from it may still be offered. "
    "So never say 'no' to anything in or near the business's own field (for an AI company: any AI, data, "
    "software, app or web development, IT, automation, chatbot, analytics, cybersecurity, training or "
    "hiring service). For those, for details like a price, salary or date, and whenever you are not sure, "
    "say you could not find it in the documents. "
    "In all these cases then print the contact details.\n"
    "3. If the question is not about the business at all (random topics like the weather, sports, "
    "general knowledge, coding help), say briefly that you can only help with questions about the business "
    "and then print the contact details.\n"
    "4. If you can only answer part of the question, give what you do know, "
    "say which part you could not find, and then print the contact details.\n"
    "5. In every other case, do NOT print the contact details. When the context answers the question, "
    "just give the answer: no contact details, no 'feel free to contact us' line at the end. "
    "Only print them if the user asks how to contact the business.\n"
    "6. If the user asks for a specific language (for example 'in English', 'Kannada mein', 'in Arabic please'), "
    "reply in that language. Otherwise, always reply in the same language AND the same script the user wrote in. "
    "If the question is in Arabic, answer fully in Arabic. "
    "If the user writes Hindi in English letters (for example 'kya haal hain bhai' or "
    "'booking ke liye kaunse documents chahiye'), answer in Hindi written in English letters, "
    "never in Devanagari script. "
    "Translate the fixed sentences from the rules above into the user's language and script as well. "
    "Keep names, numbers and contact details unchanged.\n"
    # Kept short on purpose: a longer "check every condition step by step" rule
    # made the model think ~3-6s longer per answer for no better answers (tested)
    "7. If the question assumes something the context contradicts, say so first, briefly.\n"
    "8. Answer every part of the question, using only facts about the business it is about.\n\n"
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

    "Contact details: use only the ones given at the end of the context."
)


# The businesses in the PDFs and websites. Each key must be how that PDF's file name
# starts ("Azuria Properties.pdf" -> "azuria"), or the company word in data/websites.txt;
# the search uses it to read only that business's documents.
# A new business also needs a line in the COMPANY part of TRANSLATE_PROMPT below.
COMPANIES = {
    # TRANSACCELORS ONLY for now: remove the # from these 2 lines to bring them back
    # "azuria": "Azuria - 800 298743 or care@azuriaproperties.ae",
    # "wayfarra": "Wayfarra - 800 929 3277 or care@wayfarraholidays.ae",
    # The website lists no phone number, only this email and the contact page
    "transaccelors": "TransAccelors - teamtransaccelors@transaccelors.com or https://transaccelors.com/contact-us",
}


# What each business does, in one line. Added to every question's context, so the model
# can say "no, we don't do that" to questions about things the business clearly doesn't do
# (the search finds nothing for those, so without this it could only say "could not find").
ABOUT = {
    # TRANSACCELORS ONLY for now: remove the # from these 2 lines to bring them back
    # "azuria": "Azuria Properties is a Dubai property developer (apartments, townhouses and villas, "
    #           "off-plan sales, payment plans, handover, service charges, resale).",
    # "wayfarra": "Wayfarra Holidays is a travel agency (flights, hotels, holiday packages, visas, "
    #             "travel insurance, WayCash, refunds).",
    # REMOVED: the bot must learn what TransAccelors does from the website only, not from this summary
    # "transaccelors": "TransAccelors is an AI, data and technology company in Bengaluru and Dubai: "
    #                  "AI and data consulting, custom AI solutions built for a business's problems "
    #                  "(for example AI agents / chatbots that answer customer questions and complete tasks, "
    #                  "and automation), the FinAccelors AI financial analysis product, "
    #                  "the AI Immersion Lab, AI labs for schools and colleges, hiring solutions "
    #                  "(AI/ML talent), careers, and a blog about AI. "
    #                  "This list is not complete.",
}


def contact_details(company=None) -> str:
    """What the business does + its contact line, given to the model at the end of the context.
    Only that business's, or all of them if the business is not known."""
    keys = [company] if company in COMPANIES else list(COMPANIES)
    about = " ".join(ABOUT[k] for k in keys if k in ABOUT)
    lines = [COMPANIES[k] for k in keys]
    # return "Contact details: " + " OR ".join(lines)
    # The label says when to use them, so the model does not add them to every answer
    return ((f"About the business: {about}\n\n" if about else "")
            + "Contact details (ONLY for rules 2, 3 and 4, or if the user asks how to contact us): "
            + " OR ".join(lines))


def company_key(text: str):
    """'Azuria' / 'Azura' -> 'azuria', 'Wayfarra' -> 'wayfarra', 'Both' or unclear -> None."""
    found = [key for key in COMPANIES if key[:4] in text.lower()]
    return found[0] if len(found) == 1 else None

 
 
HISTORY_TURNS = 3          # how many past question/answer pairs to remember
HISTORY_ANSWER_CHARS = 600  # past answers are shortened to this length for the rewrite step

SAME_TOPIC = "SAME_TOPIC"

TRANSLATE_PROMPT = (
    "You prepare the user's latest message for a document search.\n"
    "Reply in exactly this format, nothing else:\n"
    "LANGUAGE: <language>, <script>\n"
    # TRANSACCELORS ONLY for now (old line below)
    # "COMPANY: <Azuria, Wayfarra, TransAccelors or Both>\n"
    "COMPANY: <TransAccelors or Both>\n"
    "QUESTION: <question>\n"
    "TOPICS: <topic> | <topic> | <topic>\n\n"
    "LANGUAGE is the language the reply must be written in. If the message asks for a specific "
    "language ('in English', 'Kannada mein', 'in Arabic please'), use that one. Otherwise use the "
    "language and script the message is written in. Always write it like these examples: "
    "'English, Latin' / 'Hindi, Latin' (Hindi written in English letters) / 'Hindi, Devanagari' / "
    "'Kannada, Kannada' / 'Arabic, Arabic'.\n\n"
    # TRANSACCELORS ONLY for now: the Azuria and Wayfarra lines are commented out
    # "COMPANY is the business the message is about. The documents cover three businesses:\n"
    # "- Azuria: Azuria Properties, a Dubai property developer (buying apartments, townhouses and villas, "
    # "off-plan booking, payment plans, property fees, mortgages, residency or Golden visas for property "
    # "buyers, handover, warranties, service charges, pets, renting out, renovations, resale).\n"
    # "- Wayfarra: Wayfarra Holidays, a travel agency (flights, hotels, holiday packages, tourist and visit "
    # "visas, including UAE visit visas for family or friends, travel insurance, WayCash, trip refunds).\n"
    "COMPANY is the business the message is about. The documents cover one business:\n"
    "- TransAccelors: an AI, data and technology company in Bengaluru and Dubai (AI and data consulting, "
    "services, the FinAccelors financial analysis product, AI Immersion Lab, AI labs for education, "
    "hiring solutions, careers, its blog articles about AI).\n"
    # "Names may be misspelled (for example 'Azura', 'Wayfara' or 'Transacelors'). For a follow-up, use "
    # "the business of the earlier conversation. Write Both for greetings, for questions about more than "
    # "one business, or if you are not sure.\n\n"
    "Names may be misspelled (for example 'Transacelors' or 'Finacelors'). "
    "Write Both for greetings, for questions not about TransAccelors, or if you are not sure.\n\n"
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
    "- Keep every detail of the message: dates, holidays and seasons, ages, numbers, amounts, places, "
    "names, and who is travelling or buying. Do not shorten, summarise or add things up.\n"
    "- The QUESTION line must ALWAYS be in English, whatever language the message is in.\n\n"
    "TOPICS: always write this line. List each separate thing the message asks about as a short "
    "English search phrase of 2 to 6 words, separated by ' | ' (at most 3). Keep names, places and "
    "dates in them. One phrase if the message asks about one thing. "
    f"If the QUESTION is {SAME_TOPIC}, write: TOPICS: none\n\n"
    "Examples:\n"
    # TRANSACCELORS ONLY for now: the Azuria / Wayfarra examples are commented out
    # and replaced by TransAccelors examples in the same languages
    # "Message: mein kutta rakh saktha hoon kya?\n"
    # "LANGUAGE: Hindi, Latin\n"
    # "COMPANY: Azuria\n"
    # "QUESTION: Can I keep a dog?\n"
    # "TOPICS: keeping a dog\n\n"
    # "Message: holiday package cancel karne ka charge kitna hai?\n"
    # "LANGUAGE: Hindi, Latin\n"
    # "COMPANY: Wayfarra\n"
    # "QUESTION: What are the cancellation charges for a holiday package?\n"
    # "TOPICS: holiday package cancellation charges\n\n"
    # "Message: आप कौन से हॉलिडे पैकेज देते हैं?\n"
    # "LANGUAGE: Hindi, Devanagari\n"
    # "COMPANY: Wayfarra\n"
    # "QUESTION: Which holiday packages do you offer?\n"
    # "TOPICS: holiday packages offered\n\n"
    # "Message: ನಾನು ನಾಯಿಯನ್ನು ಸಾಕಬಹುದೇ?\n"
    # "LANGUAGE: Kannada, Kannada\n"
    # "COMPANY: Azuria\n"
    # "QUESTION: Can I keep a dog?\n"
    # "TOPICS: keeping a dog\n\n"
    # "Message: How much are the service charges?\n"
    # "LANGUAGE: English, Latin\n"
    # "COMPANY: Azuria\n"
    # "QUESTION: How much are the service charges?\n"
    # "TOPICS: service charges\n\n"
    # "Message: how much bagage can i take on my flight to bangkok and what documents do i need at hotel check in\n"
    # "LANGUAGE: English, Latin\n"
    # "COMPANY: Wayfarra\n"
    # "QUESTION: How much baggage can I take on my flight to Bangkok, and what documents do I need at hotel check-in?\n"
    # "TOPICS: flight baggage allowance | hotel check-in documents"
    "Message: finaccelors kya karta hai?\n"
    "LANGUAGE: Hindi, Latin\n"
    "COMPANY: TransAccelors\n"
    "QUESTION: What does FinAccelors do?\n"
    "TOPICS: FinAccelors product\n\n"
    "Message: आप कौन सी AI सेवाएं देते हैं?\n"
    "LANGUAGE: Hindi, Devanagari\n"
    "COMPANY: TransAccelors\n"
    "QUESTION: Which AI services do you offer?\n"
    "TOPICS: AI services offered\n\n"
    "Message: ನೀವು ಯಾವ AI ಸೇವೆಗಳನ್ನು ನೀಡುತ್ತೀರಿ?\n"
    "LANGUAGE: Kannada, Kannada\n"
    "COMPANY: TransAccelors\n"
    "QUESTION: Which AI services do you offer?\n"
    "TOPICS: AI services offered\n\n"
    "Message: What is the AI Immersion Lab?\n"
    "LANGUAGE: English, Latin\n"
    "COMPANY: TransAccelors\n"
    "QUESTION: What is the AI Immersion Lab?\n"
    "TOPICS: AI Immersion Lab\n\n"
    "Message: wat does transacelors do for schools and how do i apply for a job there\n"
    "LANGUAGE: English, Latin\n"
    "COMPANY: TransAccelors\n"
    "QUESTION: What does TransAccelors do for schools, and how do I apply for a job there?\n"
    "TOPICS: AI labs for schools | job applications careers"
)


SAME_QUESTION_PROMPT = (
    "Decide if one saved answer can be reused for both questions.\n"
    "Reply YES only if they ask for exactly the same information, worded differently.\n"
    "Reply NO if either question has any detail the other lacks or changes: an amount, "
    "number, date or time, a person type (child, non-resident), a project, place, package, "
    "product, fare class or payment method, a condition (if I cancel / if I default), "
    "or an extra part. Also NO if one is general and the other specific "
    "(refunds vs EOI refunds, the price vs the price for 2 adults).\n"
    "When unsure, reply NO. Reply with only YES or NO."
)


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
    # English names inside another script ("Pearl Residences விலை என்ன?") must not
    # make the message count as English letters: any other script wins
    other = {script: n for script, n in counts.items() if script != "latin"}
    if other:
        return max(other, key=other.get)
    return "latin"


MAX_PARTS = 3  # most topics searched separately for one message

# "LANGUAGE: ...", also when the model writes "- TOPICS: ..." or "**Question:** ..."
LABELED_LINE = re.compile(r"^[\s\-*•]*(LANGUAGE|COMPANY|QUESTION|TOPICS?)\s*\**\s*:\s*\**\s*(.*)$", re.I)


def parse_prepared(text: str) -> dict:
    """Reads the 'LANGUAGE / COMPANY / QUESTION / TOPICS' reply."""
    out = {"language": None, "company": None, "question": None, "parts": []}
    for line in text.splitlines():
        match = LABELED_LINE.match(line)
        if not match:
            continue
        label, value = match.group(1).upper(), match.group(2).strip().strip("'\"")
        if label == "LANGUAGE":
            out["language"] = value.lower() or None
        elif label == "COMPANY":
            out["company"] = company_key(value)
        elif label == "QUESTION":
            out["question"] = value or None
        else:
            topics = [t.strip(" .'\"") for t in value.split("|")]
            out["parts"] = [t for t in topics if t and t.lower() != "none"]
    if out["question"] is None and out["language"] is None:
        out["question"] = text.strip() or None  # model ignored the format: use the whole reply
    # One topic is just the question again: searching it twice only costs time
    out["parts"] = out["parts"][:MAX_PARTS] if len(out["parts"]) >= 2 else []
    return out


TRANSLATE_ANSWER_PROMPT = (
    "You are a translator. Translate the text the user sends into {target}.\n"
    "Rules:\n"
    "- Translate ALL the words, including the table headers and the text inside table cells.\n"
    "- Keep the same structure: Markdown tables stay Markdown tables with the same rows and "
    "columns, bullet points stay bullet points, bold stays bold.\n"
    "- Keep numbers, prices, dates, percentages, phone numbers and emails exactly as they are.\n"
    "- NEVER translate names. Copy every company, community, project, package, plan and "
    "place name exactly as written, in English letters, even inside text in another alphabet. "
    # TRANSACCELORS ONLY for now (old examples below)
    # "For example 'Azuria Creek', 'Pearl Residences', 'Maldives Island Escape', "
    # "'Baku Getaway, Azerbaijan', 'TravelSafe Plus', 'Wayfarra', 'WayCash' and 'Dubai (DXB)' stay unchanged.\n"
    "For example 'TransAccelors', 'FinAccelors', 'AI Immersion Lab', 'Bengaluru' and 'Dubai' stay unchanged.\n"
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
