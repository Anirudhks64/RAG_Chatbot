import os
import time
from dotenv import load_dotenv
from google import genai
from typing import List, Dict, Any
from google.genai import types

### CALL GEMINI API USING THE RETRIEVED CHUNKS

# Load API Key from .env file
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Give up after 30s of silence and retry only once, so a busy Google server
# shows an error instead of hanging forever
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30_000,  # milliseconds
        retry_options=types.HttpRetryOptions(attempts=2, max_delay=3),
    ),
)


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


MODEL = "gemini-3.8-flash"


def build_prompt(question: str, context: str) -> str:
    return f"""
    Answer the user's question using the retrieved context below.

    If the context does not contain enough information to answer,
    say that the information is unavailable. 

    Retrieved context:
    {context}

    User question:
    {question}
    """


def stream_answer(question: str, context: str, history=None):
    """Yields the answer piece by piece, as soon as Gemini sends each part.
    history: recent (question, answer) pairs, so follow-ups make sense."""
    start = time.perf_counter()
    got_any = False
    prompt = build_prompt(question, context)
    if history:
        past = "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
        prompt = f"Earlier conversation (for follow-up questions):\n{past}\n\n{prompt}"
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=prompt,
        ):
            if chunk.text:
                if not got_any:
                    print(f"Gemini first words after {time.perf_counter() - start:.2f}s")
                got_any = True
                yield chunk.text
    except Exception as error:
        print(f"Gemini failed after {time.perf_counter() - start:.2f}s: {error}")
        yield (
            "\n\nGemini is busy or not responding right now. "
            f"Please try again in a minute. (Details: {str(error)[:150]})"
        )


def generate_answer(question: str, context: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(question, context),
    )


    print("\n\n")
    print("Response from Gemini LLM --> ")
    print("\n\n")
    print(response.text)
    print("\n\n")
    return response.text
