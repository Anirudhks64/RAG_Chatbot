import os 
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

client = genai.Client(api_key=api_key)


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


def generate_answer(question: str, context: list[dict]) -> str:    
    prompt = f"""
    Answer the user's question using the retrieved context below.

    If the context does not contain enough information to answer,
    say that the information is unavailable.

    Retrieved context:
    {context}

    User question:
    {question}
    """

    


    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt, 
    )
    

    print("\n\n")
    print("Response from Gemini LLM --> ")
    print("\n\n")
    print(response.text)
    print("\n\n")
