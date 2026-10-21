import time
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
 
SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided by the user. "
    "If the answer is not in the context, say that you could not find it in the documents"

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
        )
    except Exception as error:
        return (
            "I could not reach the local model. Is the Ollama app open, "
            f"and did you run 'ollama pull {MODEL}'? Details: {error}"
        )
 
    answer = response["message"]["content"]
    print(f"Ollama took {time.perf_counter() - start:.2f}s")
    return answer
 