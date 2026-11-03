"""
Answer cache that works across languages.

Every question is first turned into an English search question
("aap kaunse packages offer karte ho?" -> "Which holiday packages do you offer?").
The cache matches on the MEANING of that English question, so the same question
asked with other words, or in another language, finds the saved answer.

Matching has two steps, because similarity alone is not reliable
(on this project "Can I keep a dog?" vs "...a cat?" scored 0.84, while true
rephrasings scored as low as 0.70):
  1. exact same English question (ignoring case/punctuation) -> match
  2. otherwise the most similar saved question above SHORTLIST_THRESHOLD is
     checked by a quick yes/no question to the LLM (same_question_fn)

Each entry keeps the answer in every language it has been given in so far:
  - same language as a saved version -> returned instantly
  - another language                 -> the saved answer is translated (no PDF search)

The cache lives in memory, is shared by all chats, and is empty after a restart,
so it never serves answers from old versions of the PDFs.
"""
import re
import threading
from typing import Callable, Optional

import numpy as np

# Saved questions at least this similar (cosine, 0 to 1) are checked with the LLM.
# Below it they are treated as different without asking (saves time).
SHORTLIST_THRESHOLD = 0.70
MAX_CHECKS = 2  # at most this many LLM yes/no checks per lookup

MAX_ENTRIES = 500  # oldest entries are dropped after this


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


class AnswerCache:
    def __init__(self, embedding_manager, same_question_fn: Callable[[str, str], bool]):
        self.model = embedding_manager.model  # same embedding model as the PDF search
        self.same_question = same_question_fn
        self.entries = []  # each: {"search", "key", "vector", "answers": {language: answer}}
        self.lock = threading.Lock()  # chats can use the cache at the same time

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

    def lookup(self, search_question: str) -> Optional[dict]:
        """Returns the saved entry for the same question, else None."""
        with self.lock:
            entries = list(self.entries)
        if not entries:
            return None

        key = normalize(search_question)
        for e in entries:
            if e["key"] == key:
                print(f"Cache: exact match '{e['search']}'")
                return e

        vector = self._embed(search_question)
        scores = np.array([float(vector @ e["vector"]) for e in entries])
        for i in scores.argsort()[::-1][:MAX_CHECKS]:
            if scores[i] < SHORTLIST_THRESHOLD:
                break
            same = self.same_question(search_question, entries[i]["search"])
            print(f"Cache: '{entries[i]['search']}' similarity {scores[i]:.3f}, "
                  f"same question: {'yes' if same else 'no'}")
            if same:
                return entries[i]
        return None

    def add(self, search_question: str, language: str, answer: str, entry: Optional[dict] = None):
        """Saves an answer. Pass entry to add a new language version to an existing entry."""
        language = language or "unknown"
        with self.lock:
            if entry is not None:
                entry["answers"][language] = answer
                return
        vector = self._embed(search_question)
        with self.lock:
            self.entries.append({
                "search": search_question,
                "key": normalize(search_question),
                "vector": vector,
                "answers": {language: answer},
            })
            if len(self.entries) > MAX_ENTRIES:
                self.entries.pop(0)
        print(f"Cache: saved answer for '{search_question}' ({language})")
