"""
Answer cache that works across languages.

Every question is first turned into an English search question
("aap kaunse packages offer karte ho?" -> "Which holiday packages do you offer?").
The cache matches on the MEANING of that English question, so the same question
asked with other words, or in another language, finds the saved answer.

Reuse is strict, because a wrong reuse gives a confident wrong answer.
In the 3 Nov test 25 answers were wrong this way: different questions on the
same topic ("default at 50%" vs "at 70%", "business class" vs "economy fare")
scored up to 0.92 and the yes/no check said yes. A saved answer is reused only if
  - the company is the same (or unknown on one side), and
  - the numbers in both questions are the same ("AED 350", "2 adults", "50%"), and
  - it is the exact same English question, or similarity >= SHORTLIST_THRESHOLD
    and the LLM yes/no check (same_question_fn) agrees.
Missing a reuse only costs time; the question is then answered from the PDFs.

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
# Below it they are treated as different without asking. In the 3 Nov test all
# wrong reuses scored 0.92 or less; the same question in another language
# usually scores 0.94 or more.
SHORTLIST_THRESHOLD = 0.93
MAX_CHECKS = 2  # at most this many LLM yes/no checks per lookup

MAX_ENTRIES = 500  # oldest entries are dropped after this


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def numbers(text: str) -> set:
    """'AED 3,200,000 at 50%' -> {'3200000', '50'}. Different numbers = different question."""
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,.]*", text)}


def same_company(a, b) -> bool:
    return a is None or b is None or a == b


class AnswerCache:
    def __init__(self, embedding_manager, same_question_fn: Callable[[str, str], bool]):
        self.model = embedding_manager.model  # same embedding model as the PDF search
        self.same_question = same_question_fn
        self.entries = []  # each: {"search", "key", "vector", "company", "numbers", "answers": {language: answer}}
        self.lock = threading.Lock()  # chats can use the cache at the same time

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

    def lookup(self, search_question: str, company: Optional[str] = None) -> Optional[dict]:
        """Returns the saved entry for the same question, else None."""
        with self.lock:
            nums = numbers(search_question)
            entries = [e for e in self.entries
                       if same_company(company, e["company"]) and e["numbers"] == nums]
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

    def clear(self):
        """Forgets every saved answer (used when the website changed, so no old answers are reused)."""
        with self.lock:
            count = len(self.entries)
            self.entries = []
        print(f"Cache: cleared {count} saved answers")

    def add(self, search_question: str, language: str, answer: str,
            entry: Optional[dict] = None, company: Optional[str] = None):
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
                "company": company,
                "numbers": numbers(search_question),
                "answers": {language: answer},
            })
            if len(self.entries) > MAX_ENTRIES:
                self.entries.pop(0)
        print(f"Cache: saved answer for '{search_question}' ({language})")
