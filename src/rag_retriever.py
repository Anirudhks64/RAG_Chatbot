import re
from typing import List, Dict, Any, Optional, Set

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from embedding import EmbeddingManager
from vector_store import VectorStore


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # free, runs locally, multilingual
RRF_K = 60  # standard constant for merging ranked lists

# A question's company label is trusted when that company's best chunk scores at least
# this share of the best chunk overall. Measured on 28 test questions: with the right
# label the share was 0.5 or more, with a wrong label 0.01 or less (except one topic
# both companies cover).
OWN_PDF_RATIO = 0.25


def tokenize(text: str) -> List[str]:
    """Lowercase words and numbers. Works for English, Arabic and Hinglish."""
    return re.findall(r"\w+", text.lower())


class RAGRetriever:
    """
    Hybrid search + reranker:
      1. Meaning search (embeddings in Chroma)
      2. Keyword search (BM25)
      3. Merge both lists (Reciprocal Rank Fusion)
      4. Reranker picks the best chunks from the merged list
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        # Keyword index over the same chunks that are in Chroma
        print("Building keyword (BM25) index ...")
        self.ids = vector_store.ids
        self.texts = vector_store.texts
        self.metadatas = vector_store.metadatas
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts]) if self.texts else None

        print(f"Loading reranker model: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        print("Reranker loaded.\n")

    # ---------- 0. which PDFs to search ----------
    def _files_of(self, company: Optional[str]) -> Optional[Set[str]]:
        """The PDF files of one company (their names start with it, e.g. 'azuria').
        None means search all PDFs: company unknown, or no file matches."""
        if not company:
            return None
        files = {m.get("source_file", "") for m in self.metadatas}
        files = {f for f in files if f.lower().startswith(company.lower())}
        return files or None

    # ---------- 1. meaning search ----------
    def _dense_search(self, query: str, n: int) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            {
                "id": doc_id,
                "content": doc,
                "metadata": meta,
                "dense_score": 1 - dist,  # cosine similarity
            }
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ---------- 2. keyword search ----------
    def _keyword_search(self, query: str, n: int, extra_words: str = "") -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        words = tokenize(query)
        # The user's own words too, in case the rewritten question dropped one
        words += [w for w in dict.fromkeys(tokenize(extra_words)) if w not in words]
        scores = self.bm25.get_scores(words)
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            {
                "id": self.ids[i],
                "content": self.texts[i],
                "metadata": self.metadatas[i],
                "bm25_score": float(scores[i]),
            }
            for i in best
            if scores[i] > 0  # skip chunks that share no words with the question
        ]

    # ---------- 3. merge ----------
    @staticmethod
    def _merge(dense: List[Dict], keyword: List[Dict]) -> List[Dict]:
        merged: Dict[str, Dict] = {}
        for source, results in (("dense", dense), ("bm25", keyword)):
            for rank, doc in enumerate(results, start=1):
                entry = merged.setdefault(doc["id"], {**doc, "rrf_score": 0.0})
                entry.update({k: v for k, v in doc.items() if k.endswith("_score")})
                entry[f"{source}_rank"] = rank
                entry["rrf_score"] += 1 / (RRF_K + rank)
        return sorted(merged.values(), key=lambda d: d["rrf_score"], reverse=True)

    # ---------- main entry point (same name and output as before) ----------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        candidates: int = 20,
        company: Optional[str] = None,
        extra_words: str = "",
        rerank_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            query: The question
            top_k: How many chunks to return after reranking
            score_threshold: Minimum reranker score (0 to 1) to keep a chunk
            candidates: How many chunks each search returns before merging
            company: the business the question is about (e.g. 'azuria'). Only that company's
                chunks are returned, unless its PDF matches much worse than another PDF: then
                the label is probably wrong and chunks from all PDFs are returned. None = all PDFs.
            extra_words: more words for the keyword search only (the user's original message)
            rerank_limit: only rerank this many of the best merged chunks (the reranker is
                the slow step, about 0.07s per chunk); None = rerank all of them
        """
        files = self._files_of(company)
        print(f"Retrieving documents for query: '{query}'"
              + (f" (prefer {', '.join(sorted(files))})" if files else ""))
        try:
            dense = self._dense_search(query, candidates)
            keyword = self._keyword_search(query, candidates, extra_words)
            merged = self._merge(dense, keyword)
            print(f"Meaning search: {len(dense)}, keyword search: {len(keyword)}, merged: {len(merged)}")

            if not merged:
                print("No documents found")
                return []
            if rerank_limit:
                merged = merged[:rerank_limit]  # already sorted best first by the merge

            # 4. reranker reads question + chunk together and scores the match
            scores = self.reranker.predict([(query, d["content"]) for d in merged])
            for doc, score in zip(merged, scores):
                doc["rerank_score"] = float(score)
            merged.sort(key=lambda d: d["rerank_score"], reverse=True)

            # 5. keep only the company's own chunks (no mixing of the two businesses),
            #    unless its best chunk is far worse than the best overall (wrong label)
            if files:
                own = [d for d in merged if d["metadata"].get("source_file") in files]
                if own and own[0]["rerank_score"] >= OWN_PDF_RATIO * merged[0]["rerank_score"]:
                    merged = own
                else:
                    print(f"'{company}' PDF matches much worse than another PDF: using all PDFs")

            retrieved_docs = []
            for doc in merged:
                if doc["rerank_score"] < score_threshold:
                    continue
                doc["similarity_score"] = doc["rerank_score"]  # used by build_context
                doc["rank"] = len(retrieved_docs) + 1
                retrieved_docs.append(doc)
                if len(retrieved_docs) == top_k:
                    break

            print(f"Retrieved {len(retrieved_docs)} documents after reranking")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
