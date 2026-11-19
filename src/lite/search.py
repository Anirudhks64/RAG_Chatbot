"""
LITE version of the search (src/full: embedding.py + vector_store.py + rag_retriever.py in one file),
small enough for a 512 MB server (Render free plan).

Differences from the full version:
  - small English models through fastembed (ONNX), so no PyTorch is needed:
      search:   bge-small-en-v1.5, 8-bit     (~34 MB instead of bge-m3, ~2.3 GB)
      reranker: ms-marco-MiniLM-L-6-v2, 8-bit (~23 MB instead of bge-reranker-v2-m3, ~2.3 GB)
    English-only models are fine: every question is rewritten into English before
    the search (prepare_question), and the documents are English.
  - no Chroma: the chunks and their vectors are kept in memory (a few hundred chunks)

Same steps and same output as RAGRetriever.retrieve, so the rest of the code is unchanged.
"""
import os
import re
import threading
import uuid
from typing import List, Dict, Any, Optional, Set

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from fastembed.common.model_description import ModelSource, PoolingType

# 8-bit ("quantized") versions of the two small models: ~4x smaller files, the reranker uses
# ~40 MB of memory instead of ~134 MB. Tested: 99.8% identical vectors, same reranking order.
TextEmbedding.add_custom_model(
    model="Xenova/bge-small-en-v1.5", pooling=PoolingType.CLS, normalization=True,
    sources=ModelSource(hf="Xenova/bge-small-en-v1.5"), dim=384, model_file="onnx/model_quantized.onnx")
TextCrossEncoder.add_custom_model(
    model="Xenova/ms-marco-MiniLM-L-6-v2-int8",
    sources=ModelSource(hf="Xenova/ms-marco-MiniLM-L-6-v2"), model_file="onnx/model_quantized.onnx")

EMBED_MODEL = os.getenv("LITE_EMBED_MODEL", "Xenova/bge-small-en-v1.5")              # 8-bit bge-small-en-v1.5
RERANK_MODEL = os.getenv("LITE_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2-int8")  # 8-bit ms-marco MiniLM
RRF_K = 60            # standard constant for merging ranked lists
OWN_PDF_RATIO = 0.25  # same rule as rag_retriever.py

# Memory savers for a 512 MB server: 1 CPU thread per model, small batches, and only one
# question uses the models at a time (in parallel, each one takes its own working memory;
# with 0.1 CPU on the Render free plan parallel runs would not be faster anyway)
THREADS = 1
BATCH = 4
MODEL_LOCK = threading.Lock()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class LiteEmbedder:
    """Small embedding model. Also looks like SentenceTransformer to answer_cache.py
    (it calls embedding_manager.model.encode([...], normalize_embeddings=True, ...))."""

    def __init__(self, model_name: str = EMBED_MODEL):
        print(f"Loading lite embedding model: {model_name}")
        self.model_name = model_name
        self._model = TextEmbedding(model_name, threads=THREADS)
        self.model = self  # answer_cache.py uses embedding_manager.model.encode

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        print(f"Generating embeddings for {len(texts)} texts...")
        # OLD: one lock for all texts (a rebuild while running would make visitors wait for all of it)
        # with MODEL_LOCK:
        #     return _normalize(np.array(list(self._model.embed(texts, batch_size=BATCH)), dtype=np.float32))
        vectors = []
        for start in range(0, len(texts), BATCH):  # lock per small batch: questions can run in between
            with MODEL_LOCK:
                vectors += list(self._model.embed(texts[start:start + BATCH], batch_size=BATCH))
        return _normalize(np.array(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        with MODEL_LOCK:
            return _normalize(np.array(list(self._model.query_embed(text))[0], dtype=np.float32))

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        """Used by the answer cache to compare questions with each other."""
        with MODEL_LOCK:
            return _normalize(np.array(list(self._model.embed(list(texts), batch_size=BATCH)), dtype=np.float32))


class LiteRetriever:
    """Hybrid search + reranker, like RAGRetriever:
      1. meaning search (vectors in memory)  2. keyword search (BM25)
      3. merge both (Reciprocal Rank Fusion)  4. reranker picks the best chunks"""

    def __init__(self, chunks, embedder: LiteEmbedder, old: Optional["LiteRetriever"] = None):
        """old: the retriever this one replaces (after the website changed). Its reranker model
        and the vectors of unchanged text are reused, so only new or changed text is embedded."""
        self.embedder = embedder
        self.ids, self.texts, self.metadatas = [], [], []
        for i, doc in enumerate(chunks):
            self.ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            self.texts.append(doc.page_content)
            self.metadatas.append({**doc.metadata, "doc_index": i})
        # OLD: every text embedded again
        # self.vectors = embedder.embed_documents(self.texts) if self.texts else np.zeros((0, 1))
        known = dict(zip(old.texts, old.vectors)) if old is not None and len(old.texts) else {}
        new_texts = list(dict.fromkeys(t for t in self.texts if t not in known))
        if new_texts:
            known.update(zip(new_texts, embedder.embed_documents(new_texts)))
        self.vectors = np.array([known[t] for t in self.texts], dtype=np.float32) if self.texts else np.zeros((0, 1))
        print("Building keyword (BM25) index ...")
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts]) if self.texts else None
        if old is not None:
            self.reranker = old.reranker
        else:
            print(f"Loading lite reranker model: {RERANK_MODEL}")
            self.reranker = TextCrossEncoder(RERANK_MODEL, threads=THREADS)
            print("Reranker loaded.\n")

    def _files_of(self, company: Optional[str]) -> Optional[Set[str]]:
        if not company:
            return None
        files = {m.get("source_file", "") for m in self.metadatas}
        files = {f for f in files if f.lower().startswith(company.lower())}
        return files or None

    def _doc(self, i: int, **scores) -> Dict[str, Any]:
        return {"id": self.ids[i], "content": self.texts[i], "metadata": self.metadatas[i], **scores}

    def _dense_search(self, query: str, n: int) -> List[Dict[str, Any]]:
        if not len(self.texts):
            return []
        similarity = self.vectors @ self.embedder.embed_query(query)
        best = np.argsort(-similarity)[:n]
        return [self._doc(int(i), dense_score=float(similarity[i])) for i in best]

    def _keyword_search(self, query: str, n: int, extra_words: str = "") -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        words = tokenize(query)
        words += [w for w in dict.fromkeys(tokenize(extra_words)) if w not in words]
        scores = self.bm25.get_scores(words)
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self._doc(i, bm25_score=float(scores[i])) for i in best if scores[i] > 0]

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

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0,
                 candidates: int = 20, company: Optional[str] = None,
                 extra_words: str = "", rerank_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Same inputs and output as RAGRetriever.retrieve (see rag_retriever.py)."""
        files = self._files_of(company)
        print(f"Retrieving documents for query: '{query}'")
        try:
            dense = self._dense_search(query, candidates)
            keyword = self._keyword_search(query, candidates, extra_words)
            merged = self._merge(dense, keyword)
            print(f"Meaning search: {len(dense)}, keyword search: {len(keyword)}, merged: {len(merged)}")
            if not merged:
                return []
            if rerank_limit:
                merged = merged[:rerank_limit]

            # ms-marco gives raw scores; the sigmoid turns them into 0..1 like the full reranker
            with MODEL_LOCK:
                raw = np.array(list(self.reranker.rerank(query, [d["content"] for d in merged], batch_size=BATCH)),
                               dtype=np.float32)
            for doc, score in zip(merged, 1 / (1 + np.exp(-raw))):
                doc["rerank_score"] = float(score)
            merged.sort(key=lambda d: d["rerank_score"], reverse=True)

            if files:
                own = [d for d in merged if d["metadata"].get("source_file") in files]
                if own and own[0]["rerank_score"] >= OWN_PDF_RATIO * merged[0]["rerank_score"]:
                    merged = own

            retrieved = []
            for doc in merged:
                if doc["rerank_score"] < score_threshold:
                    continue
                doc["similarity_score"] = doc["rerank_score"]
                doc["rank"] = len(retrieved) + 1
                retrieved.append(doc)
                if len(retrieved) == top_k:
                    break
            print(f"Retrieved {len(retrieved)} documents after reranking")
            return retrieved
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
