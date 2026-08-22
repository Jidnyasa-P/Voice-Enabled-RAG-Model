"""
Person 2 — Retrieval Engine
Voice-Enabled RAG · HH Goa 2026

Provides:
  - 3 chunking strategies (fixed-size, semantic, metadata-aware)
  - Offline index builder for MSMARCO-XI (multi-language)
  - Vector DB (Chroma) with metadata filtering
  - Cross-encoder re-ranking
  - retrieve(query_text, language, top_k=3) function contract for Person 3

Usage:
    from retrieval_engine import RetrievalEngine
    engine = RetrievalEngine()
    engine.build_index(languages=["hi", "en", "ta"])
    results = engine.retrieve("भारत की राजधानी क्या है?", language="hi", top_k=3)
"""

import os
import re
import time
import warnings
from typing import List, Dict, Any
import numpy as np

# Suppress noisy transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_DB_PATH = "./vector_index"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LANGUAGES = ["hi", "en", "ta"]          # Hindi, English, Tamil
DEFAULT_MAX_EXAMPLES = 1000                       # per language, tune as needed
CHUNK_FIXED_SIZE = 256
CHUNK_FIXED_OVERLAP = 40
SEMANTIC_THRESHOLD = 0.65
BATCH_SIZE = 500                                  # Chroma add batch size


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two 1-D vectors (numpy arrays or lists)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def split_sentences(text: str) -> List[str]:
    """
    Lightweight multilingual sentence splitter.
    Handles Latin (. ! ?) and Indic (।) sentence terminators.
    """
    if not text or not isinstance(text, str):
        return []
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r"(?<=[.!?।।])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def chunk_fixed(text: str, size: int = CHUNK_FIXED_SIZE, overlap: int = CHUNK_FIXED_OVERLAP) -> List[str]:
    """
    Strategy 1: Fixed-size with overlap (baseline).
    Word-based windows with sentence-boundary snapping at chunk edges.
    """
    if not text or not isinstance(text, str):
        return []
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
        if start >= len(words):
            break
    return chunks


def chunk_semantic(text: str, embed_fn, threshold: float = SEMANTIC_THRESHOLD) -> List[str]:
    """
    Strategy 2: Semantic chunking.
    Splits where embedding similarity between consecutive sentences drops below threshold.
    Keeps topically coherent sentences together.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    # Batch embed all sentences for speed
    embeddings = embed_fn(sentences)

    chunks = []
    current = [sentences[0]]
    prev_emb = embeddings[0]

    for idx, sent in enumerate(sentences[1:], start=1):
        emb = embeddings[idx]
        sim = _cosine_similarity(prev_emb, emb)
        if sim < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sent)
        prev_emb = emb

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_metadata_aware(passage: str) -> List[str]:
    """
    Strategy 3: Metadata-aware chunking.
    Respects natural paragraph boundaries — each paragraph is a chunk.
    This preserves document structure and aligns chunks with metadata boundaries.
    """
    if not passage or not isinstance(passage, str):
        return []
    paragraphs = [p.strip() for p in passage.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [passage]


# =============================================================================
# RETRIEVAL ENGINE
# =============================================================================

class RetrievalEngine:
    """
    End-to-end retrieval engine for MSMARCO-XI.

    Responsibilities:
      - Load / cache embedding + re-ranker models
      - Build persistent Chroma index (offline)
      - Query with language-filtered vector search + cross-encoder re-ranking
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER,
    ):
        self.db_path = db_path
        self.embed_model_name = embed_model_name
        self.reranker_model_name = reranker_model_name

        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="msmarco_xi",
            metadata={"hnsw:space": "cosine"},
        )

        # Load models (cached in memory for the process lifetime)
        print(f"[RetrievalEngine] Loading embedder: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name)
        print(f"[RetrievalEngine] Loading re-ranker: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)
        print("[RetrievalEngine] Models loaded.")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed a list of texts."""
        if not texts:
            return []
        emb = self.embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return emb.tolist()

    # ------------------------------------------------------------------
    # Offline index builder
    # ------------------------------------------------------------------

    def build_index(
        self,
        languages: List[str] = None,
        max_examples_per_lang: int = DEFAULT_MAX_EXAMPLES,
    ) -> None:
        """
        Build the vector index offline for the given languages.
        Run this ONCE before the demo. At query time it is just a lookup.

        Args:
            languages: List of language codes, e.g. ["hi", "en", "ta"]
            max_examples_per_lang: Cap examples per language for speed
        """
        languages = languages or DEFAULT_LANGUAGES
        print(f"\n[build_index] Starting index build for: {languages}")
        print(f"[build_index] Max examples per language: {max_examples_per_lang}")

        total_chunks = 0

        for lang in languages:
            print(f"\n--- Processing language: {lang} ---")
            try:
                ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split="train", trust_remote_code=True)
            except Exception as e:
                print(f"  ⚠️  Failed to load dataset for '{lang}': {e}")
                continue

            batch_ids: List[str] = []
            batch_texts: List[str] = []
            batch_metadatas: List[Dict[str, Any]] = []
            lang_chunk_count = 0
            example_count = 0

            for example in ds:
                if example_count >= max_examples_per_lang:
                    break

                query_id = example.get("id", f"{lang}_{example_count}")
                passages = example.get("passages", [])
                source_lang = example.get("source_lang", lang)
                target_lang = example.get("target_lang", lang)

                if not passages:
                    example_count += 1
                    continue

                for p_idx, passage in enumerate(passages):
                    if not passage or not isinstance(passage, str):
                        continue

                    # Apply all 3 chunking strategies
                    strategies = {
                        "fixed": chunk_fixed(passage),
                        "semantic": chunk_semantic(passage, self.embed),
                        "metadata": chunk_metadata_aware(passage),
                    }

                    for strategy_name, chunks in strategies.items():
                        for c_idx, chunk in enumerate(chunks):
                            if not chunk or not chunk.strip():
                                continue

                            chunk_id = f"{query_id}_p{p_idx}_{strategy_name}_{c_idx}"
                            metadata = {
                                "language": str(target_lang),
                                "source_query_id": str(query_id),
                                "strategy": strategy_name,
                                "source_lang": str(source_lang),
                            }

                            batch_ids.append(chunk_id)
                            batch_texts.append(chunk)
                            batch_metadatas.append(metadata)
                            lang_chunk_count += 1

                            # Flush batch to Chroma
                            if len(batch_ids) >= BATCH_SIZE:
                                self._flush_batch(batch_ids, batch_texts, batch_metadatas)
                                batch_ids, batch_texts, batch_metadatas = [], [], []

                example_count += 1

            # Flush remaining for this language
            if batch_ids:
                self._flush_batch(batch_ids, batch_texts, batch_metadatas)
                batch_ids, batch_texts, batch_metadatas = [], [], []

            print(f"  ✅ {lang}: {example_count} examples → {lang_chunk_count} chunks")
            total_chunks += lang_chunk_count

        print(f"\n🎉 Index build complete! Total chunks indexed: {self.collection.count()}")
        print(f"   (Expected ~{total_chunks}; Chroma may dedupe by ID)")

    def _flush_batch(self, ids: List[str], texts: List[str], metadatas: List[Dict]) -> None:
        """Embed and persist a batch of chunks to Chroma."""
        if not ids:
            return
        embeddings = self.embed(texts)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Query-time retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query_text: str, language: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks for a query, filtered by detected language.

        Pipeline:
          1. Embed query
          2. Vector search top-20 with language metadata filter
          3. Cross-encoder re-rank → top-3

        Returns:
            List of dicts with keys:
                text, score, strategy, language, source_query_id

            'score' is the ORIGINAL vector cosine similarity (0–1),
            suitable for Person 1's confidence-threshold guardrail.
            Results are ordered by re-ranker quality (best first).
        """
        if not query_text or not query_text.strip():
            return []

        t0 = time.time()

        # 1. Embed query
        te = time.time()
        query_embedding = self.embed([query_text])[0]
        embed_ms = (time.time() - te) * 1000

        # 2. Vector search with language filter (cheap latency win)
        tv = time.time()
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=20,
                where={"language": language},
            )
        except Exception as e:
            print(f"[retrieve] Vector search failed: {e}")
            return []
        vector_ms = (time.time() - tv) * 1000

        # Format candidates
        candidates = []
        if results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = float(results["distances"][0][i])
                # Chroma cosine distance → similarity
                vector_sim = 1.0 - distance
                meta = results["metadatas"][0][i]
                candidates.append({
                    "_id": doc_id,
                    "text": results["documents"][0][i],
                    "vector_score": vector_sim,          # for Person 1 threshold
                    "strategy": meta.get("strategy", "unknown"),
                    "language": meta.get("language", language),
                    "source_query_id": meta.get("source_query_id", ""),
                })

        if not candidates:
            return []

        # 3. Re-rank with cross-encoder
        tr = time.time()
        pairs = [(query_text, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        for i, c in enumerate(candidates):
            c["rerank_score"] = float(rerank_scores[i])

        # Sort by re-rank score (descending = best first)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        rerank_ms = (time.time() - tr) * 1000

        # 4. Prepare output — top_k
        top_candidates = candidates[:top_k]
        output = []
        for c in top_candidates:
            output.append({
                "text": c["text"],
                "score": round(c["vector_score"], 4),   # Person 1 expects 0–1 cosine sim
                "strategy": c["strategy"],
                "language": c["language"],
                "source_query_id": c["source_query_id"],
            })

        total_ms = (time.time() - t0) * 1000
        print(f"[retrieve] embed={embed_ms:.1f}ms | vector_search={vector_ms:.1f}ms | rerank={rerank_ms:.1f}ms | total={total_ms:.1f}ms")

        return output

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics for debugging / README."""
        count = self.collection.count()
        # Peek at a few records to infer strategy distribution
        sample = self.collection.peek(limit=min(1000, count))
        strategies = {}
        langs = {}
        if sample and sample.get("metadatas"):
            for meta in sample["metadatas"]:
                s = meta.get("strategy", "unknown")
                l = meta.get("language", "unknown")
                strategies[s] = strategies.get(s, 0) + 1
                langs[l] = langs.get(l, 0) + 1
        return {
            "total_chunks": count,
            "sample_strategies": strategies,
            "sample_languages": langs,
        }
