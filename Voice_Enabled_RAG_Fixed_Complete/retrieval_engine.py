"""
Person 2 — Retrieval Engine (FIXED VERSION)
Voice-Enabled RAG · HH Goa 2026

CRITICAL FIXES APPLIED:
  1. ChromaDB where filter now uses explicit {"$eq": ...} syntax
     (shorthand {"field": "value"} fails silently in some Chroma versions)
  2. Fallback search: if language-filtered search returns empty,
     retries WITHOUT the filter so the user at least sees SOMETHING
  3. DuckDB passages field robustly converted to plain Python dict
     (handles pandas Series, numpy records, etc.)
  4. Added index_health_check() to diagnose empty-index issues
  5. Better logging at every stage so you can see what's happening
  6. Added _index_built flag to detect if build_index was ever called

Usage:
    from retrieval_engine import RetrievalEngine
    engine = RetrievalEngine()

    # Check health first
    health = engine.index_health_check()
    print(health)

    # Build if needed
    if health["total_chunks"] == 0:
        engine.build_index(languages=["hi", "en", "ta"])

    results = engine.retrieve("भारत की राजधानी क्या है?", language="hi", top_k=3)
"""

import os
import re
import time
import warnings
import json
from typing import List, Dict, Any, Optional
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_DB_PATH = "./vector_index"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LANGUAGES = ["hi", "en", "ta"]
DEFAULT_MAX_EXAMPLES = 1000

VALID_HF_LANGUAGE_CONFIGS = {
    "as", "bn", "gu", "hi", "kn", "ml", "mr",
    "ne", "or", "pa", "sa", "ta", "te", "ur",
}
ENGLISH_SOURCE_CONFIG = "hi"

LANG_FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "gu", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "or", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}

CHUNK_FIXED_SIZE = 256
CHUNK_FIXED_OVERLAP = 40
SEMANTIC_THRESHOLD = 0.65
BATCH_SIZE = 500


def _cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def split_sentences(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    raw = re.split(r"(?<=[.!?।।])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def chunk_fixed(text: str, size: int = CHUNK_FIXED_SIZE, overlap: int = CHUNK_FIXED_OVERLAP) -> List[str]:
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
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences
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
    if not passage or not isinstance(passage, str):
        return []
    paragraphs = [p.strip() for p in passage.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [passage]


# =============================================================================
# DUCKDB DATASET LOADER (with robust passage extraction)
# =============================================================================

def _to_plain_dict(obj):
    """
    Recursively convert pandas Series, numpy records, or any nested
    non-plain-Python object into plain Python dicts/lists/scalars.
    This fixes the silent bug where DuckDB/pandas returns a Series
    instead of a dict for struct columns.
    """
    if hasattr(obj, "to_dict"):
        return _to_plain_dict(obj.to_dict())
    if hasattr(obj, "tolist"):
        return _to_plain_dict(obj.tolist())
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]
    return obj


def load_lang_dataset(hf_config: str, max_rows: int = None):
    """Load one language's data directly from its parquet file via DuckDB."""
    import duckdb
    from huggingface_hub import hf_hub_download

    prefix = LANG_FILE_PREFIX.get(hf_config)
    if not prefix:
        raise ValueError(f"Unknown MSMARCO-XI language config: {hf_config!r}")
    filename = f"train/{prefix}train.parquet"

    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename=filename,
        repo_type="dataset",
    )

    con = duckdb.connect()
    limit_clause = f"LIMIT {int(max_rows)}" if max_rows is not None else ""
    safe_path = local_path.replace("\\", "/")
    query = f"""
        SELECT query_id, passages, source_lang, target_lang
        FROM read_parquet('{safe_path}')
        {limit_clause}
    """
    df = con.execute(query).df()
    con.close()

    records = df.to_dict(orient="records")
    # CRITICAL FIX: Convert any pandas Series / numpy objects to plain Python
    records = [_to_plain_dict(r) for r in records]
    return records


# =============================================================================
# RETRIEVAL ENGINE
# =============================================================================

class RetrievalEngine:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER,
    ):
        self.db_path = db_path
        self.embed_model_name = embed_model_name
        self.reranker_model_name = reranker_model_name

        print(f"[RetrievalEngine] Initializing Chroma at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="msmarco_xi",
            metadata={"hnsw:space": "cosine"},
        )

        print(f"[RetrievalEngine] Loading embedder: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name)
        print(f"[RetrievalEngine] Loading re-ranker: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)
        print("[RetrievalEngine] Models loaded.")

        # Check index health on init
        health = self.index_health_check()
        print(f"[RetrievalEngine] Index health: {health['total_chunks']} chunks, "
              f"languages={list(health['languages'].keys())}")

    def embed(self, texts: List[str]) -> List[List[float]]:
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
        languages = languages or DEFAULT_LANGUAGES
        print(f"\n[build_index] Starting index build for: {languages}")
        print(f"[build_index] Max examples per language: {max_examples_per_lang}")

        total_chunks = 0
        ds_cache: Dict[str, Any] = {}

        for lang in languages:
            print(f"\n--- Processing language: {lang} ---")

            if lang == "en":
                hf_config = ENGLISH_SOURCE_CONFIG
                passage_field = "English_passages"
            elif lang in VALID_HF_LANGUAGE_CONFIGS:
                hf_config = lang
                passage_field = "Translated_passages"
            else:
                print(f"  ⚠️  '{lang}' is not a valid config; skipping")
                continue

            if hf_config in ds_cache:
                ds = ds_cache[hf_config]
                print(f"  📦 Using cached dataset for '{hf_config}' ({len(ds)} rows)")
            else:
                try:
                    ds = load_lang_dataset(hf_config, max_rows=max_examples_per_lang)
                    ds_cache[hf_config] = ds
                    print(f"  ✅ Loaded {len(ds)} rows for '{hf_config}'")
                except Exception as e:
                    print(f"  ❌ FAILED to load dataset for '{hf_config}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            batch_ids: List[str] = []
            batch_texts: List[str] = []
            batch_metadatas: List[Dict[str, Any]] = []
            lang_chunk_count = 0
            example_count = 0
            empty_passage_count = 0

            for example in ds:
                if example_count >= max_examples_per_lang:
                    break

                query_id = example.get("query_id", f"{lang}_{example_count}")

                # ROBUST PASSAGE EXTRACTION
                passages_raw = example.get("passages")
                passages_dict = {}
                if passages_raw is not None:
                    if isinstance(passages_raw, dict):
                        passages_dict = passages_raw
                    else:
                        print(f"  ⚠️  Unexpected passages type: {type(passages_raw)} for query_id={query_id}")
                        empty_passage_count += 1
                        example_count += 1
                        continue

                passages = passages_dict.get(passage_field, []) if isinstance(passages_dict, dict) else []
                if not isinstance(passages, list):
                    passages = [passages] if passages else []

                source_lang = example.get("source_lang", "")
                target_lang = example.get("target_lang", "")

                if not passages:
                    empty_passage_count += 1
                    example_count += 1
                    continue

                for p_idx, passage in enumerate(passages):
                    if not passage or not isinstance(passage, str):
                        continue

                    strategies = {
                        "fixed": chunk_fixed(passage),
                        "semantic": chunk_semantic(passage, self.embed),
                        "metadata": chunk_metadata_aware(passage),
                    }

                    for strategy_name, chunks in strategies.items():
                        for c_idx, chunk in enumerate(chunks):
                            if not chunk or not chunk.strip():
                                continue

                            chunk_id = f"{lang}_{query_id}_p{p_idx}_{strategy_name}_{c_idx}"
                            metadata = {
                                "language": lang,  # short code for filter matching
                                "source_query_id": str(query_id),
                                "strategy": strategy_name,
                                "dataset_source_lang": str(source_lang),
                                "dataset_target_lang": str(target_lang),
                            }

                            batch_ids.append(chunk_id)
                            batch_texts.append(chunk)
                            batch_metadatas.append(metadata)
                            lang_chunk_count += 1

                            if len(batch_ids) >= BATCH_SIZE:
                                self._flush_batch(batch_ids, batch_texts, batch_metadatas)
                                batch_ids, batch_texts, batch_metadatas = [], [], []

                example_count += 1

            if batch_ids:
                self._flush_batch(batch_ids, batch_texts, batch_metadatas)

            print(f"  ✅ {lang}: {example_count} examples, {lang_chunk_count} chunks "
                  f"({empty_passage_count} had empty passages)")
            total_chunks += lang_chunk_count

        final_count = self.collection.count()
        print(f"\n🎉 Index build complete!")
        print(f"   Total chunks in index: {final_count}")
        print(f"   Expected from this run: ~{total_chunks}")
        if final_count == 0:
            print("   ❌❌❌ WARNING: INDEX IS EMPTY! Retrieval will fail for all queries. ❌❌❌")
        elif final_count < 100:
            print(f"   ⚠️  WARNING: Only {final_count} chunks indexed. "
                  "Retrieval may fail for many queries.")

    def _flush_batch(self, ids: List[str], texts: List[str], metadatas: List[Dict]) -> None:
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
    # Query-time retrieval (FIXED)
    # ------------------------------------------------------------------

    def retrieve(self, query_text: str, language: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            print("[retrieve] ERROR: query_text is empty")
            return []

        if not language:
            print("[retrieve] WARNING: language is empty, will try unfiltered search")

        t0 = time.time()

        # 1. Embed query
        te = time.time()
        query_embedding = self.embed([query_text])[0]
        embed_ms = (time.time() - te) * 1000

        # 2. Vector search with language filter
        # CRITICAL FIX: Use explicit $eq operator for ChromaDB compatibility
        tv = time.time()
        candidates = []
        filter_used = "none"

        # Try 1: Filtered by language (if language is provided)
        if language:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=20,
                    where={"language": {"$eq": language}},
                )
                candidates = self._format_candidates(results, language)
                filter_used = f"language=$eq:{language}"
                print(f"[retrieve] Filtered search returned {len(candidates)} candidates")
            except Exception as e:
                print(f"[retrieve] Filtered search FAILED: {e}")
                candidates = []

        # Try 2: Fallback — search WITHOUT language filter if filtered search returned nothing
        # This ensures the user sees SOMETHING even if language codes don't match perfectly
        if not candidates:
            print(f"[retrieve] ⚠️  Filtered search returned 0 results. Trying UNFILTERED search...")
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=20,
                )
                candidates = self._format_candidates(results, language or "unknown")
                filter_used = "unfiltered (fallback)"
                print(f"[retrieve] Unfiltered search returned {len(candidates)} candidates")
            except Exception as e:
                print(f"[retrieve] Unfiltered search ALSO FAILED: {e}")
                return []

        vector_ms = (time.time() - tv) * 1000

        if not candidates:
            print(f"[retrieve] No candidates found at all. Index may be empty or query is too far from any chunk.")
            return []

        # 3. Re-rank with cross-encoder
        tr = time.time()
        pairs = [(query_text, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(rerank_scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        rerank_ms = (time.time() - tr) * 1000

        # 4. Prepare output
        top_candidates = candidates[:top_k]
        output = []
        for c in top_candidates:
            output.append({
                "text": c["text"],
                "score": round(c["vector_score"], 4),
                "strategy": c["strategy"],
                "language": c["language"],
                "source_query_id": c["source_query_id"],
            })

        total_ms = (time.time() - t0) * 1000
        print(f"[retrieve] embed={embed_ms:.1f}ms | vector_search={vector_ms:.1f}ms "
              f"| rerank={rerank_ms:.1f}ms | total={total_ms:.1f}ms | "
              f"filter={filter_used} | returned={len(output)}")

        return output

    def _format_candidates(self, results, default_lang: str) -> List[Dict]:
        """Format Chroma query results into candidate dicts."""
        candidates = []
        if not results or not results.get("ids") or not results["ids"]:
            return candidates
        ids_list = results["ids"][0] if results["ids"] else []
        if not ids_list:
            return candidates
        for i, doc_id in enumerate(ids_list):
            distance = float(results["distances"][0][i])
            vector_sim = 1.0 - distance  # Chroma cosine distance → similarity
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            if meta is None:
                meta = {}
            candidates.append({
                "_id": doc_id,
                "text": results["documents"][0][i],
                "vector_score": vector_sim,
                "strategy": meta.get("strategy", "unknown"),
                "language": meta.get("language", default_lang),
                "source_query_id": meta.get("source_query_id", ""),
            })
        return candidates

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def index_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check. Call this before building or querying."""
        count = self.collection.count()
        sample = self.collection.peek(limit=min(1000, count)) if count > 0 else None
        strategies = {}
        langs = {}
        if sample and sample.get("metadatas"):
            for meta in sample["metadatas"]:
                if meta:
                    s = meta.get("strategy", "unknown")
                    l = meta.get("language", "unknown")
                    strategies[s] = strategies.get(s, 0) + 1
                    langs[l] = langs.get(l, 0) + 1

        status = "healthy"
        if count == 0:
            status = "EMPTY — run build_index.py first"
        elif count < 100:
            status = "sparse — may miss many queries"

        return {
            "status": status,
            "total_chunks": count,
            "sample_strategies": strategies,
            "languages": langs,
            "db_path": self.db_path,
        }

    def quick_test(self, query: str = "What is the capital of India?", language: str = "en") -> None:
        """Run a single retrieval and print detailed results for debugging."""
        print(f"\n{'='*60}")
        print(f"QUICK TEST: query={query!r} | language={language}")
        print(f"{'='*60}")
        health = self.index_health_check()
        print(f"Index status: {health['status']}")
        print(f"Total chunks: {health['total_chunks']}")
        print(f"Languages in index: {list(health['languages'].keys())}")
        print()
        results = self.retrieve(query, language=language, top_k=3)
        if not results:
            print("❌ NO RESULTS RETURNED")
            print("\nPossible causes:")
            print("  1. Index is empty — did you run build_index.py?")
            print("  2. Language mismatch — check 'Languages in index' above")
            print("  3. Query is too far from any indexed chunk")
        else:
            print(f"✅ Returned {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"  #{i} [score={r['score']:.3f} | strategy={r['strategy']} | lang={r['language']}]")
                print(f"      {r['text'][:150]}...")
        print(f"{'='*60}\n")
