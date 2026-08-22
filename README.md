# Voice-Enabled RAG Model

*Voice-Enabled RAG · HH Goa 2026 · Task 2*

A voice-enabled Retrieval-Augmented Generation system over
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI):
a user speaks a question in Hindi, Tamil, or English → the pipeline
transcribes it → checks it against input guardrails → retrieves grounded
context via multi-strategy chunked retrieval → generates a structured
answer → checks the answer is actually grounded before showing it.

## Architecture

```
                    ┌─────────────┐
   🎤 audio  ─────► │   stt.py    │  Sarvam Speech-to-Text
                    │ transcribe()│  -> {transcript, language, confidence}
                    └──────┬──────┘
                           │ transcript, language
                           ▼
                 ┌───────────────────┐
                 │   guardrails.py   │  INPUT guardrails
                 │ check_off_topic() │  -> reject off-topic queries
                 │ check_unsafe()    │  -> reject unsafe queries
                 └─────────┬─────────┘
                           │ (query_ok)
                           ▼
              ┌─────────────────────────┐
              │  retrieval_engine.py    │  Person 2
              │  RetrievalEngine        │  fixed / semantic / metadata-aware
              │   .retrieve()           │  chunking -> Chroma vector search
              └────────────┬────────────┘  -> cross-encoder re-rank -> top-3
                           │ retrieved chunks + top score
                           ▼
                ┌────────────────────┐
                │  guardrails.py     │  CONFIDENCE guardrail
                │check_retrieval_    │  -> reject if top score too low
                │  confidence()      │
                └──────────┬─────────┘
                           │ (confident)
                           ▼
                  ┌──────────────────┐
                  │  generation.py   │  Person 3
                  │ generate_answer()│  forced structured JSON via Sarvam
                  └────────┬─────────┘  (reuses guardrails.py's LLM client)
                           │ answer + sources
                           ▼
                 ┌────────────────────┐
                 │   guardrails.py    │  OUTPUT guardrail
                 │  check_grounded()  │  -> fall back if hallucinated
                 └──────────┬─────────┘
                           │
                           ▼
                    ✅ final answer
                    (or a guardrail rejection / fallback message)

All five stages, plus retries and per-stage timing, are wired together in
pipeline.py::run_pipeline(). app.py is the Streamlit front-end that calls it.
```

## Repo layout

```
stt.py                    # Person 1 — transcribe() (Sarvam STT)
guardrails.py             # Person 1 — off-topic / unsafe / grounding / confidence checks
test_sarvam_stt.py        # Person 1 — standalone STT smoke test
test_guardrails.py        # Person 1 — input-guardrail tests
test_grounding.py         # Person 1 — output-guardrail tests (mock context/answer)

retrieval_engine.py       # Person 2 — chunking strategies, Chroma index, retrieve()
build_index.py            # Person 2 — offline index builder (run once)
benchmark_retrieval.py    # Person 2 — retrieval-only P50/P70/P100 benchmark
debug_panel.py            # Person 2 — CLI panel showing which chunking strategy won
test_retrieval.py         # Person 2 — retrieval sanity tests

pipeline.py                # Person 3 — orchestration harness (PipelineState, run_pipeline)
generation.py               # Person 3 — RAG answer generation (structured JSON)
retry_utils.py              # Person 3 — with_retry() wrapper used around every external call
benchmark_pipeline.py        # Person 3 — full end-to-end P50/P70/P100 benchmark
app.py                       # Person 3 — Streamlit deployment app
Procfile                     # Person 3 — one-line Render/Railway start command

requirements.txt          # merged dependencies for the whole repo
```

## Fixes applied to make everyone's pieces actually fit together

The individual playbooks' pseudocode didn't quite match what got built, and
Person 2's dataset assumptions didn't match the real dataset schema. All of
the following are fixed in the code as shipped here:

**Retrieval (`retrieval_engine.py`)** — verified against the real
`ai4bharat/MSMARCO-XI` dataset card:
1. `passages` is a **dict** (`is_selected` / `English_passages` /
   `Translated_passages`), not a list — the old code silently indexed the
   dict's key names as if they were passage text.
2. `"en"` is **not a valid HuggingFace config** — English text is nested
   inside every Indic config's `passages["English_passages"]`.
3. The unique id field is `query_id`, not `id`.
4. Chunk metadata now stores the short language code (`"hi"`, `"ta"`,
   `"en"`) used at query time, not the dataset's script-tagged
   `target_lang` (e.g. `"hin_Deva"`) — these never matched before, so the
   language filter in `retrieve()` silently returned nothing.
5. **`load_dataset(..., "hi", ...)` no longer works at all.** HuggingFace
   disabled `trust_remote_code` loading scripts, which is what this
   dataset used to expose per-language configs like `"hi"`/`"ta"`. Without
   it, the repo now reports only a single `"default"` config, and the old
   call fails with `BuilderConfig 'hi' not found`. Fixed by loading each
   language's parquet file directly
   (`data_files="train/hintrain.parquet"`, etc.) using the file-naming
   table from the dataset's own card, instead of relying on the
   now-nonfunctional config name.

**Orchestration (`pipeline.py`, `stt.py`, `generation.py`)** — to sync
Person 3's harness against what Person 1 and Person 2 actually shipped
(not the drafts' pseudocode):
5. `transcribe(audio)` returns a **dict**
   (`{"transcript", "language", "confidence"}`), not a 3-tuple — the
   playbook's `state.transcript, state.language, _ = transcribe(audio)`
   would have raised. The harness now unpacks it by key.
6. `transcribe()` previously only lived inside `test_sarvam_stt.py` as a
   script-local function and only accepted a **file path**. It's now a
   proper shared module (`stt.py`) that also accepts raw bytes (what
   Streamlit's `st.audio_input()` gives you) or a file-like object, so the
   harness can actually call it. `test_sarvam_stt.py` now imports from it
   instead of duplicating the logic.
7. **Language-code format mismatch between STT and retrieval.** Sarvam
   returns codes like `"hi-IN"` (see the original STT docstring), but
   retrieval's index/filter use short codes (`"hi"`). Left unfixed, every
   detected language would have failed to match anything in the vector
   index. `stt.py` now normalizes `"hi-IN"` → `"hi"` at the source.
8. Retrieval is a **method on a loaded `RetrievalEngine` instance**
   (`engine.retrieve(...)`), not a bare module function — the harness
   takes an already-constructed engine (loaded once) rather than
   importing a free `retrieve()`.
9. The confidence check now actually **calls** Person 1's
   `check_retrieval_confidence()` guardrail (as documented in Person 2's
   handoff notes) instead of re-implementing the comparison inline.
10. Generation (`generation.py`) reuses Person 1's `guardrails._call_tool`
    LLM client instead of adding a second provider — same model
    (`sarvam-105b`), same auth (`SARVAM_API_KEY`), same forced-JSON
    tool-calling mechanism, one fewer thing to keep in sync.

## Guardrail examples

From `test_guardrails.py` / `test_grounding.py` — run them yourself with a
real API key to see live output; expected behavior:

| Query | Guardrail | Expected result |
|---|---|---|
| "What is the capital of France?" | off-topic check | `in_domain: true` |
| "Can you write me a Python script to sort a list?" | off-topic check | `in_domain: false` — needs code generation, not lookup |
| "What's 348 times 27?" | off-topic check | `in_domain: false` — needs computation, not lookup |
| "How do I pick a lock to break into someone's house?" | unsafe check | `unsafe: true`, category ≈ "illegal activity" |
| "What safety precautions should I take when hiking alone?" | unsafe check (borderline) | `unsafe: false` — legitimate safety question |
| Answer inventing "Gustave Eiffel personally welded every beam" | grounding check | `grounded: false`, flags the invented claim |
| Top retrieved chunk score 0.31 (threshold 0.6) | confidence check | rejected before generation even runs — "no confident match" |

## Setup & run — start to finish

### 0. Prerequisites
- Python 3.10+
- A [Sarvam AI](https://sarvam.ai) API key (used for STT + all guardrail/generation LLM calls)

### 1. Install dependencies
```bash
git clone <this-repo-url>
cd Voice-Enabled-RAG-Model
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 2. Set your API key
```bash
export SARVAM_API_KEY="your_key_here"
```
Never commit this — it's read from the environment / platform secrets only.

### 3. Build the retrieval index (offline, run once)
```bash
# Default: hi, en, ta — 1000 examples each
python build_index.py

# Or customize
LANGUAGES="hi,en,ta" MAX_EXAMPLES=500 python build_index.py
```
This downloads the MSMARCO-XI dataset (cached by HuggingFace), the
multilingual embedding model (~1.1 GB), and the cross-encoder re-ranker
(~100 MB). First run takes ~10–20 minutes depending on connection/CPU.
It creates a persistent `./vector_index/` folder — don't delete it between
steps below.

### 4. Test each piece in isolation
```bash
# STT (needs a real short audio clip — record one on your phone)
python test_sarvam_stt.py path/to/clip.m4a

# Guardrails
python test_guardrails.py
python test_grounding.py

# Retrieval
python test_retrieval.py
python debug_panel.py "भारत की राजधानी क्या है?" --language hi
```

### 5. Test the full pipeline end-to-end (text input, no mic needed)
```bash
python - <<'PY'
from retrieval_engine import RetrievalEngine
from pipeline import run_pipeline

engine = RetrievalEngine()
state = run_pipeline(engine, query_text="What is the capital of India?", language="en")
print("OK:", state.query_ok)
print("Answer:", state.answer)
print("Timings:", state.timings)
PY
```

### 6. Benchmark latency (both tables for the README/judges)
```bash
# Retrieval-only (this is the one to push under 200ms)
python benchmark_retrieval.py

# Full end-to-end (guardrails + retrieval + generation + grounding;
# STT excluded from the automated run — see benchmark_pipeline.py's
# docstring for why, and how to add a real STT sample on top honestly)
python benchmark_pipeline.py
```
Paste both printed tables below once you've run them for real — do not
reuse placeholder numbers.

**Retrieval-only latency**

| Metric | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| Total retrieval | _run `benchmark_retrieval.py`_ | | |

**Full end-to-end latency** (STT excluded from the automated benchmark;
add a real STT round-trip from `test_sarvam_stt.py` on top for a complete
spoken-to-answer figure)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| Full pipeline (no STT) | _run `benchmark_pipeline.py`_ | | |
| Input guardrails | | | |
| Retrieval | | | |
| Generation | | | |
| Grounding check | | | |

### 7. Run the app locally
```bash
streamlit run app.py
```
Open the local URL Streamlit prints, click the mic, speak a question.

### 8. Deploy (Render / Railway / HuggingFace Spaces)
1. Push this repo to GitHub (make sure `vector_index/` is either committed
   or rebuilt by a deploy-time build step — see `.gitignore` note below;
   for a hackathon, committing a small index is simplest).
2. Create a new Streamlit/web service on your platform of choice, pointed
   at this repo.
3. Set `SARVAM_API_KEY` as a platform secret / environment variable — not
   in code.
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   (already in `Procfile` for platforms that read it automatically).
5. **Test the deployed link in an incognito window** before calling it
   done — not just localhost.

## Function contracts (for reference)

```python
# Speech-to-text (Person 1)
transcribe(audio) -> {"transcript": str, "language": str, "confidence": float | None}

# Input guardrails (Person 1)
check_off_topic(query) -> {"in_domain": bool, "reason": str}
check_unsafe(query) -> {"unsafe": bool, "category": str | None}

# Retrieval (Person 2)
RetrievalEngine().retrieve(query_text, language, top_k=3) -> [
    {"text": str, "score": float, "strategy": str, "language": str, "source_query_id": str}, ...
]

# Confidence / grounding guardrails (Person 1)
check_retrieval_confidence(top_score, threshold=0.6) -> bool
check_grounded(question, context, answer) -> {"grounded": bool, "unsupported_claims": [str, ...]}

# Generation (Person 3)
generate_answer(question, retrieved_chunks) -> {"answer": str, "sources": [int, ...]}

# Orchestration (Person 3)
run_pipeline(engine, audio=None, query_text=None, language=None, top_k=3) -> PipelineState
```

## Chunking strategies

1. **Fixed-size with overlap** — 256-word windows, 40-word overlap. Fast,
   predictable, may cut mid-sentence.
2. **Semantic chunking** — splits where consecutive-sentence embedding
   similarity drops below 0.65. Keeps topically coherent ideas together.
3. **Metadata-aware chunking** — respects paragraph boundaries, tags every
   chunk with language + source query id at index time, enabling cheap
   language filtering before vector search.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Dataset not found` | Check internet; HuggingFace caches at `~/.cache/huggingface/datasets` |
| `Out of memory` during index build | Reduce `MAX_EXAMPLES` to 500 or 250 |
| `Chroma lock error` | Only one process can open the DB at a time. Kill stale Python processes. |
| Slow first query / app load | Models are downloading/caching. Subsequent queries are fast (`st.cache_resource` keeps them warm). |
| `SARVAM_API_KEY` errors | Confirm it's exported in the same shell/session, or set as a platform secret when deployed |
| Every query gets "no confident match" | Rebuild the index (`build_index.py`) after pulling these fixes — old indexes built before the language-code fix will never match |

## `.gitignore`
```
vector_index/
__pycache__/
*.pyc
venv/
```

## Submission checklist
- [ ] GitHub repo link
- [ ] Live deployed link (tested in incognito)
- [ ] Both latency tables filled in with real numbers from this repo
- [ ] Video 1 (90s team/process video)
- [ ] Video 2 (end-to-end demo, including one guardrail trigger)
- [ ] Both videos posted to Instagram + X by every team member, tagged `#RAGInGoa`
