"""
Person 3 — Deployment.

Streamlit app wiring the full pipeline (STT -> guardrails -> retrieval ->
generation -> grounding) together, with a chunking-strategy debug panel so
judges can see which strategy/language won for each answer.

Run locally:
    streamlit run app.py

Deploy: push this repo to Render / Railway / HuggingFace Spaces (Streamlit
runtime) and set SARVAM_API_KEY as a platform secret / env var — never
hardcode it. The vector index (./vector_index/) must be built and
committed (or built at deploy time via build_index.py) before deploying;
this app does not build the index itself, it only loads it.
"""

import streamlit as st

from retrieval_engine import RetrievalEngine
from pipeline import run_pipeline

st.set_page_config(page_title="Voice RAG — Ask in your language", page_icon="🎙️")
st.title("🎙️ Voice RAG — Ask in your language")
st.caption("Speak a question in Hindi, Tamil, or English. Answers are grounded in MSMARCO-XI passages.")


@st.cache_resource(show_spinner="Loading models & index (first load only)...")
def get_engine():
    return RetrievalEngine()


engine = get_engine()

audio = st.audio_input("Speak your question")

if audio:
    with st.spinner("Thinking..."):
        state = run_pipeline(engine, audio=audio.read())

    st.markdown(f"**Heard:** {state.transcript or '_(nothing transcribed)_'}")
    if state.language:
        st.caption(f"Detected language: `{state.language}`")

    if not state.query_ok:
        st.warning(state.reject_reason)
    else:
        st.success(state.answer)
        if not state.grounded:
            st.caption("⚠️ This answer did not pass the grounding check.")

        with st.expander("🔍 Sources & chunking strategy used"):
            for i, c in enumerate(state.retrieved, 1):
                used = " ✅ cited" if i in (state.sources or []) else ""
                st.caption(
                    f"[{c['strategy']} | score={c['score']:.3f} | lang={c['language']}]{used}\n\n{c['text'][:300]}"
                )

    with st.expander("⏱️ Stage timings (ms)"):
        st.json(state.timings)
        st.caption(f"Total: {sum(state.timings.values()):.1f} ms")

st.divider()
st.caption("Voice-Enabled RAG · HH Goa 2026 · #RAGInGoa")
