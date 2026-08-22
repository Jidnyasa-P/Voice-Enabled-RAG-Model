"""
Person 3 — Deployment (FIXED VERSION).

Streamlit app with:
  - Index health check displayed at startup
  - Clear error messages telling you WHY retrieval failed
  - Retrieval debug panel showing scores, chunk counts, filter used
  - Chunking strategy visibility

Run locally:
    streamlit run app.py

Deploy: push to Render / Railway / HuggingFace Spaces.
Set SARVAM_API_KEY as a platform secret.
The vector index (./vector_index/) must be built BEFORE deploying.
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

# Show index health at the top so you know immediately if something is wrong
health = engine.index_health_check()
if health["total_chunks"] == 0:
    st.error("🚨 INDEX IS EMPTY — Run `python build_index.py` first! Retrieval will fail for all queries.")
elif health["total_chunks"] < 100:
    st.warning(f"⚠️  Index only has {health['total_chunks']} chunks. Retrieval may miss many queries.")
else:
    st.success(f"✅ Index loaded: {health['total_chunks']} chunks across languages {list(health['languages'].keys())}")

audio = st.audio_input("Speak your question")

if audio:
    with st.spinner("Thinking..."):
        state = run_pipeline(engine, audio=audio.read())

    st.markdown(f"**Heard:** {state.transcript or '_(nothing transcribed)_'}")
    if state.language:
        st.caption(f"Detected language: `{state.language}`")

    if not state.query_ok:
        st.warning(state.reject_reason)
        # Show retrieval debug to help troubleshoot
        if state.retrieval_debug:
            with st.expander("🔧 Retrieval debug info"):
                st.json(state.retrieval_debug)
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
