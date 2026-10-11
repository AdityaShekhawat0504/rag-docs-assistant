"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from rag.qa import ask

st.set_page_config(page_title="RAG Docs Assistant", page_icon="📄")
st.title("📄 RAG Docs Assistant")
st.caption("Answers only from your PDFs, with page citations and a grounding check.")

q = st.text_input("Ask a question about the indexed documents")
if q:
    with st.spinner("Retrieving and answering..."):
        a = ask(q)
    st.markdown(a.text)
    if a.sources:
        with st.expander("Sources"):
            for n, src, page, score in a.sources:
                st.write(f"[{n}] {src}, p. {page} (relevance {score})")
    if a.flags:
        st.warning("Grounding check flagged:\n\n" + "\n".join(f"- {w}: {s}" for w, s in a.flags))
    elif not a.refused:
        st.success("Grounding check passed: every sentence is cited and supported.")
