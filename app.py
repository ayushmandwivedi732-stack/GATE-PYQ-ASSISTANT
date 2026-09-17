"""
app.py
------
Step 3: Streamlit UI so you can deploy this as a live web app
(e.g. on Hugging Face Spaces or Streamlit Community Cloud — both free).

Run locally:
    streamlit run app.py
"""

import streamlit as st
from query import retrieve_chunks, generate_answer

st.set_page_config(page_title="GATE PYQ Assistant", page_icon="📘")
st.title("📘 GATE PYQ Assistant")
st.caption("Ask a doubt from your ML / DBMS notes or past GATE papers.")

db_dir = "./chroma_db"

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Ask a question, e.g. 'Explain 3NF with an example'")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            try:
                chunks, sources = retrieve_chunks(question, db_dir)
                answer = generate_answer(question, chunks)
                st.markdown(answer)
                st.caption(f"Sources: {', '.join(set(sources))}")
                st.session_state.history.append(("assistant", answer))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info("Did you run `python ingest.py` first to build the database?")
