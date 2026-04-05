import streamlit as st
import os
import tempfile
from pipeline import process_document, COLLECTION_NAME
from retrieval import retrieve
from memory import update_memory, prompt_with_memory
from prompt import generate_response_stream

st.set_page_config(page_title="Document RAG", layout="wide")
st.title("Document RAG")

# ---- SESSION STATE INITIALIZATION ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ---- SIDEBAR: Configuration & Upload ----
with st.sidebar:
    st.header("Upload Document")
    uploaded_files = st.file_uploader("Upload Documents", type="pdf", accept_multiple_files=True)
    method = st.selectbox("Chunking Strategy", ["semantic", "recursive"])

    if uploaded_files and st.button("Build Knowledge Base"):
        with st.status("Processing documents...") as status:
            for uploaded_file in uploaded_files:
                try:
                    st.write(f"Processing {uploaded_file.name}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    try:
                        process_document(tmp_path, method=method)
                        if uploaded_file.name not in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.append(uploaded_file.name)
                    finally:
                        os.remove(tmp_path)
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
            status.update(label="Processing complete!", state="complete", expanded=False)

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Knowledge Base")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- {fname}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.rerun()

# ---- CHAT INTERFACE ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.uploaded_files:
    if query := st.chat_input("Ex: What are the termination clauses?"):
        with st.chat_message("user"):
            st.markdown(query)

        # 1. Update Memory
        st.session_state.messages, st.session_state.summary = update_memory(
            st.session_state.messages, st.session_state.summary
        )

        # 2. Retrieval with Status UI
        with st.spinner("Searching document..."):
            retrieved_chunks = retrieve(COLLECTION_NAME, query, top_k=5)

        # 3. Prompt Construction
        messages = prompt_with_memory(query, retrieved_chunks, st.session_state.summary, st.session_state.messages)

        # 4. Streaming Response
        with st.chat_message("assistant"):
            response = st.write_stream(generate_response_stream(messages))

        # 5. Display Sources neatly
        with st.expander("View Evidence (Sources)"):
            for i, chunk in enumerate(retrieved_chunks):
                st.info(f"**Source {i+1}:**\n\n{chunk}")

        # Update Session State
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👋 Welcome! Please upload a PDF in the sidebar to begin the analysis.")
