import streamlit as st
import os
from pipeline import process_document
from retrieval import retrieve
from memory import update_memory, prompt_with_memory
from prompt import generate_response_stream

st.title("Legal Document RAG")

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# ---- SIDEBAR ----
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    method = st.selectbox("Chunking method", ["semantic", "recursive"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            collection_name = f"{os.path.splitext(uploaded_file.name)[0]}_{method}"
            st.session_state.collection_name = collection_name
            process_document("temp.pdf", collection_name=collection_name, method=method)
            st.session_state.processed = True
            st.success("Document processed! You can now ask questions.")

# ---- CHAT HISTORY ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---- CHAT INPUT ----
if st.session_state.processed:
    if query := st.chat_input("Ask a question about the document"):
        # show user message
        with st.chat_message("user"):
            st.write(query)

        # update memory
        st.session_state.messages, st.session_state.summary = update_memory(
            st.session_state.messages, st.session_state.summary
        )

        # retrieve
        retrieved_chunks = retrieve(st.session_state.collection_name, query, top_k=5)

        # build prompt
        messages = prompt_with_memory(query, retrieved_chunks, st.session_state.summary, st.session_state.messages)

        # stream response
        with st.chat_message("assistant"):
            response = st.write_stream(generate_response_stream(messages))

        # show sources
        with st.expander("Sources"):
            for i, chunk in enumerate(retrieved_chunks):
                st.write(f"**{i+1}.** {chunk}")

        # append to memory
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload and process a document first.")