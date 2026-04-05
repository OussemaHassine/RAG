import streamlit as st
import os
import tempfile
from pipeline import process_document
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
if "processed" not in st.session_state:
    st.session_state.processed = False

# ---- SIDEBAR: Configuration & Upload ----
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a Document", type="pdf")
    method = st.selectbox("Chunking Strategy", ["semantic", "recursive"])
    
    if uploaded_file and st.button("Build Knowledge Base"):
        # Use tempfile to prevent multi-user collisions on Hugging Face
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        with st.status("Processing document...") as status:
            st.write("Extracting text and chunking...")
            collection_name = f"{os.path.splitext(uploaded_file.name)[0]}_{method}"
            st.session_state.collection_name = collection_name
            
            # Run your pipeline
            process_document(tmp_path, collection_name=collection_name, method=method)
            
            st.session_state.processed = True
            status.update(label="Processing complete!", state="complete", expanded=False)
            os.remove(tmp_path) # Clean up the temp file

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.rerun()

# ---- CHAT INTERFACE ----
# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.processed:
    if query := st.chat_input("Ex: What are the termination clauses?"):
        with st.chat_message("user"):
            st.markdown(query)

        # 1. Update Memory
        st.session_state.messages, st.session_state.summary = update_memory(
            st.session_state.messages, st.session_state.summary
        )

        # 2. Retrieval with Status UI
        with st.spinner("Searching document..."):
            retrieved_chunks = retrieve(st.session_state.collection_name, query, top_k=5)

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