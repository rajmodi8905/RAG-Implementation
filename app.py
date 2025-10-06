# app.py

import streamlit as st
import time
import shutil
import os
from backend import (
    initialize_vectorstore, 
    process_uploaded_file, 
    get_rag_chain, 
    get_image_search_results,
    StreamlitCallbackHandler
)

# --- Session Initialization Block ---
if "session_started" not in st.session_state:
    st.session_state.session_started = True
    st.session_state.mode = "rag" # Default mode
    persist_directory = "chroma_db_streamlit"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    st.session_state.vectorstore = initialize_vectorstore()
    st.session_state.messages = []
    st.session_state.source_files = []
    print("New session started.")

# --- Page Configuration & CSS (no changes) ---
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="💡",
    layout="wide"
)
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: #1E1E1E;
        color: #FFFFFF;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2E2E2E;
    }
    /* Chat input box */
    .stTextInput > div > div > input {
        background-color: #2E2E2E;
        color: #FFFFFF;
        border: 1px solid #444;
    }
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    /* Uploaded file display */
    .uploaded-file-container {
        background-color: #2E2E2E;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-family: monospace;
    }
    div[data-testid="stHorizontalBlock"] img {
    max-height: 250px;      /* Set a maximum height for the images */
    object-fit: contain;    /* Preserve aspect ratio, don't stretch */
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 5px;
    margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.title("💡 Multimodal RAG Assistant")

# --- NEW: Mode switching UI ---
mode_selection = st.radio(
    "Select Mode:",
    ("Text (PDF/Audio) RAG", "Image Search"),
    horizontal=True,
    key='mode_selector'
)
# Update session state based on radio button
if mode_selection == "Text (PDF/Audio) RAG":
    st.session_state.mode = "rag"
else:
    st.session_state.mode = "image_search"

st.markdown("---")

sources_col, chat_col = st.columns([1, 2])

# --- Sources Column ---
with sources_col:
    st.header("📚 Sources")
    
    upload_type = ["pdf", "mp3", "wav", "m4a"] if st.session_state.mode == "rag" else ["jpg", "png", "jpeg"]
    upload_label = "Add Text/Audio Documents" if st.session_state.mode == "rag" else "Add Images to Search Library"
    
    uploaded_files = st.file_uploader(
        upload_label,
        type=upload_type,
        accept_multiple_files=True
    )

    if uploaded_files:
        # Check if a new file has been uploaded
        new_files = [f for f in uploaded_files if f.name not in [sf['name'] for sf in st.session_state.source_files]]
        for file in new_files:
            with st.spinner(f"Processing {file.name}..."):
                progress_bar = st.empty()
                status = process_uploaded_file(file, progress_bar)
                st.session_state.source_files.append({"name": file.name, "type": file.type})
                st.success(status)

    st.markdown("---")
    if st.session_state.source_files:
        st.subheader("Processed Files")
        for file_info in st.session_state.source_files:
            st.info(f"{'📄' if 'pdf' in file_info['type'] else '🔊' if 'audio' in file_info['type'] else '🖼️'} {file_info['name']}")

# --- Chat Column ---
with chat_col:
    st.header("💬 Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if content is a list of image paths
            if isinstance(message["content"], list):
                st.image(message["content"], width=200)
            else:
                st.markdown(message["content"])

    # Determine placeholder text based on mode
    placeholder = "Ask about your documents..." if st.session_state.mode == "rag" else "Describe an image to search for..."
    
    if prompt := st.chat_input(placeholder):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- MODIFIED: Conditional logic based on mode ---
        with st.chat_message("assistant"):
            if st.session_state.mode == "rag":
                # --- RAG Mode Logic (Streaming Text) ---
                stream_handler_container = st.empty()
                stream_handler = StreamlitCallbackHandler(stream_handler_container)
                rag_chain = get_rag_chain(st.session_state.vectorstore, stream_handler)
                response = rag_chain({"query": prompt})
                final_answer = response['result']
                # Add sources
                if 'source_documents' in response and response['source_documents']:
                    sources = "\n\n---\n**Sources:**\n"
                    for doc in response['source_documents']:
                        sources += f"- **{doc.metadata.get('source', 'Unknown')}**: \"*{doc.page_content[:100]}*...\"\n"
                    final_answer += sources
                stream_handler_container.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            elif st.session_state.mode == "image_search":
                # --- Image Search Mode Logic (Displaying Images) ---
                with st.spinner("Searching for matching images..."):
                    image_paths = get_image_search_results(prompt, k=1)
                    if image_paths:
                        st.success(f"Found {len(image_paths)} matching images:")
                        # Display images in columns
                        cols = st.columns(len(image_paths))
                        for i, path in enumerate(image_paths):
                            with cols[i]:
                                st.image(path, caption=os.path.basename(path), use_container_width=True)
                        # Save list of paths to message history for redisplay
                        st.session_state.messages.append({"role": "assistant", "content": image_paths})
                    else:
                        st.warning("No matching images found for your query.")
                        st.session_state.messages.append({"role": "assistant", "content": "No matching images found."})