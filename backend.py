# backend.py

import os
import time
import shutil
import tempfile
import streamlit as st
import torch
from transformers import CLIPModel, CLIPProcessor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama 
from IMGModule.indexing import index_images
from IMGModule.search import perform_semantic_search
from audio_tools.stt_whisper import transcribe

# --- DEFINE CONSTANTS FOR IMAGE MODULE ---
IMAGES_DIR = 'all_images'
INDEX_DIR = 'ImgIndexStore'
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'faiss_image_index.bin')
METADATA_PATH = os.path.join(INDEX_DIR, 'image_metadata.npy')
LOCAL_MODEL_PATH = 'clip-model'
REMOTE_MODEL_NAME = "openai/clip-vit-base-patch32"

# --- NEW: Cached function to load the CLIP model ---
@st.cache_resource
def load_clip_model():
    """
    Loads and caches the CLIP model and processor.
    This function should NOT have any Streamlit UI calls inside it.
    """
    print("Attempting to load CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = CLIPModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
        processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_PATH)
        print("✅ CLIP model loaded successfully from local path.")
    except OSError:
        print(f"⚠️ Model not found locally. Downloading '{REMOTE_MODEL_NAME}'...")
        model = CLIPModel.from_pretrained(REMOTE_MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(REMOTE_MODEL_NAME)
        model.save_pretrained(LOCAL_MODEL_PATH)
        processor.save_pretrained(LOCAL_MODEL_PATH)
        print("✅ CLIP model downloaded and saved.")
        model.to(device)
    return model, processor, device

# (StreamlitCallbackHandler, initialize_vectorstore are unchanged)
class StreamlitCallbackHandler(BaseCallbackHandler):
    # ... (no changes here) ...
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def initialize_vectorstore():
    # ... (no changes here) ...
    print("Initializing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "chroma_db_streamlit"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("Vector store initialized.")
    return vectorstore

# --- MODIFIED: process_uploaded_file now handles images ---
def process_uploaded_file(uploaded_file, progress):
    """Processes any uploaded file and routes to the correct handler."""
    file_type = uploaded_file.type
    
    # Handle Images
    if "image" in file_type:
        return process_image_upload(uploaded_file, progress)
    
    # Handle PDF/Audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        vectorstore = st.session_state.vectorstore
        if file_type == "application/pdf":
            documents = process_pdf(tmp_file_path, progress)
        elif "audio" in file_type:
            documents = process_audio(tmp_file_path, progress)
        else:
            return f"Unsupported file type: {file_type}"
        
        if documents:
            progress.text("🧠 Creating text embeddings...")
            vectorstore.add_documents(documents)
            vectorstore.persist()
            time.sleep(1)
            return f"✅ Text document '{uploaded_file.name}' processed."
    finally:
        os.remove(tmp_file_path)

# --- NEW: Function to handle image saving and indexing ---
def process_image_upload(uploaded_file, progress):
    """Saves the uploaded image and updates the FAISS index."""
    progress.text(f"Saving image: {uploaded_file.name}...")
    
    # Save the image to the 'all_images' directory
    save_path = os.path.join(IMAGES_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    time.sleep(1)

    # --- UI UPDATE: Inform the user before loading the model ---
    progress.text("Loading image processing model...")
    model, processor, device = load_clip_model() # Call the clean, cached function
    progress.text("✅ Model loaded.")
    time.sleep(1) # For UX

    # Update the image index
    progress.text("🧠 Indexing image...")
    index_images(
        model=model,
        processor=processor,
        device=device,
        images_dir=IMAGES_DIR,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH
    )
    return f"✅ Image '{uploaded_file.name}' saved and indexed."

# --- NEW: Function to perform image search ---
def get_image_search_results(query, k=1):
    """Performs semantic search and returns a list of image paths."""
    model, processor, device = load_clip_model() # Load without progress bar
    
    retrieved_filenames = perform_semantic_search(
        query=query,
        k=k,
        model=model,
        processor=processor,
        device=device,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH
    )
    
    # Return full paths to the images for display
    return [os.path.join(IMAGES_DIR, fname) for fname in retrieved_filenames]

# (process_pdf, process_audio, and get_rag_chain are unchanged)
def process_pdf(file_path, progress):
    # ... (no changes here) ...
    progress.text(f"⏳ Loading PDF: {os.path.basename(file_path)}...")
    time.sleep(1) # For UX
    loader = PyPDFLoader(file_path)
    data = loader.load()
    progress.text("✂️ Splitting document into chunks...")
    time.sleep(1) # For UX
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    return docs

def process_audio(file_path, progress):
    # ... (no changes here) ...
    transcribed_text = transcribe(file_path, progress_container=progress)
    if "error" in transcribed_text.lower() or not transcribed_text.strip():
        progress.error(f"Transcription failed for {os.path.basename(file_path)}. Details: {transcribed_text}")
        return None
    progress.text("✅ Transcription complete. Creating document from transcript...")
    time.sleep(1)
    from langchain.docstore.document import Document
    data = [Document(page_content=transcribed_text, metadata={"source": os.path.basename(file_path)})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    return docs

# backend.py

# ... (rest of the file remains the same) ...

def get_rag_chain(vectorstore, stream_handler):
    """Initializes and returns a streaming RAG chain using Ollama."""
    
    llm = ChatOllama(
        model="phi3:mini-128k",
        temperature=0.7,
        streaming=True,
        callbacks=[stream_handler]
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful RAG assistant. Answer the user's question based ONLY on the context provided. If you don't know the answer, say you don't have enough information. Do not make things up."""),
        ("user", """
        Context:
        {context}

        Question: 
        {question}
        """),
    ])
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    
    return chain