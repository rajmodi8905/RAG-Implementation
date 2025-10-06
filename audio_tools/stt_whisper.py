import whisper
import os
import streamlit as st

# MODIFIED: The function now accepts a container to write status messages to
@st.cache_resource
def load_whisper_model(model_name="base", _progress_container=None):
    """
    Loads a Whisper model and caches it, reporting progress to a container.
    """
    # Use the provided container to display messages
    if _progress_container:
        _progress_container.text(f"⏳ Loading Whisper model '{model_name}' for the first time...")
    else:
        print(f"Loading Whisper model '{model_name}' for the first time...")

    try:
        model = whisper.load_model(model_name)
        if _progress_container:
            _progress_container.text(f"✅ Whisper model '{model_name}' loaded.")
        else:
            print(f"Whisper model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        if _progress_container:
            _progress_container.error(f"Could not load Whisper model. Error: {e}")
        else:
            print(f"Error loading Whisper model: {e}")
        return None

# MODIFIED: The function now accepts and passes the progress container
def transcribe(audio_path: str, progress_container=None) -> str:
    """
    Transcribes speech from an audio file using the cached Whisper model.
    """
    # Pass the container to the loading function
    model = load_whisper_model(_progress_container=progress_container)

    if model is None:
        return "Error: Whisper model is not available."

    try:
        if progress_container:
            progress_container.text(f"🎤 Transcribing audio... (this may take a moment)")
            
        result = model.transcribe(audio_path)
        transcribed_text = result["text"]
        return transcribed_text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error during transcription: {e}"