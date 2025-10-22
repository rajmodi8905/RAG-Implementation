# 🤖 Offline Multimodal RAG

> **Multimodal Retrieval-Augmented Generation (RAG)** — an offline desktop/edge-first pipeline for contextual question-answering over PDFs + audio, and natural-language image search. Fast, privacy-friendly, and designed to run locally using Ollama + local embedding/index stores.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Core Features](#-core-features)
- [Architecture & Components](#-architecture--components)
- [Requirements](#-requirements)
- [Quickstart / Setup](#-quickstart--setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works-high-level)
- [Contributing](#-contributing)
- [Acknowledgments & References](#-acknowledgments--references)

---

## 🎯 Project Overview

This repository implements a **local, offline multimodal RAG pipeline** that:

- 📄 Extracts and indexes text from **PDFs** and **transcribed audio**, storing chunks in a vector DB (Chromadb).
- 💬 Uses a local LLM (phi3:mini-128k via Ollama) to condition answer generation on relevant retrieved chunks.
- 🖼️ Provides **natural-language image querying**: describe an image in natural language and the top-k matching images are returned (using CLIP embeddings & local search).
- 🎨 Bundled with a Streamlit GUI for uploading assets and querying the system locally.

> 🔒 Designed for privacy-conscious environments and fast local experimentation without sending data to the cloud.

---

## ✨ Core Features

### 1. 📚 PDF + Audio Contextual QA
   - PDF text extraction → chunking → embed → store in Chromadb.
   - Audio → Whisper transcription → chunk → embed → store in Chromadb.
   - Query flow: retrieve relevant chunks → augment prompt → LLM generates answers (phi3:mini-128k via Ollama).

### 2. 🔍 Natural Language Image Search
   - Uses CLIP embeddings (downloaded locally) to embed images in `all_images` (configurable).
   - Search by natural-language prompt; returns top-k images ranked by cosine similarity.

---

## 🏗️ Architecture & Components

- 🖥️ **Frontend / UI**: Streamlit app (`app.py`) — upload PDFs/audio/images, run search/QA.
- 🧠 **Embeddings & Index**:
  - CLIP (local download via `download_clip.py`) for image embeddings.
  - Text embeddings and storage: **Chromadb** (local/embedded mode).
- 🎙️ **Speech-to-Text**: `openai-whisper` for audio transcription (local).
- 🤖 **Local LLM**: Ollama running `phi3:mini-128k` for response generation (local inference).
- 📊 **Indexing**: Chromadb for text embeddings indexing and FAISS for image embeddings indexing.  
- 💾 **Storage**:
  - Text embeddings & metadata are stored in chromadb files.
  - Image embeddings are stored in a bin file and metadata in an npy file.

---

## 📋 Requirements

- Python 3.10+ (recommended)
- `pip` + virtual environment
- ~600 MB free for CLIP model download
- Sufficient RAM for running Ollama + model

---

## 🚀 Quickstart / Setup

### 1. 📥 Clone repository

```bash
git clone https://github.com/rajmodi8905/RAG-Implementation.git
cd RAG-Implementation
```

### 2. 🐍 Create & activate Python venv

```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. 📦 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. ⬇️ Download CLIP model

```bash
python download_clip.py
```

This downloads the CLIP model (~577 MB) to `clip-model/`.

### 5. 🛠️ Install Ollama (local LLM runner)

Download and install Ollama following their official instructions:  
👉 [https://ollama.com/download](https://ollama.com/download)

**Example (macOS / Linux):**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 6. 🤖 Download / run the phi3:mini-128k model with Ollama

Once Ollama is installed you can pull/run the model locally. Example:

```bash
ollama run phi3:mini-128k
```

### 7. ▶️ Run the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` (or the URL Streamlit prints) to use the GUI.

---

## 💻 Usage

### 🎨 UI

- 📤 Upload PDFs, audio files (wav/mp3), or images via the Streamlit GUI.
- 📝 For PDFs/audio: the system will extract/transcribe and index.
- 🔎 For queries:
  - **Text QA**: ask a question — the app will retrieve relevant chunks from Chromadb, augment the LLM prompt, and return an answer.
  - **Image Search**: enter a natural-language description; top-k matching images (from `IMAGES_DIR`) are returned.

---

## ⚙️ Configuration

You can change these configurations in the `backend.py`:

- 📁 `IMAGES_DIR` — default `'all_images'` (change to your image folder's relative path).

---

## 🔄 How It Works (high level)

### 1. 📥 Ingestion

- **PDFs**: Extract text (e.g., `pdfminer`/`PyMuPDF`), clean, chunk (overlap + sliding window), embed each chunk, store embeddings and metadata in Chromadb.
- **Audio**: Transcribe with `openai-whisper` locally, chunk transcript, embed and store in Chromadb.
- **Images**: Compute CLIP image embeddings and persist to a local index (Chromadb or FAISS).

### 2. 🔍 Retrieval

On user query, compute embedding for the query (text → embedding) and perform k-NN search in the vector DB to retrieve the most relevant chunks/images.

### 3. 🎯 Generation

Build an augmented prompt that includes retrieved chunks and the user question; send to local LLM (phi3:mini-128k running on Ollama) to generate an answer.

### 4. ✅ Return

Present LLM output and provenance (e.g., which PDF or audio chunk(s) the answer used — ensure you surface source metadata where helpful).

---

## 🤝 Contributing

Thank you for considering contributing! Add step-by-step guidance:

1. 🍴 Fork the repository.
2. 🌿 Create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
3. ✏️ Make changes and add tests where applicable.
4. 📦 Ensure `requirements.txt` is updated if dependencies change.
5. 🔀 Create a pull request with a clear description of the change and motivation.

---

## 🙏 Acknowledgments & References

- **Ollama** — local LLM runner & model registry  
  👉 [https://ollama.com](https://ollama.com)
- **Phi-3 model** — small and efficient 128k context window LLM used for generation
- **OpenAI Whisper** — used for offline speech-to-text transcription  
  👉 [https://github.com/openai/whisper](https://github.com/openai/whisper)
- **CLIP (OpenAI)** — used for image-text embeddings  
  👉 [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

---

<div align="center">
Made with ❤️ by Team Phoenix
</div>
