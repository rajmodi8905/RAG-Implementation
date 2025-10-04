# 🎓 SIH Hackathon - Multi-Project Repository

This repository contains two AI-powered projects:

1. **📄 PDF RAG Chatbot** - Question-answering system for PDF documents
2. **🔍 Multimodal Face Recognition System** - Face matching and similarity detection

---

# 📄 Project 1: PDF RAG Bot with Ollama

A Retrieval-Augmented Generation (RAG) chatbot that reads PDF files and answers questions based on their content using the Phi3 mini model via Ollama.

## Setup Complete ✅

Your virtual environment is set up with all necessary packages!

## How to Use

### 1. Add Your PDF Files
Place your PDF files in the `data/` directory that has been created for you.

### 2. Run the Notebook Cells in Order

**Cell 1:** Import all required libraries
- Imports PDF loaders, embeddings, vector stores, and LLM components

**Cell 2:** Comment cell (no action needed)

**Cell 3:** Initialize Ollama LLM
- Make sure you have Ollama installed and running
- Pull the model: `ollama pull phi3:mini-128k`
- Uses phi3:mini-128k model with 128k context window

**Cell 4:** Define the RAG prompt template
- Sets up the system prompt for retrieval-augmented generation
- Ensures the model only uses context from your PDFs

**Cell 5:** Load PDF data
- Option 1: Load a single PDF by uncommenting and setting the path
- Option 2: Load all PDFs from the `data/` directory (default)

**Cell 6-7:** Split documents into chunks
- Chunks documents with 1000 characters per chunk
- 500 character overlap to maintain context

**Cell 8-9:** View the chunks
- Check how many chunks were created
- View the first chunk

**Cell 10:** Create embeddings
- Uses HuggingFace's `all-MiniLM-L6-v2` model for embeddings
- Creates a FAISS vector store for fast similarity search

**Cell 11:** Save the vector index
- Saves the index to `vector_index.pkl` for reuse

**Cell 12:** Load the vector index
- Loads the saved index (useful for subsequent runs)

**Cell 13:** Create the retrieval chain
- Sets up the question-answering chain with sources

**Cell 14:** Define query function
- Helper function to query the RAG system

**Cell 15:** Test your RAG bot!
- Ask questions about your PDF content
- Modify the question to ask about your specific documents

## Example Usage

```python
# After running all setup cells, you can ask questions like:
print(gen("What is the main topic of the document?"))
print(gen("Summarize the key findings"))
print(gen("What are the conclusions?"))
```

## Requirements

- Python 3.13.2
- Ollama with phi3:mini-128k model
- All Python packages are installed in the virtual environment

## Directory Structure

```
sih hackathon/
├── data/                    # Place your PDF files here
├── venv/                    # Virtual environment
├── retrieval.ipynb          # Main notebook
├── requirements.txt         # Python dependencies
├── vector_index.pkl         # Generated after running the notebook
└── README.md               # This file
```

## Notes

- The first run will download the embedding model (~90MB)
- Processing time depends on the size and number of PDFs
- The vector index is saved locally for faster subsequent runs
- Make sure Ollama is running before executing the notebook

---

# 🔍 Project 2: Multimodal Face Recognition System

A complete face recognition and similarity matching system that extracts faces from PDFs, associates them with names, and enables intelligent face matching using CLIP embeddings.

## 🚀 Features

- **PDF Face Extraction**: Automatically extracts face images from PDF documents
- **Smart Name Matching**: Uses spatial analysis to associate names with faces
- **AI-Powered Similarity**: CLIP-based embedding for accurate face matching  
- **Offline Operation**: Works completely offline once set up
- **Multiple Interfaces**: Interactive and simple command-line interfaces
- **Intelligent Filtering**: Removes non-face images and academic text

## 📁 Project Structure

```
RAG-Implementation/
├── image_matcher.py              # Interactive face similarity matcher
├── simple_image_matcher.py       # Simple drag-and-drop interface  
├── spatial_name_matcher.py       # PDF spatial analysis for name-photo matching
├── extract_images_from_pdf.py    # PDF image extraction with face filtering
├── download_model.py             # CLIP model downloader
├── smart_name_extractor.py       # Name extraction from PDF
├── test_images/                  # Face database (251 named + 120 unnamed)
├── models/                       # Local CLIP model (download required)
└── requirements.txt              # Python dependencies
```

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/rajmodi8905/RAG-Implementation.git
cd RAG-Implementation
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch transformers pillow opencv-python PyMuPDF pytesseract pdf2image
```

### 4. Download CLIP Model
```bash
python download_model.py
```
This downloads the CLIP model (~577MB) to `models/clip-vit-base-patch32/`

### 5. Add Your PDF (Optional)
Place your PDF document in the root directory to extract faces and names.

## 🎯 Usage

### Face Similarity Matching

**Interactive Mode:**
```bash
python image_matcher.py
```

**Simple Mode:**
```bash
python simple_image_matcher.py path/to/your/query/image.jpg
```

### Processing New PDFs

**Extract Images from PDF:**
```bash
python extract_images_from_pdf.py
```

**Match Names with Faces:**
```bash
python spatial_name_matcher.py
```

## 📊 Performance

- **Database Size**: 371 face images (251 named, 120 unnamed)
- **Similarity Accuracy**: CLIP-based cosine similarity
- **Processing Speed**: ~50 images/second on modern hardware
- **Memory Usage**: ~2GB RAM (including model)

## 🎨 Confidence Levels

- 🟢 **Excellent (>0.85)**: Very likely the same person
- 🟡 **Good (>0.70)**: Likely the same person  
- 🟠 **Moderate (>0.55)**: Could be the same person
- 🔴 **Low (<0.55)**: Likely different people

## 🔧 Technical Details

- **Model**: OpenAI CLIP ViT-Base-Patch32
- **Face Detection**: OpenCV Haar Cascades
- **PDF Processing**: PyMuPDF (fitz)
- **Embeddings**: 512-dimensional CLIP features
- **Similarity**: Cosine similarity matching

## 📚 Example Results

```
🎯 BEST MATCH FOUND!
Query Image: my_photo.jpg
Matched Person: Raj Amit Modi
Similarity Score: 0.8543
🟢 EXCELLENT - Very likely the same person!
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI CLIP for powerful vision-language embeddings
- OpenCV for face detection capabilities
- PyMuPDF for PDF processing

---

## 📦 Repository Contents

### PDF RAG Chatbot Files:
- `chatbot.py` - Interactive PDF question-answering chatbot
- `retrieval.ipynb` - Jupyter notebook with RAG implementation
- `requirements.txt` - Python dependencies
- `CHATBOT_IMPROVEMENTS.md` - Documentation of chatbot enhancements

### Face Recognition Files:
- `image_matcher.py` - Interactive face similarity matcher
- `simple_image_matcher.py` - Simple drag-and-drop interface
- `spatial_name_matcher.py` - PDF spatial analysis
- `extract_images_from_pdf.py` - PDF image extraction
- `test_images/` - Face database (370+ images)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
