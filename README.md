# 🎓 RAG Implementation - Complete Multimodal System

This repository contains a comprehensive multimodal RAG (Retrieval-Augmented Generation) system with two main components:

1. **📄 PDF RAG Chatbot** - Question-answering system for PDF documents using Ollama
2. **🔍 Face Recognition System** - Face matching and similarity detection using CLIP embeddings

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
├── audio_tools/ 
└──  stt_whispercpp.py            # Wrapper for Whisper.cpp (Speech → Text)
└──  tts_piper.py                 # Wrapper for Piper (Text → Speech)
└──README.md
└── sample_audio                  # Test input/output WAV files
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
