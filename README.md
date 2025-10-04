# PDF RAG Bot with Ollama

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
