"""
PDF RAG Chatbot - Interactive Command Line Interface
A complete chatbot that answers questions from your PDF documents using RAG
"""

import os
import pickle
import time
import langchain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler


# Custom streaming callback
class StreamCallback(BaseCallbackHandler):
    """Custom callback handler for streaming output."""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Print each new token as it's generated."""
        print(token, end="", flush=True)


def load_or_create_vectorstore():
    """Load vector store from disk or create new one from PDFs."""
    file_path = "vector_index.pkl"
    
    # Try to load existing vector store
    if os.path.exists(file_path):
        print("📦 Loading existing vector store...")
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
        print("✅ Vector store loaded!\n")
        return vectorIndex
    
    # Create new vector store
    print("📚 No existing vector store found. Creating new one...")
    print("📂 Loading PDFs from 'data/' directory...")
    
    # Load PDFs
    loader = DirectoryLoader(
        "data/",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    data = loader.load()
    print(f"✅ Loaded {len(data)} pages from PDF(s)")
    
    # Split into chunks
    print("✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )
    docs = text_splitter.split_documents(data)
    print(f"✅ Created {len(docs)} chunks")
    
    # Create embeddings and vector store
    print("🧠 Creating embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorindex = FAISS.from_documents(docs, embeddings)
    
    # Save to disk
    print("💾 Saving vector store to disk...")
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex, f)
    print("✅ Vector store saved!\n")
    
    return vectorindex


def initialize_chatbot(vectorIndex, use_streaming=True):
    """Initialize the RAG chatbot with LLM and retriever."""
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and friendly RAG (Retrieval-Augmented Generation) assistant specialized in answering questions about the provided PDF documents.

IMPORTANT INSTRUCTIONS:
1. If the user greets you (says "hi", "hello", "hey", etc.) or asks general questions not related to the documents:
   - Respond warmly and briefly
   - Ask them what they'd like to know about the documents
   - DO NOT make up information from the context

2. If the user asks about the documents:
   - Answer ONLY using the information in the context below
   - If the answer is not in the context, say "I don't have that information in the documents I've been given"
   - Always cite facts from the context
   - Never speculate or use knowledge beyond the provided context

3. Be conversational but always ground your document-related answers in the provided context."""),
        ("user", """Context from documents:
{context}

User's message: {question}

Your response:"""),
    ])
    
    # Create LLM
    if use_streaming:
        llm = ChatOllama(
            model="phi3:mini-128k",
            temperature=0.7,
            streaming=True,
            callbacks=[StreamCallback()]
        )
    else:
        llm = ChatOllama(
            model="phi3:mini-128k",
            temperature=0.7
        )
    
    # Create RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorIndex.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return chain


def print_header():
    """Print chatbot header."""
    print("\n" + "=" * 80)
    print("🤖  PDF RAG CHATBOT - Ask questions about your documents!")
    print("=" * 80)
    print("Commands:")
    print("  • Type your question to get an answer")
    print("  • 'quit' or 'exit' - Exit the chatbot")
    print("  • 'clear' - Clear the screen")
    print("  • 'help' - Show this help message")
    print("=" * 80 + "\n")


def chatbot_loop(chain, use_streaming=True):
    """Main chatbot conversation loop."""
    
    print_header()
    
    # Track if this is first interaction
    first_interaction = True
    
    while True:
        try:
            # Get user input
            user_input = input("💬 You: ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye! Thanks for using the PDF RAG Chatbot!")
                break
            
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
                first_interaction = True
                continue
            
            if user_input.lower() == 'help':
                print_header()
                continue
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Handle simple greetings without calling the LLM
            greeting_responses = {
                'hi': "👋 Hello! I'm your PDF document assistant. What would you like to know about your documents?",
                'hello': "👋 Hi there! I'm here to help you understand your PDF documents. What can I answer for you?",
                'hey': "👋 Hey! I can answer questions about your PDF documents. What would you like to know?",
                'good morning': "🌅 Good morning! Ready to explore your documents? What questions do you have?",
                'good afternoon': "☀️ Good afternoon! How can I help you with your documents today?",
                'good evening': "🌆 Good evening! What would you like to learn from your documents?",
                'how are you': "I'm doing great, thanks for asking! 😊 I'm ready to help you with your PDF documents. What would you like to know?",
                'thanks': "You're welcome! 😊 Feel free to ask me anything else about your documents.",
                'thank you': "You're very welcome! Happy to help! 🎉 Any other questions about your documents?",
            }
            
            # Check if it's a simple greeting
            if user_input.lower() in greeting_responses:
                print(f"\n🤖 Bot: {greeting_responses[user_input.lower()]}\n")
                print("-" * 80 + "\n")
                first_interaction = False
                continue
            
            # Get response
            print("\n🤖 Bot: ", end="", flush=True)
            
            try:
                result = chain({"query": user_input}, return_only_outputs=True)['result']
                
                # If not streaming, print the result
                if not use_streaming:
                    print(result)
                
                print("\n" + "-" * 80 + "\n")
                
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Please try rephrasing your question.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")


def main():
    """Main function to run the chatbot."""
    
    print("\n🚀 Starting PDF RAG Chatbot...")
    print("-" * 80)
    
    try:
        # Check if data directory exists
        if not os.path.exists("data/"):
            print("❌ Error: 'data/' directory not found!")
            print("📁 Please create a 'data/' directory and add your PDF files.")
            return
        
        # Check if there are PDFs
        pdf_files = [f for f in os.listdir("data/") if f.endswith('.pdf')]
        if not pdf_files:
            print("❌ Error: No PDF files found in 'data/' directory!")
            print("📄 Please add some PDF files to the 'data/' directory.")
            return
        
        print(f"📄 Found {len(pdf_files)} PDF file(s) in data/ directory")
        
        # Load or create vector store
        vectorIndex = load_or_create_vectorstore()
        
        # Initialize chatbot
        print("🔧 Initializing chatbot with Phi3 model...")
        chain = initialize_chatbot(vectorIndex, use_streaming=True)
        print("✅ Chatbot ready!\n")
        
        # Start conversation loop
        chatbot_loop(chain, use_streaming=True)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
