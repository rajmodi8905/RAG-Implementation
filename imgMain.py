import os
import torch
from transformers import CLIPModel, CLIPProcessor

# --- 1. Import your custom modules ---
# This assumes your project structure is set up correctly
from IMGModule.indexing import index_images
from IMGModule.search import perform_semantic_search

# --- 2. DEFINE CONSTANTS AND PATHS ---

# Directory where your images are stored
IMAGES_DIR = 'all_images'

# Directory to save the generated index and metadata
INDEX_DIR = 'ImgIndexStore'
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'faiss_image_index.bin')
METADATA_PATH = os.path.join(INDEX_DIR, 'image_metadata.npy')

# Path to your locally saved CLIP model
LOCAL_MODEL_PATH = os.path.abspath('./clip-model') 
# The model identifier from Hugging Face, used for downloading if needed
REMOTE_MODEL_NAME = "openai/clip-vit-base-patch32"

# Ensure the index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)


# --- 3. MAIN EXECUTION SCRIPT ---

if __name__ == "__main__":
    # --- A. Setup Model and Device ---

    # Check for GPU availability and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Load the CLIP model and processor from your local folder
    print(f"Attempting to load model from local path: {LOCAL_MODEL_PATH}")
    try:
        model = CLIPModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
        processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_PATH)
        print("✅ Model loaded successfully from local path.")
    except OSError:
        print(f"⚠️ Model not found locally. Downloading from Hugging Face...")
        print(f"This will save the model to '{LOCAL_MODEL_PATH}' for future use.")
        try:
            # Download and save the model and processor
            model = CLIPModel.from_pretrained(REMOTE_MODEL_NAME)
            processor = CLIPProcessor.from_pretrained(REMOTE_MODEL_NAME)

            model.save_pretrained(r"{}".format(LOCAL_MODEL_PATH))
            processor.save_pretrained(r"{}".format(LOCAL_MODEL_PATH))
            
            print("✅ Model downloaded and saved successfully.")
            # Move the newly downloaded model to the correct device
            model.to(device)

        except Exception as e:
            print(f"❌ An error occurred during model download: {e}")
            exit()

    # --- B. Indexing ---

    print("\n--- Starting Image Indexing ---")
    # This function will process images in IMAGES_DIR and save the index files
    index_images(
        model=model,
        processor=processor,
        device=device,
        images_dir=IMAGES_DIR,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH
    )
    print("--- Indexing Complete ---\n")

    # --- C. Searching ---

    print("--- Performing a Semantic Search ---")
    # Define a sample query to search for
    search_query = "image of a team playing football"
    print(f"Searching for query: '{search_query}'")

    # This function will load the index and find the most relevant images
    # It returns a list of filenames
    retrieved_images = perform_semantic_search(
        query=search_query,
        k=3,  # Number of top results to retrieve
        model=model,
        processor=processor,
        device=device,
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    # --- D. Display Results ---

    if retrieved_images:
        print("\n✅ Search Results (most relevant first):")
        for i, image_name in enumerate(retrieved_images):
            # The result is just the filename. You can construct the full path if needed.
            # full_path = os.path.join(IMAGES_DIR, image_name)
            print(f"  {i+1}. {image_name}")
    else:
        print("\n⚠️ No results found for the query.")

