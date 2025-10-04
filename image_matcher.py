# image_matcher.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# 1. Define the path to your LOCAL model files.
local_model_path = "./models/clip-vit-base-patch32"

print("Loading local CLIP model...")
# 2. Load the model and processor from the local folder. No internet is needed.
model = CLIPModel.from_pretrained(local_model_path)
processor = CLIPProcessor.from_pretrained(local_model_path, use_fast=False)
print("Model loaded successfully.")

def get_image_embedding(image_path):
    """Takes an image file path and returns its embedding vector."""
    try:
        # 3. Open the image file using Pillow.
        image = Image.open(image_path)
        # 4. Prepare the image for the model using the processor.
        inputs = processor(images=image, return_tensors="pt")
        
        # 5. Get the embedding. `torch.no_grad()` is a performance optimization.
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# --- Main script logic ---

def find_similar_image(query_image_path, database_folder='test_images'):
    """
    Find the most similar image to the query image from the database folder.
    """
    # Check if database folder exists
    if not os.path.exists(database_folder):
        print(f"Error: The folder '{database_folder}' does not exist.")
        print("Please create the folder and add some images to search against.")
        return None
    
    # Get all database images
    database_image_paths = sorted([os.path.join(database_folder, f) for f in os.listdir(database_folder) 
                                  if f.endswith(('png', 'jpg', 'jpeg'))])
    
    # Check if we have any database images
    if len(database_image_paths) == 0:
        print(f"Error: No image files found in '{database_folder}' folder.")
        print("Please add some images (PNG, JPG, or JPEG) to the folder.")
        return None
    
    print(f"Found {len(database_image_paths)} images in database folder.")
    print(f"Querying with image: {os.path.basename(query_image_path)}")
    print(f"Searching against {len(database_image_paths)} database images...")
    
    return database_image_paths

def interactive_mode():
    """
    Interactive mode to upload/specify a query image and find matches.
    """
    print("=" * 60)
    print("FACE SIMILARITY MATCHER")
    print("=" * 60)
    print("This tool finds the most similar face image from your database.")
    print()
    
    while True:
        # Get query image path from user
        query_path = input("Enter the path to your query image (or 'quit' to exit): ").strip()
        
        if query_path.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Remove quotes if present
        query_path = query_path.strip('"\'')
        
        # Check if query image exists
        if not os.path.exists(query_path):
            print(f"Error: Image file '{query_path}' not found.")
            print("Please check the path and try again.")
            continue
        
        # Check if it's a valid image file
        if not query_path.lower().endswith(('png', 'jpg', 'jpeg')):
            print("Error: Please provide a valid image file (PNG, JPG, or JPEG).")
            continue
        
        print(f"\n" + "-" * 60)
        print(f"Processing query image: {os.path.basename(query_path)}")
        print("-" * 60)
        
        # Find similar images
        database_image_paths = find_similar_image(query_path)
        
        if database_image_paths is None:
            continue
        
        # Process the matching
        match_result = process_image_matching(query_path, database_image_paths)
        
        if match_result:
            print(f"\n" + "=" * 60)
            print("BEST MATCH FOUND!")
            print("=" * 60)
            print(f"Query Image: {os.path.basename(query_path)}")
            print(f"Best Match: {match_result['match_name']}")
            print(f"Similarity Score: {match_result['score']:.4f}")
            print(f"Match File: {match_result['match_path']}")
            
            # Interpretation of similarity score
            if match_result['score'] > 0.85:
                print("🟢 Excellent match! Very likely the same person.")
            elif match_result['score'] > 0.70:
                print("🟡 Good match! Likely the same person.")
            elif match_result['score'] > 0.55:
                print("🟠 Moderate match. Could be the same person.")
            else:
                print("🔴 Low match. Likely different people.")
        else:
            print("❌ No matches could be processed.")
        
        print("\n" + "=" * 60)

def process_image_matching(query_image_path, database_image_paths):
    """
    Process image matching and return the best match.
    """
    # 8. Generate and store embeddings for all database images in a list.
    database_embeddings = []
    valid_database_paths = []
    
    print("Generating embeddings for database images...")
    for i, path in enumerate(database_image_paths):
        embedding = get_image_embedding(path)
        if embedding is not None:
            database_embeddings.append(embedding)
            valid_database_paths.append(path)
        if (i + 1) % 50 == 0:  # Progress indicator
            print(f"Processed {i + 1}/{len(database_image_paths)} images...")
    
    print(f"Successfully generated {len(database_embeddings)} embeddings.")
    
    # 9. Generate the embedding for your single query image.
    print("Generating embedding for query image...")
    query_embedding = get_image_embedding(query_image_path)
    
    if query_embedding is not None and database_embeddings:
        # 10. Prepare embeddings for calculation.
        database_embeddings_tensor = torch.cat(database_embeddings, dim=0)
        query_embedding_norm = query_embedding / query_embedding.norm()
        database_embeddings_norm = database_embeddings_tensor / database_embeddings_tensor.norm(dim=-1, keepdim=True)

        # 11. Calculate cosine similarity between the query and all database images.
        similarity_scores = torch.matmul(query_embedding_norm, database_embeddings_norm.T)
        
        # 12. Find the highest score and its index to identify the best match.
        best_match_index = int(torch.argmax(similarity_scores).item())
        best_match_score = float(similarity_scores[0][best_match_index].item())
        best_match_path = valid_database_paths[best_match_index]
        
        # Extract name from filename (remove extension and clean up)
        match_filename = os.path.basename(best_match_path)
        match_name = os.path.splitext(match_filename)[0].replace('_', ' ')
        
        return {
            'match_path': best_match_path,
            'match_name': match_name,
            'score': best_match_score,
            'filename': match_filename
        }
    else:
        print("\n--- ERROR ---")
        if query_embedding is None:
            print("Failed to generate embedding for the query image.")
        if not database_embeddings:
            print("No valid embeddings were generated for the database images.")
        print("Cannot perform image matching.")
        return None

# Main execution
if __name__ == "__main__":
    print("🔍 Face Image Similarity Matcher")
    print("=" * 50)
    print("Upload a query image to find the most similar person in the database!")
    print()
    
    interactive_mode()