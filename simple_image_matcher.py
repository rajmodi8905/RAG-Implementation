# simple_image_matcher.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import sys

# Load the model once at startup
print("Loading CLIP model...")
local_model_path = "./models/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(local_model_path)
processor = CLIPProcessor.from_pretrained(local_model_path, use_fast=False)
print("Model loaded successfully!")

def get_image_embedding(image_path):
    """Generate embedding for an image."""
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def find_best_match(query_image_path, database_folder='test_images'):
    """Find the best matching image for the query."""
    
    # Check if query image exists
    if not os.path.exists(query_image_path):
        return {"error": f"Query image '{query_image_path}' not found."}
    
    # Check if database folder exists
    if not os.path.exists(database_folder):
        return {"error": f"Database folder '{database_folder}' not found."}
    
    # Get database images
    database_images = [os.path.join(database_folder, f) 
                      for f in os.listdir(database_folder) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not database_images:
        return {"error": f"No images found in '{database_folder}' folder."}
    
    print(f"Searching through {len(database_images)} images...")
    
    # Generate embeddings
    database_embeddings = []
    valid_paths = []
    
    for path in database_images:
        embedding = get_image_embedding(path)
        if embedding is not None:
            database_embeddings.append(embedding)
            valid_paths.append(path)
    
    if not database_embeddings:
        return {"error": "No valid database image embeddings could be generated."}
    
    # Generate query embedding
    query_embedding = get_image_embedding(query_image_path)
    if query_embedding is None:
        return {"error": "Could not generate embedding for query image."}
    
    # Calculate similarities
    database_embeddings_tensor = torch.cat(database_embeddings, dim=0)
    query_embedding_norm = query_embedding / query_embedding.norm()
    database_embeddings_norm = database_embeddings_tensor / database_embeddings_tensor.norm(dim=-1, keepdim=True)
    
    similarity_scores = torch.matmul(query_embedding_norm, database_embeddings_norm.T)
    
    # Find best match
    best_match_index = int(torch.argmax(similarity_scores).item())
    best_match_score = float(similarity_scores[0][best_match_index].item())
    best_match_path = valid_paths[best_match_index]
    
    # Extract person name from filename
    filename = os.path.basename(best_match_path)
    person_name = os.path.splitext(filename)[0].replace('_', ' ')
    
    return {
        "success": True,
        "query_image": os.path.basename(query_image_path),
        "best_match": {
            "person_name": person_name,
            "filename": filename,
            "file_path": best_match_path,
            "similarity_score": best_match_score
        }
    }

def main():
    print("\n" + "="*60)
    print("🔍 SIMPLE FACE MATCHER")
    print("="*60)
    
    # Check if image path was provided as command line argument
    if len(sys.argv) > 1:
        query_path = sys.argv[1]
        print(f"Using command line image: {query_path}")
    else:
        # Interactive mode
        print("Drop your image file here or enter the path:")
        query_path = input("Image path: ").strip().strip('"\'')
    
    if not query_path:
        print("No image path provided. Exiting.")
        return
    
    print(f"\nProcessing: {os.path.basename(query_path)}")
    print("-" * 40)
    
    # Find the best match
    result = find_best_match(query_path)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Display results
    match = result["best_match"]
    score = match["similarity_score"]
    
    print(f"\n🎯 BEST MATCH FOUND!")
    print(f"Query Image: {result['query_image']}")
    print(f"Matched Person: {match['person_name']}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Match File: {match['filename']}")
    
    # Score interpretation
    print(f"\nConfidence Level:")
    if score > 0.85:
        print("🟢 EXCELLENT - Very likely the same person!")
    elif score > 0.70:
        print("🟡 GOOD - Likely the same person")
    elif score > 0.55:
        print("🟠 MODERATE - Could be the same person")
    else:
        print("🔴 LOW - Likely different people")
    
    print(f"\nMatch file location: {match['file_path']}")
    print("="*60)

if __name__ == "__main__":
    main()