import os
import numpy as np
import faiss
import torch

def perform_semantic_search(query, k, model, processor, device, index_path, metadata_path):
    """
    Performs a pure semantic search using CLIP embeddings and FAISS.
    """
    if not os.path.exists(index_path):
        print("Index not found. Please index your images first.")
        return []

    # 2. Load the pre-built FAISS index and corresponding metadata.
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)

    # 3. Process the text query to create a vector embedding.
    inputs = processor(text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    query_vector = (text_features / text_features.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()
    
    distances, indices = index.search(query_vector, k)
    
    results = [metadata[i] for i in indices[0]]
    
    return results
