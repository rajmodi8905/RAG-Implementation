import os
import glob
import numpy as np
import faiss
import torch
from PIL import Image
from tqdm.auto import tqdm

# import streamlit as st # Still need for st.progress, st.warning etc.

def index_images(model, processor, device, images_dir, index_path, metadata_path):
    """Generates embeddings for all images and builds a FAISS index."""
    image_paths = glob.glob(os.path.join(images_dir, '*.*'))
    all_embeddings = []
    all_metadata = []

    if not image_paths:
        # st.warning(f"No images found in '{images_dir}'. Please upload some.")
        print(f"No images found in '{images_dir}'. Please upload some.")
        return

    # st.write(f"Found {len(image_paths)} images. Starting indexing...")
    # progress_bar = st.progress(0)
    print(f"Found {len(image_paths)} images. Starting indexing...")

    for i, image_path in enumerate(tqdm(image_paths, desc="Embedding images", unit="image")):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embedding_np = embedding.cpu().numpy().flatten()
            
            all_embeddings.append(embedding_np)
            all_metadata.append(os.path.basename(image_path))
        except Exception as e:
            # st.error(f"Error processing image {os.path.basename(image_path)}: {e}")
            print(f"Error processing image {os.path.basename(image_path)}: {e}")
        
        # progress_bar.progress((i + 1) / len(image_paths))

    if all_embeddings:
        embeddings_matrix = np.stack(all_embeddings)
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_matrix)
        
        faiss.write_index(index, index_path)
        np.save(metadata_path, np.array(all_metadata))
        # st.success(f"Indexing complete! {index.ntotal} images indexed.")
        print(f"Indexing complete! {index.ntotal} images indexed.")
