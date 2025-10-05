import os
import glob
import numpy as np
import faiss
import torch
from PIL import Image
from tqdm.auto import tqdm

def index_images(model, processor, device, images_dir, index_path, metadata_path):
    """
    Synchronizes the FAISS index with the image directory using an efficient strategy:
    - Appends embeddings for new images.
    - Rebuilds the index only if images have been deleted.
    - Skips if no changes are detected.
    """
    image_extensions = ['jpg', 'jpeg', 'png', 'webp']
    all_image_paths = []
    
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(images_dir, f'*.{ext}')))

    if not all_image_paths and not os.path.exists(index_path):
        print(f"⚠️ No images found in '{images_dir}' and no index exists.")
        return

    # Get the list of image filenames currently on disk
    current_filenames = {os.path.basename(p) for p in all_image_paths}
    
    # --- Step 1: Check if an index exists and determine the update strategy ---
    if os.path.exists(index_path):
        print("Found existing index. Checking for updates...")
        existing_metadata = list(np.load(metadata_path, allow_pickle=True))
        existing_filenames = set(existing_metadata)
        
        new_files = current_filenames - existing_filenames
        deleted_files = existing_filenames - current_filenames
        
        if not new_files and not deleted_files:
            print("✅ Index is already up-to-date. No changes needed.")
            return
            
        # --- STRATEGY 1: EFFICIENT APPEND (Only new files) ---
        if new_files and not deleted_files:
            print(f"Update summary: {len(new_files)} new files found. Appending to index.")
            
            index = faiss.read_index(index_path)
            new_image_paths = [os.path.join(images_dir, fname) for fname in new_files]
            new_embeddings = []
            new_metadata = []

            for image_path in tqdm(new_image_paths, desc="Embedding new images", unit="image"):
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs)
                    
                    embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    new_embeddings.append(embedding.cpu().numpy().flatten())
                    new_metadata.append(os.path.basename(image_path))
                except Exception as e:
                    print(f"❌ Error processing new image {os.path.basename(image_path)}: {e}")

            if new_embeddings:
                embeddings_matrix = np.stack(new_embeddings)
                index.add(embeddings_matrix)
                
                updated_metadata = np.array(existing_metadata + new_metadata)
                faiss.write_index(index, index_path)
                np.save(metadata_path, updated_metadata)
                print(f"✅ Index updated successfully. Total images indexed: {index.ntotal}")
            return # Stop here after the append operation

        # --- STRATEGY 2: REBUILD (Deletions were detected) ---
        else:
            print(f"Update summary: {len(deleted_files)} files deleted, {len(new_files)} files added. Rebuilding index.")
            final_filenames = list(existing_filenames - deleted_files) + list(new_files)
            if not final_filenames:
                if os.path.exists(index_path): os.remove(index_path)
                if os.path.exists(metadata_path): os.remove(metadata_path)
                print("🗑️ All images were deleted. Index has been removed.")
                return
            final_image_paths = [os.path.join(images_dir, fname) for fname in final_filenames]
            
    else:
        # --- INITIAL INDEXING MODE ---
        print("No existing index found. Building a new one from scratch.")
        final_image_paths = all_image_paths

    # --- This block is now used for both initial indexing and rebuilding ---
    all_embeddings = []
    all_metadata = []

    for image_path in tqdm(final_image_paths, desc="Processing images", unit="image"):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(embedding.cpu().numpy().flatten())
            all_metadata.append(os.path.basename(image_path))
        except Exception as e:
            print(f"❌ Error processing image {os.path.basename(image_path)}: {e}")

    if all_embeddings:
        embeddings_matrix = np.stack(all_embeddings)
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_matrix)
        
        faiss.write_index(index, index_path)
        np.save(metadata_path, np.array(all_metadata))
        print(f"✅ Indexing complete! {index.ntotal} images are now indexed.")

