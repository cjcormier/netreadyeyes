import imagehash as ih
import numpy as np
from PIL import Image
import pandas as pd
from tkinter import messagebox

import os
import time
import concurrent.futures

def _compute_image_hash(path, hash_size=32):
    """Compute a perceptual hash for an image."""
    try:
        image = Image.open(path).convert("RGB")
        image_hash = ih.phash(image, hash_size=hash_size).hash.flatten()
        return (path, image_hash)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def generate_hash_pool(path):
    if not path:
        return

    start_time = time.time()
    image_paths = [
        os.path.join(path, img) for img in os.listdir(path) if img.lower().endswith(('.png', '.jpg'))]
    
    if not image_paths:
        messagebox.showerror("Error", "No PNG or JPG images found in the selected folder.")
        return

    hash_pool = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {executor.submit(_compute_image_hash, img_path): img_path for img_path in image_paths}
        
        for future in concurrent.futures.as_completed(future_to_path):
            result = future.result()
            if result:
                hash_pool.append(result)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Generated Perceptual Hashes for {len(hash_pool)} images in {elapsed_time:.2f}s.")

    # Convert to Pandas DataFrame for easy access
    hash_pool_df = pd.DataFrame([{"name": os.path.basename(path), "card_hash_32": hash} for path, hash in hash_pool])

    return hash_pool_df


def find_minimum_hash_difference(image, hash_pool_df, hash_size=32):
    image_object = Image.fromarray(image.astype('uint8'), 'RGB')
    card_hash = _compute_image_hash(image_object, hash_size)
    hash_pool = pd.DataFrame(hash_pool_df)
    #add a new in the hash pool to store the difference between the computed hash and each stored hash
    hash_pool['diff'] = hash_pool['card_hash_%d' % hash_size]
    # Calculate the Hamming distance between the image hash and each hash in the pool
    hash_pool['diff'] = hash_pool['diff'].apply(lambda x: np.count_nonzero(x != card_hash))
    # Return the row with the smallest hash difference and the minimum difference value
    return hash_pool[hash_pool['diff'] == min(hash_pool['diff'])].iloc[0], \
           min(hash_pool['diff'])
