import cv2
import imagehash as ih
import numpy as np
from PIL import Image
import pandas as pd
from tkinter import messagebox

import os
import time
import concurrent.futures

from utils.utils import convert_image_to_opencv, convert_image_to_pil

def compute_image_hash_from_file(image_path, hash_size=32):
    """Compute a perceptual hash for an image.
    
    Args:
        image_path (str): Path to a .png or .jpg image file.
        hash_size (int, optional): Hash size for perceptual hashing. Defaults to 32.
    
    Returns:
        tuple: (image path (string), computed hash) or None on failure.
    """
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path)

        image, image_hash = compute_image_hash(image)
        return (image_path, image_hash)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def compute_image_hash(image, hash_size=32):
    """Compute a perceptual hash for an image.
    
    Args:
        image (PIL.Image.Image): Image object.
        hash_size (int, optional): Hash size for perceptual hashing. Defaults to 32.
    
    Returns:
        tuple: (image path or object, computed hash) or None on failure.
    """
    #image_path should be a path to a png or jpg image
    try:
        image_hash = ih.phash(image, hash_size=hash_size).hash.flatten()
        return (image, image_hash)
    except Exception as e:
        print(f"Error processing {image}: {e}")
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

        future_to_path = {executor.submit(compute_image_hash_from_file, img_path): img_path for img_path in image_paths}
        
        for future in concurrent.futures.as_completed(future_to_path):
            result = future.result()
            if result:
                hash_pool.append(result)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Generated Perceptual Hashes for {len(hash_pool)} images in {elapsed_time:.2f}s.")

    # Convert to Pandas DataFrame for easy access
    hash_pool_df = pd.DataFrame([{"name": os.path.basename(img_path), "card_hash_32": hash} for img_path, hash in hash_pool])

    return hash_pool_df


def find_minimum_hash_difference(query_image, hash_pool_df, hash_size=32):
    
    if query_image is None:
        print("Error - find_minimum_hash_difference(): query_image is None")  # Debugging
        return None, None
    
    card_hash_data = compute_image_hash(query_image, hash_size)
    if card_hash_data is None:
        print("Error: compute_image_hash returned None")  # Debugging
        return None, None
    
    card_path, card_hash = compute_image_hash(query_image, hash_size)
    hash_pool = pd.DataFrame(hash_pool_df)

    #add a new in the hash pool to store the difference between the computed hash and each stored hash
    hash_pool['diff'] = hash_pool['card_hash_%d' % hash_size]
    # Calculate the Hamming distance between the image hash and each hash in the pool
    hash_pool['diff'] = hash_pool['diff'].apply(lambda x: np.count_nonzero(x != card_hash))
    
    # Return the row with the smallest hash difference and the minimum difference value
    return hash_pool[hash_pool['diff'] == min(hash_pool['diff'])].iloc[0], \
           min(hash_pool['diff'])
