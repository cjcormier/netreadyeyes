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

        image, image_hash = compute_image_hash(image, hash_size)
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
        image_hash = ih.phash(image, hash_size=hash_size)
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
    if hash_pool_df is None or hash_pool_df.empty:
        print("Error: The hash pool dataframe is empty.")
        return None, None
    
    if query_image is None:
        print("Error - find_minimum_hash_difference(): query_image is None")  # Debugging
        return None, None

    # if isinstance(query_image, Image.Image):
    #     img_copy = query_iamge.copy()
    #     img_copy = convert_image_to_opencv(img_copy)
    #     cv2.imshow('query_image', img_copy)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Convert OpenCV image to PIL format if needed
    if isinstance(query_image, np.ndarray):
        # cv2.imshow('query_image', query_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        query_image = convert_image_to_pil(query_image)

    card_path, card_hash = compute_image_hash(query_image, hash_size)

    if card_hash is None:
        print("Error: compute_image_hash returned None")  # Debugging
        return None, None
    else:
        print(f"card_hash for the query_image = {card_hash}")
    
    hash_pool = pd.DataFrame(hash_pool_df)

    # ///////// DEBUGGING DEVIL CHARM SPECIFICALLY

    # Find the row where 'card_name' matches 'devil_charm.png'
    devil_charm_row = hash_pool_df[hash_pool_df['name'].str.startswith('devil_charm.')]

    if not devil_charm_row.empty:
        devil_charm_hash = devil_charm_row['card_hash_%d' % hash_size].values[0]
        
        # Compute Hamming distance
        devil_charm_diff = devil_charm_hash - card_hash

        print(f"\nComparing query image to 'devil_charm.png':")
        print(f"Query Hash (Hex): {card_hash}")
        print(f"Devil Charm Hash (Hex): {devil_charm_hash}")
        print(f"Hamming Distance: {devil_charm_diff}\n")
        # Print bitwise differences
       
    else:
        print("Error: 'devil_charm.png' not found in hash pool.")

    # ///////// END DEBUGGING DEVIL CHARM SPECIFICALLY


    #add a new value in the hash pool to store the difference between the computed hash and each stored hash
    hash_pool['diff'] = hash_pool['card_hash_%d' % hash_size]
    # Calculate the Hamming distance between the image hash and eac h hash in the pool
    hash_pool['diff'] = hash_pool['diff'].apply(lambda x: x - card_hash)

    # Return the row with the smallest hash difference and the minimum difference value
    minimum = hash_pool['diff'].min()
    return hash_pool[hash_pool['diff'] == minimum].iloc[0], minimum
