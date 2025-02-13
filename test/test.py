import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
from pygrabber.dshow_graph import FilterGraph
import numpy as np

import os
import threading
import json
import queue

from image_matcher.detect_image import find_cards
from image_matcher.hash import generate_hash_pool


# Initialize variables
thresh=127
thresh_max_value = 255
block_size = 26
offset_c = 5
ksize = 4

#opencv uses BGR (Blue, Green, Red) by default
match_color = (132, 255, 0) #green
no_match_color = (0, 0, 255) #red

recognition_thread = None
recognition_queue = queue.Queue() # Queue for handling recognition results

high_res_image_folder = "C:\\Users\\erich\\net_ready_eyes\\media\\high_res_images"
# Set the current folder to the default image folder
low_res_image_folder = "C:\\Users\\erich\\net_ready_eyes\\media\\low_res_images"
test_image_folder = "C:\\Users\\erich\\net_ready_eyes\\media\\test_images"
query_image_path = os.path.join(test_image_folder, "table_capture2.png")
# Define the size of the "playing card" area (width, height)
card_width = 200
card_height = 300

hash_pool = []

def display_matched_image(matched_image_path):
    if matched_image_path:
        image = Image.open(matched_image_path)
        image_resized = image.resize((300, 419), Image.LANCZOS)
        #image_resized = image #debugging scaling
        photo = ImageTk.PhotoImage(image=image_resized)

        # Save the high-res matched image to a file for OBS to import
        export_path = os.path.join(os.path.dirname(__file__), "obs_export_image.png")
        image.save(export_path)
        print(f"Saved high-res matched image to {export_path}")

def main():
    global hash_pool, query_image_path, recognition_thread

    if low_res_image_folder:
        hash_pool = generate_hash_pool(low_res_image_folder)

    #print(hash_pool)

    if not os.path.exists(query_image_path):
        print(f"Error: Query image {query_image_path} not found!")
        return

    #query_image = cv2.imread(query_image_path)
    #query_image = Image.open(query_image_path)

      # Start recognition in a separate thread if not already running
    if recognition_thread is None or not recognition_thread.is_alive():
        # Pass the frame, hash_pool we've calculated for the cards in the pool, 
        # and a pointer to the recognition queue to the find_cards function
        recognition_thread = threading.Thread(target=find_cards(query_image_path, hash_pool, recognition_queue))
        recognition_thread.daemon = True
        recognition_thread.start()

    try:
        while not recognition_queue.empty():
            match_found = recognition_queue.get_nowait()
            if match_found:
                matched_image_path = match_found
                display_matched_image(matched_image_path)
                print(f"Image match detected - {matched_image_path}")
    except queue.Empty:
        pass

if __name__ == "__main__":
    main()