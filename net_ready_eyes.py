# import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from pygrabber.dshow_graph import FilterGraph

import threading
import queue
import os

import utils.const as const
from ui.ui_class import UI
from image_matcher.detect_image import find_cards

class NetReadyEyesApp():
    def __init__(self, root):
        self.root = root
        self.root.title("Net Ready Eyes")                
        # Initialize the core app variables
        self.video_source = None  # This will be set after video_source selection
        self.available_video_sources = self.find_video_sources()
        self.hash_pool = {}

         # Create the UI
        self.ui = UI(root, self)

        self._is_running = False

        self.recognition_queue = queue.Queue() # Queue for handling recognition results
        self.recognition_thread = None
        self.target_images = []
        self.dragging_point = None # Stores which point of the polygon is being dragged

        self.rec_params = {
            "thresh_max_value": 255,
            "block_size": 27, # must be an odd number
            "offset_c": 5,
            "kernel_size": (5,5),
            "match_threshold": 20,
            "stddev": 0,
            "card_min_area": 60000,
            "card_max_area": 80000
        }

        self.thresh_max_value = 255
        self.block_size = 27 #must be an odd number
        self.offset_c = 5
        #self.ksize = 4
        self.kernel_size = (5,5) #must be a tuple of odd numbers

        # this defines how big we want to display the matching image (high res images can sometimes be too big)
        self.card_width = 300
        self.card_height = 419
        
        # Default value for the frequency (in milliseconds)
        self.processing_period = int(self.ui.proc_period_slider.get())
        # self.processing_period = 700
        self.worker_threads = 4 # todo: make configurable in UI

    def find_video_sources(self):
        """Find available video_sources and get their descriptive names."""
        video_sources = []
        graph = FilterGraph()
        devices = graph.get_input_devices()  # List of video_source names
        
        for index, device_name in enumerate(devices):
            video_sources.append((index, device_name))  # Store index and name as a tuple
        
        return video_sources

    def update_proc_period(self, value):
        self.processing_period = int(float(value))

    def update_threshold(self, value):
        self.rec_params["match_threshold"] = float(value)
        self.ui.log_msg(f"Updated match Threshold to {self.rec_params["match_threshold"]}")

    def update_recognition(self):
        """ Process recognition separately from video update. """

        if not self.is_running:
            return # Stop if app isn't running
        
        roi_frame = self.ui.get_roi_frame()
      
        self.handle_recognition(roi_frame)
        # Schedule next recognition check
        self.root.after(self.processing_period, self.update_recognition)

    def handle_recognition(self, roi_frame):
        """Handle recognition thread and queue processing."""
        # Start recognition in a separate thread if not already running 
        # (avoid creating multiple threads)
        if self.recognition_thread is None or not self.recognition_thread.is_alive():
            # Pass the frame, hash_pool we've calculated for the cards in the pool, 
            # and a pointer to the recognition queue to the find_cards function
            self.recognition_thread = threading.Thread(target=find_cards, args=(
                roi_frame, self.rec_params, self.hash_pool, 
                self.recognition_queue, self.ui.display_image, self.ui.display_mode
                ), daemon=True)
            
            self.recognition_thread.start()

        # Process results from the queue
        self.process_recognition_results()


    def process_recognition_results(self):
        """ Wait for matching results from the recognition queue and display them """
        try:
            while not self.recognition_queue.empty():
                match_card_name = self.recognition_queue.get_nowait()
                match_img_path = os.path.join(const.HIGH_RES_DIR, match_card_name)
                if match_img_path:
                    self.ui.display_match(match_img_path)

        except queue.Empty:
            pass


    def update_hash_pool(self, new_hash_pool):
        self.hash_pool = new_hash_pool

    def export_to_obs(self, image_path):
        """ Export the matched image to OBS as an image source. """ 
        # OBS integration (if needed) can go here

    def start(self):
        self._is_running = True
        self.update_recognition()

    def stop(self):
        self._is_running = False

    def is_running(self):
        return self._is_running

if __name__ == "__main__":
    # root = tk.Tk()
    root = ttk.Window(themename="darkly")
    app = NetReadyEyesApp(root)
    root.mainloop()