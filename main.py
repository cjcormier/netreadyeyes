import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
from pygrabber.dshow_graph import FilterGraph
import numpy as np

import os
import threading
import queue

from image_matcher.detect_image import find_cards
import utils.const as const
from image_matcher.hash import generate_hash_pool

class net_ready_eyes:
    def __init__(self, root):
        self.root = root
        self.root.title("Net-Ready Eyes")
        
        #choose from rectangle, polygon, or auto - to do: make this selectable from a drop down
        #self.detect_mode = "polygon"
        self.detect_mode = "rectangle"
        #self.detect_mode = "auto"

        #default display mode (what shows up in the video frame)
        self.display_mode = "thresholding"
        #self.display_mode = "unfiltered contours"
        #self.display_mode = "filtered contours"
        #self.display_mode = "rectangular contours"

        # Initialize variables
        self.vid_stream = None  # This will be set after webcam selection
        self.is_running = False

        self.match_occured = False

        #opencv uses BGR (Blue, Green, Red) by default
        self.match_color = (132, 255, 0) #green
        self.no_match_color = (0, 0, 255) #red

        self.recognition_queue = queue.Queue() # Queue for handling recognition results
        self.recognition_thread = None
        self.target_images = []
        self.matched_image_path = None
        self.available_webcams = self.find_webcams()
        self.dragging_point = None # Stores which point of the polygon is being dragged

        self.thresh_max_value = 255
        self.block_size = 27 #must be an odd number
        self.offset_c = 5
        #self.ksize = 4
        self.kernel_size = (5,5) #must be a tuple of odd numbers

        # Get the current script's directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Set the default image folders relative to the script directory - low resolution images for parsing/matching,
        # and high resolution files for displaying once a match is found.
        # IMPORTANT: The file names in these folders must match exactly. I have found Windows PowerToys Image Resizer 
        # to be helpful for scaling files:
        # https://learn.microsoft.com/en-us/windows/powertoys/install
        self.default_image_folder = const.HIGH_RES_DIR  # Folder named 'images'
        self.high_res_image_folder = const.HIGH_RES_DIR  # Folder named 'images'
        self.low_res_image_folder = const.LOW_RES_DIR

        if self.low_res_image_folder:
            self.hash_pool = generate_hash_pool(self.low_res_image_folder)

        # Coordinates for the ROI (Region of Interest) - where the playing card sized area will be placed
        self.roi_x = 2418  # X coordinate for the top-left corner
        self.roi_y = 846  # Y coordinate for the top-left corner

        # Define the size of the "region of interest"
        self.roi_width = 400
        self.roi_height = 600

        self.roi_color = self.no_match_color

        # this defines how big we want to display the matching image (high res images can sometimes be too big)
        self.card_width = 300
        self.card_height = 419

        self.video_width = 1280  # Default width, update dynamically if needed
        self.video_height = 720  # Default height, update dynamically if needed

        # Initialize polygon with 4 points (modify as needed)
        self.polygon = np.array([
            [self.roi_x, self.roi_y],  # Top-left
            [self.roi_x+self.roi_width, self.roi_y],  # Top-right
            [self.roi_x+self.roi_width, self.roi_y+self.roi_height],  # Bottom-right
            [self.roi_x, self.roi_y+self.roi_height]   # Bottom-left
        ], dtype=np.int32)

        # Create GUI components
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        # Video frame on the left
        self.video_frame = tk.Label(self.main_frame)
        self.video_frame.grid(row=0, column=0)

        #left-click in the video_frame
        self.video_frame.bind("<ButtonPress-1>", self.on_mouse_press)
        #move while holding left-click in the video_frame
        self.video_frame.bind("<B1-Motion>", self.on_mouse_drag)
        #release left-click
        self.video_frame.bind("<ButtonRelease-1>", self.on_mouse_release)
        #move the mouse while not clicking in the video_frame
        self.video_frame.bind("<Motion>", self.on_mouse_move)

        # Debug log frame on the right
        self.debug_frame = tk.Frame(self.main_frame)
        self.debug_frame.grid(row=0, column=1, padx=10)
                
        self.match_frame = tk.Label(self.main_frame)
        self.match_frame.grid(row=0, column=2, padx=10)

        # Scrollable Text widget for the debug log
        self.debug_log = tk.Text(self.debug_frame, height=20, width=100, wrap=tk.WORD, state=tk.DISABLED)
        self.debug_log.grid(row=0, column=0)

        self.start_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam)
        self.stop_button = tk.Button(self.root, text="Stop Webcam", command=self.stop_webcam, state=tk.DISABLED)
        self.select_button = tk.Button(self.root, text="Load Image Folder", command=self.select_image_folder)
        self.folder_label = tk.Label(self.root, text=f"Current Folder: {self.low_res_image_folder}")
        
        # Webcam selection dropdown
        self.webcam_label = tk.Label(self.root, text="Select Webcam:")
        self.webcam_combobox = ttk.Combobox(self.root, values=self.available_webcams)

        # Dropdown menu to select detect_mode
        self.detect_mode_label = tk.Label(self.root, text="Detection Mode:")
        self.detect_mode_label.pack(pady=5)

        self.detect_mode_combobox = ttk.Combobox(self.root, values=["polygon", "rectangle", "auto"])
        self.detect_mode_combobox.set(self.detect_mode)  # populate the box with the current value
        self.detect_mode_combobox.pack(pady=5)

        # Bind the combobox change event to update detect_mode
        self.detect_mode_combobox.bind("<<ComboboxSelected>>", self.on_detect_mode_change)

        
        # Dropdown menu to select what is displayed in the video frame
        # (thresholded image, unfiltered contours, filtered contours, rectangular contours)
        self.display_mode_label = tk.Label(self.root, text="Display Mode:")
        self.display_mode_label.pack(pady=5)

        self.display_mode_combobox = ttk.Combobox(self.root, values=["none", "thresholding", "unfiltered contours", "filtered contours", "rectangular_contours"])
        self.display_mode_combobox.set(self.display_mode)  # populate the box with the current value
        self.display_mode_combobox.pack(pady=5)

        # Bind the combobox change event to update display_mode
        self.display_mode_combobox.bind("<<ComboboxSelected>>", self.on_display_mode_change)
        
        # Create a frame to hold buttons more compactly
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.select_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Place the folder label and webcam selection next to each other
        self.folder_label.pack(pady=5)
        self.webcam_label.pack(side=tk.LEFT, padx=5)
        self.webcam_combobox.pack(side=tk.LEFT, padx=5)
        
        # Default to first webcam in the list
        if len(self.available_webcams) >= 2:
            self.webcam_combobox.set(self.available_webcams[2]) #default to #2 - camlink 4k on eric's systems
        elif self.available_webcams:
            self.webcam_combobox.set(self.available_webcams[0])

        # Bind mouse events for moving/resizing the ROI
        self.video_frame.grid(row=0, column=0, sticky="nsew")  # Allow expansion
        self.main_frame.columnconfigure(0, weight=1)  # Expand to fill space
        self.main_frame.rowconfigure(0, weight=1)

        self.roi_dragging = False
        self.roi_resizing = False
        self.roi_drag_offset = (0, 0)

        # Slider to control the image recognition frequency
        self.freq_label = tk.Label(self.root, text="Image Recognition Frequency (ms):")
        self.freq_label.pack()

        self.freq_slider = tk.Scale(self.root, from_=10, to_=2000, orient=tk.HORIZONTAL, label="Frequency (ms)", command=self.update_frequency)
        self.freq_slider.set(500)  # Default frequency in milliseconds
        self.freq_slider.pack()

        # Default value for the frequency (in milliseconds)
        self.recognition_frequency = self.freq_slider.get()

        # Slider to adjust the image detection threshold
        self.threshold_label = tk.Label(self.root, text="Image Detection Threshold (perc of keypoints:")
        self.threshold_label.pack()

        self.threshold_slider = tk.Scale(self.root, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, label="Threshold", command=self.update_threshold)
        self.threshold_slider.set(20)  # Default threshold
        self.threshold_slider.pack()
        self.match_threshold = self.threshold_slider.get()  # Set initial match threshold in case it isn't used.
        
        self.match_label = tk.Label(self.root, text="", font=("Arial", 12, "bold"), fg="green")
        self.match_label.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.worker_threads = 4 # todo: make configurable in UI

        #create image recognition objects for repeated use
        self.orb = cv2.ORB_create()
        # FLANN Matcher Parameters (optimized for ORB/SIFT)
        index_params = dict(algorithm=6,  # FLANN LSH (Locality Sensitive Hashing) for ORB
                            table_number=6,  # Number of hash tables
                            key_size=12,  # Size of the key in bits
                            multi_probe_level=1)  # Number of probes per table

        search_params = dict(checks=50)  # Number of nearest neighbors to check

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def on_detect_mode_change(self, event):
            """This function is triggered when the user selects a different detection mode from the dropdown."""
            selected_mode = self.detect_mode_combobox.get()
            self.detect_mode = selected_mode  # Update the detect_mode variable

            # Optionally, you can update the UI or log the change
            print(f"Detection Mode set to: {self.detect_mode}")
            self.log_debug_message(f"Detection Mode set to: {self.detect_mode}")
            
            # You can then update other parts of the program that depend on the detect_mode if needed
            # Example: Update the ROI drawing logic based on the selected mode
            self.update_frame()

    def on_display_mode_change(self, event):
        """This function is triggered when the user selects a different detection mode from the dropdown."""
        selected_mode = self.display_mode_combobox.get()
        self.display_mode = selected_mode  # Update the display_mode variable

        # Optionally, you can update the UI or log the change
        print(f"Detection Mode set to: {self.display_mode}")
        self.log_debug_message(f"Detection Mode set to: {self.display_mode}")
        
        # You can then update other parts of the program that depend on the display_mode if needed
        # Example: Update the ROI drawing logic based on the selected mode
        self.update_frame()


    def on_mouse_press(self, event):
        """Determine if resizing or moving the whole ROI."""
        
        # Scale event coordinates
        x = event.x * self.scale_x
        y = event.y * self.scale_y
        
        margin = 10  # Margin for resizing detection

        # Check for resizing on edges/corners
        if (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.resizing_direction = "nw"
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.resizing_direction = "ne"
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.resizing_direction = "sw"
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.resizing_direction = "se"
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.resizing_direction = "top"
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.resizing_direction = "bottom"
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.resizing_direction = "left"
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.resizing_direction = "right"
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.resizing_direction = "move"  # Entire ROI should move
        else:
            self.resizing_direction = None

        # Store starting position
        self.start_x, self.start_y = x, y


    def on_mouse_drag(self, event):
        """Resize or move ROI dynamically as the mouse is dragged."""
        
        if self.resizing_direction is None:
            return  # Not resizing or moving

        # Scale event coordinates
        x = event.x * self.scale_x
        y = event.y * self.scale_y

        dx = x - self.start_x
        dy = y - self.start_y

        if self.resizing_direction == "move":  # Moving the whole ROI
            self.roi_x += dx
            self.roi_y += dy
        elif self.resizing_direction == "nw":  # Top-left corner
            self.roi_x += dx
            self.roi_width -= dx
            self.roi_y += dy
            self.roi_height -= dy
        elif self.resizing_direction == "ne":  # Top-right corner
            self.roi_width += dx
            self.roi_y += dy
            self.roi_height -= dy
        elif self.resizing_direction == "sw":  # Bottom-left corner
            self.roi_x += dx
            self.roi_width -= dx
            self.roi_height += dy
        elif self.resizing_direction == "se":  # Bottom-right corner
            self.roi_width += dx
            self.roi_height += dy
        elif self.resizing_direction == "top":  # Top edge
            self.roi_y += dy
            self.roi_height -= dy
        elif self.resizing_direction == "bottom":  # Bottom edge
            self.roi_height += dy
        elif self.resizing_direction == "left":  # Left edge
            self.roi_x += dx
            self.roi_width -= dx
        elif self.resizing_direction == "right":  # Right edge
            self.roi_width += dx

        # Ensure ROI stays within bounds
        self.roi_x = max(0, self.roi_x)
        self.roi_y = max(0, self.roi_y)
        self.roi_width = max(10, self.roi_width)  # Prevent shrinking too small
        self.roi_height = max(10, self.roi_height)

        self.start_x, self.start_y = x, y  # Update starting point

        self.update_display()  # Redraw with updated ROI

    def on_mouse_release(self, event):
        
        self.log_debug_message(f"ROI moved to position ({self.roi_x},{self.roi_y})")
    
        """End any dragging or resizing action."""
        self.dragging_point = None
        self.roi_dragging = False
        self.roi_resizing = None

    def on_mouse_move(self, event):
        """Change cursor when near ROI edges to indicate resizing."""
        
        # Scale mouse event coordinates
        x = event.x * self.scale_x
        y = event.y * self.scale_y
        
        margin = 30

        # Corner resizing
        if (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_frame.config(cursor="size_nw_se")
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_frame.config(cursor="size_ne_sw")
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.video_frame.config(cursor="size_ne_sw")
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.video_frame.config(cursor="size_nw_se")
        
        # **NEW: Vertical resizing (top and bottom edges)**
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_frame.config(cursor="size_ns")  # Top edge
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.video_frame.config(cursor="size_ns")  # Bottom edge
        
        # **NEW: Horizontal resizing (left and right edges)**
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.video_frame.config(cursor="size_we")  # Left edge
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.video_frame.config(cursor="size_we")  # Right edge
        
        # Move cursor when inside ROI
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.video_frame.config(cursor="fleur")  # Move cursor
        else:
            self.video_frame.config(cursor="")  # Default

    def find_webcams(self):
        """Find available webcams and get their descriptive names."""
        webcams = []
        graph = FilterGraph()
        devices = graph.get_input_devices()  # List of webcam names
        
        for index, device_name in enumerate(devices):
            webcams.append((index, device_name))  # Store index and name as a tuple
        
        return webcams

    def start_webcam(self):
        """ Start the selected webcam feed and recognition loop. """
        selected_webcam = self.webcam_combobox.get()
        webcam_index = int(selected_webcam.split(" ")[0])  # Extract webcam index
        
        self.vid_stream = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)  # Use DirectShow backend
        if not self.vid_stream.isOpened():
            messagebox.showerror("Error", "Failed to open the selected webcam.")
            return

        # Read a frame to get the dimensions
        ret, frame = self.vid_stream.read()
        self.log_debug_message(f"Started video stream with dimensions (height, width, channels): {frame.shape}")  # Should print (height, width, channels)

        self.scale_x = frame.shape[1] / self.video_width  # Original width / Display width
        self.scale_y = frame.shape[0] / self.video_height  # Original height / Display height

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_frame()

        # Log message
        self.log_debug_message(f"Started webcam: {selected_webcam} (ROI centered at: {self.roi_x}, {self.roi_y})")
        self.log_debug_message(f"Detect mode set to {self.detect_mode}")

    def stop_webcam(self):
        """ Stop the webcam feed. """
        self.is_running = False
        if self.vid_stream is not None:
            self.vid_stream.release()
            self.vid_stream = None
        self.video_frame.config(image="")
        self.match_frame.config(image="")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Log message
        self.log_debug_message("Webcam stopped.")

    def on_closing(self):
        self.stop_webcam()
        self.root.destroy()

    def update_frequency(self, value):
        self.recognition_frequency = int(value)

    def update_threshold(self, value):
        self.match_threshold = float(value)
        self.log_debug_message(f"Updated match Threshold to {self.match_threshold}")

    def update_frame(self):
        if self.is_running:
            ret, frame = self.vid_stream.read()
            if ret:
                # pass the full frame for display (will be resized)
                self.draw_roi_frame(frame)

                roi_x = int(self.roi_x)
                roi_y = int(self.roi_y)
                roi_width = int(self.roi_width)
                roi_height = int(self.roi_height)

                # Crop the frame to only contain the ROI
                roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                #roi_frame = frame[self.roi_y:self.roi_y + self.roi_height, self.roi_x:self.roi_x + self.roi_width]

                self.roi_color = self.match_color if self.match_occured else self.no_match_color

                
                #self.log_debug_message("in ret loop")

                # Start recognition in a separate thread if not already running
                if self.recognition_thread is None or not self.recognition_thread.is_alive():
                    # Pass the frame, hash_pool we've calculated for the cards in the pool, 
                    # and a pointer to the recognition queue to the find_cards function
                    self.recognition_thread = threading.Thread(target=find_cards(roi_frame,
                                                                                 self.thresh_max_value, 
                                                                                 self.block_size, 
                                                                                 self.offset_c,
                                                                                 self.kernel_size,
                                                                                 self.match_threshold,
                                                                                 self.hash_pool, 
                                                                                 self.recognition_queue,
                                                                                 self.display_mode))
                    
                    self.recognition_thread.daemon = True
                    self.recognition_thread.start()

                # Process results from the queue
                self.process_recognition_results()

                self.root.after(self.recognition_frequency, self.update_frame)

    def draw_roi_frame(self, frame):
        # Convert frame to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(image)

        if self.detect_mode == "rectangle":
            # Draw rectangle using PIL
            draw.rectangle(
                [(self.roi_x, self.roi_y),
                 (self.roi_x + self.roi_width, self.roi_y + self.roi_height)],
                outline=self.roi_color, width=4
            )

        elif self.detect_mode == "polygon":
            # Draw polygon outline
            draw.polygon([tuple(point) for point in self.polygon], outline=self.roi_color, width=4)

            point_radius = 7
            # Draw points for dragging
            for (px, py) in self.polygon:
                draw.ellipse((px - point_radius, py - point_radius, px + point_radius, py + point_radius), fill=(self.roi_color))  # Red dots

        elif self.detect_mode == "auto":
            pass #to do: create automatic detect mode

        # Resize for Tkinter display
        image_resized = image.resize((self.video_width, self.video_height), Image.LANCZOS)

       
        # Convert back to Tkinter-compatible format
        photo = ImageTk.PhotoImage(image=image_resized)

        # Update video frame in Tkinter
        self.video_frame.config(image=photo)
        self.video_frame.image = photo

    def process_recognition_results(self):
        """ Safely update UI from the main thread. """
        try:
            while not self.recognition_queue.empty():
                match_found = self.recognition_queue.get_nowait()

                if match_found:
                    self.match_occured = True
                    self.matched_image_path = match_found
                    self.match_label.config(text=f"Matched {self.matched_image_path}")
                    self.display_matched_image()
                    self.log_debug_message(f"Image match detected - {self.matched_image_path}")
                    self.root.after(1000, self.clear_match_label)
        except queue.Empty:
            pass

    def clear_match_label(self):
        self.match_label.config(text="")
        self.match_occured = False

    def display_matched_image(self):
        if self.matched_image_path:
            image = Image.open(self.matched_image_path)
            image_resized = image.resize((self.card_width, self.card_height), Image.LANCZOS)
            #image_resized = image #debugging scaling
            photo = ImageTk.PhotoImage(image=image_resized)
            self.match_frame.config(image=photo)
            self.match_frame.image = photo

            # Save the high-res matched image to a file for OBS to import
            export_path = os.path.join(os.path.dirname(__file__), "obs_export_image.png")
            image.save(export_path)
            self.log_debug_message(f"Saved high-res matched image to {export_path}")

    def select_image_folder(self):
        """ Let the user select a folder of images. """
        folder_path = filedialog.askdirectory(initialdir=self.low_res_image_folder)
        if folder_path:
            self.hash_pool = generate_hash_pool(folder_path)

    def log_debug_message(self, message):
        """ Log debug messages to the Text widget. """
        self.debug_log.config(state=tk.NORMAL)  # Enable text widget for editing
        self.debug_log.insert(tk.END, message + "\n")  # Insert the message at the end
        self.debug_log.yview(tk.END)  # Scroll to the end to show the latest message
        self.debug_log.config(state=tk.DISABLED)  # Disable text widget to prevent manual editing

    def export_to_obs(self, image_path):
        """ Export the matched image to OBS as an image source. """ 
        # OBS integration (if needed) can go here

if __name__ == "__main__":
    root = tk.Tk()
    app = net_ready_eyes(root)
    root.mainloop()