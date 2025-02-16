import cv2
# import tkinter as tk
# from tkinter import ttk
from ttkbootstrap.dialogs import Messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np

import os
import time

import utils.const as const
from image_matcher.hash import generate_hash_pool

class UI:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        
        # Initialize other attributes
        self.last_update_time = time.time()  # Store the initial timestamp
        # The desired frame rate in milliseconds
        self.frame_rate_ms = 100
        self.frame_rate_seconds = self.frame_rate_ms / 1000  # Convert milliseconds to seconds

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # this defines how big we want to display the matching image (high res images can sometimes be too big)
        self.card_width = 300
        self.card_height = 419

        #choose from rectangle, polygon, or auto - to do: make this selectable from a drop down
        #self.detect_mode = "polygon"
        self.detect_mode = "rectangle"
        #self.detect_mode = "auto"

        #default display mode (what shows up in the video frame)
        #self.display_mode = "thresholding"
        #self.display_mode = "unfiltered contours"
        self.display_mode = "filtered contours"
        #self.display_mode = "approx contours"

        # Coordinates for the ROI (Region of Interest) - where the playing card sized area will be placed
        self.roi_x = 2418  # X coordinate for the top-left corner
        self.roi_y = 846  # Y coordinate for the top-left corner

        # Define the size of the "region of interest"
        self.roi_width = 400
        self.roi_height = 600

        # Initialize polygon with 4 points (modify as needed)
        self.polygon = np.array([
            [self.roi_x, self.roi_y],  # Top-left
            [self.roi_x+self.roi_width, self.roi_y],  # Top-right
            [self.roi_x+self.roi_width, self.roi_y+self.roi_height],  # Bottom-right
            [self.roi_x, self.roi_y+self.roi_height]   # Bottom-left
        ], dtype=np.int32)

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

        if self.high_res_image_folder:
            new_hash_pool = generate_hash_pool(self.high_res_image_folder)
            self.app.update_hash_pool(new_hash_pool)

        # The delay between frames in millisecond for the video feed to update
        # We don't need this to be as fast as the camera as we don't want to tax the system
        self.frame_rate_ms = 100

        self.root.geometry("1920x1080")

        self.video_width = 854  # Default width, update dynamically if needed
        self.video_height = 480  # Default height, update dynamically if needed

        # padding values for Tkinter (ui layout)
        self.padx = 1
        self.pady = 1

        # Create GUI components
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid_rowconfigure(0, weight=1, minsize=self.video_height)
        self.main_frame.grid_columnconfigure(0, weight=1, minsize=self.video_width)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        self.video_label = ttk.Label(self.main_frame, text="Video Feed")
        self.control_frame = ttk.Frame(self.main_frame)
        self.process_frame = ttk.Label(self.main_frame)
        self.match_frame = ttk.Label(self.main_frame)
        self.debug_log = ttk.Text(self.main_frame, height=20, width=100, wrap=WORD, state=DISABLED)
        self.video_source_label = ttk.Label(self.control_frame, text="Select Video Source:")
        self.video_source_combobox = ttk.Combobox(self.control_frame, values=self.app.available_video_sources)
        self.detect_mode_label = ttk.Label(self.control_frame, text="Detection Mode:")
        # self.detect_mode_combobox = ttk.Combobox(self.control_frame, values=["polygon", "rectangle", "auto"]) 
        self.detect_mode_combobox = ttk.Combobox(self.control_frame, values=["rectangle"]) #rectangle only for now
        self.display_mode_label = ttk.Label(self.control_frame, text="Display Mode:")
        self.display_mode_combobox = ttk.Combobox(self.control_frame, values=["none", "thresholding", "unfiltered contours", "filtered contours", "approx contours"])
        self.toggle_video_button = ttk.Button(self.control_frame, text="Start video source", command=self.toggle_video_source, bootstyle=SUCCESS)
        self.select_button = ttk.Button(self.control_frame, text="Change Image Folder", command=self.select_image_folder)
        self.folder_label = ttk.Label(self.control_frame, text=f"Current: {self.low_res_image_folder}")
        self.proc_period_label = ttk.Label(self.control_frame, text="Image proc delay (.1 to 2 sec):")
        self.proc_period_slider = ttk.Scale(self.control_frame, from_=100, to_=2000, orient=HORIZONTAL, command=self.app.update_proc_period)
        self.block_size_label = ttk.Label(self.control_frame, text="Block Size: 27")
        self.offset_c_label = ttk.Label(self.control_frame, text="Offset C: 5")
        self.kernel_size_label = ttk.Label(self.control_frame, text="Kernel Size (nxn): 5")
        self.match_threshold_label = ttk.Label(self.control_frame, text="Match Threshold: 20")
        self.card_min_area_label = ttk.Label(self.control_frame, text="Card Min Area: 1000")
        self.card_max_area_label = ttk.Label(self.control_frame, text="Card Max Area: 10000")
        # self.threshold_label = ttk.Label(self.control_frame, text="Image Detection Threshold (perc of keypoints:")
        # self.threshold_slider = ttk.Scale(self.control_frame, from_=0, to=100, orient=HORIZONTAL, command=self.app.update_threshold)
        self.block_size_spinbox = ttk.Spinbox(
            self.control_frame, from_=3, to=101, increment=2, command=lambda: self.update_block_size_label(self.block_size_spinbox.get())
        )
        self.block_size_spinbox.set(11)  # Default value
        self.offset_c_spinbox = ttk.Spinbox(
            self.control_frame, from_=1, to=100, increment=1, command=lambda: self.update_offset_c_label(self.offset_c_spinbox.get())
        )
        self.offset_c_spinbox.set(5)  # Default value
        self.kernel_size_spinbox = ttk.Spinbox(
            self.control_frame, from_=1, to=51, increment=2, command=lambda: self.update_kernel_size_label(self.kernel_size_spinbox.get())
        )
        self.kernel_size_spinbox.set(5)  # Default value
        self.match_threshold_spinbox = ttk.Spinbox(
            self.control_frame, from_=1, to=100, increment=1, command=lambda: self.update_match_threshold_label(self.match_threshold_spinbox.get())
        )
        self.match_threshold_spinbox.set(20)  # Default value
        self.card_min_area_spinbox = ttk.Spinbox(
                                                                                          
            self.control_frame, from_=5000, to=200000, increment=1000, command=lambda: self.update_card_min_area_label(self.card_min_area_spinbox.get())
        )
        self.card_min_area_spinbox.set(60000)  # Default value
        self.card_max_area_spinbox = ttk.Spinbox(
            self.control_frame, from_=5000, to=200000, increment=1000, command=lambda: self.update_card_max_area_label(self.card_max_area_spinbox.get())
        )
        self.card_max_area_spinbox.set(80000)  # Default value

        self.match_label = ttk.Label(self.control_frame, text="", font=("Arial", 12, "bold"), foreground="green")

        self.detect_mode_combobox.set(self.detect_mode)  # populate the box with the current value
        self.display_mode_combobox.set(self.display_mode)  # populate the box with the current value

        self.detect_mode_combobox.bind("<<ComboboxSelected>>", self.on_detect_mode_change)
        self.display_mode_combobox.bind("<<ComboboxSelected>>", self.on_display_mode_change)
        #left-click in the video_label
        self.video_label.bind("<ButtonPress-1>", self.on_mouse_press)
        #move while holding left-click in the video_label
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)
        #release left-click
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_release)
        #move the mouse while not clicking in the video_label
        self.video_label.bind("<Motion>", self.on_mouse_move)        

        self.proc_period_slider.set(500)  # Default in milliseconds

        # Place the sub-frames inside main_frame
        self.video_label.grid(row=0, column=0, padx=self.padx, pady=self.pady, sticky="nsew")
        self.process_frame.grid(row=0, column=1, padx=self.padx, pady=self.pady)
        self.match_frame.grid(row=0, column=2, padx=self.padx, pady=self.pady)
        self.control_frame.grid(row=1, column=0, padx=self.padx, pady=self.pady, sticky="ew")

        
        # self.debug_frame.grid(row=1, column=0, padx=self.padx, pady=self.pady)
        # self.debug_log.grid(row=0, column=0, padx=self.padx, pady=self.pady)
        
        
        # Place buttons inside control_frame
        self.video_source_label.grid(row=0, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.video_source_combobox.grid(row=0, column=1, padx=self.padx, pady=self.pady, sticky="w")
        self.toggle_video_button.grid(row=0, column=2, padx=self.padx, pady=self.pady)
        self.select_button.grid(row=1, column=0, padx=self.padx, pady=self.pady)
        self.folder_label.grid(row=1, column=1, padx=self.padx, pady=self.pady)
        self.detect_mode_label.grid(row=3, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.detect_mode_combobox.grid(row=3, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.display_mode_label.grid(row=4, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.display_mode_combobox.grid(row=4, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.proc_period_label.grid(row=5, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.proc_period_slider.grid(row=5, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.block_size_label.grid(row=6, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.block_size_spinbox.grid(row=6, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.offset_c_label.grid(row=7, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.offset_c_spinbox.grid(row=7, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.kernel_size_label.grid(row=8, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.kernel_size_spinbox.grid(row=8, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.match_threshold_label.grid(row=9, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.match_threshold_spinbox.grid(row=9, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.card_min_area_label.grid(row=10, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.card_min_area_spinbox.grid(row=10, column=1,padx=self.padx, pady=self.pady, sticky="w")
        self.card_max_area_label.grid(row=11, column=0, padx=self.padx, pady=self.pady, sticky="e")
        self.card_max_area_spinbox.grid(row=11, column=1,padx=self.padx, pady=self.pady, sticky="w")

        self.img_on_process_frame = None  # Keep track of the current image on the canvas

        # Default to first video_source in the list
        if len(self.app.available_video_sources) >= 2:
            self.video_source_combobox.set(self.app.available_video_sources[2]) #default to #2 - camlink 4k on eric's systems
        elif self.app.available_video_sources:
            self.video_source_combobox.set(self.app.available_video_sources[0])

        # Bind mouse events for moving/resizing the ROI

        self.roi_dragging = False
        self.roi_resizing = False
        self.roi_drag_offset = (0, 0)

        # Slider to control the image processing_period

        # Slider to adjust the image detection threshold

        # self.threshold_label.grid()


        # self.threshold_slider.set(20)  # Default threshold
        # self.threshold_slider.grid()
        # self.match_threshold = self.threshold_slider.get()  # Set initial match threshold in case it isn't used.
        
        self.match_label.grid()

        #opencv uses BGR (Blue, Green, Red) by default
        self.roi_color = (132, 255, 0) #green

        self.show_default_screen()

    def update_block_size_label(self, value):
        """Update the block_size_label with current spinbox value."""
        self.block_size_label.config(text=f"Block Size: {value}")
        self.app.rec_params['block_size'] = int(value)

    def update_offset_c_label(self, value):
        """Update the offset_c_label with current spinbox value."""
        self.offset_c_label.config(text=f"Offset C: {value}")
        self.app.rec_params['offset_c'] = int(value)

    def update_kernel_size_label(self, value):
        """Update the kernel_size_label with current spinbox value."""
        self.kernel_size_label.config(text=f"Kernel Size: {value}")
        self.app.rec_params['kernel_size'] = (int(value), int(value))

    def update_match_threshold_label(self, value):
        """Update the match_threshold_label with current spinbox value."""
        self.match_threshold_label.config(text=f"Match Threshold: {value}")
        self.app.rec_params['match_threshold'] = int(value)

    def update_card_min_area_label(self, value):
        """Update the card_min_area_label with current spinbox value."""
        self.card_min_area_label.config(text=f"Min. Card Area: {value}")
        self.app.rec_params['card_min_area'] = int(value)

    def update_card_max_area_label(self, value):
        """Update the card_max_area_label with current spinbox value."""
        self.card_max_area_label.config(text=f"Max. Card Area: {value}")
        self.app.rec_params['card_max_area'] = int(value)


    #Tkinter isn't thread safe, so we need to schedule an update to the image in the main thread
    def display_image(self, img):
        """Schedule the image update in the main thread."""
        self.root.after(0, self._update_image, img)  # Schedule the update in the main thread


    def _update_image(self, img):
        """Update the UI with the image in the main thread."""
        pil_image = Image.fromarray(img)  # Convert the image array to PIL format
        tk_image = ImageTk.PhotoImage(pil_image)  # Convert PIL image to Tkinter format

        # Update the canvas image (this will update it every time a new image is passed)
        if self.process_frame:
            self.process_frame.config(image=tk_image) #update the image
            self.process_frame.image = tk_image # keep reference to avoid garbage collection
        
    def toggle_video_source(self):
        """ Start the selected video_source feed and recognition loop. """
        if self.app.is_running():
            self.toggle_video_button.config(text="Start Video Source", bootstyle=SUCCESS)
            # first stop the app from trying to process images
            self.app.stop()

            if self.video_source is not None:
                self.video_source.release()
                self.video_source = None
            self.match_frame.config(image="")

            self.show_default_screen()

            # Log message
            self.log_msg("video_source stopped.")

        else:
            selected_video_source = self.video_source_combobox.get()
            video_source_index = int(selected_video_source.split(" ")[0])  # Extract video_source index
            
            self.video_source = cv2.VideoCapture(video_source_index, cv2.CAP_DSHOW)  # Use DirectShow backend
            
            if not self.video_source.isOpened():
                messagebox.showerror("Error", "Failed to open the selected video_source.")
                return

            # Read a frame to get the dimensions
            ret, frame = self.video_source.read()
            self.log_msg(f"Started video stream with dimensions (height, width, channels): {frame.shape}")  # Should print (height, width, channels)

            self.set_scale_factors()

            # Update button to indicate it's now "Stop"
            self.toggle_video_button.config(text="Stop Video Source", bootstyle=DANGER)

            # kicks off a process of periodically updating the video frame
            self.update_frame()

            #start image recognition in main app
            self.app.start()

            # Log message
            self.log_msg(f"Started video_source: {selected_video_source} (ROI centered at: {self.roi_x}, {self.roi_y})")
            self.log_msg(f"Detect mode set to {self.detect_mode}")


    def on_mouse_press(self, event):
        """Determine if resizing or moving the whole ROI."""
        if not (hasattr(self, 'video_source') and self.video_source and self.video_source.isOpened()):
            return # none of the below should occur if no video feed is active
        
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
        if not (hasattr(self, 'video_source') and self.video_source and self.video_source.isOpened()):
            return # none of the below should occur if no video feed is active
        
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

    def on_mouse_release(self, event):
        if not (hasattr(self, 'video_source') and self.video_source and self.video_source.isOpened()):
            return # none of the below should occur if no video feed is active
        
        self.log_msg(f"ROI moved to position ({self.roi_x},{self.roi_y})")
    
        """End any dragging or resizing action."""
        self.dragging_point = None
        self.roi_dragging = False
        self.roi_resizing = None

    def on_mouse_move(self, event):
        """Change cursor when near ROI edges to indicate resizing."""
        if not (hasattr(self, 'video_source') and self.video_source and self.video_source.isOpened()):
            return # none of the below should occur if no video feed is active
        
        # Scale mouse event coordinates
        x = event.x * self.scale_x
        y = event.y * self.scale_y
        
        margin = 30

        # Corner resizing
        if (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_label.config(cursor="size_nw_se")
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_label.config(cursor="size_ne_sw")
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.video_label.config(cursor="size_ne_sw")
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.video_label.config(cursor="size_nw_se")
        
        # Vertical resizing (top and bottom edges)**
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_label.config(cursor="size_ns")  # Top edge
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y + self.roi_height - margin <= y <= self.roi_y + self.roi_height + margin):
            self.video_label.config(cursor="size_ns")  # Bottom edge
        
        # Horizontal resizing (left and right edges)**
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.video_label.config(cursor="size_we")  # Left edge
        elif (self.roi_x + self.roi_width - margin <= x <= self.roi_x + self.roi_width + margin and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.video_label.config(cursor="size_we")  # Right edge
        
        # Move cursor when inside ROI
        elif (self.roi_x <= x <= self.roi_x + self.roi_width and
            self.roi_y <= y <= self.roi_y + self.roi_height):
            self.video_label.config(cursor="fleur")  # Move cursor
        else:
            self.video_label.config(cursor="")  # Default

    def update_video_label(self, photo):
        """ Update the video frame with the passed image. """
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def update_match_frame(self, photo):
        """ Update the matched image frame. """
        self.match_frame.config(image=photo)
        self.match_frame.image = photo

    def update_match_label(self, text):
        """ Update the match label with text. """
        self.match_label.config(text=text)

    def select_image_folder(self):
        """ Let the user select a folder of images. """
        folder_path = filedialog.askdirectory(initialdir=self.low_res_image_folder)
        if folder_path:
            new_hash_pool = generate_hash_pool(folder_path)
            self.app.update_hash_pool(new_hash_pool)

    def on_detect_mode_change(self, event):
            """This function is triggered when the user selects a different detection mode from the dropdown."""
            selected_mode = self.detect_mode_combobox.get()
            self.detect_mode = selected_mode  # Update the detect_mode variable

            # Optionally, you can update the UI or log the change
            print(f"Detection Mode set to: {self.detect_mode}")
            self.log_msg(f"Detection Mode set to: {self.detect_mode}")
        

    def on_display_mode_change(self, event):
        """This function is triggered when the user selects a different detection mode from the dropdown."""
        selected_mode = self.display_mode_combobox.get()
        self.display_mode = selected_mode  # Update the display_mode variable

        # Optionally, you can update the UI or log the change
        print(f"Detection Mode set to: {self.display_mode}")
        self.log_msg(f"Detection Mode set to: {self.display_mode}")


    def get_roi_frame(self):
        ret, frame = self.video_source.read()
        if ret:
            roi_x_int = int(self.roi_x)
            roi_y_int = int(self.roi_y)
            roi_width_int = int(self.roi_width)
            roi_height_int = int(self.roi_height)
            return frame[roi_y_int:roi_y_int + roi_height_int, roi_x_int:roi_x_int + roi_width_int]
        
    def update_frame(self):
        """Update the video frame with the latest frame from the video source."""

        # # DEBUG LOGIC: Frame rate seemed too slow
        # call_time = time.time()  # Get the current time
        # frame_time = call_time - self.last_update_time  # Calculate time since last update
        # # Update the last update time
        # self.last_update_time = call_time
        # print(f"Time elapsed since last update: {frame_time:.4f} seconds")

        start_time = time.time()  # Start the timer

        ret, frame = self.video_source.read()

        if frame is None:
            print("Warning: Received None frame in update_frame")  # Debugging message
            return
        
        if ret:
            # Process frame to draw ROI (function handles resizing)
            self.draw_roi_frame(frame)

        # Calculate how long the frame processing took
        elapsed_time = time.time() - start_time
        # print(f"Time elapsed for processing the frame: {elapsed_time:.4f} seconds")

        # Calculate the remaining time to maintain the target frame rate
        remaining_time = self.frame_rate_seconds - elapsed_time

        # If we need to wait (which we should), schedule the next frame update
        if remaining_time > 0:
            # Schedule the next frame update (Tkinter scheduling callback)
            self.root.after(int(remaining_time * 1000), self.update_frame)  # Convert to milliseconds
        else:
            # If processing took longer than the target frame rate, update immediately
            print("WARNING: Frame Update processing taking longer than frame display rate allows.")
            self.root.after(1, self.update_frame()) # Ensures Tkinter event loop remains stable


    def draw_roi_frame(self, frame):
        
        # Ensure ROI coordinates are within the frame bounds
        self.roi_x = max(0, min(self.roi_x, frame.shape[1] - self.roi_width))
        self.roi_y = max(0, min(self.roi_y, frame.shape[0] - self.roi_height))
        # Ensure ROI width and height do not exceed the frame size
        self.roi_width = min(self.roi_width, frame.shape[1] - self.roi_x)
        self.roi_height = min(self.roi_height, frame.shape[0] - self.roi_y)

        # Convert frame to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(image)

        if self.detect_mode == "rectangle":
            # Draw rectangle using PIL
            draw.rectangle(
                [(self.roi_x, self.roi_y),
                 (self.roi_x + self.roi_width, self.roi_y + self.roi_height)],
                outline=self.roi_color, width=5
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
        else:
            print("ROI Display Error: No detect mode set")

        # Resize for Tkinter display
        image_resized = image.resize((self.video_width, self.video_height), Image.LANCZOS)

        # Convert back to Tkinter-compatible format
        photo = ImageTk.PhotoImage(image=image_resized)

        # # Display the image in a separate OpenCV window
        # cvimage = np.array(image_resized)
        # cv2.imshow('DEBUG', cvimage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        self.video_label.config(image=photo) # Update video frame in Tkinter
        self.video_label.image = photo # Keep a reference to prevent garbage collection

    def log_msg(self, message):
        """ Log debug messages to the Text widget. """
        self.debug_log.config(state=ttk.NORMAL)  # Enable text widget for editing
        self.debug_log.insert(ttk.END, message + "\n")  # Insert the message at the end
        self.debug_log.yview(ttk.END)  # Scroll to the end to show the latest message
        self.debug_log.config(state=ttk.DISABLED)  # Disable text widget to prevent manual editing

    def set_scale_factors(self):
        ret, frame = self.video_source.read()
        if ret:
            self.scale_x = frame.shape[1] / self.video_width  # Original width / Display width
            self.scale_y = frame.shape[0] / self.video_height  # Original height / Display height

    def display_match(self, match_img_path):
        if match_img_path:
            # self.match_label.config(text=f"Matched {match_img_path}")
            self.log_msg(f"Image match detected - {match_img_path}")
            self.root.after(1000, self.clear_match_label)

            image = Image.open(match_img_path)
            image_resized = image.resize((self.card_width, self.card_height), Image.LANCZOS)
            #image_resized = image #debugging scaling
            photo = ImageTk.PhotoImage(image=image_resized)
            self.match_frame.config(image=photo)
            self.match_frame.image = photo

            # Save the high-res matched image to a file for OBS to import
            export_path = os.path.join(os.path.dirname(__file__), "obs_export_image.png")
            image.save(export_path)
            # self.log_msg(f"Saved high-res matched image to {export_path}")

    def clear_match_label(self):
        self.match_label.config(text="")
        self.match_occured = False

    def show_default_screen(self):
        """Display a black box with 'Start a video feed' text before a video is selected."""
        # Create a black image
        width, height = self.video_width, self.video_height
        image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(image)

        # Define text and font
        text = "Start a Video Source"
        text_color = "white"
        font_size = 24

        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # Use system font
        except IOError:
            font = ImageFont.load_default()  # Fallback if arial.ttf is not found

        # Get text bounding box for centering
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate centered position
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2

        # Draw the text
        draw.text((text_x, text_y), text, fill=text_color, font=font)

        # Convert to Tkinter-compatible format
        photo = ImageTk.PhotoImage(image)

        # if image:
        #     print("displaying default video_label")
        # else:
        #     print("problem with displaying default video_label")

        # Display the default image in the video frame
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Keep reference to prevent garbage collection

    def on_closing(self):
        if self.app.is_running():
            self.toggle_video_source()
        self.root.destroy()
