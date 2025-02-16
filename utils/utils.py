import numpy as np
import cv2
from PIL import Image

def convert_image_to_opencv(image):
    # If the input image is a PIL Image, convert it to a NumPy array
    if isinstance(image, Image.Image):
        print(f"image is a PIL Image, converting to opencv - aka NumPy array")
        image = np.array(image)
    elif is_opencv(image):
        print(f"image is already an opencv_image aka NumPy array - nothing to do")
    else:
        print(f"image is neither a PIL Image nor an opencv_image")
    return image

def convert_image_to_pil(image):
    # If the input image is an OpenCV Image (NumPy array), convert it to a PIL Image
    if is_opencv(image):  
        print(f"image is an opencv_image, converting to PIL")
        # Convert from BGR to RGB (since OpenCV uses BGR and PIL uses RGB)
        opencv_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Fixed here: 'image' instead of 'opencv_image'
        # Convert the NumPy array (RGB) to a PIL Image object
        image = Image.fromarray(opencv_image_rgb)
    elif is_pil(image):
        print(f"image is already a PIL Image")
    else:
        print(f"image is neither a PIL Image nor an opencv_image")
    return image

def is_opencv(image):
    return isinstance(image, np.ndarray)

def is_pil(image):
    return isinstance(image, Image.Image)