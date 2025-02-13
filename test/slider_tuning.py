import cv2
import numpy as np

from image_matcher.detect_image import *
from image_matcher.hash import *

# Initialize global variables for slider values
thresh=127
thresh_max_value = 255
block_size = 26
offset_c = 5
ksize = 4
kernel_size_single = 3
stddev = 0
d = 9
sigma_color = 75
sigma_space = 75

# Load image
image = cv2.imread("C:\\Users\\erich\\net_ready_eyes\\media\\test_images\\table_capture2.png")  # Change to your image pathf

height, width = image.shape[:2]

# Define cropping margins (adjust as needed)
left_margin = 235  # Exclude pixels from the left
right_margin = 690  # Exclude pixels from the right
top_margin = 0 # Exclude pixels from the top
bottom_margin = 0 # Exclude pixels from the bottom

playmats = []

# Crop image (excluding left & right margins)
playmats.append(image[top_margin:height - bottom_margin, left_margin:width - right_margin])

# Define cropping margins (adjust as needed)
left_margin = 690  # Exclude pixels from the left
right_margin = 250  # Exclude pixels from the right
top_margin = 0 # Exclude pixels from the top
bottom_margin = 0 # Exclude pixels from the bottom

playmats.append(image[top_margin:height - bottom_margin, left_margin:width - right_margin])

thresh_playmats = [None] * len(playmats)  # Creates a list of the right size
hashes = [None] * len(playmats)  # Creates a list of the right size

# for mat in playmats:
#     cv2.imshow('Image', mat)
#     # Wait for a key press
#     key = cv2.waitKey(0) & 0xFF

def nothing(x):
    pass

# Create a window to show the image and add sliders for parameters
cv2.namedWindow('Image')

# Threshold Max Value
cv2.createTrackbar('Thresh Max Value', 'Image', thresh_max_value, 255, nothing)

# Threshold Block Size (must be odd)
cv2.createTrackbar('Block Size', 'Image', block_size, 500, nothing)

# Threshold C value
cv2.createTrackbar('Offset C', 'Image', offset_c, 110, nothing)

# Ksize value
cv2.createTrackbar('Ksize', 'Image', ksize, 40, nothing)

# Kernel size value
cv2.createTrackbar('Kernel Size', 'Image', kernel_size_single, 40, nothing)

# d size value
cv2.createTrackbar('d', 'Image', d, 100, nothing)

# sigmaColor size value
cv2.createTrackbar('SigmaColor', 'Image', sigma_color, 200, nothing)

# sigmaSpace size value
cv2.createTrackbar('SigmaSpace', 'Image', sigma_space, 200, nothing)


# Get current values from the sliders
thresh_max_value = cv2.getTrackbarPos('Thresh Max Value', 'Image')

block_size = cv2.getTrackbarPos('Block Size', 'Image')

# Enforce the odd condition
if block_size % 2 == 0:
    block_size += 1

# Ensure it's greater than 1
if block_size <= 1:
    block_size = 3

# Get the other values from sliders
offset_c = cv2.getTrackbarPos('Offset C', 'Image')

ksize = cv2.getTrackbarPos('Ksize', 'Image')

# Enforce the odd condition
if ksize % 2 == 0:
    ksize += 1

# Ensure it's greater than 1
if ksize <= 1:
    ksize = 3

kernel_size_single = cv2.getTrackbarPos('Kernel Size', 'Image')
# Enforce the odd condition
if kernel_size_single % 2 == 0:
    kernel_size_single += 1

# Ensure it's greater than 1
if kernel_size_single <= 1:
    kernel_size_single = 3
kernel_size = (kernel_size_single, kernel_size_single)

d = cv2.getTrackbarPos('d', 'Image')
if d < 1:
    d = 1

sigma_color = cv2.getTrackbarPos('SigmaColor', 'Image')

sigma_space = cv2.getTrackbarPos('SigmaSpace', 'Image')

#we've got two playmats, so iterate over them each independently for easier thresholding
for i, img in enumerate(playmats):
    print(f"i = {i}")
    #gray, blur, and adaptive threshold
    thresh_playmats[i] = preprocess_image(img, thresh_max_value=thresh_max_value, block_size=block_size,
                                                offset_c=offset_c, ksize=ksize, kernel_size=kernel_size,  d=d, sigma_color=sigma_color, sigma_space=sigma_space)
    
    card_images = extract_card_bounding_boxes(img, thresh_playmats[i])

    print(f"found {len(card_images)} cards on playmat {i} (zero-based)")
    
    for card in card_images:    
        cv2.imshow('Card Image', card)
        key = cv2.waitKey(0) & 0xFF
        hashes.append(compute_image_hash(card))
        
    #Display the result
    # cv2.imshow('Image', pre_proc_playmats[i])
    # key = cv2.waitKey(0) & 0xFF

#hashes = generate_perceptual_hashes(card_images)



# Apply contour detection with the current parameters
# image_contours, contours = find_contours(image, ksize=ksize, thresh_max_value=thresh_max_value, block_size=block_size,
#                                              offset_c=offset_c, kernel_size=kernel_size)

#Display the result
# cv2.imshow('Image', image_contours)

#image_gray, threshold = interactive_threshold(image_gray, thresh=thresh, thresh_max_value=thresh_max_value, block_size=block_size, offset_c=offset_c)

# Show original and thresholded images side by side
# cv2.imshow("Original Image", image_gray)
#cv2.imshow("Image", threshold)

# Wait for a key press
key = cv2.waitKey(0) & 0xFF

# Close all windows
cv2.destroyAllWindows()