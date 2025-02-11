import cv2
import numpy as np

# Initialize global variables for slider values
thresh_block_size = 101
thresh_c = 10
ksize = 3
kernel_size = (5, 5)

# Load image
image = cv2.imread("C:\\Users\\erich\\net_ready_eyes\\media\\test_images\\table_capture2.png")  # Change to your image path


# Convert image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the window and sliders
def nothing(x):
    pass

# Create a window to show the image and add sliders for parameters
cv2.namedWindow('Contour Detection')

# Threshold Block Size (must be odd)
cv2.createTrackbar('Thresh Block Size', 'Contour Detection', thresh_block_size, 255, nothing)

# Threshold C value
cv2.createTrackbar('Thresh C', 'Contour Detection', thresh_c, 50, nothing)

# Kernel Size (blur size)
cv2.createTrackbar('Ksize', 'Contour Detection', ksize, 9, nothing)

# Kernel Size for Morphological operations
cv2.createTrackbar('Kernel Size', 'Contour Detection', 5, 10, nothing)


def find_contours(image, ksize=3, thresh_max_value=255, thresh_block_size=101, thresh_c=10, kernel_size=(5, 5)):
    # Convert the image to grayscale and apply median blur
    image_blur = cv2.medianBlur(image, ksize)

    # Apply adaptive thresholding
    image_thresh = cv2.adaptiveThreshold(image_blur, thresh_max_value,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         thresh_block_size, thresh_c)

    # Perform morphological operations (closing and opening)
    kernel = np.ones(kernel_size, np.uint8)
    image_morph = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(image_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

    return image_contours, contours

def save_parameters():
    # Save current slider values to a JSON file
    params = {
        "thresh_block_size": thresh_block_size,
        "thresh_c": thresh_c,
        "ksize": ksize,
        "kernel_size": kernel_size[0]  # Assuming square kernel size
    }

    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {params_file}")

while True:
    # Get current values from the sliders
    thresh_block_size = cv2.getTrackbarPos('Thresh Block Size', 'Contour Detection')

    # Enforce the odd condition
    if thresh_block_size % 2 == 0:
        thresh_block_size += 1

    # Ensure it's greater than 1
    if thresh_block_size <= 1:
        thresh_block_size = 3
    
    # Get the other values from sliders
    thresh_c = cv2.getTrackbarPos('Thresh C', 'Contour Detection')
    ksize = cv2.getTrackbarPos('Ksize', 'Contour Detection')
    kernel_size_value = cv2.getTrackbarPos('Kernel Size', 'Contour Detection')
    
    if ksize % 2 == 0:  # Ensure ksize is always odd
        ksize += 1
    if kernel_size_value % 2 == 0:  # Ensure kernel_size is always odd
        kernel_size_value += 1
    
    kernel_size = (kernel_size_value, kernel_size_value)

    # Apply contour detection with the current parameters
    image_contours, contours = find_contours(image_gray, ksize=ksize, thresh_block_size=thresh_block_size,
                                             thresh_c=thresh_c, kernel_size=kernel_size)

    # Display the result
    cv2.imshow('Contour Detection', image_contours)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key ==ord('s'): # 's' key to save parameters
        save_parameters()
        
# Close all windows
cv2.destroyAllWindows()