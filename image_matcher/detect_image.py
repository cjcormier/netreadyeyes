import cv2
import numpy as np
import pandas as pd
#from PIL import Image, ImageDraw, ImageTk

from image_matcher.hash import find_minimum_hash_difference
from utils.utils import convert_image_to_opencv, convert_image_to_pil

def find_cards(query_image_path, hash_pool, recognition_queue):
    print("called find_cards")
    query_image = cv2.imread(query_image_path)

    height, width = query_image.shape[:2]
    
    # Define cropping margins (adjust as needed)
    left_margin = 235  # Exclude pixels from the left
    right_margin = 250  # Exclude pixels from the right
    top_margin = 0 # Exclude pixels from the top
    bottom_margin = 0 # Exclude pixels from the bottom

    # Crop image (excluding left & right margins)
    image_cropped = query_image[top_margin:height - bottom_margin, left_margin:width - right_margin]

    # Display the image using OpenCV
    #query_image_resized = cv2.resize(query_image, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
    # Convert to grayscale for drawing
    # query_image_copy = query_image.copy()

    # cv2.imshow(f"query_image - {query_image}", query_image)
    # cv2.imshow(f"query_image_copy - {query_image_copy}", query_image_copy)
    # print(f"query_image_path = {query_image_path}")
    # print(f"query_image = {query_image}")
    # print(f"query_image_copy = {query_image_copy}")

    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    #query_image = Image.open(query_image_path)
    contours = find_contours(image_cropped)

    for n, contour in enumerate(contours):
        print(f"in loop: n = {n}, contour = {contour}")

        # Get the rectangle points from the contour
        rectangle_points = _get_rectangle_points_from_contour(contour)

        print(f"rectangle_points = {rectangle_points}")

        # Draw the rectangle on the query_image_resized
        # Convert points to integer for drawing (they may be floats after _get_rectangle_points_from_contour())
        rectangle_points_int = rectangle_points.astype(int)

        # Draw the rectangle (quadrilateral) on the image
        cv2.polylines(query_image_copy, [rectangle_points_int], isClosed=True, color=(0, 255, 0), thickness=2)

        # You can also draw individual points (optional, for visual verification)
        for point in rectangle_points_int:
            cv2.circle(query_image_copy, tuple(point), 5, (0, 0, 255), -1)

        # Show the image with the rectangle overlayed
        cv2.imshow(f"Rectangle Overlay - {query_image_path}", query_image_copy)
            
        # cv2.imshow(query_image_path, query_image_resized)

        # Wait for a key press to close the window
        cv2.waitKey(0)

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        card_image = _four_point_transform(query_image, rectangle_points)
        card, diff = find_minimum_hash_difference(query_image_path, hash_pool)
        if _possible_match(diff):

            print(f"find_minimum_hash_difference returned {card['name']} with a diff of {diff}.")

            # # Display the image using OpenCV
            # cv2.imshow(card['name'], card_image)

            # # Wait for a key press to close the window
            # cv2.waitKey(0)

            # # Close all OpenCV windows
            # cv2.destroyAllWindows()

            # Unpack card path and diff from find_minimum_hash_difference
            min_dist_df, diff =  (card_image, hash_pool)

            # Send the result back to the recognition_queue
            recognition_queue.put(min_dist_df['name'])      
    

def _possible_match(diff):
    if diff < 450: #To-do: make this number based on the threshold slider
        return True


def _get_rectangle_points_from_contour(contour):
    return np.float32([p[0] for p in contour])


def _four_point_transform(image, pts, for_display=False):
    """Transform a quadrilateral section of an image into a rectangular area.
    Parameters
    ----------
    image : Image
        source image
    pts : np.array

    Returns
    -------
    Image
        Transformed rectangular image
    """
    rect = _order_points(pts)

    spacing_around_card = 0
    double_spacing_around_card = 0
    if for_display:
        spacing_around_card = 100
        double_spacing_around_card = 2 * spacing_around_card

    max_height, max_width = _get_edges(double_spacing_around_card, rect)
    transformed_image = _warp_image(image, max_height, max_width, rect,
                                    spacing_around_card)
    if _image_is_horizontal(max_width, max_height):
        transformed_image = rotate_image(max_height, max_width, transformed_image)
    return transformed_image


def _order_points(pts):
    """Initialize a list of coordinates that will be ordered such that the first entry in the list is the top-left,
        the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left.

    Parameters
    ----------
    pts : np.array

    Returns
    -------
    : ordered list of 4 points
    """

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # the top-left point will have the smallest sum, whereas
    rect[2] = pts[np.argmax(s)]  # the bottom-right point will have the largest sum

    diff = np.diff(pts, axis=1)     # now, compute the difference between the points, the
    rect[1] = pts[np.argmin(diff)]  # top-right point will have the smallest difference,
    rect[3] = pts[np.argmax(diff)]  # whereas the bottom-left will have the largest difference
    return rect


def _get_edges(double_spacing_around_card, rect):
    (tl, tr, br, bl) = rect
    max_width = max(int(_get_edge(bl, br)), int(_get_edge(tl, tr)))
    max_width += double_spacing_around_card
    max_height = max(int(_get_edge(br, tr)), int(_get_edge(bl, tl)))
    max_height += double_spacing_around_card
    return max_height, max_width


def _get_edge(bl, br):
    return np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))


def _warp_image(image, max_height, max_width, rect, spacing_around_card):
    transformation_array = np.array([
        [0 + spacing_around_card, 0 + spacing_around_card],
        [max_width - spacing_around_card, 0 + spacing_around_card],
        [max_width - spacing_around_card, max_height - spacing_around_card],
        [0 + spacing_around_card, max_height - spacing_around_card]
    ],
        dtype="float32"
    )
    applied_transformation_matrix = cv2.getPerspectiveTransform(rect,
                                                                transformation_array)
    warped_matrix = cv2.warpPerspective(image, applied_transformation_matrix,
                                        (max_width, max_height))
    return warped_matrix


def _image_is_horizontal(max_width, max_height):
    return max_width > max_height


def rotate_image(max_height, max_width, transformed_image):
    center = (max_width / 2, max_height / 2)
    rotated_applied_transformation_matrix = cv2.getRotationMatrix2D(center, 270, 1.0)
    transformed_image = cv2.warpAffine(transformed_image,
                                       rotated_applied_transformation_matrix,
                                       (max_height, max_width))
    return transformed_image



# ksize = (Kernel Size for Median Blur)
# Used in: cv2.medianBlur(image_gray, ksize)

# Effect:
# Controls how much the image is smoothed before edge detection.
# Larger values (e.g., ksize=7, 9) remove more noise but might blur out small details.
# Smaller values (e.g., ksize=3) preserve details but keep more noise.
# Tuning Advice:
# If small noise is interfering with contour detection, try increasing ksize (e.g., 7 or 9).
# If contours are disappearing, reduce ksize (e.g., 3 or 5).

# thresh_max_value (Maximum Threshold Value)
# Used in: cv2.adaptiveThreshold(image_blur, thresh_max_value, ...)

# Effect:
# Sets the maximum intensity assigned to pixels that pass the thresholding condition.
# Usually left at 255 (max intensity for 8-bit images).
# Lower values will make thresholding less aggressive.
# Tuning Advice:
# Keep it at 255 unless you need a softer binary mask.
# If you see too much white noise, you might reduce it slightly (e.g., 200-230).


# def find_contours(image, ksize=5, thresh_max_value=255, thresh_block_size=199, thresh_c=5,
# #                   kernel_size=(3, 3)):

# 3️⃣ thresh_block_size (Neighborhood Size for Adaptive Thresholding)
# 🔹 Used in: cv2.adaptiveThreshold(..., thresh_block_size, thresh_c)

# Effect:
# Defines the size of the local region (block) used to calculate the threshold for each pixel.
# Must be an odd number (e.g., 3, 5, 199).
# Larger values result in smoother thresholding.
# Smaller values make thresholding more localized, which helps with small details.
# Tuning Advice:
# If some contours are lost, reduce thresh_block_size (e.g., from 199 → 51).
# If background noise appears too much, increase thresh_block_size (e.g., 199 or more).
# 4️⃣ thresh_c (Constant Subtracted in Adaptive Thresholding)
# 🔹 Used in: cv2.adaptiveThreshold(..., thresh_block_size, thresh_c)

# Effect:
# A fine-tuning parameter that shifts the threshold up or down.
# Positive values make the threshold more aggressive, turning more pixels black.
# Negative values make the threshold less aggressive, keeping more pixels white.
# Tuning Advice:
# If you are missing contours, decrease thresh_c (e.g., 5 → 3).
# If too much background noise is detected, increase thresh_c (e.g., 5 → 10).
# 5️⃣ kernel_size (Morphological Kernel Size)
# 🔹 Used in:

# python
# Copy
# Edit
# kernel = np.ones(kernel_size, np.uint8)
# image_dilate = cv2.dilate(image_thresh, kernel, iterations=1)
# image_erode = cv2.erode(image_dilate, kernel, iterations=1)
# Effect:
# Defines the size of the structuring element used for morphological operations (dilate & erode).
# Larger kernels (e.g., (5,5)) remove small noise but can distort small contours.
# Smaller kernels (e.g., (3,3)) preserve details but may not clean up noise well.
# Tuning Advice:
# If small noise remains, increase kernel size (e.g., (3,3) → (5,5)).
# If contours become too thick or connected, reduce kernel size.

def find_contours(image, ksize=3, # Smooth out noise but keep card edges
                  thresh_max_value=255, # Keep maximum threshold intensity
                  thresh_block_size=101, # Adapt to local contrast without over-smoothing - must be odd
                  thresh_c=10, # Remove some noise
                  kernel_size=(5, 5) # Strengthen edges for better contour detection
                  ):

    print("in find_contours")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.medianBlur(image_gray, ksize)

    # old algorithm
    # image_thresh = cv2.adaptiveThreshold(image_blur, thresh_max_value,
    #                                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
    #                                    thresh_block_size, thresh_c)
    
    image_thresh = cv2.adaptiveThreshold(image_blur, thresh_max_value,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    thresh_block_size, thresh_c)
    
    kernel = np.ones(kernel_size, np.uint8)


    # old algorithm
    # image_dilate = cv2.dilate(image_thresh, kernel, iterations=1)
    # image_erode = cv2.erode(image_dilate, kernel, iterations=1)
    # contours, hierarchy = cv2.findContours(image_erode, cv2.RETR_EXTERNAL,
    #                                        cv2.CHAIN_APPROX_TC89_KCOS)

    # Apply stronger morphological operations**
    image_morph = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        image_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Draw all contours on the image to visualize
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Detected Contours", image_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if contours:
        #print(f"counters = {contours}")
        print("contours was not empty")
        rectangular_contours = find_rectangular_contours(contours, hierarchy)
        if rectangular_contours:
            print(f"rectangular_counters = {contours}")
            print("rectangular_contours was not empty")
            return rectangular_contours
        else:
            print("No rectangular_contours found")        
    
    print("No contours found")
    return []


def find_rectangular_contours(contours, hierarchy):
    print("in find_rectangular_contours")
    stack = _get_stack(hierarchy)
    rectangular_contours = []
    while len(stack) > 0:
        i_contour, h = stack.pop()
        i_next, i_prev, i_child, i_parent = h
        if i_next != -1:
            stack.append((i_next, hierarchy[0][i_next]))
        contour, area = _find_bounded_contour(contours, i_contour)
        if _threshold_size_bounded_by(area) and _is_rectangular(contour):
            rectangular_contours.append(contour)
        elif i_child != -1:
            stack.append((i_child, hierarchy[0][i_child]))
    return rectangular_contours


def _get_stack(hierarchy):
    return [
        (0, hierarchy[0][0]),
    ]


def _find_bounded_contour(contours, i_contour):
    contour = contours[i_contour]
    size = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    return approx, size


def _is_rectangular(contour):
    return len(contour) == 4


def _threshold_size_bounded_by(area, min_area=5000, max_area=200000):
    print(f"Contour area: {area}")
    return min_area <= area <= max_area