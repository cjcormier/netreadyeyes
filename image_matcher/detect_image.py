import cv2
import numpy as np
import pandas as pd
#from PIL import Image, ImageDraw, ImageTk

from image_matcher.hash import find_minimum_hash_difference
from utils.utils import convert_image_to_opencv, convert_image_to_pil

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000
# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

def find_cards(query_image, 
               thresh_max_value=255, 
               block_size=101, 
               offset_c=10, 
               kernel_size=(5,5),
               match_threshold=25,
               hash_pool=[],
               recognition_queue=[],
               display_mode=None):
    
    thresh_image = preprocess_image(query_image, thresh_max_value, block_size, offset_c, kernel_size)

    if display_mode == "thresholding":
        cv2.imshow('thresholding', thresh_image)

    card_sized_imgs = extract_card_bounding_boxes(query_image, thresh_image)

    print(f"found {len(card_sized_imgs)} card sized boxes in this image")
    
    for candidate in card_sized_imgs:
        card, diff = find_minimum_hash_difference(candidate, hash_pool)
        if _possible_match(diff):

            print(f"find_minimum_hash_difference returned {card['name']} with a diff of {diff}.")
            
            # # Display the image using OpenCV
            # cv2.imshow(card['name'], card_image)

            # # Wait for a key press to close the window
            # cv2.waitKey(0)
            # # Close all OpenCV windows
            # cv2.destroyAllWindows()

            # Unpack card path and diff from find_minimum_hash_difference
            min_dist_df, diff =  (candidate, hash_pool)

            # Send the result back to the recognition_queue
            recognition_queue.put(min_dist_df['name']) 

        # cv2.imshow('Card Image', candidate)
        # key = cv2.waitKey(0) & 0xFF
        # hashes.append(compute_image_hash(candidate))

# def find_cards(query_image_path, hash_pool, recognition_queue):
#     print("called find_cards")
#     query_image = cv2.imread(query_image_path)

#     height, width = query_image.shape[:2]
    
#     # Define cropping margins (adjust as needed)
#     left_margin = 235  # Exclude pixels from the left
#     right_margin = 250  # Exclude pixels from the right
#     top_margin = 0 # Exclude pixels from the top
#     bottom_margin = 0 # Exclude pixels from the bottom

#     # Crop image (excluding left & right margins)
#     image_cropped = query_image[top_margin:height - bottom_margin, left_margin:width - right_margin]

#     # Display the image using OpenCV
#     #query_image_resized = cv2.resize(query_image, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
#     # Convert to grayscale for drawing
#     # query_image_copy = query_image.copy()

#     # cv2.imshow(f"query_image - {query_image}", query_image)
#     # cv2.imshow(f"query_image_copy - {query_image_copy}", query_image_copy)
#     # print(f"query_image_path = {query_image_path}")
#     # print(f"query_image = {query_image}")
#     # print(f"query_image_copy = {query_image_copy}")

#     # Wait for a key press to close the window
#     cv2.waitKey(0)

#     # Close all OpenCV windows
#     cv2.destroyAllWindows()

#     #query_image = Image.open(query_image_path)
#     #contours = find_contours(image_cropped)
#     contours = find_contours_hsv(image_cropped)

#     for n, contour in enumerate(contours):
#         print(f"in loop: n = {n}, contour = {contour}")

#         # Get the rectangle points from the contour
#         rectangle_points = _get_rectangle_points_from_contour(contour)

#         print(f"rectangle_points = {rectangle_points}")

#         # Draw the rectangle on the query_image_resized
#         # Convert points to integer for drawing (they may be floats after _get_rectangle_points_from_contour())
#         rectangle_points_int = rectangle_points.astype(int)

#         # Draw the rectangle (quadrilateral) on the image
#         cv2.polylines(image_cropped, [rectangle_points_int], isClosed=True, color=(0, 255, 0), thickness=2)

#         # You can also draw individual points (optional, for visual verification)
#         for point in rectangle_points_int:
#             cv2.circle(image_cropped, tuple(point), 5, (0, 0, 255), -1)

#         # Show the image with the rectangle overlayed
#         cv2.imshow(f"Rectangle Overlay - {query_image_path}", image_cropped)
            
#         # cv2.imshow(query_image_path, query_image_resized)

#         # Wait for a key press to close the window
#         cv2.waitKey(0)

#         # Close all OpenCV windows
#         cv2.destroyAllWindows()

#         card_image = _four_point_transform(query_image, rectangle_points)
#         card, diff = find_minimum_hash_difference(card_image, hash_pool)
#         if _possible_match(diff):

#             print(f"find_minimum_hash_difference returned {card['name']} with a diff of {diff}.")

#             # # Display the image using OpenCV
#             # cv2.imshow(card['name'], card_image)

#             # # Wait for a key press to close the window
#             # cv2.waitKey(0)

#             # # Close all OpenCV windows
#             # cv2.destroyAllWindows()

#             # Unpack card path and diff from find_minimum_hash_difference
#             min_dist_df, diff =  (card_image, hash_pool)

#             # Send the result back to the recognition_queue
#             recognition_queue.put(min_dist_df['name'])      
    

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
                  block_size=18, # Adapt to local contrast without over-smoothing - must be odd
                  offset_c=5, # Remove some noise
                  kernel_size=(5, 5) # Strengthen edges for better contour detection
                  ):

# Convert to grayscale if it's a color image
    if len(image.shape) == 3:  # Check if the image has multiple channels
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image  # If already grayscale, use it directly

    # image_blur = cv2.medianBlur(image_gray, ksize)

    # old algorithm
    # image_thresh = cv2.adaptiveThreshold(image_blur, thresh_max_value,
    #                                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
    #                                    thresh_block_size, thresh_c)
    
    image_thresh = cv2.adaptiveThreshold(image_gray, thresh_max_value,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    block_size, offset_c)
    
    # kernel = np.ones(kernel_size, np.uint8)


    #mode = cv2.RETR_EXTERNAL
    # mode = cv2.RETR_LIST
    mode = cv2.RETR_CCOMP
    # mode = cv2.RETR_TREE

    # Find contours and sort their indices by contour size
    dummy,contours,hier = cv2.findContours(image_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card



    # Draw all contours on the image to visualize
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

    # cv2.imshow("Detected Contours", image_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # if contours:
    #     #print(f"counters = {contours}")
    #     print("contours was not empty")
    #     rectangular_contours = find_rectangular_contours(contours, hierarchy)
    #     if rectangular_contours:
    #         print(f"rectangular_counters = {contours}")
    #         print("rectangular_contours was not empty")
    #         return rectangular_contours
    #     else:
    #         print("No rectangular_contours found")        
    
    # print("No contours found")
    # return []
    return image_contours, contours


def interactive_threshold(image, thresh=127, thresh_max_value=255, block_size=101, offset_c=10):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:  # Check if the image has multiple channels
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image  # If already grayscale, use it directly

    print(f"interactive_threshold called with thresh = {thresh}, thresh_max_value = {thresh_max_value}")
    #_, image_thresh = cv2.threshold(image_gray, thresh, thresh_max_value, cv2.THRESH_BINARY)
    image_thresh = cv2.adaptiveThreshold(image_gray, thresh_max_value,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         block_size, offset_c)

    # contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Draw contours on the original color image
    # image_contours = image.copy()
    # cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

    return image_gray, image_thresh

def find_contours_hsv(image, ksize=3, thresh_max_value=255, block_size=101, offset_c=10, kernel_size=(5, 5)):
    # Ensure ksize is always odd and greater than 1
    if ksize % 2 == 0:
        ksize += 1  # Make ksize odd
    if ksize <= 1:
        ksize = 3  # Set ksize to the minimum valid odd value

    # Apply median blur to the color image (no grayscale conversion)
    image_blur = cv2.medianBlur(image, ksize)

    # Convert the blurred image to grayscale just for thresholding (adaptive thresholding works best in grayscale)
    image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

    # Split the HSV image into its individual channels (Hue, Saturation, Value)
    h, s, v = cv2.split(image_hsv)

    # Apply adaptive thresholding on the V channel (or try S channel, depending on your need)
    image_thresh = cv2.adaptiveThreshold(v, thresh_max_value,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         block_size, offset_c)

    # # Perform morphological operations (closing and opening)
    # kernel = np.ones(kernel_size, np.uint8)
    # image_morph = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original color image
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

    return image_contours, contours

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

# def find_cards(thresh_image):
#     """Finds all card-sized contours in a thresholded camera image.
#     Returns the number of cards, and a list of card contours sorted
#     from largest to smallest."""

#     # Find contours and sort their indices by contour size
#     dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

#     # If there are no contours, do nothing
#     if len(cnts) == 0:
#         return [], []
    
#     # Otherwise, initialize empty sorted contour and hierarchy lists
#     cnts_sort = []
#     hier_sort = []
#     cnt_is_card = np.zeros(len(cnts),dtype=int)

#     # Fill empty lists with sorted contour and sorted hierarchy. Now,
#     # the indices of the contour list still correspond with those of
#     # the hierarchy list. The hierarchy array can be used to check if
#     # the contours have parents or not.
#     for i in index_sort:
#         cnts_sort.append(cnts[i])
#         hier_sort.append(hier[0][i])

#     # Determine which of the contours are cards by applying the
#     # following criteria: 1) Smaller area than the maximum card size,
#     # 2), bigger area than the minimum card size, 3) have no parents,
#     # and 4) have four corners

#     for i in range(len(cnts_sort)):
#         size = cv2.contourArea(cnts_sort[i])
#         peri = cv2.arcLength(cnts_sort[i],True)
#         approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
#         if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
#             and (hier_sort[i][3] == -1) and (len(approx) == 4)):
#             cnt_is_card[i] = 1

#     return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts, w, h)

    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    
    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    dummy, Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # Find suit contour and bounding rectangle, isolate and find largest contour
    dummy, Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea,reverse=True)
    
    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard

def preprocess_image(image, thresh_max_value=255, block_size=101, offset_c=10, kernel_size=(5,5), stddev=0):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,kernel_size,stddev)
    # ksize = (Kernel Size for Median Blur)
    # blur = cv2.medianBlur(gray, ksize)
    #blur = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)  # d = 9, sigmaColor = 75, sigmaSpace = 75

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    
    # print(f"bkg_level = {bkg_level}")
    #thresh_level = bkg_level + BKG_THRESH

    #retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(blur, thresh_max_value,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset_c)
    

    return thresh


def extract_card_bounding_boxes(image, thresh, display_mode=None):
    """ Detect contours, filter them, and get bounding boxes using approxPolyDP. """
    
     #mode = cv2.RETR_EXTERNAL
    # mode = cv2.RETR_LIST
    mode = cv2.RETR_CCOMP
    # mode = cv2.RETR_TREE

    # Find contours
    contours,hier = cv2.findContours(thresh,mode,cv2.CHAIN_APPROX_SIMPLE)

    #we'll want to sort their indices by contour size

    # DEBUG - display contours that we found
    if display_mode == "unfiltered_contours":
        image_contours = image.copy()
        cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Unfiltered Contours", image_contours)

    # filter the contours based on the area and area-to-perimeter ratio 
    # (cards luckily have a consistent shape)
    filtered_contours = filter_contours(contours)
    
    # DEBUG - display filtered contours
    if display_mode == "filtered_contours":
        image_filtered_contours = image.copy()
        cv2.drawContours(image_filtered_contours, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Filtered Contours", image_filtered_contours)

    card_images = []
    
    for contour in filtered_contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)  # Approximate shape
        
        # # Wait for a key press to close the window
        # cv2.waitKey(0)

        if len(approx) == 4:  # Ensuring it's a quadrilateral (card)
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.imshow("Approximated Contours", image)
            
            x, y, w, h = cv2.boundingRect(approx)
            card = image[y:y+h, x:x+w]  # Crop card region
            
            card_images.append(card)
    
    return card_images

def filter_contours(contours, min_area=1000, max_area=50000, area_perimeter_ratio_thresh=19):
    """ Filter contours based on area and area-to-perimeter ratio. """
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:  # Avoid division by zero
            continue
        
        area_perimeter_ratio = area / perimeter
        
        if min_area < area < max_area and area_perimeter_ratio > area_perimeter_ratio_thresh:
            print(f"qualitifed a contour with area = {area}, area_perimeter_ratio = {area_perimeter_ratio}")
            filtered_contours.append(contour) 
    
    # print(f"len(contours) = {len(contours)}") 
    # print(f"len(filtered_contours) = {len(filtered_contours)}")

    return filtered_contours
