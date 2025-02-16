import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from image_matcher.hash import find_minimum_hash_difference
from utils.utils import convert_image_to_opencv, convert_image_to_pil, is_opencv, is_pil

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000
# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

def find_cards(query_image, 
               rec_params,
               hash_pool,
               recognition_queue,
               display_image=None, # display_image function passed in from the UI
               display_mode=None):
    
    thresh_image = preprocess_image(query_image, rec_params)

    if display_mode == "thresholding" and display_image:
        display_image(thresh_image)

    card_sized_imgs = extract_card_bounding_boxes(query_image, thresh_image, 
                                                  rec_params,
                                                  display_mode, display_image)

    # rotation_corrected_imgs = preprocess_rotate(card_sized_imgs)

    print(f"found {len(card_sized_imgs)} card sized boxes in this image")
    
    for candidate in card_sized_imgs
        card, diff = find_minimum_hash_difference(candidate, hash_pool)
        if _possible_match(diff):

            print(f"find_minimum_hash_difference returned {card['name']} with a diff of {diff}.")
            
            # # Display the image using OpenCV
            # cv2.imshow(card['name'], card_image)

            # # Wait for a key press to close the window
            # cv2.waitKey(0)
            # # Close all OpenCV windows
            # cv2.destroyAllWindows()

            # # Unpack card path and diff from find_minimum_hash_difference
            # min_dist_df, diff =  (candidate, hash_pool)

            # Send the result back to the recognition_queue
            recognition_queue.put(card['name']) 

        # cv2.imshow('Card Image', candidate)
        # key = cv2.waitKey(0) & 0xFF
        # hashes.append(compute_image_hash(candidate))

    
def _possible_match(diff):
    if diff < 450: #To-do: make this number based on the threshold slider
        return True

def preprocess_rotate(images):
    rotated_imgs = []

    for image in images:
        if isinstance(image, Image.Image):
            converted_img = convert_image_to_opencv(image)
        elif isinstance(image, np.ndarray):
            converted_img = image
        else:
            return
        
        # Find edges using Canny edge detection to detect orientation
        edges = cv2.Canny(converted_img, 50, 150, apertureSize=3)
        
        # Find the rotation angle
        coords = np.column_stack(np.where(edges > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct the image rotation
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(converted_img, M, (w, h))

        # # Display the image using OpenCV
        cv2.imshow("rotated_image", rotated_image)

        # Wait for a key press to close the window
        cv2.waitKey(0)
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        rotated_imgs.append(convert_image_to_pil(rotated_image))

    return rotated_imgs


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


def filter_contour_size(contours, min_area=1000, max_area=50000):
    """ Filter contours based on area and area-to-perimeter ratio. """
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            filtered_contours.append(contour) 
    
    return filtered_contours


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
    return min_area <= area <= max_area

def preprocess_image(image, rec_params=[]):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    kernel_size = rec_params["kernel_size"]
    thresh_max_value = rec_params["thresh_max_value"]
    block_size = rec_params["block_size"]
    offset_c = rec_params["offset_c"]
    stddev = rec_params["stddev"]

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,kernel_size,stddev)
    thresh = cv2.adaptiveThreshold(blur, thresh_max_value,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset_c)
    
    return thresh


def extract_card_bounding_boxes(image, thresh, rec_params,
                                display_mode=None, display_image=None):
    """ Detect contours, filter them, and get bounding boxes using approxPolyDP. """
    
     #mode = cv2.RETR_EXTERNAL
    # mode = cv2.RETR_LIST
    mode = cv2.RETR_CCOMP
    # mode = cv2.RETR_TREE

    # Find contours
    contours,hier = cv2.findContours(thresh,mode,cv2.CHAIN_APPROX_SIMPLE)

    #we'll want to sort their indices by contour size

    # display contours that we found
    if display_mode == "unfiltered contours" and display_image:
        image_contours = image.copy()
        image_rgb = cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB)
        cv2.drawContours(image_rgb, contours, -1, (0, 255, 0), 2)
        display_image(image_rgb)

    # filter the contours based on the area and area-to-perimeter ratio 
    # (cards luckily have a consistent shape)
    # filtered_contours = filter_contours(contours)
    filtered_contours = find_rectangular_contours(contours, hier)
    card_sized_contours = filter_contour_size(filtered_contours,
                                              rec_params['card_min_area'],
                                              rec_params['card_max_area'])

    # display filtered contours
    if display_mode == "filtered contours" and display_image:
        image_filtered_contours = image.copy()
        image_rgb = cv2.cvtColor(image_filtered_contours, cv2.COLOR_BGR2RGB)
        cv2.drawContours(image_rgb, card_sized_contours, -1, (0, 255, 0), 2)
        display_image(image_rgb)

    card_images = []
    
    for contour in card_sized_contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)  # Approximate shape
        
        # # Wait for a key press to close the window
        # cv2.waitKey(0)

        if len(approx) == 4:  # Ensuring it's a quadrilateral (card)
            # if display_mode == "approx contours" and display_image:
            #     image_approx_contours = image.copy()
            #     image_rgb = cv2.cvtColor(image_approx_contours, cv2.COLOR_BGR2RGB)
            #     cv2.drawContours(image_rgb, [approx], -1, (0, 255, 0), 2)
            #     display_image(image_rgb)
            
            x, y, w, h = cv2.boundingRect(approx)
            card = image[y:y+h, x:x+w]  # Crop card region
            
            card_images.append(card)

    # print(f"Total contours found: {len(contours)}")
    # print(f"After filtering nested contours: {len(filtered_contours)}")
    # print(f"After filtering by size: {len(card_sized_contours)}")
    # print(f"Final number of card images: {len(card_images)}")

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
            print(f"qualified a contour with area = {area}, area_perimeter_ratio = {area_perimeter_ratio}")
            filtered_contours.append(contour) 
    
    return filtered_contours
