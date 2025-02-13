# import cv2
# import numpy as np

# from image_matcher.detect_image import *

# # Initialize global variables for slider values
# thresh=127
# thresh_max_value = 255
# block_size = 18
# offset_c = 5
# ksize = 3
# kernel_size = (5, 5)

# # Load image
# image = cv2.imread("C:\\Users\\erich\\net_ready_eyes\\media\\test_images\\table_capture2.png")  # Change to your image pathf

# height, width = image.shape[:2]

# # Define cropping margins (adjust as needed)
# left_margin = 235  # Exclude pixels from the left
# right_margin = 250  # Exclude pixels from the right
# top_margin = 0 # Exclude pixels from the top
# bottom_margin = 0 # Exclude pixels from the bottom

# # Crop image (excluding left & right margins)
# image_cropped = image[top_margin:height - bottom_margin, left_margin:width - right_margin]

# # Create the window and sliders
# def nothing(x):
#     pass

# # Create a window to show the image and add sliders for parameters
# cv2.namedWindow('Image')

# # Threshold
# cv2.createTrackbar('Thresh', 'Image', thresh, 255, nothing)

# # Threshold Max Value
# cv2.createTrackbar('Thresh Max Value', 'Image', thresh_max_value, 255, nothing)

# # Threshold Block Size (must be odd)
# cv2.createTrackbar('Block Size', 'Image', block_size, 101, nothing)

# # Threshold C value
# cv2.createTrackbar('Offset C', 'Image', offset_c, 50, nothing)

# # # Kernel Size (blur size)
# # cv2.createTrackbar('Ksize', 'Image', ksize, 9, nothing)

# # Kernel Size for Morphological operations
# # cv2.createTrackbar('Kernel Size', 'Image', 5, 10, nothing)

# # def save_parameters():
# #     # Save current slider values to a JSON file
# #     params = {
# #         "block_size": block_size,
# #         "offset_c": offset_c,
# #         "ksize": ksize,
# #         "kernel_size": kernel_size[0]  # Assuming square kernel size
# #     }

# #     with open(params_file, 'w') as f:
# #         json.dump(params, f, indent=4)
# #     print(f"Parameters saved to {params_file}")

# while True:
#     # Get current values from the sliders
#     thresh = cv2.getTrackbarPos('Thresh', 'Image')

#     thresh_max_value = cv2.getTrackbarPos('Thresh Max Value', 'Image')

#     block_size = cv2.getTrackbarPos('Block Size', 'Image')

#     # Enforce the odd condition
#     if block_size % 2 == 0:
#         block_size += 1

#     # Ensure it's greater than 1
#     if block_size <= 1:
#         block_size = 3
    
#     # Get the other values from sliders
#     offset_c = cv2.getTrackbarPos('Offset C', 'Image')

#     # Start timer (for calculating frame rate)
#     t1 = cv2.getTickCount()

#     # Pre-process camera image (gray, blur, and threshold it)
#     pre_proc = preprocess_image(image)
	
#     cv2.imshow("Image", pre_proc)

#     # # Find and sort the contours of all cards in the image (query cards)
#     # cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

#     # # If there are no contours, do nothing
#     # if len(cnts_sort) != 0:

#     #     # Initialize a new "cards" list to assign the card objects.
#     #     # k indexes the newly made array of cards.
#     #     cards = []
#     #     k = 0

#     #     # For each contour detected:
#     #     for i in range(len(cnts_sort)):
#     #         if (cnt_is_card[i] == 1):

#     #             # Create a card object from the contour and append it to the list of cards.
#     #             # preprocess_card function takes the card contour and contour and
#     #             # determines the cards properties (corner points, etc). It generates a
#     #             # flattened 200x300 image of the card, and isolates the card's
#     #             # suit and rank from the image.
#     #             cards.append(Cards.preprocess_card(cnts_sort[i],image))

#     #             # Find the best rank and suit match for the card.
#     #             cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

#     #             # Draw center point and match result on the image.
#     #             image = Cards.draw_results(image, cards[k])
#     #             k = k + 1
	    
#     #     # Draw card contours on image (have to do contours all at once or
#     #     # they do not show up properly for some reason)
#     #     if (len(cards) != 0):
#     #         temp_cnts = []
#     #         for i in range(len(cards)):
#     #             temp_cnts.append(cards[i].contour)
#     #         cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
        
        
#     # # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
#     # # so the first time this runs, framerate will be shown as 0.
#     # cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

#     # # Finally, display the image with the identified cards!
#     # cv2.imshow("Card Detector",image)


#     # Wait for a key press
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:  # ESC to exit
#         break
#     elif key ==ord('s'): # 's' key to save parameters
#         pass
#         # save_parameters()

# # Close all windows
# cv2.destroyAllWindows()