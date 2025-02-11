import cv2

def draw_text_and_contours_image(card_name, contour, input_image, rectangle_points,
                                 contour_idx=-1, edge_color=(0, 255, 0), edge_thickness=2,
                                 font_scale=0.4, font_color=(143, 0, 255),
                                 font_thickness=2):
    cv2.drawContours(input_image, [contour], contour_idx, edge_color, edge_thickness)
    cv2.putText(input_image, card_name,
                _minimum_width_by_minimum_height(rectangle_points),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)


def _minimum_width_by_minimum_height(rectangle_points):
    return minimum_width(rectangle_points), minimum_height(rectangle_points)


def minimum_height(rectangle_points):
    return min(rectangle_points[0][1], rectangle_points[1][1])


def minimum_width(rectangle_points):
    return min(rectangle_points[0][0], rectangle_points[1][0])

def draw_and_pause(self, image1, keypoints1, image2, keypoints2, matches):
    """ Draws matches and pauses execution until a key is pressed """
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matched features
    cv2.imshow("Matches", matched_image)

    # Wait for user input to continue (press any key)
    print("Press any key to continue to the next image...")
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

def draw(self, image1, keypoints1, image2, keypoints2, matches):
    """ Draws matches and pauses execution until a key is pressed """

    #destroy the last window
    cv2.destroyAllWindows()

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matched features
    cv2.imshow("Matches", matched_image)

    # Wait for user input to continue (press any key)
