import cv2
import numpy as np

from image_matcher.draw_image import draw_text_and_save_card_image
from image_matcher.hash_matcher import find_minimum_hash_difference


def find_cards(image, hash_pool):
    contours = find_contours(image.copy())
    # card_models = []
    for n, contour in enumerate(contours):
        rectangle_points = _get_rectangle_points_from_contour(contour)
        #Transform the detected card region into a rectangle
        card_image = _four_point_transform(image, rectangle_points)
        
        card, diff = find_minimum_hash_difference(card_image, hash_pool)
        if _possible_match(card['name'], diff):
            label_card_image = _four_point_transform(image, rectangle_points,
                                                     for_display=True)
            card_image_path = draw_text_and_save_card_image(card['name'],
                                                            label_card_image, n)
            del card_image
            del label_card_image
            # details = CardListingDetails.objects.create(
            #     scryfall_id=card['id'], name=card['name'], set=card['set'])
            # card_models.append(ImageUpload.objects.create(image_input=card_image_path,
            #                                               image_name=card['name'],
            #                                               listing_details=details))
        else:
            del card_image
    return card_image_path


def _possible_match(card_name, diff):
    if diff < 450:
        return card_name


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


def find_contours(image, ksize=5, thresh_max_value=255, thresh_block_size=199, thresh_c=5,
                  kernel_size=(3, 3)):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.medianBlur(image_gray, ksize)
    image_thresh = cv2.adaptiveThreshold(image_blur, thresh_max_value,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       thresh_block_size, thresh_c)
    kernel = np.ones(kernel_size, np.uint8)
    image_dilate = cv2.dilate(image_thresh, kernel, iterations=1)
    image_erode = cv2.erode(image_dilate, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(image_erode, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_TC89_KCOS)
    if contours:
        return find_rectangular_contours(contours, hierarchy)
    return []


def find_rectangular_contours(contours, hierarchy):
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


def _threshold_size_bounded_by(area, thresh_size=10000):
    return area >= thresh_size