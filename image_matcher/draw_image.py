import cv2

from net_ready_eyes.settings import MEDIA_ROOT


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


def draw_text_and_save_card_image(card_name, card_image, n):
    """ NO IMAGE SHOULD BE APPENDED ON CARD FOR EBAY LISTING
    cv2.putText(card_image, card_label, (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (143, 0, 255), 2)
    HOWEVER, IF YOU WOULD LIKE TO ADD THIS FEATURE, THE WORK IS
    ALREADY DONE FOR YOU. YOUR WELCOME.
    """

    advertising_watermark = 'Listed with mtg-vision.com'
    height = card_image.shape[0] - 40
    width = int(card_image.shape[1] / 2) - 100
    cv2.putText(card_image, advertising_watermark, (width, height),
                cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 2)
    image_path = f'{card_name}_{n}.jpg'
    full_path = f'{MEDIA_ROOT}/{image_path}'
    cv2.imwrite(full_path, card_image)
    return image_path