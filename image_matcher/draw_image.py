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