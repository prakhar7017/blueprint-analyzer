import cv2
import numpy as np

from wall_detection import MIN_CONTOUR_AREA


def detect_rooms(wall_mask: np.ndarray) -> list:
    if wall_mask.size == 0:
        return []

    inv = cv2.bitwise_not(wall_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
