from __future__ import annotations

import logging

import cv2
import numpy as np

from model import get_model, get_onnx_model

logger = logging.getLogger(__name__)

MIN_CONTOUR_AREA = 500


def _filter_contours(contours: list, min_area: int = MIN_CONTOUR_AREA) -> list:
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _masks_to_binary_wall_mask(image: np.ndarray, results) -> tuple[np.ndarray, bool]:
    h, w = image.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        return combined, False

    masks_data = r.masks.data.cpu().numpy()
    for i in range(masks_data.shape[0]):
        m = masks_data[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (m > 0.5).astype(np.uint8) * 255
        combined = np.maximum(combined, binary)
    return combined, True


def _fallback_canny_wall_mask(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    return dilated


def detect_walls(image: np.ndarray) -> tuple[np.ndarray, list]:
    h, w = image.shape[:2]
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    used_yolo = False

    for loader in (get_onnx_model, get_model):
        try:
            model = loader()
            results = model(image, verbose=False)
            wall_mask, used_yolo = _masks_to_binary_wall_mask(image, results)
            if used_yolo:
                break
        except Exception:
            logger.warning("YOLO inference failed with %s, trying next backend", loader.__name__, exc_info=True)
            wall_mask = np.zeros((h, w), dtype=np.uint8)
            used_yolo = False

    if not used_yolo or np.count_nonzero(wall_mask) == 0:
        logger.info("No YOLO masks produced, falling back to Canny edge detection")
        wall_mask = _fallback_canny_wall_mask(image)

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wall_contours = _filter_contours(contours)

    clean_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_mask, wall_contours, -1, 255, thickness=3)
    if np.count_nonzero(clean_mask) == 0 and wall_contours:
        cv2.drawContours(clean_mask, wall_contours, -1, 255, thickness=cv2.FILLED)

    if np.count_nonzero(clean_mask) == 0:
        clean_mask = wall_mask

    return clean_mask, wall_contours


def detect_fixtures(image: np.ndarray, wall_mask: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    fixtures: dict[str, list] = {"lights": [], "doors": [], "windows": []}

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=5, maxRadius=30,
    )
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            fixtures["lights"].append({"x": int(x), "y": int(y), "radius": int(r)})

    edges = cv2.Canny(gray, 50, 150)
    non_wall = cv2.bitwise_and(edges, cv2.bitwise_not(wall_mask))
    contours, _ = cv2.findContours(non_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 200 or area > 8000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)

        if 0.05 < circularity < 0.4 and 0.3 < aspect < 3.0:
            fixtures["doors"].append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        elif circularity > 0.4 and (aspect < 0.35 or aspect > 2.8):
            fixtures["windows"].append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return fixtures


def estimate_avg_wall_thickness(wall_mask: np.ndarray) -> float:
    m = (wall_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    vals = dt[m > 0]
    return float(2.0 * np.mean(vals))
