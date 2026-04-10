import cv2
import numpy as np

from wall_detection import MIN_CONTOUR_AREA


def _estimate_kernel_size(wall_mask: np.ndarray) -> int:
    h, w = wall_mask.shape[:2]
    size = max(7, min(h, w) // 60)
    return size | 1


def _fill_wall_gaps(wall_mask: np.ndarray) -> np.ndarray:
    k = _estimate_kernel_size(wall_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def _watershed_rooms(wall_mask: np.ndarray) -> np.ndarray:
    inv = cv2.bitwise_not(wall_mask)

    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dt_norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, markers_bin = cv2.threshold(dt_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, _estimate_kernel_size(wall_mask) // 2)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    markers_bin = cv2.morphologyEx(markers_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, markers = cv2.connectedComponents(markers_bin)
    if num_labels <= 1:
        return inv

    markers = markers + 1
    markers[wall_mask > 0] = 0

    bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    markers_32 = markers.astype(np.int32)
    cv2.watershed(bgr, markers_32)

    result = np.zeros_like(wall_mask)
    result[markers_32 > 1] = 255
    return result


def _filter_rooms_by_shape(contours: list, image_area: int) -> list:
    max_room_ratio = 0.85
    min_area = max(MIN_CONTOUR_AREA, int(image_area * 0.001))
    max_aspect = 8.0
    min_solidity = 0.3

    rooms = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > image_area * max_room_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > max_aspect:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / max(hull_area, 1)
        if solidity < min_solidity:
            continue

        rooms.append(c)

    return rooms


def _remove_border_contours(contours: list, h: int, w: int) -> list:
    filtered = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 2 and y <= 2 and bw >= w - 4 and bh >= h - 4:
            continue
        filtered.append(c)
    return filtered


def _merge_overlapping_rooms(contours: list) -> list:
    if len(contours) <= 1:
        return contours

    bboxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]
    used = [False] * len(contours)
    merged = []

    sorted_idx = sorted(range(len(contours)), key=lambda i: areas[i], reverse=True)

    for i in sorted_idx:
        if used[i]:
            continue
        used[i] = True
        current = contours[i]
        x1, y1, w1, h1 = bboxes[i]

        for j in sorted_idx:
            if used[j] or j == i:
                continue
            x2, y2, w2, h2 = bboxes[j]
            ix1, iy1 = max(x1, x2), max(y1, y2)
            ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller = min(areas[i], areas[j])
                if smaller > 0 and inter / smaller > 0.5:
                    used[j] = True

        merged.append(current)

    return merged


def detect_rooms(wall_mask: np.ndarray) -> list:
    if wall_mask.size == 0:
        return []

    h, w = wall_mask.shape[:2]
    image_area = h * w

    filled_walls = _fill_wall_gaps(wall_mask)

    inv = cv2.bitwise_not(filled_walls)
    k = _estimate_kernel_size(wall_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    morph_cleaned = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    morph_cleaned = cv2.morphologyEx(morph_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    morph_contours, _ = cv2.findContours(
        morph_cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )

    try:
        ws_mask = _watershed_rooms(filled_walls)
        ws_contours, _ = cv2.findContours(
            ws_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
    except Exception:
        ws_contours = []

    all_contours = list(morph_contours) + list(ws_contours)

    all_contours = _remove_border_contours(all_contours, h, w)
    all_contours = _filter_rooms_by_shape(all_contours, image_area)
    all_contours = _simplify_room_contours(all_contours)
    all_contours = _merge_overlapping_rooms(all_contours)

    return all_contours


def _simplify_room_contours(contours: list) -> list:
    simplified = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        eps = 0.008 * peri
        approx = cv2.approxPolyDP(c, eps, True)
        if cv2.contourArea(approx) >= MIN_CONTOUR_AREA:
            simplified.append(approx)
    return simplified
