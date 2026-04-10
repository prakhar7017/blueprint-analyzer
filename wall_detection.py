from __future__ import annotations

import logging

import cv2
import numpy as np

from model import get_model, get_onnx_model
from utils import preprocess_blueprint

logger = logging.getLogger(__name__)

MIN_CONTOUR_AREA = 500


def _filter_contours(contours: list, min_area: int = MIN_CONTOUR_AREA) -> list:
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def _enhance_blueprint(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _adaptive_threshold_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    block_size = max(11, (min(gray.shape[:2]) // 40) | 1)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 10,
    )
    return thresh


def _otsu_threshold_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def _morphological_gradient_mask(gray: np.ndarray, image_shape: tuple) -> np.ndarray:
    h, w = image_shape[:2]
    k = max(2, min(h, w) // 400)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _directional_wall_mask(gray: np.ndarray, image_shape: tuple) -> np.ndarray:
    h, w = image_shape[:2]
    length = max(10, min(h, w) // 40)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    combined = cv2.bitwise_or(h_walls, v_walls)
    return combined


def _detect_lines_mask(gray: np.ndarray, image_shape: tuple) -> np.ndarray:
    h, w = image_shape[:2]
    edges = _auto_canny(gray)

    mask = np.zeros((h, w), dtype=np.uint8)
    axis_mask = np.zeros((h, w), dtype=np.uint8)

    scales = [
        (max(20, min(h, w) // 30), max(5, min(h, w) // 100), 50),
        (max(40, min(h, w) // 15), max(8, min(h, w) // 60), 80),
    ]

    for min_length, max_gap, threshold in scales:
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=threshold, minLineLength=min_length, maxLineGap=max_gap,
        )
        if lines is None:
            continue
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            is_axis_aligned = angle < 5 or angle > 175 or abs(angle - 90) < 5
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
            if is_axis_aligned:
                cv2.line(axis_mask, (x1, y1), (x2, y2), 255, 3)

    boosted = cv2.addWeighted(mask, 0.5, axis_mask, 0.5, 0)
    _, boosted = cv2.threshold(boosted, 50, 255, cv2.THRESH_BINARY)
    return boosted


def _merge_collinear_segments(lines: np.ndarray | None, image_shape: tuple) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if lines is None:
        return mask

    angle_tol = 5.0
    gap_tol = max(10, min(h, w) // 50)

    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        segments.append((x1, y1, x2, y2, angle, length))

    merged_used = [False] * len(segments)

    for i, (x1, y1, x2, y2, ang_i, len_i) in enumerate(segments):
        if merged_used[i]:
            continue
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

        for j in range(i + 1, len(segments)):
            if merged_used[j]:
                continue
            x3, y3, x4, y4, ang_j, _ = segments[j]
            angle_diff = min(abs(ang_i - ang_j), 180 - abs(ang_i - ang_j))
            if angle_diff > angle_tol:
                continue

            endpoint_pairs = [
                ((x2, y2), (x3, y3)),
                ((x2, y2), (x4, y4)),
                ((x1, y1), (x3, y3)),
                ((x1, y1), (x4, y4)),
            ]
            dists = [
                np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                for a, b in endpoint_pairs
            ]
            best_idx = int(np.argmin(dists))
            min_dist = dists[best_idx]

            if min_dist < gap_tol:
                pa, pb = endpoint_pairs[best_idx]
                cv2.line(mask, pa, pb, 255, 2)
                merged_used[j] = True

    return mask


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8,
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned


def _combine_wall_masks(
    adaptive_mask: np.ndarray,
    otsu_mask: np.ndarray,
    line_mask: np.ndarray,
    canny_mask: np.ndarray,
    gradient_mask: np.ndarray,
    directional_mask: np.ndarray,
) -> np.ndarray:
    h, w = adaptive_mask.shape[:2]
    score = np.zeros((h, w), dtype=np.float32)
    score += (adaptive_mask > 0).astype(np.float32) * 0.20
    score += (otsu_mask > 0).astype(np.float32) * 0.15
    score += (line_mask > 0).astype(np.float32) * 0.20
    score += (canny_mask > 0).astype(np.float32) * 0.10
    score += (gradient_mask > 0).astype(np.float32) * 0.15
    score += (directional_mask > 0).astype(np.float32) * 0.20

    combined = (score >= 0.30).astype(np.uint8) * 255
    return combined


def _refine_wall_mask(mask: np.ndarray, image_shape: tuple) -> np.ndarray:
    h, w = image_shape[:2]
    scale = max(1, min(h, w) // 500)
    min_component = max(50, int(h * w * 0.00003))

    close_k = max(3, 3 * scale)
    if close_k % 2 == 0:
        close_k += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    mask = _remove_small_components(mask, min_component)

    open_k = max(2, scale)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return mask


def _simplify_contours(contours: list) -> list:
    epsilon_factor = 0.005
    simplified = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        eps = epsilon_factor * peri
        approx = cv2.approxPolyDP(c, eps, True)
        if cv2.contourArea(approx) >= MIN_CONTOUR_AREA:
            simplified.append(approx)
    return simplified


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


def _fallback_canny_wall_mask(gray: np.ndarray) -> np.ndarray:
    edges = _auto_canny(gray)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated


def detect_walls(image: np.ndarray) -> tuple[np.ndarray, list]:
    h, w = image.shape[:2]

    preprocessed = preprocess_blueprint(image)
    enhanced = _enhance_blueprint(preprocessed)

    adaptive_mask = _adaptive_threshold_mask(enhanced)
    otsu_mask = _otsu_threshold_mask(enhanced)
    line_mask = _detect_lines_mask(enhanced, image.shape)
    canny_mask = _fallback_canny_wall_mask(enhanced)
    gradient_mask = _morphological_gradient_mask(enhanced, image.shape)
    directional_mask = _directional_wall_mask(enhanced, image.shape)

    edges_for_merge = _auto_canny(enhanced)
    merge_lines = cv2.HoughLinesP(
        edges_for_merge, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=max(20, min(h, w) // 30),
        maxLineGap=max(5, min(h, w) // 100),
    )
    collinear_mask = _merge_collinear_segments(merge_lines, image.shape)

    combined = _combine_wall_masks(
        adaptive_mask, otsu_mask, line_mask, canny_mask,
        gradient_mask, directional_mask,
    )
    combined = cv2.bitwise_or(combined, collinear_mask)

    yolo_mask = np.zeros((h, w), dtype=np.uint8)
    for loader in (get_onnx_model, get_model):
        try:
            model = loader()
            results = model(image, conf=0.25, iou=0.45, verbose=False)
            yolo_mask, had_masks = _masks_to_binary_wall_mask(image, results)
            if had_masks:
                break
        except Exception:
            logger.warning(
                "YOLO inference failed with %s, trying next backend",
                loader.__name__, exc_info=True,
            )
            yolo_mask = np.zeros((h, w), dtype=np.uint8)

    if np.count_nonzero(yolo_mask) > 0:
        yolo_weight = 0.15
        score = (combined > 0).astype(np.float32) * (1 - yolo_weight)
        score += (yolo_mask > 0).astype(np.float32) * yolo_weight
        combined = (score > 0.3).astype(np.uint8) * 255

    wall_mask = _refine_wall_mask(combined, image.shape)

    contours, _ = cv2.findContours(
        wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    wall_contours = _filter_contours(contours)
    wall_contours = _simplify_contours(wall_contours)

    clean_mask = np.zeros((h, w), dtype=np.uint8)
    if wall_contours:
        cv2.drawContours(clean_mask, wall_contours, -1, 255, thickness=cv2.FILLED)

    if np.count_nonzero(clean_mask) == 0:
        clean_mask = wall_mask

    return clean_mask, wall_contours


def _validate_circle(gray: np.ndarray, x: int, y: int, r: int) -> bool:
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    num_samples = max(16, int(2 * np.pi * r))
    edge_hits = 0
    for k in range(num_samples):
        theta = 2 * np.pi * k / num_samples
        px = int(round(x + r * np.cos(theta)))
        py = int(round(y + r * np.sin(theta)))
        if 0 <= px < w and 0 <= py < h:
            if edges[py, px] > 0:
                edge_hits += 1
    return (edge_hits / max(num_samples, 1)) >= 0.30


def _detect_door_arcs(
    gray: np.ndarray,
    wall_mask: np.ndarray,
    min_fixture_area: int,
    max_fixture_area: int,
) -> list[dict]:
    doors: list[dict] = []
    edges = _auto_canny(gray)
    non_wall = cv2.bitwise_and(edges, cv2.bitwise_not(wall_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    non_wall = cv2.morphologyEx(non_wall, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        non_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_fixture_area or area > max_fixture_area:
            continue
        if len(c) < 5:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area

        ellipse = cv2.fitEllipse(c)
        (_, _), (ma, MA), _ = ellipse
        if MA == 0:
            continue
        ellipse_ratio = ma / MA

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        is_arc = (
            0.02 < circularity < 0.5
            and solidity < 0.7
            and 0.3 < ellipse_ratio < 1.0
        )
        if is_arc and _contour_near_wall(c, wall_mask):
            x, y, bw, bh = cv2.boundingRect(c)
            doors.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)})

    return doors


def _nms_fixtures(detections: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda d: d["w"] * d["h"], reverse=True)
    keep = []
    for det in detections:
        x1, y1 = det["x"], det["y"]
        x2, y2 = x1 + det["w"], y1 + det["h"]
        area = det["w"] * det["h"]
        suppressed = False
        for kept in keep:
            kx1, ky1 = kept["x"], kept["y"]
            kx2, ky2 = kx1 + kept["w"], ky1 + kept["h"]
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = area + kept["w"] * kept["h"] - inter
                if union > 0 and inter / union > iou_threshold:
                    suppressed = True
                    break
        if not suppressed:
            keep.append(det)
    return keep


def _detect_wall_breaks(wall_mask: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape[:2]
    margin = max(15, min(h, w) // 60)

    dilated = cv2.dilate(wall_mask, np.ones((margin, margin), np.uint8), iterations=1)
    breaks = cv2.subtract(dilated, wall_mask)
    return breaks


def detect_fixtures(image: np.ndarray, wall_mask: np.ndarray) -> dict:
    enhanced = _enhance_blueprint(image)
    h, w = image.shape[:2]
    img_area = h * w

    fixtures: dict[str, list] = {"lights": [], "doors": [], "windows": []}

    min_radius = max(3, min(h, w) // 200)
    max_radius = max(15, min(h, w) // 50)
    min_dist = max(20, min(h, w) // 40)

    for blur_size, param2 in [(7, 25), (9, 30), (11, 35)]:
        blurred = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
            param1=50, param2=param2,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is not None:
            for cx, cy, cr in np.round(circles[0]).astype(int):
                if not _validate_circle(enhanced, int(cx), int(cy), int(cr)):
                    continue
                if not any(
                    abs(cx - l["x"]) < min_dist and abs(cy - l["y"]) < min_dist
                    for l in fixtures["lights"]
                ):
                    fixtures["lights"].append(
                        {"x": int(cx), "y": int(cy), "radius": int(cr)}
                    )

    min_fixture_area = max(100, int(img_area * 0.00005))
    max_fixture_area = max(5000, int(img_area * 0.01))

    arc_doors = _detect_door_arcs(enhanced, wall_mask, min_fixture_area, max_fixture_area)

    try:
        break_mask = _detect_wall_breaks(wall_mask)
    except Exception:
        break_mask = np.zeros((h, w), dtype=np.uint8)

    edges = _auto_canny(enhanced)
    non_wall = cv2.bitwise_and(edges, cv2.bitwise_not(wall_mask))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    non_wall = cv2.morphologyEx(non_wall, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(
        non_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    shape_doors: list[dict] = []
    shape_windows: list[dict] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_fixture_area or area > max_fixture_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / max(hull_area, 1)

        is_near_wall = _contour_near_wall(c, wall_mask)
        near_break = False
        if np.count_nonzero(break_mask) > 0:
            near_break = _contour_near_mask(c, break_mask)

        if 0.03 < circularity < 0.45 and 0.2 < aspect < 4.0 and solidity < 0.7:
            if is_near_wall or near_break:
                shape_doors.append(
                    {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}
                )
        elif circularity > 0.3 and (aspect < 0.3 or aspect > 3.0) and is_near_wall:
            shape_windows.append(
                {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}
            )

    all_doors = arc_doors + shape_doors
    fixtures["doors"] = _nms_fixtures(all_doors)
    fixtures["windows"] = _nms_fixtures(shape_windows)

    return fixtures


def _contour_near_wall(contour: np.ndarray, wall_mask: np.ndarray) -> bool:
    h, w = wall_mask.shape[:2]
    margin = max(10, min(h, w) // 100)
    x, y, bw, bh = cv2.boundingRect(contour)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)
    roi = wall_mask[y1:y2, x1:x2]
    return np.count_nonzero(roi) > 0


def _contour_near_mask(contour: np.ndarray, mask: np.ndarray) -> bool:
    h, w = mask.shape[:2]
    margin = max(5, min(h, w) // 150)
    x, y, bw, bh = cv2.boundingRect(contour)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)
    roi = mask[y1:y2, x1:x2]
    return np.count_nonzero(roi) > 0


def estimate_avg_wall_thickness(wall_mask: np.ndarray) -> float:
    m = (wall_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    vals = dt[m > 0]
    if len(vals) == 0:
        return 0.0
    p25, p75 = np.percentile(vals, [25, 75])
    iqr = p75 - p25
    inlier = (vals >= p25 - 1.5 * iqr) & (vals <= p75 + 1.5 * iqr)
    filtered = vals[inlier]
    if len(filtered) == 0:
        filtered = vals
    return float(2.0 * np.mean(filtered))
