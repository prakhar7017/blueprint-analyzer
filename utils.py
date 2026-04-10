import io
import random

import cv2
import numpy as np
from PIL import Image

PDF_MAGIC = b"%PDF"


def _is_pdf(data: bytes) -> bool:
    return data[:4] == PDF_MAGIC


def _pdf_first_page_to_bytes(data: bytes) -> bytes:
    import fitz 

    with fitz.open(stream=data, filetype="pdf") as doc:
        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        return pix.tobytes("png")


def _deskew(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 720, threshold=min(gray.shape[:2]) // 4)
    if lines is None:
        return gray

    angles = []
    for rho, theta in lines[:, 0]:
        deg = np.degrees(theta)
        if deg < 10 or abs(deg - 90) < 10 or abs(deg - 180) < 10:
            offset = deg if deg < 45 else (deg - 90 if deg < 135 else deg - 180)
            angles.append(offset)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return gray
    if abs(median_angle) > 10:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _remove_border(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    margin = max(3, min(h, w) // 100)

    edges = cv2.Canny(gray, 50, 150)
    top_density = np.count_nonzero(edges[:margin, :]) / max(1, margin * w)
    bot_density = np.count_nonzero(edges[h - margin:, :]) / max(1, margin * w)
    left_density = np.count_nonzero(edges[:, :margin]) / max(1, margin * h)
    right_density = np.count_nonzero(edges[:, w - margin:]) / max(1, margin * h)

    result = gray.copy()
    threshold = 0.15
    bg = int(np.percentile(gray, 95))
    if top_density > threshold:
        result[:margin, :] = bg
    if bot_density > threshold:
        result[h - margin:, :] = bg
    if left_density > threshold:
        result[:, :margin] = bg
    if right_density > threshold:
        result[:, w - margin:] = bg

    return result


def _normalize_blueprint_colors(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    channels = [l_ch, a_ch, b_ch]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    channels.append(gray)

    best = gray
    best_std = float(np.std(gray))
    for ch in channels:
        s = float(np.std(ch))
        if s > best_std:
            best_std = s
            best = ch

    if best.dtype != np.uint8:
        best = cv2.normalize(best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mean_val = np.mean(best)
    if mean_val < 127:
        best = cv2.bitwise_not(best)

    return best


def preprocess_blueprint(image_bgr: np.ndarray) -> np.ndarray:
    gray = _normalize_blueprint_colors(image_bgr)
    gray = _deskew(gray)
    gray = _remove_border(gray)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return gray


def load_image_from_bytes(data: bytes) -> np.ndarray:
    if _is_pdf(data):
        data = _pdf_first_page_to_bytes(data)
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_wall_contours(image: np.ndarray, contours: list) -> np.ndarray:
    out = image.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def draw_room_contours(image: np.ndarray, contours: list) -> np.ndarray:
    out = image.copy()
    rng = random.Random(42)
    for c in contours:
        color = (rng.randint(32, 255), rng.randint(32, 255), rng.randint(32, 255))
        cv2.drawContours(out, [c], -1, color, 2)
    return out


def draw_fixtures(image: np.ndarray, fixtures: dict) -> np.ndarray:
    out = image.copy()
    for f in fixtures.get("lights", []):
        cv2.circle(out, (f["x"], f["y"]), f["radius"], (255, 255, 0), 2)
        cv2.putText(out, "light", (f["x"] - 15, f["y"] - f["radius"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for f in fixtures.get("doors", []):
        cv2.rectangle(out, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]),
                      (255, 0, 255), 2)
        cv2.putText(out, "door", (f["x"], f["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    for f in fixtures.get("windows", []):
        cv2.rectangle(out, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]),
                      (0, 255, 255), 2)
        cv2.putText(out, "window", (f["x"], f["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def encode_image_to_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image")
    return buf.tobytes()
