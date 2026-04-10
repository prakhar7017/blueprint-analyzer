import io
import random

import cv2
import numpy as np
from PIL import Image

PDF_MAGIC = b"%PDF"


def _is_pdf(data: bytes) -> bool:
    return data[:4] == PDF_MAGIC


def _pdf_first_page_to_bytes(data: bytes) -> bytes:
    """Render first page of a PDF to PNG bytes at 200 DPI."""
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        return pix.tobytes("png")


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
