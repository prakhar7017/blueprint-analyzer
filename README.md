# Blueprint analyzer (walls + rooms)

Minimal demo: pretrained YOLOv8-seg (with ONNX Runtime support) + OpenCV fallback, simple room regions from the wall mask, and a FastAPI service.

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On first run, Ultralytics downloads `yolov8n-seg.pt` (pretrained COCO weights) and exports it to ONNX automatically.

## Run the API

```bash
uvicorn main:app --reload
```

Or directly:

```bash
python main.py
```

Open `http://127.0.0.1:8000/docs` for the interactive OpenAPI UI.

## Endpoints

### GET /

Health check. Returns `{"status": "ok"}`.

### POST /predict

Returns JSON with counts and base64-encoded annotated image. Accepts **image files** (PNG, JPEG, etc.) or **PDF** blueprints (first page is rendered at 200 DPI).

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@blueprint.png" -o response.json
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@blueprint.pdf" -o response.json
```

Response fields:

- `num_walls`, `num_rooms`: counts after filtering small contours (area under 500 pixels).
- `avg_wall_thickness`: rough estimate in pixels from the distance transform on `wall_mask`; not calibrated.
- `fixtures`: object with `lights`, `doors`, `windows` counts (heuristic, see Limitations).
- `image_base64`: annotated PNG, base64-encoded. Decode with `base64.b64decode` to get raw PNG bytes.

### POST /predict/image

Returns the annotated image directly as `image/png`. Metadata in response headers (`X-Num-Walls`, `X-Num-Rooms`, `X-Avg-Wall-Thickness`, `X-Lights`, `X-Doors`, `X-Windows`). Also accepts PDF input.

```bash
curl -X POST "http://127.0.0.1:8000/predict/image" -F "file=@blueprint.png" -o result.png
curl -X POST "http://127.0.0.1:8000/predict/image" -F "file=@blueprint.pdf" -o result.png
```

## Inference pipeline

1. Try **ONNX Runtime** inference (faster, exported automatically on first use).
2. If ONNX fails, fall back to **PyTorch** inference.
3. If both fail or return no masks, fall back to **OpenCV Canny + morphology**.

## Fixtures (bonus)

Detected via OpenCV heuristics (no trained model):

- **Lights**: small circles found with `HoughCircles` — annotated in cyan.
- **Doors**: arc-shaped contours (low circularity, near-square bounding box) — annotated in magenta.
- **Windows**: elongated rectangular contours outside wall regions — annotated in yellow.

**Wall types / tags** would require OCR (e.g. Tesseract), which is outside the specified tech stack. Not implemented.

## Limitations

- The model is **not** trained on blueprints; COCO segmentation rarely matches architectural walls, so the pipeline often relies on **Canny + morphology** as a generic fallback.
- Room regions are a **heuristic** (inverted wall mask + morphology + contours), not true floor-plan parsing.
- Wall thickness is a rough pixel estimate, not a calibrated measurement.
- Fixture detection is **heuristic-based** and will produce false positives/negatives on real blueprints.
- **Accuracy is not guaranteed**; this is a minimal assignment-style baseline, not production CAD software.
