# Blueprint analyzer (walls + rooms)

Service that preprocesses blueprint images, fuses **classical CV wall evidence** (adaptive/Otsu thresholds, Hough line segments, auto-Canny) with optional **YOLOv8-seg** masks, refines the binary wall mask, then derives **rooms** (morphology + watershed) and **fixtures** (heuristics). FastAPI exposes `/predict` and `/predict/image`.

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

   PDF uploads: the first page is rasterized at **300 DPI** (see `utils._pdf_first_page_to_bytes`).

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

Returns JSON with counts and base64-encoded annotated image. Accepts **image files** (PNG, JPEG, etc.) or **PDF** blueprints (first page is rendered at **300 DPI**).

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@blueprint.png" -o response.json
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@blueprint.pdf" -o response.json
```

Response fields:

- `num_walls`: wall contours kept after **minimum area 500 px²** (see `MIN_CONTOUR_AREA` in `wall_detection`).
- `num_rooms`: room contours after **morphology + watershed** candidates, **border removal** (full-sheet rectangles dropped), **shape filters** (min area ≈ max(500, 0.1% of image), max ≈ 85% of image, aspect and solidity checks), and **bbox overlap merging**—not the same rule as walls alone.
- `avg_wall_thickness`: pixels from the distance transform on the binary `wall_mask`, using **IQR-trimmed** interior distances then **twice the mean** of the inliers; not calibrated to real units.
- `fixtures`: object with `lights`, `doors`, `windows` counts (heuristic; see **Fixtures** below).
- `image_base64`: annotated PNG, base64-encoded. Decode with `base64.b64decode` to get raw PNG bytes.

### POST /predict/image

Returns the annotated image directly as `image/png`. Metadata in response headers (`X-Num-Walls`, `X-Num-Rooms`, `X-Avg-Wall-Thickness`, `X-Lights`, `X-Doors`, `X-Windows`). Also accepts PDF input.

```bash
curl -X POST "http://127.0.0.1:8000/predict/image" -F "file=@blueprint.png" -o result.png
curl -X POST "http://127.0.0.1:8000/predict/image" -F "file=@blueprint.pdf" -o result.png
```

## Preprocessing (`utils.preprocess_blueprint`)

Before wall detection, grayscale is derived by picking the **highest-contrast** channel among L/A/B from LAB, raw gray, and BGR gray; the image may be **inverted** if it reads as dark-on-light. **Deskewing** uses `HoughLines` on edges (small angles only). **Border stripping** clears margins where edge density suggests a frame. A **bilateral filter** reduces noise while preserving edges.

## Inference pipeline (`detect_walls`)

1. **Classical wall mask**: On CLAHE-enhanced grayscale, build four binary maps—**adaptive Gaussian threshold**, **Otsu** (inverted), **HoughLinesP** line segments (two scales; axis-aligned lines get extra weight), and **median-based auto-Canny** with light dilation—then **fuse** them with fixed weights into one score map (threshold ≈ 0.35).
2. **YOLOv8-seg**: Run **ONNX** first, then **PyTorch** if needed (`conf=0.3`, `iou=0.5`). If any instance masks exist, they are **blended** into the classical map with a small weight (~15%) and re-thresholded.
3. **Refine**: **Morphological close/open**, drop tiny connected components, then **findContours** and filter by area; the returned mask is contour-**filled** when possible.

Rooms and fixtures use the resulting `wall_mask` as described below.

## Fixtures (bonus)

Detected via OpenCV heuristics on **CLAHE-enhanced** grayscale (no fixture model):

- **Lights**: `HoughCircles` with **resolution-dependent** radii and spacing; multiple blur/`param2` passes; duplicate centers suppressed — cyan circles.
- **Doors**: (1) **Arc-shaped** contours on non-wall Canny edges—convexity/solidity, ellipse fit, and **near-wall** check; (2) **shape** candidates from non-wall contours with **circularity/aspect/solidity** rules, gated by **near wall** or **near wall-break** regions (`dilate(wall) - wall`); combined list passed through **bounding-box IoU NMS** — magenta rectangles.
- **Windows**: elongated contours (aspect thresholds) **near walls**, also **NMS**-deduplicated — yellow rectangles.

**Wall types / labels** would require OCR (e.g. Tesseract); not implemented.

## Approach & Design Choices

- **Input normalization (`utils`)**: Blueprints are normalized for robust line extraction—**channel selection**, optional **invert**, **deskew**, **frame/border suppression**, and **bilateral** smoothing—before `wall_detection` runs.

- **Walls (`wall_detection`)**: A **multi-cue classical mask** (adaptive threshold, Otsu, multi-scale Hough segments, auto-Canny) is the primary signal. **YOLOv8-seg** (COCO, all instance masks OR’d together—no architectural class) **augments** that mask when masks are available; it is not an exclusive “DL then else CV” path. The result is **refined** with morphology and small-component removal, then **filled** wall contours drive the binary mask returned to rooms/fixtures/thickness.

- **Rooms (`room_detection`)**: **Gaps in the wall mask** are closed first. Free space uses **morphological open/close** on the inverted mask and **CCOMP** contours, combined with a **watershed** split of the inverted mask (distance-transform seeds, `cv2.watershed` on free space). Candidates are **filtered** by area/aspect/solidity, **full-image border** contours removed, and **highly overlapping** room boxes merged.

- **Wall thickness**: **Distance transform** on the binary wall mask; interior distances are **filtered with an IQR rule** before averaging, then doubled to approximate thickness in **pixels** (uncalibrated).

- **Structure**: Modules stay separated—`utils` (load, preprocess, draw), `wall_detection` (walls + fixtures + thickness), `room_detection`, `model` (YOLO singletons + ONNX export), `main` (FastAPI)—so weights, fusion weights, and heuristics can evolve independently.
