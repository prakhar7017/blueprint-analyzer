# Blueprint analyzer (walls + rooms)

Service that preprocesses blueprint images (including **text-region suppression**), fuses **six classical wall cues** (adaptive/Otsu thresholds, Hough line segments, auto-Canny, **morphological gradient**, **directional** horizontal/vertical structure) plus **collinear Hough segment bridging**, then optionally blends **YOLOv8s-seg** instance masks. The refined wall mask feeds **rooms** (morphology + watershed + simplified contours) and **fixtures** (heuristics with circle validation on lights). FastAPI exposes `/predict` and `/predict/image`.

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

   On first run, Ultralytics downloads **`yolov8s-seg.pt`** (small segmentation, pretrained COCO weights) and exports it to ONNX automatically.

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
- `num_rooms`: room contours after **morphology + watershed** candidates, **border removal** (full-sheet rectangles dropped), **shape filters** (min area ≈ max(500, 0.1% of image), max ≈ 85% of image, aspect and solidity checks), **`approxPolyDP` simplification**, and **bbox overlap merging**—not the same rule as walls alone.
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

Grayscale uses the **highest-contrast** channel among L/A/B from LAB, raw gray, and BGR gray; the image may be **inverted** if it reads as dark-on-light. **Deskewing** uses `HoughLines` on edges (small angles only). **Border stripping** clears margins where edge density suggests a frame. **`_mask_text_regions`** detects high **local variance** blobs (Otsu on variance map, size/aspect filters) and paints them toward the background luminance to reduce lettering noise before walls are extracted. A **bilateral filter** runs last.

## Inference pipeline (`detect_walls`)

1. **Classical fusion**: On CLAHE-enhanced grayscale, build **six** binary cues—**adaptive Gaussian threshold**, **Otsu** (inverted), **HoughLinesP** (two scales; axis-aligned strokes boosted), **median-based auto-Canny** with light dilation, **morphological gradient** (Otsu on gradient magnitude), and **directional** masks (horizontal/vertical morphological opens on an Otsu binarization, then OR’d). Fixed weights sum into a score map; pixels with score **≥ ~0.30** are kept.
2. **Line bridging**: **HoughLinesP** on auto-Canny edges feeds **`_merge_collinear_segments`**, which links nearly parallel segments whose endpoints are close; the result is **OR’d** into the fused mask to reinforce long wall lines.
3. **YOLOv8s-seg**: Run **ONNX** first, then **PyTorch** if needed (`conf=0.25`, `iou=0.45`). If any instance masks exist, they are **blended** at **~15%** into the classical map and re-thresholded (**> 0.3** score).
4. **Refine & output**: **Morphological close/open**, drop tiny connected components, **findContours**, area filter (**≥ 500 px²**), then **`approxPolyDP` simplification** on wall contours; the returned mask is contour-**filled** when possible.

Rooms and fixtures use the resulting `wall_mask` as described below.

## Fixtures (bonus)

Detected via OpenCV heuristics on **CLAHE-enhanced** grayscale (no fixture model):

- **Lights**: `HoughCircles` with **resolution-dependent** radii and spacing; multiple blur/`param2` passes; each candidate is accepted only if **`_validate_circle`** finds enough **Canny edge** samples along the circumference; duplicate centers suppressed — cyan circles.
- **Doors**: (1) **Arc-shaped** contours on non-wall Canny edges—convexity/solidity, ellipse fit, and **near-wall** check; (2) **shape** candidates from non-wall contours with **circularity/aspect/solidity** rules, gated by **near wall** or **near wall-break** regions (`dilate(wall) - wall`); combined list passed through **bounding-box IoU NMS** — magenta rectangles.
- **Windows**: elongated contours (aspect thresholds) **near walls**, also **NMS**-deduplicated — yellow rectangles.

**Wall types / labels** would require OCR (e.g. Tesseract); not implemented.

## Approach & Design Choices

- **Input normalization (`utils`)**: Same as **Preprocessing** above: **channel selection**, **invert**, **deskew**, **border suppression**, **text-blob masking** (variance-based), then **bilateral** smoothing before `wall_detection`.

- **Walls (`wall_detection`)**: **Six classical cues** plus **collinear Hough bridging** form the main signal; **`yolov8s-seg.pt`** (COCO, all instance masks OR’d—no architectural class) **augments** at low weight when masks exist. Output wall contours are **simplified** with `approxPolyDP` for cleaner overlays.

- **Rooms (`room_detection`)**: **Gaps in the wall mask** are closed first. Free space uses **morphological open/close** on the inverted mask and **CCOMP** contours, combined with **watershed** regions from distance-transform seeds. Candidates are **filtered**, **border** sheet contours dropped, contours **simplified** (`approxPolyDP`), then **overlapping** room boxes merged.

- **Wall thickness**: **Distance transform** on the binary wall mask; interior distances use an **IQR inlier mask** before averaging, then **doubled** for thickness in **pixels** (uncalibrated).

- **Model (`model.py`)**: Singleton **PyTorch** and **ONNX** loaders for **`yolov8s-seg`** with automatic ONNX export (`imgsz=640`, `simplify=True`).

- **Structure**: `utils` (load, preprocess, draw), `wall_detection` (walls + fixtures + thickness), `room_detection`, `model`, `main` (FastAPI)—fusion weights, YOLO `conf`/`iou`, and heuristics can be tuned independently.
