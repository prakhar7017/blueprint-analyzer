from __future__ import annotations

import base64
import logging

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel

from room_detection import detect_rooms
from utils import (
    draw_fixtures,
    draw_room_contours,
    draw_wall_contours,
    encode_image_to_bytes,
    load_image_from_bytes,
)
from wall_detection import detect_fixtures, detect_walls, estimate_avg_wall_thickness

logger = logging.getLogger(__name__)

app = FastAPI(title="Blueprint analyzer")


class FixtureCounts(BaseModel):
    lights: int
    doors: int
    windows: int


class PredictResponse(BaseModel):
    num_walls: int
    num_rooms: int
    avg_wall_thickness: float
    fixtures: FixtureCounts
    image_base64: str


def _run_pipeline(
    data: bytes,
) -> tuple[bytes, int, int, float, FixtureCounts]:
    image = load_image_from_bytes(data)
    wall_mask, wall_contours = detect_walls(image)
    avg_thickness = estimate_avg_wall_thickness(wall_mask)
    room_contours = detect_rooms(wall_mask)
    fixtures = detect_fixtures(image, wall_mask)
    out = draw_room_contours(image, room_contours)
    out = draw_wall_contours(out, wall_contours)
    out = draw_fixtures(out, fixtures)
    png = encode_image_to_bytes(out)
    fcounts = FixtureCounts(
        lights=len(fixtures["lights"]),
        doors=len(fixtures["doors"]),
        windows=len(fixtures["windows"]),
    )
    return png, len(wall_contours), len(room_contours), avg_thickness, fcounts


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        png, num_walls, num_rooms, avg_thickness, fcounts = _run_pipeline(data)
    except Exception:
        logger.exception("Pipeline failed for uploaded file: %s", file.filename)
        raise HTTPException(status_code=400, detail="Could not process the uploaded file. Ensure it is a valid image or PDF.")
    return PredictResponse(
        num_walls=num_walls,
        num_rooms=num_rooms,
        avg_wall_thickness=avg_thickness,
        fixtures=fcounts,
        image_base64=base64.b64encode(png).decode("ascii"),
    )


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        png, num_walls, num_rooms, avg_thickness, fcounts = _run_pipeline(data)
    except Exception:
        logger.exception("Pipeline failed for uploaded file: %s", file.filename)
        raise HTTPException(status_code=400, detail="Could not process the uploaded file. Ensure it is a valid image or PDF.")
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Num-Walls": str(num_walls),
            "X-Num-Rooms": str(num_rooms),
            "X-Avg-Wall-Thickness": f"{avg_thickness:.2f}",
            "X-Lights": str(fcounts.lights),
            "X-Doors": str(fcounts.doors),
            "X-Windows": str(fcounts.windows),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
