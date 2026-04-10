import os

from ultralytics import YOLO

_pt_model = None
_onnx_model = None
_onnx_path = None


def get_model():
    """Return singleton YOLOv8n-seg with pretrained COCO weights (PyTorch)."""
    global _pt_model
    if _pt_model is None:
        _pt_model = YOLO("yolov8n-seg.pt")
    return _pt_model


def export_to_onnx() -> str:
    """Export YOLO model to ONNX if not already exported. Returns actual path."""
    global _onnx_path
    if _onnx_path is not None and os.path.exists(_onnx_path):
        return _onnx_path
    pt_model = get_model()
    _onnx_path = str(pt_model.export(format="onnx", imgsz=640, simplify=True))
    return _onnx_path


def get_onnx_model():
    """Return singleton YOLO loaded from ONNX (uses onnxruntime under the hood)."""
    global _onnx_model
    if _onnx_model is None:
        path = export_to_onnx()
        _onnx_model = YOLO(path)
    return _onnx_model
