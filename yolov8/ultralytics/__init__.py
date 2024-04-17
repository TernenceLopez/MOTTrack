# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = "8.0.38"

from yolov8.ultralytics.yolo.engine.model import YOLO
from yolov8.ultralytics.yolo.utils.checks import check_yolo as checks

__all__ = ["__version__", "YOLO", "checks"]  # allow simpler import
