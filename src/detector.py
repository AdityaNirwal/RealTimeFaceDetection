"""OpenCV DNN face detector using the ResNet-10 SSD Caffe model."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
PROTO_PATH = MODEL_DIR / "face_detector.prototxt"
MODEL_PATH = MODEL_DIR / "face_detector.caffemodel"
PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


@dataclass(frozen=True)
class Detection:
    """A detected face in pixel coordinates and its confidence score."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    print(f"Downloading {path.name} ...")
    try:
        with urlopen(url, timeout=60) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary_path.stat().st_size < 1024:
            raise RuntimeError(f"Downloaded model file is unexpectedly small: {path.name}")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


class FaceDetector:
    """Load the pretrained OpenCV Caffe model and detect faces in frames."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        _download_if_missing(PROTO_PATH, PROTO_URL)
        _download_if_missing(MODEL_PATH, MODEL_URL)
        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return faces found in a BGR image, ordered by confidence."""
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        output = self.net.forward()
        faces: list[Detection] = []
        for item in output[0, 0]:
            confidence = float(item[2])
            if confidence < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = (item[3:7] * np.array([width, height, width, height])).astype(int)
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            if x2 > x1 and y2 > y1:
                faces.append(Detection(x1, y1, x2, y2, confidence))
        return sorted(faces, key=lambda face: face.confidence, reverse=True)
