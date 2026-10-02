"""Drawing and FPS helpers for the face detection demo."""

import time

import cv2  # type: ignore[import-not-found]
import numpy as np

from detector import Detection


class FPSCounter:
    def __init__(self) -> None:
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed > 0:
            instantaneous = 1.0 / elapsed
            self._fps = instantaneous if self._fps == 0 else 0.85 * self._fps + 0.15 * instantaneous
        return self._fps


def draw_overlay(frame: np.ndarray, faces: list[Detection], fps: float) -> np.ndarray:
    for face in faces:
        cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), (40, 220, 40), 2)
        cv2.putText(
            frame,
            f"Face {face.confidence:.0%}",
            (face.x1, max(24, face.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Faces: {len(faces)}  |  FPS: {fps:.1f}  |  Q: quit",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
