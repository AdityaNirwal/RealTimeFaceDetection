"""Small wrapper around OpenCV video capture."""

import cv2


class VideoSource:
    def __init__(self, source: str | int = 0) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                f"Could not open video source {source!r}. Check the camera index or file path."
            )

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()
