import argparse
from pathlib import Path

import cv2

from camera import VideoSource
from detector import FaceDetector
from utils import FPSCounter, draw_overlay


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time OpenCV DNN face detection")
    parser.add_argument("--source", default="0", help="Camera index (0), video path, or image path")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence from 0 to 1")
    parser.add_argument("--output", help="Optional output video path (video/camera sources only)")
    args = parser.parse_args()

    source = parse_source(args.source)
    detector = FaceDetector(args.confidence)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if isinstance(source, str) and Path(source).suffix.lower() in image_extensions:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        faces = detector.detect(frame)
        draw_overlay(frame, faces, 0.0)
        cv2.imshow("Real-Time Face Detection", frame)
        print(f"Detected {len(faces)} face(s). Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    video = VideoSource(source)
    writer = None
    fps_counter = FPSCounter()
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            faces = detector.detect(frame)
            fps = fps_counter.tick()
            draw_overlay(frame, faces, fps)
            if args.output and writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {args.output}")
            if writer is not None:
                writer.write(frame)
            cv2.imshow("Real-Time Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
