import cv2
from ultralytics import YOLO

from AppConfig import AppConfig


class YOLODetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.confidence_threshold = config.confidence_threshold

    def initialize(self):
        try:
            self.model = YOLO(self.config.model_path)
            if not self.model:
                raise RuntimeError("Model not initialized. Call initialize() first.")
            print(f"Model loaded successfully on {self.config.detector}")
            print(f"Confidence threshold: {self.confidence_threshold}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, frame):
        try:
            # Run detection
            results = self.model(frame, device=self.config.detector, verbose=False)[0]

            # Draw detections on frame with averaged confidences
            annotated_frame = frame.copy()

            for box in results.boxes:
                try:
                    # Get averaged confidence and check threshold
                    confidence = box.conf.mean().item()
                    if confidence < self.confidence_threshold:
                        continue

                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get class
                    cls = int(box.cls[0])

                    # Draw bounding box
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    # Create label with class name and averaged confidence
                    label = f"{results.names[cls]} {confidence:.1f}"

                    # Calculate text size and position
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )

                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1 - text_height - 5)),
                        (int(x1 + text_width), int(y1)),
                        (0, 255, 0),
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(x1), int(y1 - 5)),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness
                    )
                except Exception as e:
                    print(f"Warning: Error drawing detection: {e}")
                    continue

            return annotated_frame

        except Exception as e:
            print(f"Warning: Detection error: {e}")
            return frame
