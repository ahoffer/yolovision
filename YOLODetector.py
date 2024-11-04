import cv2
import torch
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, config, confidence_tracker):
        self.config = config
        self.confidence_tracker = confidence_tracker
        self.model = None
        self.class_names = None

    def initialize(self):
        try:
            self.model = YOLO(self.config.model_path)
            print(f"Model loaded successfully on {self.config.device}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, frame):
        try:
            # Run detection
            results = self.model(frame, device=self.config.device)[0]
            self.confidence_tracker.update(results.boxes)
            averaged_results = self._average_confidences(results)

            # Draw detections on frame
            annotated_frame = frame.copy()
            for box in averaged_results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Create label with class name and confidence
                label = f"{results.names[cls]} {conf:.2f}"

                # Calculate text size and position
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw label background
                cv2.rectangle(annotated_frame, (int(x1), int(y1 - text_height - 5)), (int(x1 + text_width), int(y1)),
                              (0, 255, 0), -1)

                # Draw label text
                cv2.putText(annotated_frame, label, (int(x1), int(y1 - 5)), font, font_scale, (0, 0, 0), thickness)

            return annotated_frame

        except Exception as e:
            print(f"Warning: Detection error: {e}")
            return frame

    def _average_confidences(self, results):
        if not hasattr(results, 'boxes') or len(results.boxes) == 0:
            return results

        try:
            averaged_results = results
            for i, box in enumerate(results.boxes):
                try:
                    avg_conf = self.confidence_tracker.get_averaged_confidence(box)
                    averaged_results.boxes[i].conf = torch.tensor([avg_conf]).to(self.config.device)
                except Exception as e:
                    print(f"Warning: Error averaging confidence for detection {i}: {e}")
            return averaged_results
        except Exception as e:
            print(f"Warning: Error creating averaged results: {e}")
            return results
