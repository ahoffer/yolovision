from collections import deque

import numpy as np


class ConfidenceTracker:
    def __init__(self, history_size):
        self.history_size = history_size
        self.confidence_history = {}

    def update(self, boxes):
        current_detections = {}

        for box in boxes:
            try:
                # Get box coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                if xyxy is None or len(xyxy) != 4:
                    continue

                # Get class and confidence
                cls = int(box.cls[0]) if len(box.cls) > 0 else -1
                conf = float(box.conf[0]) if len(box.conf) > 0 else 0.0

                # Create key
                det_key = self._create_detection_key(cls, xyxy)
                current_detections[det_key] = conf

                # Initialize or update confidence history
                if det_key not in self.confidence_history:
                    self.confidence_history[det_key] = deque(maxlen=self.history_size)
                self.confidence_history[det_key].append(conf)

            except Exception as e:
                print(f"Warning: Error processing detection in confidence tracker: {e}")
                continue

        # Remove old tracks that weren't detected in current frame
        self.confidence_history = {k: v for k, v in self.confidence_history.items()
                                   if k in current_detections}

    def get_averaged_confidence(self, box):
        try:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0]) if len(box.cls) > 0 else -1
            det_key = self._create_detection_key(cls, xyxy)

            if det_key in self.confidence_history:
                return np.mean(list(self.confidence_history[det_key]))
            return float(box.conf[0]) if len(box.conf) > 0 else 0.0

        except Exception as e:
            print(f"Warning: Error getting averaged confidence: {e}")
            return float(box.conf[0]) if len(box.conf) > 0 else 0.0

    def _create_detection_key(self, cls, xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        return f"{cls}_{x1}_{y1}_{x2}_{y2}"
