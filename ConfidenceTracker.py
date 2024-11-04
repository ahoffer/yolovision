from collections import deque

import numpy as np


class ConfidenceTracker:
    def __init__(self, history_size):
        self.history_size = history_size
        self.confidence_history = {}

    def update(self, detections):
        current_detections = {}

        for det in detections:
            det_key = self._create_detection_key(det)
            current_detections[det_key] = det.confidence

            if det_key not in self.confidence_history:
                self.confidence_history[det_key] = deque(maxlen=self.history_size)
            self.confidence_history[det_key].append(det.confidence)

        # Cleanup old detections
        self.confidence_history = {k: v for k, v in self.confidence_history.items() if k in current_detections}

    def get_averaged_confidence(self, detection):
        det_key = self._create_detection_key(detection)
        if det_key in self.confidence_history:
            return np.mean(list(self.confidence_history[det_key]))
        return detection.confidence

    def _create_detection_key(self, detection):
        return f"{detection.class_id}_{int(detection.x1)}_{int(detection.y1)}_{int(detection.x2)}_{int(detection.y2)}"
