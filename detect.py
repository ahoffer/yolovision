#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Object Detection Script
This script performs object detection on video files using YOLO models.
Author: Assistant
License: MIT
"""

import cv2
import torch
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize the object detector
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def process_video(self, input_path, output_path):
        """
        Process video file and save output with detected objects
        Args:
            input_path: Path to input video
            output_path: Path to save output video
        """
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform detection
                results = self.model(frame, device=self.device)[0]

                # Process detections
                annotated_frame = self.draw_detections(frame, results)

                # Write frame
                out.write(annotated_frame)

                # Display progress (optional)
                cv2.imshow('Processing...', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        Args:
            frame: Input frame
            results: YOLO detection results
        Returns:
            Annotated frame
        """
        for detection in results.boxes.data:
            if detection[4] >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, detection[:4])
                conf = float(detection[4])
                class_id = int(detection[5])

                # Get class name
                class_name = results.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Create label with class name and confidence
                label = f'{class_name} {conf:.2f}'

                # Calculate label size and position
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width + 10, y1), (0, 255, 0), -1)

                # Draw label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame


def main():
    # Initialize detector
    detector = ObjectDetector(model_path='./yolov8n.pt',  # Use tiny model for faster inference
        conf_threshold=0.5  # Adjust confidence threshold as needed
    )

    # Process video
    input_video = '/home/aaron/Videos/tank-cars-people.mp4'  # Replace with your input video path
    output_video = '/home/aaron/Videos/output.mp4'  # Replace with desired output path

    detector.process_video(input_video, output_video)


if __name__ == '__main__':
    main()
