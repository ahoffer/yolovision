#!/usr/bin/env python3

import torch
import cv2
from ultralytics import YOLO
from pathlib import Path
import time
from abc import ABC, abstractmethod


class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None

    def initialize(self):
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open input video: {self.input_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        return self

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Video capture not initialized")
        return self.cap.read()

    def write_frame(self, frame):
        if self.writer is None:
            raise RuntimeError("Video writer not initialized")
        self.writer.write(frame)

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

    def get_video_info(self):
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames,
            'input_path': self.input_path,
            'output_path': self.output_path
        }


class DetectionModel(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def detect(self, frame, show_confidence=True):
        pass


class YOLOModel(DetectionModel):
    def __init__(self, model_path, confidence_update_interval=1):
        self.model_path = model_path
        self.model = None
        self.device = self._get_optimal_device()
        self.confidence_update_interval = confidence_update_interval
        self.frame_count = 0
        self.last_results = None

    def _get_optimal_device(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.current_device()
                return 'cuda'
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                print("Falling back to CPU")
                return 'cpu'
        return 'cpu'

    def initialize(self):
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully on {self.device}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, frame):
        """
        Perform detection with confidence update interval
        """
        self.frame_count += 1

        # Determine if we should update detections this frame
        should_update = (self.frame_count % self.confidence_update_interval) == 0

        if should_update:
            # Perform new detection
            results = self.model(frame, device=self.device)[0]
            self.last_results = results
            return results.plot()
        else:
            # Use last results but with new frame
            if self.last_results is not None:
                # Plot last results on current frame
                return self.last_results.plot()
            else:
                # First frame case
                results = self.model(frame, device=self.device)[0]
                self.last_results = results
                return results.plot()


class DisplayManager:
    def __init__(self, window_name="Detection"):
        self.window_name = window_name

    def initialize(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        return self

    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)

    def handle_input(self, wait_time):
        return cv2.waitKey(wait_time) & 0xFF == ord('q')


class DetectionApp:
    def __init__(self, model_path, input_video, output_video, confidence_update_interval=1):
        self.video_processor = VideoProcessor(input_video, output_video)
        self.model = YOLOModel(model_path, confidence_update_interval)
        self.display = DisplayManager()

    def initialize(self):
        self.video_processor.initialize()
        self.model.initialize()
        self.display.initialize()

        info = self.video_processor.get_video_info()
        print(f"Input video: {info['input_path']}")
        print(f"Output video: {info['output_path']}")
        print(f"FPS: {info['fps']}")
        print(f"Resolution: {info['width']}x{info['height']}")

    def run(self):
        try:
            frame_count = 0
            frame_time = 1 / self.video_processor.fps

            while True:
                start_time = time.time()

                ret, frame = self.video_processor.read_frame()
                if not ret:
                    break

                processed_frame = self.model.detect(frame)

                self.video_processor.write_frame(processed_frame)
                self.display.show_frame(processed_frame)

                elapsed_time = time.time() - start_time
                wait_time = max(1, int((frame_time - elapsed_time) * 1000))
                if self.display.handle_input(wait_time):
                    print("\nProcessing interrupted by user")
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / self.video_processor.total_frames) * 100
                    current_fps = 1.0 / (time.time() - start_time)
                    print(f"Processing: {progress:.1f}% complete (FPS: {current_fps:.1f})")

        finally:
            self.video_processor.release()


def main():
    try:
        app = DetectionApp(
            model_path="yolov8n.pt",
            input_video='/home/aaron/Videos/tank-cars-people.mp4',
            output_video='/home/aaron/Videos/output.mp4',
            confidence_update_interval=8  # Update confidence every 4 frames
        )
        app.initialize()
        app.run()
        print("\nVideo processing completed")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    while  True:
        main()