#!/usr/bin/env python3

import time
from abc import ABC, abstractmethod
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Base Video I/O Classes
class VideoSource(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def read_frame(self):
        pass

    @abstractmethod
    def release(self):
        pass


class VideoSink(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def write_frame(self, frame):
        pass

    @abstractmethod
    def release(self):
        pass


class FileVideoSource(VideoSource):
    def __init__(self, input_path):
        self.input_path = input_path
        self.cap = None
        self.metadata = {}

    def initialize(self):
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open input video: {self.input_path}")

        self.metadata = {'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        return self

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Video capture not initialized")
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def reset(self):
        """Reset video to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_metadata(self):
        return self.metadata


class FileVideoSink(VideoSink):
    def __init__(self, output_path, fps, width, height):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = None

    def initialize(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        return self

    def write_frame(self, frame):
        if self.writer is None:
            raise RuntimeError("Video writer not initialized")
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()


# Display Classes
class DisplayWindow:
    def __init__(self, window_name="Detection", window_size=(1280, 720)):
        self.window_name = window_name
        self.window_size = window_size

    def initialize(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)
        return self

    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)

    def close(self):
        cv2.destroyWindow(self.window_name)


class UserInterface:
    def __init__(self, display_window):
        self.display = display_window

    def show_frame(self, frame):
        self.display.show_frame(frame)

    def check_quit(self, wait_time):
        return cv2.waitKey(wait_time) & 0xFF == ord('q')

    def cleanup(self):
        self.display.close()
        cv2.destroyAllWindows()


# Performance Monitoring
class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.processing_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.loop_count = 0

    def start_frame(self):
        return time.time()

    def end_frame(self, start_time):
        self.frame_count += 1
        self.processing_times.append(time.time() - start_time)

    def increment_loop(self):
        self.loop_count += 1

    def get_stats(self):
        if not self.processing_times:
            return None

        avg_time = sum(self.processing_times) / len(self.processing_times)
        return {'avg_fps': 1.0 / avg_time, 'avg_frame_time': avg_time * 1000,  # in ms
            'frame_count': self.frame_count, 'loop_count': self.loop_count}


# Detection Classes
class DetectionConfig:
    def __init__(self, model_path, device=None, confidence_interval=1):
        self.model_path = model_path
        self.device = device or self._get_optimal_device()
        self.confidence_interval = confidence_interval

    def _get_optimal_device(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.current_device()
                return 'cuda'
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                return 'cpu'
        return 'cpu'


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


class YOLODetector:
    def __init__(self, config, confidence_tracker):
        self.config = config
        self.confidence_tracker = confidence_tracker
        self.model = None

    def initialize(self):
        try:
            self.model = YOLO(self.config.model_path)
            print(f"Model loaded successfully on {self.config.device}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, frame):
        try:
            results = self.model(frame, device=self.config.device)[0]
            self.confidence_tracker.update(results.boxes)
            averaged_results = self._average_confidences(results)
            return averaged_results.plot()
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


# Main Application Classes
class ProcessingPipeline:
    def __init__(self, source, sink, detector, performance_monitor):
        self.source = source
        self.sink = sink
        self.detector = detector
        self.performance_monitor = performance_monitor

    def process_frame(self):
        ret, frame = self.source.read_frame()
        if not ret:
            return False, None

        processed_frame = self.detector.detect(frame)
        self.sink.write_frame(processed_frame)
        return True, processed_frame


class DetectionApp:
    def __init__(self, config):
        self.config = config
        self.source = FileVideoSource(config.input_video)
        self.ui = UserInterface(DisplayWindow(window_size=config.display_size))
        self.performance_monitor = PerformanceMonitor()
        self.detector = None
        self.sink = None
        self.pipeline = None

    def initialize(self):
        # Initialize video source
        self.source.initialize()
        metadata = self.source.get_metadata()

        # Initialize detector
        confidence_tracker = ConfidenceTracker(self.config.confidence_interval)
        detector_config = DetectionConfig(self.config.model_path)
        self.detector = YOLODetector(detector_config, confidence_tracker)
        self.detector.initialize()

        # Initialize output and display
        self.sink = FileVideoSink(self.config.output_video, metadata['fps'], metadata['width'], metadata['height'])
        self.sink.initialize()
        self.ui.display.initialize()

        # Create pipeline
        self.pipeline = ProcessingPipeline(self.source, self.sink, self.detector, self.performance_monitor)

        # Print configuration
        print(f"Input video: {self.config.input_video}")
        print(f"Output video: {self.config.output_video}")
        print(f"FPS: {metadata['fps']}")
        print(f"Resolution: {metadata['width']}x{metadata['height']}")
        print(f"Loop mode: {'enabled' if self.config.loop_video else 'disabled'}")

    def run(self):
        try:
            print("\nProcessing video...")
            print("Press 'q' to quit")

            while True:
                frame_start = self.performance_monitor.start_frame()

                # Process frame
                success, frame = self.pipeline.process_frame()

                # Handle end of video
                if not success:
                    if self.config.loop_video:
                        self.source.reset()
                        self.performance_monitor.increment_loop()
                        print(f"\nStarting loop {self.performance_monitor.loop_count}")
                        continue
                    break

                # Display frame
                self.ui.show_frame(frame)

                # Update performance stats
                self.performance_monitor.end_frame(frame_start)

                # Show stats periodically
                if self.performance_monitor.frame_count % 30 == 0:
                    stats = self.performance_monitor.get_stats()
                    if stats:
                        print(f"\rLoop: {stats['loop_count']} | "
                              f"FPS: {stats['avg_fps']:.1f}", end="")

                # Check for user quit
                if self.ui.check_quit(1):
                    print("\nProcessing interrupted by user")
                    break

        except Exception as e:
            print(f"\nError during video processing: {e}")
            raise

        finally:
            self.cleanup()

            # Print final statistics
            stats = self.performance_monitor.get_stats()
            if stats:
                print(f"\n\nProcessing Statistics:")
                print(f"Total frames processed: {stats['frame_count']}")
                print(f"Number of complete loops: {stats['loop_count']}")
                print(f"Average processing time per frame: {stats['avg_frame_time']:.1f}ms")
                print(f"Average FPS: {stats['avg_fps']:.1f}")

    def cleanup(self):
        self.source.release()
        self.sink.release()
        self.ui.cleanup()


class AppConfig:
    def __init__(self, model_path, input_video, output_video, confidence_interval=1, display_size=(1920, 1080),
                 loop_video=False):
        self.model_path = model_path
        self.input_video = input_video
        self.output_video = output_video
        self.confidence_interval = confidence_interval
        self.display_size = display_size
        self.loop_video = loop_video


def main():
    try:
        config = AppConfig(model_path="yolov8n.pt", input_video='/home/aaron/Videos/tank-cars-people.mp4',
            output_video='/home/aaron/Videos/output.mp4', confidence_interval=4, display_size=(1920, 1080),
            loop_video=True)

        app = DetectionApp(config)
        app.initialize()
        app.run()
        print("\nVideo processing completed")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
