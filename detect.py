#! /usr/bin/python3

import torch
import cv2
from ultralytics import YOLO
from pathlib import Path
import time


class YOLODetector:
    def __init__(self, model_path):
        """
        Initialize YOLO detector with automatic device selection

        Args:
            model_path (str): Path to the YOLO model weights
        """
        self.model_path = model_path
        self.device = self._get_optimal_device()
        self.model = None
        self.setup_model()

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

    def setup_model(self):
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def process_video(self, input_video, output_video):
        """
        Process video with YOLO detection while maintaining original FPS and displaying output

        Args:
            input_video (str): Path to input video file
            output_video (str): Path to output video file
        """
        try:
            # Open the input video
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                raise ValueError(f"Could not open input video: {input_video}")

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Input video: {input_video}")
            print(f"Output video: {output_video}")
            print(f"Input video FPS: {fps}")
            print(f"Resolution: {width}x{height}")

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_video,
                fourcc,
                fps,
                (width, height)
            )

            frame_count = 0
            frame_time = 1 / fps  # Time per frame in seconds

            # Create window for display
            cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)

            while cap.isOpened():
                start_time = time.time()  # Start timing for FPS control

                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame with YOLO
                results = self.model(frame, device=self.device)[0]

                # Draw detection boxes
                annotated_frame = results.plot()

                # Write and display frame
                out.write(annotated_frame)
                cv2.imshow('YOLO Detection', annotated_frame)

                # FPS control and exit condition
                elapsed_time = time.time() - start_time
                wait_time = max(1, int((frame_time - elapsed_time) * 1000))
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    current_fps = 1.0 / (time.time() - start_time)
                    print(f"Processing: {progress:.1f}% complete (FPS: {current_fps:.1f})")

            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            print("\nVideo processing completed")
            print(f"Output saved to: {output_video}")

        except Exception as e:
            # Clean up resources in case of error
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
            raise RuntimeError(f"Video processing failed: {e}")


def main():
    try:
        detector = YOLODetector("yolov8n.pt")  # or your model path
        input_video = '/home/aaron/Videos/tank-cars-people.mp4'
        output_video = '/home/aaron/Videos/output.mp4'
        detector.process_video(input_video, output_video)
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()