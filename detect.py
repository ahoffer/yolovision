#!/usr/bin/env python3
import pydevd_pycharm
import torch

from AppConfig import AppConfig
from DetectionApp import DetectionApp


def main():
    try:
        config = AppConfig(
            model_path="yolov8n",
            input_video='/home/aaron/Videos/tank-cars-people.mp4',
            output_video='/home/aaron/Videos/output.mp4',
            display_size=(1920, 1080),
            loop_video=True,
            confidence_threshold=0.3,
            batch_size=1,
            detector='cuda' if torch.cuda.is_available() else 'cpu'
        )

        app = DetectionApp(config)
        app.initialize()
        app.run()
        print("\nVideo processing completed")

    except Exception as e:
        raise


if __name__ == "__main__":
    # pydevd_pycharm.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)
    main()
