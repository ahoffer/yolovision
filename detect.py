#!/usr/bin/env python3

from AppConfig import AppConfig
from DetectionApp import DetectionApp

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
