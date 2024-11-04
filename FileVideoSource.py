import cv2

from VideoSource import VideoSource


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
