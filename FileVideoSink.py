import cv2

from VideoSink import VideoSink


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
