import cv2
import time
from AppConfig import AppConfig

from FramerateController import FramerateController
from VideoSink import VideoSink


class DisplaySink(VideoSink):
    def __init__(self, config: AppConfig):
        self.config = config
        self.targetFramerate = None
        self.last_frame_time = None
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0
        self.windowSize = (640,640)

    def initialize(self):
        cv2.namedWindow("video", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("video", *self.windowSize)
        self.frameRateController  = FramerateController()
        self.last_frame_time = time.time()
        self.start_time = self.last_frame_time
        return self

    def process_frame(self, frame):
        self.frameRateController.target_fps = self.targetFramerate
        self.frameRateController.wait_for_next_frame()

        # Display frame
        cv2.imshow("video", frame)

        self.frameRateController.update_stats()

    def is_window_closed(self):
        try:
            return cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1
        except:
            return True

    def release(self):
        cv2.destroyWindow("video")

    @property
    def fps(self):
        """Return the actual frames per second being displayed"""
        return self.actual_fps