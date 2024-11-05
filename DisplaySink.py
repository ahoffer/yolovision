import cv2
import time
from AppConfig import AppConfig
from VideoSink import VideoSink


class DisplaySink(VideoSink):
    def __init__(self, config: AppConfig):
        self.config = config
        self.targetFramerate = None
        self.last_frame_time = None
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0

    def initialize(self):
        cv2.namedWindow("video", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("video", *self.config.display_size)
        self.last_frame_time = time.time()
        self.start_time = self.last_frame_time
        return self

    def process_frame(self, frame):
        if self.targetFramerate:
            # Calculate target frame time
            target_frame_time = 1.0 / self.targetFramerate

            # Get time since last frame
            elapsed = time.time() - self.last_frame_time

            # If we're ahead of schedule, wait
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

        # Display frame
        cv2.imshow("video", frame)

        # Get current time after display
        now = time.time()

        # Update number of frames display since last FPS calculation
        self.frame_count += 1
        # Mark the time the frame was displayed
        self.last_frame_time = now

        # Calculate actual FPS every second
        timeSinceLastFpsCalc = now - self.start_time
        calcPeriodSec = 3.0
        if timeSinceLastFpsCalc >= calcPeriodSec:
            self.actual_fps = self.frame_count / (timeSinceLastFpsCalc)
            print(f"FPS {self.actual_fps:.0f} over last {calcPeriodSec:.0f} seconds. Target is {self.targetFramerate:.0f}" )
            self.frame_count = 0
            self.start_time = now


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