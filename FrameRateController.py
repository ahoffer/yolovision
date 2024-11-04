import time


class FrameRateController:
    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.last_frame_time = None

    def start_frame(self):
        self.last_frame_time = time.time()

    def wait_for_next_frame(self):
        """Wait until it's time for the next frame"""
        if self.last_frame_time is None:
            return

        elapsed = time.time() - self.last_frame_time
        sleep_time = max(0, self.target_frame_time - elapsed)

        if sleep_time > 0:
            time.sleep(sleep_time)
