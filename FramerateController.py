import time


class FramerateController:
    def __init__(self, target_fps=None, calc_period_sec=3.0):
        self.target_fps = target_fps
        self.calc_period_sec = calc_period_sec
        self.last_frame_time = time.time()
        self.start_time = self.last_frame_time
        self.frame_count = 0
        self.actual_fps = 0

    def wait_for_next_frame(self):
        if self.target_fps:
            target_frame_time = 1.0 / self.target_fps
            elapsed = time.time() - self.last_frame_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

    def update_stats(self):
        now = time.time()
        self.frame_count += 1
        self.last_frame_time = now
        time_since_last_calc = now - self.start_time
        if time_since_last_calc >= self.calc_period_sec:
            self.actual_fps = self.frame_count / time_since_last_calc
            self.frame_count = 0
            self.start_time = now