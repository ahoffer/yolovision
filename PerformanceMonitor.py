import time
from collections import deque


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
