import time
from typing import Tuple, Optional

import numpy as np

from VideoSink import VideoSink


class ProcessingPipeline(VideoSink):
    def initialize(self):
        pass

    def __init__(self):
        self.source = None
        self.detector = None
        self.sinks = []
        self._reset_stats()

    def add_source(self, source: VideoSink):
        source.initialize()
        self.source = source
        return self

    def add_detector(self, detector):
        detector.initialize()
        self.detector = detector
        return self

    def process_frame(self, frame: np.ndarray) -> None:
        """Process a frame and write to downstream sinks"""
        try:
            processed_frame = self.detector.detect(frame)
            self._write_to_sinks(processed_frame)
            self.total_frames += 1
        except Exception as e:
            self.dropped_frames += 1

    def release(self) -> None:
        """Release pipeline resources and downstream sinks"""
        for sink in self.sinks:
            try:
                sink.release()
            except Exception as e:
                raise
        self.sinks.clear()

    def add_sink(self, sink: VideoSink):
        sink.initialize()
        self.sinks.append(sink)
        return self

    def remove_sink(self, sink: VideoSink) -> bool:
        try:
            sink.release()
            self.sinks.remove(sink)
            return True
        except Exception as e:
            print(f"Error releasing sink: {str(e)}")
            return False

    def _reset_stats(self) -> None:
        self.total_frames = 0
        self.dropped_frames = 0
        self.start_time = time.time()

    def _write_to_sinks(self, frame: np.ndarray) -> None:
        for sink in self.sinks:
            try:
                sink.process_frame(frame)
            except Exception as e:
                raise

    def process_source_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Process a frame from the source"""
        try:
            if not self.sinks:
                return False, None

            ret, frame = self.source.read_frame()
            if not ret:
                return False, None

            self.process_frame(frame)
            return True, frame

        except Exception as e:
            print(f"Error processing source frame: {str(e)}")
            return False, None
