from abc import ABC, abstractmethod


class VideoSink(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def process_frame(self, frame):
        pass

    @abstractmethod
    def release(self):
        pass
