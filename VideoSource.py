from abc import ABC, abstractmethod


class VideoSource(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def read_frame(self):
        pass

    @abstractmethod
    def release(self):
        pass
