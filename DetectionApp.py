from AppConfig import AppConfig
from DisplaySink import DisplaySink
from FileVideoSource import FileVideoSource
from ProcessingPipeline import ProcessingPipeline
from UserInterface import UserInterface
from YOLODetector import YOLODetector


class DetectionApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ui = None
        self.pipeline = None
        self.source = FileVideoSource(config)

    def initialize(self):
        self.source.initialize()
        metadata = self.source.get_metadata()
        displaySink = DisplaySink(self.config)
        displaySink.targetFramerate = metadata['fps']
        displaySink.windowSize=metadata['width'], metadata['height']
        self.ui = UserInterface(displaySink)

        # Create pipeline
        self.pipeline = ProcessingPipeline().add_source(self.source).add_detector(YOLODetector(self.config)).add_sink(
            displaySink)  # (FileVideoSink(self.config, metadata['fps'], metadata['width'], metadata['height']))

    def run(self):
        try:
            while True:
                success, frame = self.source.read_frame()
                if success:
                    self.pipeline.process_frame(frame)
                else:
                    # Handle end of video
                    if self.config.loop_video:
                        self.source.reset()
                        continue
                    break

                # Check for user quit - use a small wait time to not interfere with frame timing
                if self.ui.check_quit(1):
                    print("\nProcessing interrupted by user")
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        self.ui.cleanup()
