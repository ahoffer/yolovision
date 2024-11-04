from AppConfig import AppConfig
from DisplayWindow import DisplayWindow
from FileVideoSink import FileVideoSink
from FileVideoSource import FileVideoSource
from FrameRateController import FrameRateController
from PerformanceMonitor import PerformanceMonitor
from ProcessingPipeline import ProcessingPipeline
from UserInterface import UserInterface
from YOLODetector import YOLODetector


class DetectionApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.source = FileVideoSource(config)
        self.ui = UserInterface(DisplayWindow(config))
        self.performance_monitor = PerformanceMonitor()
        self.detector = None
        self.sink = None
        self.pipeline = None
        self.frame_rate_controller = None

    def initialize(self):
        # Initialize video source
        self.source.initialize()
        metadata = self.source.get_metadata()

        # Initialize frame rate controller
        self.frame_rate_controller = FrameRateController(metadata['fps'])

        # Initialize detector
        self.detector = YOLODetector(self.config)
        self.detector.initialize()

        # Initialize output and display
        self.sink = FileVideoSink(self.config.output_video, metadata['fps'], metadata['width'], metadata['height'])
        self.sink.initialize()
        self.ui.display.initialize()

        # Create pipeline
        self.pipeline = ProcessingPipeline(self.source, self.sink, self.detector, self.performance_monitor,
                                           self.frame_rate_controller)

        # Print configuration
        print(f"Input video: {self.config.input_video}")
        print(f"Output video: {self.config.output_video}")
        print(f"FPS: {metadata['fps']}")
        print(f"Resolution: {metadata['width']}x{metadata['height']}")
        print(f"Loop mode: {'enabled' if self.config.loop_video else 'disabled'}")

    def run(self):
        try:
            print("\nProcessing video...")
            print("Press 'q' to quit")

            while True:
                frame_start = self.performance_monitor.start_frame()

                # Process frame
                success, frame = self.pipeline.process_frame()

                # Handle end of video
                if not success:
                    if self.config.loop_video:
                        self.source.reset()
                        self.performance_monitor.increment_loop()
                        print(f"\nStarting loop {self.performance_monitor.loop_count}")
                        continue
                    break

                # Display frame
                self.ui.show_frame(frame)

                # Update performance stats
                self.performance_monitor.end_frame(frame_start)

                # Show stats periodically
                if self.performance_monitor.frame_count % 30 == 0:
                    stats = self.performance_monitor.get_stats()
                    if stats:
                        print(f"\rLoop: {stats['loop_count']} | "
                              f"FPS: {stats['avg_fps']:.1f}", end="")

                # Check for user quit - use a small wait time to not interfere with frame timing
                if self.ui.check_quit(1):
                    print("\nProcessing interrupted by user")
                    break

        except Exception as e:
            print(f"\nError during video processing: {e}")
            raise

        finally:
            self.cleanup()

            # Print final statistics
            stats = self.performance_monitor.get_stats()
            if stats:
                print(f"\n\nProcessing Statistics:")
                print(f"Total frames processed: {stats['frame_count']}")
                print(f"Number of complete loops: {stats['loop_count']}")
                print(f"Average processing time per frame: {stats['avg_frame_time']:.1f}ms")
                print(f"Target frame time: {1000.0 / self.source.get_metadata()['fps']:.1f}ms")
                print(f"Average FPS: {stats['avg_fps']:.1f}")
                print(f"Target FPS: {self.source.get_metadata()['fps']}")

    def cleanup(self):
        self.source.release()
        self.sink.release()
        self.ui.cleanup()
