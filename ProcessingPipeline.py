class ProcessingPipeline:
    def __init__(self, source, sink, detector, performance_monitor, frame_rate_controller):
        self.source = source
        self.sink = sink
        self.detector = detector
        self.performance_monitor = performance_monitor
        self.frame_rate_controller = frame_rate_controller

    def process_frame(self):
        # Start timing this frame
        self.frame_rate_controller.start_frame()

        # Read and process frame
        ret, frame = self.source.read_frame()
        if not ret:
            return False, None

        processed_frame = self.detector.detect(frame)
        self.sink.write_frame(processed_frame)

        # Wait for appropriate frame time
        self.frame_rate_controller.wait_for_next_frame()

        return True, processed_frame
