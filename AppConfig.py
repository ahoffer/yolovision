class AppConfig:
    def __init__(self, model_path, input_video, output_video,
                 display_size=(1920, 1080),
                 loop_video=False, confidence_threshold=0.3, batch_size=1, detector='cpu'):
        self.detector = detector
        self.model_path = model_path
        self.input_video = input_video
        self.output_video = output_video
        self.display_size = display_size
        self.loop_video = loop_video
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
