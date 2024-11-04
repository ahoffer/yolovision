class AppConfig:
    def __init__(self, model_path, input_video, output_video, confidence_interval=1, display_size=(1920, 1080),
                 loop_video=False):
        self.model_path = model_path
        self.input_video = input_video
        self.output_video = output_video
        self.confidence_interval = confidence_interval
        self.display_size = display_size
        self.loop_video = loop_video
