import torch


class DetectionConfig:
    def __init__(self, model_path, device=None, confidence_interval=1):
        self.model_path = model_path
        self.device = device or self._get_optimal_device()
        self.confidence_interval = confidence_interval

    def _get_optimal_device(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.current_device()
                return 'cuda'
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                return 'cpu'
        return 'cpu'
