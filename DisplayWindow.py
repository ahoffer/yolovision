
import cv2

from AppConfig import AppConfig


class DisplayWindow:
    def __init__(self, config: AppConfig):
        self.config = config

    def initialize(self):
        cv2.namedWindow("video", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("video", *self.config.display_size)
        cv2.setMouseCallback("video", self.mouse_callback)
        return self

    def show_frame(self, frame):
        cv2.imshow("video", frame)

    def is_window_closed(self):
        try:
            return cv2.getWindowProperty("video", cv2.WND_PROP_VISIBLE) < 1
        except:
            return True

    def close(self):
        cv2.destroyWindow("video")

    def mouse_callback(self, event, x, y, flags, param):
        # Implement actual logic for mouse events if needed
        if event == cv2.EVENT_LBUTTONDOWN:
            None
            # print(f"Mouse clicked at ({x}, {y})")