import cv2


class DisplayWindow:
    def __init__(self, window_name="Detection", window_size=(1280, 720)):
        self.window_name = window_name
        self.window_size = window_size

    def initialize(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        return self

    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)

    def is_window_closed(self):
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except:
            return True

    def close(self):
        cv2.destroyWindow(self.window_name)
