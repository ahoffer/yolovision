import cv2


class UserInterface:
    def __init__(self, display_window):
        self.display = display_window

    def show_frame(self, frame):
        self.display.show_frame(frame)

    def check_quit(self, wait_time):
        return cv2.waitKey(wait_time) & 0xFF == ord('q')

    def cleanup(self):
        self.display.close()
        cv2.destroyAllWindows()
