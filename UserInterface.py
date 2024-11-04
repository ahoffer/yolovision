import cv2


class UserInterface:
    def __init__(self, display_window):
        self.display = display_window

    def show_frame(self, frame):
        self.display.show_frame(frame)

    def check_quit(self, wait_time):
        key = cv2.waitKey(wait_time) & 0xFF
        # Return True if 'q' is pressed OR window is closed
        return key == ord('q') or self.display.is_window_closed()

    def cleanup(self):
        self.display.close()
        cv2.destroyAllWindows()
