import cv2
import time
from picamera2 import Picamera2

def init_camera():
    """Picamera2 초기화"""
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
<<<<<<< HEAD
        main={"size": (1640, 1232), "format": "RGB888"}
=======
        main={"size": (720, 480), "format": "RGB888"}
>>>>>>> 5c679181eb29a809374a90eaf64f4ab305708726
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.4)
    print("Camera initialized.")
    return picam2


def get_frame(picam2):
    """현재 프레임 반환 (BGR)"""
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame