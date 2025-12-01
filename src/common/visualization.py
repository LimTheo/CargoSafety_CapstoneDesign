import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_image(image, window_name='Image', wait=True):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)

def preview_dataset(dataset):
    for item in dataset:
        print(f"Label: {item['label']}")
        plt.imshow(item["image"])
        plt.title(item["label"])
        plt.axis('off')
        plt.show()

def draw_box(frame, x1, y1, x2, y2, color):
    """Bounding Box 그리기"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_label(frame, text, x, y, color):
    """라벨 텍스트 그리기"""
    cv2.putText(
        frame, text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        color, 2
    )


def show_frame(frame):
    """320x240으로 축소하여 표시"""
    display_frame = cv2.resize(frame, (320, 240))
    cv2.imshow("YOLOE + Fast Tilt Analyzer", display_frame)
    return cv2.waitKey(1) & 0xFF