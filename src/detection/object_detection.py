import cv2
import numpy as np
from PIL import Image

from src.models.yoloe_loader import load_yoloe_model, names

CONF_THRESHOLD = 0.25
SKIP_FRAMES = 10

def run_inference(model, frame, frame_count):
    should_infer = 1#(frame_count % SKIP_FRAMES == 0)

    if not should_infer:
        return None

    result = model.predict(
        frame,
        imgsz=256,
        verbose=False,
        conf=CONF_THRESHOLD,
        task="detect"
    )[0]

    return result

def detect_and_crop(frame, conf=0.1, iou=0.5, imgsz=640, area_threshold=10000, padding=20):
    model = load_yoloe_model()

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model.predict(img, conf=conf, iou=iou, imgsz=imgsz)
    detections = results[0]

    # 면적 필터링
    filtered_boxes = []
    for det in detections.boxes:
        xyxy = det.xyxy[0].cpu().numpy()
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        if area < area_threshold:
            filtered_boxes.append(xyxy)

    # 크롭 이미지 추출
    img_width, img_height = img.size
    cropped_images = []
    for det in detections.boxes:
        x1_orig, y1_orig, x2_orig, y2_orig = det.xyxy[0].cpu().numpy().astype(int)

        x1 = max(0, x1_orig - padding)
        y1 = max(0, y1_orig - padding)
        x2 = min(img_width, x2_orig + padding)
        y2 = min(img_height, y2_orig + padding)

        crop = img.crop((x1, y1, x2, y2))
        cropped_images.append(crop)

    # Dataset 구성
    dataset = []
    for det, crop in zip(detections.boxes, cropped_images):
        class_id = int(det.cls.cpu().numpy()[0]) if det.cls.numel() > 0 else 0
        class_name = names[class_id]
        dataset.append({"image": crop, "label": class_name})

    return {
        "boxes": filtered_boxes,
        "dataset": dataset,
        "annotated_image": detections.plot()[..., ::-1]
    }
