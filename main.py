import cv2
import os
from huggingface_hub import hf_hub_download
from src.detection.object_detection import detect_and_crop
from src.detection.masking import mask_background
from src.tilt.tilt_detection import detect_pallet_tilt_with_graph
from src.common.camera_input import get_camera_stream
from src.common.visualization import show_image, preview_dataset

# 데이터 다운로드 (처음 실행 시)
if not os.path.exists('data/yoloe-v8l-seg.pt'):
    hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg.pt", local_dir='data')
    hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg-pf.pt", local_dir='data')
if not os.path.exists('data/ram_tag_list.txt'):
    os.system('wget https://raw.githubusercontent.com/THU-MIG/yoloe/main/tools/ram_tag_list.txt -P data')
# 다른 wget 파일도 유사하게 (bus.jpg 등, 테스트용)

for frame in get_camera_stream():
    # 1. 화물 검출 & 크롭
    detection_result = detect_and_crop(frame)
    boxes = detection_result["boxes"]
    dataset = detection_result["dataset"]
    annotated_img = Image.fromarray(detection_result["annotated_image"])

    # 2. 마스킹 (선택적)
    masked_img = mask_background(annotated_img, boxes)

    # 3. 각 크롭에 기울기 계산
    for item in dataset:
        print(f"Analyzing cropped image for: {item['label']}")
        result_img, status, mean, std = detect_pallet_tilt_with_graph(item['image'])
        if result_img is not None:
            print(f"[{status}] Mean: {mean:.2f}, Std: {std:.2f}")
            show_image(result_img, f"Tilt Result for {item['label']}", wait=False)

    # 결과 표시
    show_image(annotated_img, 'Detected Boxes', wait=False)
    show_image(masked_img, 'Masked Image', wait=False)
    # preview_dataset(dataset)  # 필요시 (matplotlib, 실시간에 느림)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cv2.destroyAllWindows()