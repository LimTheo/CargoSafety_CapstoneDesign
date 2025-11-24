import cv2
from src.detection.object_detection import detect_cargo
from src.tilt.tilt_calculation import calculate_tilt
from src.overload.overload_measurement import measure_overload
from src.blind_spot.blind_spot_detection import detect_blind_spot
from src.common.camera_input import get_camera_frame
from src.common.visualization import show_results  # 시각화 함수 (별도 구현)

for frame in get_camera_frame():
    # 1. 검출
    detection_result = detect_cargo(frame)
    boxes = detection_result["boxes"]
    dataset = detection_result["dataset"]
    
    # 2. 기울기
    tilts = calculate_tilt(boxes, frame)
    
    # 3. 과적
    overload_result = measure_overload(dataset)
    
    # 결과 처리/시각화
    print(f"Tilts: {tilts}, Overload: {overload_result}, Blind Spot: {blind_spot_result}")
    show_results(frame, detection_result)  # e.g., cv2.imshow
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cv2.destroyAllWindows()