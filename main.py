import src.common
import src.tilt.tilt_detection as td
import threading
import time
from picamera2 import Picamera2, Preview
# 필요한 경우 BMI160 센서 모듈 임포트
# [1단계]에서 저장한 파일 이름으로 수정하세요 (예: `motion_detector.py`를 `md`로 임포트)
import motion_detector as md # <--- 여기에 BMI160 감지 파일 임포트


# YOLOE 및 기능 모듈 임포트
from src.models.yoloe_loader import load_yoloe_model
from src.common.camera_input import init_camera, get_frame
from src.detection.object_detection import run_inference
from src.tilt.tilt_detection import analyze_tilt_fast
from src.common.visualization import draw_box, draw_label, show_frame

# 공유 자원 및 조건 변수 생성
current_state = "STOPPED" # 상태 저장 변수
condition = threading.Condition() # Condition 객체 생성
model = load_yoloe_model()

def car_moved_task():
    """차가 움직일 때 실행되는 태스크"""
    while True:
        with condition:
            # 상태가 'MOVING'이 아니면 스레드는 여기서 멈춤 (CPU 사용 X)
            # 상태가 바뀌고 notify가 오면 깨어나서 조건을 다시 확인
            condition.wait_for(lambda: current_state == "MOVING")
        
        # --- [실제 작업 영역] ---
        print("car moved: monitoring...")




def car_stopped_task(picam2):
    frame_count = 0
    """차가 멈췄을 때 실행되는 태스크"""
    while True:
        with condition:
            # 상태가 'STOPPED'가 아니면 여기서 대기 (CPU 사용 X)
            condition.wait_for(lambda: current_state == "STOPPED")

        # --- [실제 작업 영역] ---
        print("car stopped: detecting tilt...")

        frame = get_frame(picam2)
        frame_count += 1
        result = run_inference(model, frame, frame_count)

        if result:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                status, color, angle = analyze_tilt_fast(crop)
                label = f"{cls} | {status} {angle:.1f}°"

                draw_box(frame, x1, y1, x2, y2, color)
                draw_label(frame, label, x1, max(10, y1 - 10), color)

        key = show_frame(frame)


# YOLOE + Tilt 분석
def run_yolo_tilt():
    model = load_yoloe_model()
    picam2 = init_camera()

    print("Starting YOLOE + Tilt Analyzer... Press 'q' to exit.")

    frame_count = 0
    last_result = None

    while True:
        frame = get_frame(picam2)
        frame_count += 1

        result = run_inference(model, frame, frame_count)

        if result:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                status, color, angle = analyze_tilt_fast(crop)
                label = f"{cls} | {status} {angle:.1f}°"

                draw_box(frame, x1, y1, x2, y2, color)
                draw_label(frame, label, x1, max(10, y1 - 10), color)

        key = show_frame(frame)

        if key == ord("q"):
            break

    print("Exiting.")


if __name__ == "__main__":
    # 센서 초기화는 메인 스레드에서 한 번만 수행
    try:
        md.initialize_bmi160()
    except Exception as e:
        print(f"센서 초기화에 실패하여 ZUPT 기능 없이 시작합니다: {e}")
        # 센서 초기화에 실패하면 is_moving을 항상 True로 설정하여 동작 감지 로직을 우회할 수 있습니다.


    picam2 = init_camera()

    # 👇 Preview 강제 적용
    # try:
    #     picam2.start_preview(Preview.OPENCV)
    # except Exception:
    #     print("OpenCV preview unavailable, switching to Null")
    #     picam2.start_preview(Preview.NULL)

    #picam2.start_preview(Preview.NULL)

    picam2.start()

    # 스레드 생성 및 시작
    t1 = threading.Thread(target=car_moved_task, daemon=True)
    t2 = threading.Thread(target=car_stopped_task,args=(picam2,), daemon=True)
    
    t1.start()
    t2.start()
    
    last_result = None

    last_state = None # 상태 변경 감지용

    while True:
        try:
            car_moving = md.check_motion_state()
            #car_moving = state_received() 
            
        except NameError:
            # 센서 오류 등으로 함수 호출에 실패하면 '움직임' 상태로 간주 (안전 모드)
            car_moving = True

        # 상태 결정
        new_state = "MOVING" if car_moving else "STOPPED"

        # *** 상태가 바뀌었을 때만 스레드들에게 알림 ***
        if new_state != last_state:
            with condition:
                current_state = new_state
                print(f"\n--- State changed to: {current_state} ---\n")
                condition.notify_all() # 대기 중인 모든 스레드를 깨움
            last_state = new_state
        
        # 메인 루프의 과도한 CPU 점유를 막기 위한 최소한의 sleep
        time.sleep(0.1)
        # (이건 스레드 제어용이 아니라 상태 체크 주기용입니다)