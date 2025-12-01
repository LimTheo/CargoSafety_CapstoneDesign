import time
import os
import cv2
import numpy as np
import subprocess # 외부 파일 실행용
from ultralytics import YOLO

# Picamera2 로드
try:
    from picamera2 import Picamera2
except ImportError:
    print("picamera2 라이브러리가 필요합니다.")
    exit()

# ==========================================
# [1] 설정
# ==========================================
CONFIG_FILE = "camera_config.npy"
MODEL_PATH = "yolov8n-pose.pt"

# 캘리브레이션 실제 거리 (1m~4m)
REAL_POINTS_BASE = np.array([
    [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
], dtype=np.float32)

# 현장 맞춤형 계수
ALPHA = 830.93
BETA = 1.09

# 2차 보정 계수
CORRECT_A, CORRECT_B, CORRECT_C = 0, 1, 0 

# ==========================================
# [모듈 0] 초기화 및 캘리브레이션 체크
# ==========================================
def check_calibration():
    if not os.path.exists(CONFIG_FILE):
        print("설정 파일이 없습니다. 캘리브레이션 모드를 실행합니다.")
        subprocess.run(["python", "calibration.py"])
        
        if not os.path.exists(CONFIG_FILE):
            print("캘리브레이션이 완료되지 않았습니다. 종료합니다.")
            exit()
    else:
        print("설정 파일 로드됨.")

def init_camera():
    print("메인 시스템 카메라 초기화 (Wide View)...")
    picam2 = Picamera2()
    # [수정 1] 해상도를 1280x960 (4:3)으로 높여서 넓은 시야(Full FOV) 확보
    config = picam2.create_preview_configuration(main={"size": (1640, 1232), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

# ==========================================
# [모듈 2] 계산 함수들
# ==========================================
def compute_homography(pixel_points):
    pixels = pixel_points.tolist()
    reals = REAL_POINTS_BASE.tolist()
    p1 = pixels[0]
    pixels.append([p1[0] + 100.0, p1[1]]) 
    reals.append([0.5, 1.0]) 
    
    src = np.array(pixels, dtype=np.float32)
    dst = np.array(reals, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H

def get_features(keypoints, box_h):
    kps = keypoints.data[0].cpu().numpy()
    l_ankle, r_ankle = kps[15], kps[16]
    l_knee, r_knee = kps[13], kps[14]
    l_sh, r_sh = kps[5], kps[6]
    l_hip, r_hip = kps[11], kps[12]

    foot_pt = None
    pose_type = "Unknown"
    if l_ankle[2] > 0.5 or r_ankle[2] > 0.5:
        pts = [p[:2] for p in [l_ankle, r_ankle] if p[2] > 0.5]
        foot_pt = np.mean(pts, axis=0)
        pose_type = "Real"
    elif l_knee[2] > 0.5 or r_knee[2] > 0.5:
        pts = [p[:2] for p in [l_knee, r_knee] if p[2] > 0.5]
        avg = np.mean(pts, axis=0)
        foot_pt = [avg[0], avg[1] + (box_h * 0.25)]
        pose_type = "Virtual"

    torso_len = None
    if l_sh[2] > 0.5 or r_sh[2] > 0.5 or l_hip[2] > 0.5 or r_hip[2] > 0.5:
        sh_ys = [p[1] for p in [l_sh, r_sh] if p[2] > 0.5]
        hip_ys = [p[1] for p in [l_hip, r_hip] if p[2] > 0.5]
        if sh_ys and hip_ys:
            torso_len = abs(np.mean(hip_ys) - np.mean(sh_ys))

    return foot_pt, torso_len, pose_type

def apply_correction(d):
    return max(0, (CORRECT_A * d**2) + (CORRECT_B * d) + CORRECT_C)

def calculate_ensemble_distance(foot_pt, torso_len, img_h, H):
    dist_stat = 0
    if torso_len and torso_len > 0:
        raw_stat = (ALPHA / torso_len) + BETA
        dist_stat = apply_correction(raw_stat)

    dist_homo = 0
    is_clipped = True
    
    if foot_pt is not None:
        if foot_pt[1] < (img_h * 0.95):
            is_clipped = False
            pt_px = np.array([[[foot_pt[0], foot_pt[1]]]], dtype=np.float32)
            pt_real = cv2.perspectiveTransform(pt_px, H)
            dist_homo = apply_correction(pt_real[0][0][1])
            real_x = pt_real[0][0][0]
        else:
            real_x = (foot_pt[0] - (640/2)) * dist_stat * 0.002

    final_dist = 0
    method = ""

    if is_clipped:
        final_dist = dist_stat
        method = "Stat"
    else:
        if dist_homo > 0 and dist_stat > 0:
            final_dist = (dist_homo + dist_stat) / 2
            method = "Mix"
        elif dist_homo > 0:
            final_dist = dist_homo
            method = "Homo"
        else:
            final_dist = dist_stat
            method = "Stat"

    return real_x, final_dist, method

def draw_separate_radar(objects, width=400, height=400, current_alert="Safe"):
    radar = np.zeros((height, width, 3), dtype=np.uint8)
    scale_z = height / 5.0
    cx = width // 2
    for i in range(1, 6):
        y = height - int(i * scale_z)
        col = (50, 50, 50)
        if i == 2: col = (0, 0, 150)
        cv2.line(radar, (0, y), (width, y), col, 1)
        cv2.putText(radar, f"{i}m", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150))
    cv2.circle(radar, (cx, height), 15, (255, 255, 255), -1)
    for (x, z, status) in objects:
        px = np.clip(int(cx + (x * (width / 4.0))), 0, width)
        py = np.clip(int(height - (z * scale_z)), 0, height)
        color = (0, 255, 0)
        if "DANGER" in status: color = (0, 0, 255)
        elif "WARNING" in status: color = (0, 165, 255)
        cv2.circle(radar, (px, py), 10, color, -1)
    if "DANGER" in current_alert:
        cv2.rectangle(radar, (0,0), (width,height), (0,0,255), 10)
        cv2.putText(radar, "STOP!", (cx-40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    elif "WARNING" in current_alert:
        cv2.rectangle(radar, (0,0), (width,height), (0,165,255), 5)
    return radar

# ==========================================
# [모듈 4] 메인 루프
# ==========================================
def main():
    check_calibration()
    
    pixel_points = np.load(CONFIG_FILE)
    H = compute_homography(pixel_points)
    
    picam2 = init_camera()
    model = YOLO(MODEL_PATH)
    print("\n시스템 가동! (종료: q, 설정초기화: r)")

    try:
        while True:
            # [수정 2] 고해상도(1280x960) 캡처 후 -> 640x480으로 리사이징
            raw_frame = picam2.capture_array()
            frame = cv2.resize(raw_frame, (640, 480))
            
            h, w = frame.shape[:2]

            results = model(frame, verbose=False, conf=0.5)
            detected_objects = []
            max_alert = "Safe"

            for result in results:
                if result.keypoints is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        box_h = y2 - y1
                        
                        foot_pt, torso_len, pose_type = get_features(result.keypoints[i], box_h)
                        real_x, dist, method = calculate_ensemble_distance(foot_pt, torso_len, h, H)
                        
                        if dist < 1.5:
                            status = "DANGER"
                            color = (0, 0, 255)
                            max_alert = status
                        elif dist < 2.5:
                            status = "WARNING"
                            color = (0, 165, 255)
                            if max_alert != "DANGER": max_alert = status
                        else:
                            status = "Safe"
                            color = (0, 255, 0)
                        
                        detected_objects.append((real_x, dist, status))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{status} {dist:.1f}m ({method})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            radar_img = draw_separate_radar(detected_objects, current_alert=max_alert)
            cv2.imshow('Main Camera (Wide View)', frame)
            cv2.imshow('Radar View', radar_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("설정 초기화...")
                picam2.stop()
                cv2.destroyAllWindows()
                os.remove(CONFIG_FILE)
                check_calibration() 
                pixel_points = np.load(CONFIG_FILE)
                H = compute_homography(pixel_points)
                picam2 = init_camera()

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()