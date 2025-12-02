import cv2
import numpy as np
import os

# ==========================================
# [설정] 새로 구한 파라미터 적용
# ==========================================
CONFIG_FILE = "/home/devjang/Desktop/CargoSafety_CapstoneDesign-Integration_Test/src/person_detection/camera_config.npy"

REAL_POINTS_BASE = np.array([
    [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
], dtype=np.float32)

# [사용자 업데이트 값]
ALPHA = 268.82
BETA  = -3.90

CORRECT_A = 0.030258
CORRECT_B = 0.853139
CORRECT_C = 0.151990
REALITY_SCALE = 1.0

def load_calibration_data():
    """설정 파일 로드 및 호모그래피 행렬 계산"""
    if not os.path.exists(CONFIG_FILE):
        print(f"[Warning] {CONFIG_FILE} 파일이 없습니다.")
        return None

    try:
        pixel_points = np.load(CONFIG_FILE)
        
        pixels = pixel_points.tolist()
        reals = REAL_POINTS_BASE.tolist()
        p1 = pixels[0]
        
        pixels.append([p1[0] + 100.0, p1[1]])
        reals.append([0.5, 1.0])
        
        src = np.array(pixels, dtype=np.float32)
        dst = np.array(reals, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        
        print("Calibration data loaded & Homography computed.")
        return H
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None

def get_features(keypoints, box_h):
    kps = keypoints.data[0].cpu().numpy()
    
    l_sh, r_sh = kps[0], kps[1]
    l_hip, r_hip = kps[2], kps[3]
    l_knee, r_knee = kps[4], kps[5]
    l_ankle, r_ankle = kps[6], kps[7]

    foot_pt = None
    if l_ankle[2] > 0.5 or r_ankle[2] > 0.5:
        pts = [p[:2] for p in [l_ankle, r_ankle] if p[2] > 0.5]
        foot_pt = np.mean(pts, axis=0)
    elif l_knee[2] > 0.5 or r_knee[2] > 0.5:
        pts = [p[:2] for p in [l_knee, r_knee] if p[2] > 0.5]
        avg = np.mean(pts, axis=0)
        foot_pt = [avg[0], avg[1] + (box_h * 0.25)]

    torso_len = None
    if l_sh[2] > 0.5 or r_sh[2] > 0.5 or l_hip[2] > 0.5 or r_hip[2] > 0.5:
        sh_ys = [p[1] for p in [l_sh, r_sh] if p[2] > 0.5]
        hip_ys = [p[1] for p in [l_hip, r_hip] if p[2] > 0.5]
        if sh_ys and hip_ys:
            torso_len = abs(np.mean(hip_ys) - np.mean(sh_ys))

    return foot_pt, torso_len

def apply_correction(d):
    """
    [수정됨] 음수 값이라도 그대로 반환합니다.
    (max(0, val) 제거)
    """
    val = (CORRECT_A * d**2) + (CORRECT_B * d) + CORRECT_C
    return val

def calculate_ensemble_distance(foot_pt, torso_len, img_h, H):
    dist_stat = 0
    stat_valid = False
    
    # 1. 통계적 거리 계산 (음수 허용)
    if torso_len and torso_len > 0:
        raw_stat = (ALPHA / torso_len) + BETA
        dist_stat = apply_correction(raw_stat)
        stat_valid = True

    dist_homo = 0
    homo_valid = False
    is_clipped = True
    real_x = 0
    
    # 2. 호모그래피 거리 계산
    if foot_pt is not None:
        if foot_pt[1] < (img_h * 0.95):
            is_clipped = False
            if H is not None:
                pt_px = np.array([[[foot_pt[0], foot_pt[1]]]], dtype=np.float32)
                pt_real = cv2.perspectiveTransform(pt_px, H)
                dist_homo = apply_correction(pt_real[0][0][1])
                real_x = pt_real[0][0][0]
                homo_valid = True
        else:
            # 발이 잘렸을 때 X좌표 추정
            real_x = (foot_pt[0] - 320) * dist_stat * 0.002

    final_dist = 0
    method = ""

    # [수정됨] 앙상블 로직: 값이 존재하기만 하면 계산 (음수 포함)
    if is_clipped or (not homo_valid):
        final_dist = dist_stat
        method = "Stat"
    else:
        if stat_valid and homo_valid:
            final_dist = (dist_homo + dist_stat) / 2
            method = "Mix"
        elif homo_valid:
            final_dist = dist_homo
            method = "Homo"
        else:
            final_dist = dist_stat
            method = "Stat"

    return real_x, final_dist * REALITY_SCALE, method

def get_status_info(dist):
    """
    거리별 상태 반환
    [수정] 음수 값이 나오면 'DANGER' (매우 가까움)로 처리
    """
    if dist < 1.5:  # 1.5m 미만 (음수 포함)은 모두 위험
        return "DANGER", (0, 0, 255)   # Red
    elif dist < 2.5:
        return "WARNING", (0, 165, 255) # Orange
    else:
        return "Safe", (0, 255, 0)      # Green

def process_distance_estimation(model, frame, H):
    # [중요] 640x480 리사이즈 유지
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    
    results = model(frame, verbose=False, conf=0.5)
    
    detected_objects = []

    for result in results:
        if result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                box_h = y2 - y1
                
                foot_pt, torso_len = get_features(result.keypoints[i], box_h)
                
                if foot_pt is not None:
                    real_x, dist, method = calculate_ensemble_distance(foot_pt, torso_len, h, H)
                    status, color = get_status_info(dist)
                    
                    detected_objects.append((real_x, dist, status))
                    
                    # 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 텍스트 표시 (음수도 그대로 표시됨, 예: -0.5m)
                    label = f"{status} {dist:.1f}m ({method})"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if 0 <= foot_pt[0] < w and 0 <= foot_pt[1] < h:
                        cv2.circle(frame, (int(foot_pt[0]), int(foot_pt[1])), 5, (0, 255, 255), -1)

    return frame, detected_objects