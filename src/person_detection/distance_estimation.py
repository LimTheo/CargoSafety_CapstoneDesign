import cv2
import numpy as np
import os

# 설정 파일 경로
CONFIG_FILE = "camera_config.npy"

# 캘리브레이션 상수
REAL_POINTS_BASE = np.array([
    [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
], dtype=np.float32)

ALPHA = 263.06
BETA = -3.90

CORRECT_A = -0.004430
CORRECT_B = 1.022200
CORRECT_C = -0.023389
REALITY_SCALE = 1.0

def load_calibration_data():
    """설정 파일 로드 및 호모그래피 행렬 계산"""
    if not os.path.exists(CONFIG_FILE):
        print(f"[Warning] {CONFIG_FILE} 파일을 찾을 수 없습니다. 거리 계산이 제한됩니다.")
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
    
    # 인덱스: [0:L_Sh, 1:R_Sh, 2:L_Hip, 3:R_Hip, 4:L_Knee, 5:R_Knee, 6:L_Ankle, 7:R_Ankle]
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
    return max(0, (CORRECT_A * d**2) + (CORRECT_B * d) + CORRECT_C)

def calculate_ensemble_distance(foot_pt, torso_len, img_h, H):
    dist_stat = 0
    if torso_len and torso_len > 0:
        raw_stat = (ALPHA / torso_len) + BETA
        dist_stat = apply_correction(raw_stat)

    dist_homo = 0
    is_clipped = True
    real_x = 0
    
    if H is not None and foot_pt is not None:
        if foot_pt[1] < (img_h * 0.95):
            is_clipped = False
            pt_px = np.array([[[foot_pt[0], foot_pt[1]]]], dtype=np.float32)
            pt_real = cv2.perspectiveTransform(pt_px, H)
            dist_homo = apply_correction(pt_real[0][0][1])
            real_x = pt_real[0][0][0]
        else:
            real_x = (foot_pt[0] - 320) * dist_stat * 0.002

    final_dist = 0
    method = ""

    if is_clipped or H is None:
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

    return real_x, final_dist * REALITY_SCALE, method

def get_status_info(dist):
    if dist < 1.5:
        return "DANGER", (0, 0, 255)   # Red
    elif dist < 2.5:
        return "WARNING", (0, 165, 255) # Orange
    else:
        return "Safe", (0, 255, 0)      # Green

def process_distance_estimation(model, frame, H):
    """
    메인에서 호출하는 함수: 프레임을 받아 거리 계산 후 시각화된 프레임 반환
    """
    h, w = frame.shape[:2]
    # 모델 추론
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
                    
                    # 박스 및 텍스트 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{status} {dist:.1f}m"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if 0 <= foot_pt[0] < w and 0 <= foot_pt[1] < h:
                        cv2.circle(frame, (int(foot_pt[0]), int(foot_pt[1])), 5, (0, 255, 255), -1)

    return frame, detected_objects