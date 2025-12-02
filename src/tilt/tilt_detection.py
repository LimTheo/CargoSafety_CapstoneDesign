import cv2
import numpy as np
from PIL import Image
from math import ceil

def detect_pallet_tilt(image_input, mean_threshold=3.0, std_threshold=2.0):
    """
    실시간용 빠른 기울기 계산 함수 (그래프 없음).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(data_list, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvspan(mean_val - std_val, mean_val + std_val, color='green', alpha=0.1, label=f'Std: {std_val:.2f}')
    ax.set_title('Tilt Angle Distribution')
    ax.set_xlabel('Angle (deg)')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    h_plot, w_plot = plot_img.shape[:2]
    scale = height / h_plot
    plot_img_resized = cv2.resize(plot_img, (int(w_plot * scale), height))
    return plot_img_resized

# 기울기 분석 함수
def analyze_tilt_fast(roi_img, tilt_threshold=10):
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return "NORMAL", (0, 255, 0), 0.0

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = abs(rect[-1])

    # angle normalization
    if angle > 45:
        angle = 90 - angle

    if angle > tilt_threshold:
        return "TILTED", (0, 0, 255), angle

    return "NORMAL", (0, 255, 0), angle

def analyze_tilt_hough(roi_img, tilt_threshold=3.0, std_threshold=2.0):
    """
    기존의 Hough Line 변환 방식을 사용하여 기울기를 정밀하게 분석합니다.
    최신 코드 포맷에 맞춰 (status, color, angle) 3개의 값을 반환합니다.
    """
    
    # 1. 입력 예외 처리 (반환값 3개 유지)
    if isinstance(roi_img, str):
        image = cv2.imread(roi_img)
    elif isinstance(roi_img, Image.Image):
        image = np.array(roi_img)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif isinstance(roi_img, np.ndarray):
        image = roi_img
    else:
        return "Error: Invalid Input", (0, 0, 0), 0.0

    if image is None:
        return "Error: Image None", (0, 0, 0), 0.0

    # 2. 전처리 (Resize -> Canny)
    target_height = 800
    h, w = image.shape[:2]
    
    # 이미지가 너무 작거나 비어있는 경우 방지
    if h == 0 or w == 0:
        return "Error: Empty Frame", (0, 0, 0), 0.0

    scale = target_height / h
    image_resized = cv2.resize(image, (int(w * scale), target_height))
    
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 3. 선 검출 (HoughLinesP)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=target_height / 10,
        maxLineGap=20
    )

    angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            
            # 수직에 가까운 선만 추출 (가로선 무시)
            if dy == 0 or abs(dx) > abs(dy):
                continue
                
            angle_rad = np.arctan(dx / dy)
            angle_deg = np.degrees(angle_rad) # 절대값 처리 전 각도
            
            # 각도 절대값 (기울기 정도)
            abs_angle = abs(angle_deg)
            
            if abs_angle > 45: # 45도 이상은 노이즈로 간주
                continue
                
            angles.append(abs_angle)

    # 4. 결과 분석 및 반환 (항상 3개 값 반환)
    if not angles:
        # 선이 검출되지 않음 -> 정상으로 간주하거나 별도 처리
        return "NORMAL (No lines)", (0, 255, 0), 0.0

    avg_angle = np.mean(angles)
    std_dev_angle = np.std(angles)

    # 논리 판단
    is_tilted = avg_angle > tilt_threshold
    is_unstable = std_dev_angle > std_threshold

    if is_tilted:
        # 기울어짐 (빨강)
        return "WARNING: TILTED", (0, 0, 255), avg_angle
    elif is_unstable:
        # 흔들림/불안정 (주황)
        return "WARNING: UNSTABLE", (0, 165, 255), avg_angle
    else:
        # 정상 (초록)
        return "NORMAL", (0, 255, 0), avg_angle