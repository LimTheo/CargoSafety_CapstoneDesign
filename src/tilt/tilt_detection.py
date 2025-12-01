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

def detect_pallet_tilt_with_graph(image_input, mean_threshold=3.0, std_threshold=2.0):
    """
    이미지 내 수직선을 검출하고, 각도 통계(평균, 표준편차)와 히스토그램을 시각화합니다.
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = image_input.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=20
    )

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)

            if dy == 0 or abs(dx) > abs(dy):
                continue

            angle_rad = np.arctan(dx / dy)
            angle_deg = abs(np.degrees(angle_rad))

            if angle_deg < 45:
                angles.append(angle_deg)

    if len(angles) == 0:
        return "NORMAL (no lines)", 0.0, 0.0

    mean_angle = np.mean(angles)
    std_angle = np.std(angles)

    if mean_angle > mean_threshold:
        status = "WARNING: TILTED"
    elif std_angle > std_threshold:
        status = "WARNING: UNSTABLE"
    else:
        status = "NORMAL"

    return status, mean_angle, std_angle