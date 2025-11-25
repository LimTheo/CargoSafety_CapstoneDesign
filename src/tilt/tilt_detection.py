import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from math import ceil

def draw_histogram_to_image(data_list, mean_val, std_val, height):
    """
    데이터 리스트를 받아 히스토그램을 그리고, 이를 OpenCV 이미지 형식으로 변환합니다.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(data_list, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvspan(mean_val - std_val, mean_val + std_val, color='green', alpha=0.1, label=f'Std: {std_val:.2f}')
    ax.set_title('Tilt Angle Distribution')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Count')
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
        return None, "Error: Invalid image input type", 0, 0
    if image is None:
        return None, "Error: Image not found or could not be loaded", 0, 0
    target_height = 800
    h, w = image.shape[:2]
    scale = target_height / h
    image = cv2.resize(image, (int(w * scale), target_height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=target_height / 10,
        maxLineGap=20
    )
    output_image = image.copy()
    angles = []
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if dy == 0 or abs(dx) > abs(dy):
                continue
            angle_rad = np.arctan(dx / dy)
            angle_deg = ceil(np.abs(np.degrees(angle_rad)))
            if angle_deg > 45:
                continue
            vertical_lines.append(line)
            angles.append(angle_deg)
    if not angles:
        avg_angle = 0.0
        std_dev_angle = 0.0
        status = "NORMAL (No lines)"
        color = (0, 255, 0)
    else:
        angles_np = np.array(angles)
        avg_angle = np.mean(angles_np)
        std_dev_angle = np.std(angles_np)
        is_tilted = avg_angle > mean_threshold
        is_unstable = std_dev_angle > std_threshold
        if is_tilted:
            status = "WARNING: TILTED"
            color = (0, 0, 255)
        elif is_unstable:
            status = "WARNING: UNSTABLE"
            color = (0, 165, 255)
        else:
            status = "NORMAL"
            color = (0, 255, 0)
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.rectangle(output_image, (0, 0), (450, 140), (255, 255, 255), -1)
    cv2.putText(output_image, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(output_image, f"Mean Angle: {avg_angle:.2f} (Th: {mean_threshold})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)
    cv2.putText(output_image, f"Std Dev: {std_dev_angle:.2f} (Th: {std_threshold})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)
    if angles:
        hist_img = draw_histogram_to_image(angles, avg_angle, std_dev_angle, target_height)
        final_result = np.hstack((output_image, hist_img))
    else:
        final_result = output_image
    return final_result, status, avg_angle, std_dev_angle