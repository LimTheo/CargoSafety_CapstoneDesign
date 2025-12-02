import cv2
import numpy as np

def unwarp_tilt_view(frame):
    h, w = frame.shape[:2]

    # [설정 필요] 1. 영상에서 '수직'이어야 하는데 기울어진 영역의 4점
    # 예: 앞에 있는 선반(Rack)이나 기둥의 네 모서리를 잡으세요.
    # 보통 아래쪽이 좁고 위쪽이 넓은 사다리꼴 형태일 것입니다.
    # 순서: [왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래]
    pts1 = np.float32([[200, 100], [440, 100], [400, 400], [240, 400]])

    # [설정 필요] 2. 변환 후 이 점들이 위치할 직사각형 좌표
    # 원본보다 x좌표 간격을 넓혀서 11자로 만듭니다.
    # 높이(y)는 유지하고, 너비(x)를 위쪽 너비에 맞추거나 원하는 대로 조정
    width_target = 440 - 200  # 위쪽 너비 기준
    
    # 중앙 정렬을 위해 약간의 오프셋을 줄 수도 있습니다.
    pts2 = np.float32([
        [200, 100],               # 왼쪽 위 (고정)
        [200 + width_target, 100],# 오른쪽 위 (고정)
        [200 + width_target, 400],# 오른쪽 아래 (x좌표를 위와 똑같이 -> 수직)
        [200, 400]                # 왼쪽 아래 (x좌표를 위와 똑같이 -> 수직)
    ])

    # 3. 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 4. 이미지 변환
    # 출력 크기(w, h)는 원본과 같게 하거나, 잘리는 부분이 싫다면 더 크게 설정
    result = cv2.warpPerspective(frame, matrix, (w, h))
    
    return result