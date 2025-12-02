from ultralytics import YOLO

# 모델 파일 경로 (프로젝트 루트 기준 혹은 절대 경로)
MODEL_PATH = "figure_pose.pt"

def load_pose_model():
    """
    거리 추정용 YOLO Pose 모델을 로드합니다.
    """
    print(f"Loading Pose Model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Pose model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Pose model: {e}")
        return None