#  CapstoneDesign

### 고려사항
1. Python 버전 통합을 해야하는지 실행해봐야 할듯
2. 변수 통합 하지 않고 각각의 기능들을 메인에서 불러와 돌렸을때 기능의 오류가 없을지
3. 화물검출시에 기존 이미지 -> 실시간 처리 코드 변경 시 라즈베리파이 에서 문제없이 검출이 될지
4. 현재 폴더 구조


.
├── cargo_monitoring_project/
├── ├── src/                  # 주요 소스 코드
├── │   ├── detection/        # 당신의 기능: 화물 검출
├── │   │   ├── object_detection.py  # YOLOE 기반 검출 로직
├── │   │   └── utils.py             # 공통 헬퍼 (e.g., 필터링, 마스킹)
├── │   ├── tilt/             # 팀원1: 박스 기울기 계산
├── │   │   └── tilt_calculation.py  # 기울기 측정 함수
├── │   ├── overload/         # 팀원2: 과적 측정
├── │   │   └── overload_measurement.py  # 과적 계산 로직 (e.g., 무게/부피 추정)
├── │   ├── blind_spot/       # 팀원3: 사각지대 감지
├── │   │   └── blind_spot_detection.py  # 사각지대 관련 로직 (e.g., 카메라 각도 기반)
├── │   └── common/           # 공통 모듈 (모두 공유)
├── │       ├── camera_input.py      # 실시간 카메라 입력 처리 (OpenCV)
├── │       └── visualization.py     # 결과 시각화 (imshow 등)
├── ├── data/                 # 모델 웨이트, 테스트 이미지 등
├── ├── tests/                # 각 모듈 테스트 파일 (e.g., test_detection.py)
├── ├── requirements.txt      # 의존성 (YOLOE, OpenCV 등)
├── ├── main.py               # 메인 실행: 모든 모듈 호출
└── └── README.md             # 각 모듈 사용법, 실행 방법