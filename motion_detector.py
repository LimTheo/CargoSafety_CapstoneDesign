import time
import math
import numpy as np

# BMI160 센서 통신 라이브러리 임포트 (기존 코드와 동일)
try:
    from BMI160_i2c import Driver
except ImportError:
    # 이 파일은 독립 실행보다 모듈로 사용되므로, 오류 처리 후 종료 대신 예외 처리 필요
    print("오류: BMI160-i2c 라이브러리가 설치되지 않았습니다.")
    raise

# ====================================================================
# [1] 캘리브레이션 상수 설정 (기존 코드와 동일)
# ====================================================================
GYRO_BIAS = np.array([-0.2185, 0.2443, -0.3015])
GYRO_THRESHOLD = 150.0
ACC_THRESHOLD = 500.0
TIME_HYSTERESIS_S = 0.5
I2C_ADDRESS = 0x69
ACC_1G_LSB = 16384.0

# 전역 변수로 센서 객체를 저장하여 초기화는 한 번만 수행
_sensor = None
# 상태 전환을 위한 정적 변수 (함수가 호출될 때마다 이전 상태를 기억)
_stationary_start_time = None
_motion_start_time = None
# 초기 상태는 '움직임'으로 설정 (is_moving을 전역 변수나 클래스 변수로 유지)
_is_moving_state = True

# ====================================================================
# [2] 센서 초기화 함수
# ====================================================================
def initialize_bmi160():
    global _sensor
    if _sensor is None:
        try:
            _sensor = Driver(I2C_ADDRESS)
            _sensor.set_gyro_power_mode('normal') # 센서 초기 전력 모드 설정 (선택 사항)
            _sensor.set_accel_power_mode('normal')
            print("-" * 50)
            print("--- BMI160 초기화 완료: ZUPT 감지 준비 ---")
            print("-" * 50)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] 센서 초기화 오류: {e}")
            raise e
    return _sensor

# ====================================================================
# [3] 이동/정지 상태 판별 함수
# ====================================================================
def check_motion_state():
    """
    현재 BMI160 센서 데이터를 기반으로 이동(True) 또는 정지(False) 상태를 반환합니다.
    양방향 시간 지연(Hysteresis) 로직을 적용합니다.
    """
    global _stationary_start_time, _motion_start_time, _is_moving_state, _sensor
    
    # 센서가 초기화되지 않았으면 초기화 시도
    if _sensor is None:
        try:
            initialize_bmi160()
        except Exception:
            return True # 센서 오류 시 안전하게 '움직임' 상태로 간주

    try:
        # 1. 센서 데이터 읽기
        data = _sensor.getMotion6()
        gx_raw, gy_raw, gz_raw = data[0], data[1], data[2]
        ax_raw, ay_raw, az_raw = data[3], data[4], data[5]

        # 2. 각속도 크기 계산 (바이어스 제거 적용)
        gx_clean = gx_raw - GYRO_BIAS[0]
        gy_clean = gy_raw - GYRO_BIAS[1]
        gz_clean = gz_raw - GYRO_BIAS[2]
        gyro_magnitude = math.sqrt(gx_clean**2 + gy_clean**2 + gz_clean**2)

        # 3. 가속도 움직임 성분 크기 계산 (1G 편차)
        acc_magnitude_lsb = math.sqrt(ax_raw**2 + ay_raw**2 + az_raw**2)
        acc_motion_magnitude = abs(acc_magnitude_lsb - ACC_1G_LSB)

        # 4. 정지 조건 확인 (각속도 AND 가속도 모두 임계값 미만)
        is_gyro_stationary = (gyro_magnitude < GYRO_THRESHOLD)
        is_accel_stationary = (acc_motion_magnitude < ACC_THRESHOLD)
        is_short_term_stationary = is_gyro_stationary and is_accel_stationary

        # 5. 상태 전환 로직 (Hysteresis)
        current_time = time.time()
        
        if _is_moving_state:
            # 현재 상태: 움직임. '정지'로 전환할지 확인
            if is_short_term_stationary:
                # '정지' 조건 만족 -> '정지' 타이머 시작/지속
                if _stationary_start_time is None:
                    _stationary_start_time = current_time
                _motion_start_time = None 
                
                # 정지 상태가 충분히 유지되었는지 확인
                if (current_time - _stationary_start_time) >= TIME_HYSTERESIS_S:
                    _is_moving_state = False # 상태 전환: 움직임 -> 정지
                    print(f"[{current_time:.2f}s] **[정지]** 상태 변경됨! | Gyro M: {gyro_magnitude:.4f} | Acc M: {acc_motion_magnitude:.4f}")
            else:
                # '정지' 조건 불만족 -> '정지' 타이머 리셋
                _stationary_start_time = None

        else: # not _is_moving_state (현재 정지 상태)
            # 현재 상태: 정지. '움직임'으로 전환할지 확인
            if not is_short_term_stationary:
                # '움직임' 조건 만족 -> '움직임' 타이머 시작/지속
                if _motion_start_time is None:
                    _motion_start_time = current_time
                _stationary_start_time = None 
                
                # 움직임 상태가 충분히 유지되었는지 확인
                if (current_time - _motion_start_time) >= TIME_HYSTERESIS_S:
                    _is_moving_state = True # 상태 전환: 정지 -> 움직임
                    print(f"[{current_time:.2f}s] **[움직임]** 상태 변경됨! | Gyro M: {gyro_magnitude:.4f} | Acc M: {acc_motion_magnitude:.4f}")
            else:
                # '움직임' 조건 불만족 -> '움직임' 타이머 리셋
                _motion_start_time = None
        
        # 실시간 상태 출력 (선택 사항)
        # print(f"현재 상태: {'움직임' if _is_moving_state else '정지'} | Gyro M: {gyro_magnitude:.4f} | Acc M: {acc_motion_magnitude:.4f}")

        return _is_moving_state

    except Exception as e:
        print(f"데이터 처리 오류 발생: {e}")
        return True # 오류 발생 시 안전하게 '움직임' 상태로 간주

if __name__ == "__main__":
    # 독립 실행 시 테스트 루프
    initialize_bmi160()
    while True:
        state = check_motion_state()
        print(f"현재 상태: {'움직임' if state else '정지'}")
        time.sleep(0.01)