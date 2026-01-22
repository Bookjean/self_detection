# Self Detection ROS2 패키지

정전용량 근접 센서의 자기 감지(Self Detection) 보상을 위한 ROS2 패키지입니다.

로봇 자세에 따른 센서 변동을 Two-Stage 딥러닝 모델로 실시간 보상합니다.

---

## 목차

1. [개요](#개요)
2. [Two-Stage 보상 아키텍처](#two-stage-보상-아키텍처)
3. [패키지 구조](#패키지-구조)
4. [의존성](#의존성)
5. [설치 방법](#설치-방법)
6. [사용 방법](#사용-방법)
7. [모델 학습](#모델-학습)
8. [파라미터](#파라미터)
9. [토픽](#토픽)
10. [로그 파일](#로그-파일)

---

## 개요

### 문제 상황
정전용량 근접 센서는 로봇 자세(joint angles)에 따라 센서 값이 변동합니다.
이는 로봇 본체의 금속 구조물이 센서에 영향을 주기 때문입니다.
이 변동을 "자기 감지(Self Detection)"라고 하며, 실제 물체 감지를 방해합니다.

### 해결 방법
Two-Stage 딥러닝 모델로 자기 감지 성분을 예측하고 원본 센서 값에서 빼줍니다.

```
보상된_값 = 원본_센서값 - Stage1_예측 - Stage2_예측
```

---

## Two-Stage 보상 아키텍처

### Stage 1: 정적 자세 보상 (MLP)

| 항목 | 내용 |
|------|------|
| 모델 | Stage1StaticOffsetMLP |
| 입력 | joint angles (6차원: j1~j6) |
| 출력 | baseline prediction (4차원: raw1~raw4) |
| 역할 | 로봇 자세에 따른 **정적** 센서 변동 제거 |
| 특징 | 단순한 MLP로 충분 (자세→센서 매핑) |

### Stage 2: 동적 잔차 보상 (TCN)

| 항목 | 내용 |
|------|------|
| 모델 | Stage2ResidualTCN |
| 입력 | residual sequence (K×4차원) |
| 출력 | residual correction (4차원) |
| 역할 | 히스테리시스, 유전체 이완 등 **시간적** 효과 제거 |
| 특징 | Temporal Convolutional Network로 시퀀스 패턴 학습 |

### 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                      실시간 보상 흐름                         │
└─────────────────────────────────────────────────────────────┘

/joint_states (6) ──┬──→ [Stage1 MLP] ──→ baseline_pred (4)
                    │                            │
                    │                            ▼
/raw_distance1~4 ───┼─────────────────→ residual = raw - baseline
                    │                            │
                    │                            ▼
                    │                   [Residual Buffer] (K×4)
                    │                            │
                    │                            ▼
                    │                   [Stage2 TCN] ──→ correction (4)
                    │                            │
                    │                            ▼
                    └───────────────→ compensated = raw - baseline - correction
                                                 │
                                                 ▼
                                        [100Hz 로그 저장]
```

### 왜 Two-Stage인가?

1. **분리의 이점**: 정적 효과와 동적 효과를 분리하면 각 모델이 자기 역할에 집중
2. **Stage1만으로도 동작**: Stage2 버퍼가 차기 전에도 Stage1 보상은 즉시 적용
3. **해석 가능성**: 각 단계의 기여도를 개별적으로 분석 가능
4. **모듈성**: Stage1, Stage2를 독립적으로 개선/교체 가능

---

## 패키지 구조

```
src/self_detection/
├── package.xml                 # ROS2 패키지 설정
├── setup.py                    # Python 패키지 설정
├── setup.cfg                   # 설치 경로 설정
├── README.md                   # 이 문서
│
├── resource/
│   └── self_detection          # ament 리소스 마커
│
├── launch/
│   └── two_stage.launch.py     # 런치 파일
│
├── self_detection/             # ROS2 노드 패키지
│   ├── __init__.py
│   └── realtime_monitor_two_stage.py   # 메인 노드 (실시간 보상 + 로깅)
│
└── self_detection_mlp/         # ML 모델 패키지
    ├── __init__.py
    ├── model.py                # 모델 정의 (MLP, TCN 등)
    ├── data_loader.py          # 데이터 로딩 유틸
    ├── trainer.py              # 학습 유틸
    ├── evaluator.py            # 평가 유틸
    ├── preprocessing.py        # 전처리 유틸
    └── utils.py                # 기타 유틸
```

---

## 의존성

### ROS2 의존성
- `rclpy`
- `sensor_msgs`
- `std_msgs`

### Python 의존성
- `numpy`
- `torch` (PyTorch)

### 패키지 의존성 (package.xml)
```xml
<depend>rclpy</depend>
<depend>sensor_msgs</depend>
<depend>std_msgs</depend>
<exec_depend>python3-numpy</exec_depend>
<exec_depend>python3-torch</exec_depend>
```

---

## 설치 방법

### 1. 워크스페이스로 이동
```bash
cd ~/rb10_Proximity
```

### 2. 빌드
```bash
colcon build --packages-select self_detection
```

### 3. 환경 설정
```bash
source install/setup.bash
```

### 4. 설치 확인
```bash
ros2 pkg executables self_detection
# 출력: self_detection realtime_monitor_two_stage
```

---

## 사용 방법

### 사전 조건

1. **학습된 모델 필요**: `~/rb10_Proximity/ml/self_detection/models/` 폴더에 다음 파일들이 있어야 함
   - `two_stage_config_*.pt` (설정 파일)
   - `stage1_*.pth` (Stage1 모델)
   - `stage2_*.pth` (Stage2 모델)

2. **토픽 퍼블리시 필요**:
   - `/joint_states` (로봇 관절 각도)
   - `/raw_distance1~4` (센서 원본 값)

### 실행 방법

#### 방법 1: ros2 run
```bash
# 터미널 1: 로봇 드라이버
ros2 launch rb_bringup robot.launch.py

# 터미널 2: 센서 드라이버
ros2 run ecan_driver processing_node

# 터미널 3: 보상 노드
ros2 run self_detection realtime_monitor_two_stage
```

#### 방법 2: launch 파일
```bash
ros2 launch self_detection two_stage.launch.py
```

#### 방법 3: 파라미터 지정
```bash
ros2 run self_detection realtime_monitor_two_stage --ros-args \
    -p config_file:=two_stage_config_20260121.pt \
    -p seq_len:=10
```

---

## 모델 학습

### 학습 스크립트 위치
```bash
cd ~/rb10_Proximity/ml/self_detection/examples
```

### Two-Stage 학습 실행
```bash
# 인터랙티브 모드
python3 train_two_stage.py

# 자동 모드 (기본값)
python3 train_two_stage.py --auto

# 옵션 지정
python3 train_two_stage.py --auto \
    --stage1 stage1_mlp \
    --stage2 stage2_tcn \
    --seq-len 10 \
    --epochs 200
```

### 학습 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--stage1` | stage1_mlp | Stage1 모델 (stage1_mlp, simple_mlp, deep_mlp) |
| `--stage2` | stage2_tcn | Stage2 모델 (stage2_tcn, stage2_memory) |
| `--seq-len` | 10 | Stage2 시퀀스 길이 (K) |
| `--epochs` | 200 | 학습 에폭 수 |
| `--batch-size` | 64 | 배치 크기 |
| `--lr` | 0.001 | 학습률 |
| `--data` | 자동탐색 | 데이터 파일 경로 |

### 학습 결과물
학습 완료 후 `~/rb10_Proximity/ml/self_detection/models/`에 생성:
- `stage1_{model}_{timestamp}.pth` - Stage1 모델 가중치
- `stage2_{model}_{timestamp}.pth` - Stage2 모델 가중치
- `two_stage_config_{timestamp}.pt` - 통합 설정 파일

---

## 파라미터

### 노드 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `config_file` | string | '' | 설정 파일명 (자동탐색 시 빈 문자열) |
| `stage1_file` | string | '' | Stage1 모델 파일명 |
| `stage2_file` | string | '' | Stage2 모델 파일명 |
| `seq_len` | int | 10 | Stage2 시퀀스 길이 |

### 파라미터 설정 예시 (launch 파일)
```python
Node(
    package='self_detection',
    executable='realtime_monitor_two_stage',
    parameters=[
        {'seq_len': 10},
        {'config_file': 'two_stage_config_20260121.pt'},
    ]
)
```

---

## 토픽

### 구독 토픽 (Subscribe)

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/joint_states` | sensor_msgs/JointState | 로봇 관절 각도 (j1~j6) |
| `/raw_distance1` | sensor_msgs/Range | 센서 1 원본 값 |
| `/raw_distance2` | sensor_msgs/Range | 센서 2 원본 값 |
| `/raw_distance3` | sensor_msgs/Range | 센서 3 원본 값 |
| `/raw_distance4` | sensor_msgs/Range | 센서 4 원본 값 |

### 토픽 데이터 형식

**JointState 메시지:**
```
header:
  stamp: {sec, nanosec}
name: ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
position: [라디안 값 6개]
```

**Range 메시지:**
```
header:
  stamp: {sec, nanosec}
range: float32 (센서 값)
```

---

## 로그 파일

### 저장 위치
```
~/rb10_Proximity/logs/compensated_100hz_{timestamp}.txt
```

### 파일 형식
```
# raw1 raw2 raw3 raw4 (compensated)
123.456789 234.567890 345.678901 456.789012
...
```

- 100Hz (10ms 간격)로 저장
- 공백 구분
- 4채널 보상된 센서 값

### 로그 분석 예시 (Python)
```python
import numpy as np

# 로그 파일 로드
data = np.loadtxt('~/rb10_Proximity/logs/compensated_100hz_xxx.txt')

# 각 채널
comp1 = data[:, 0]
comp2 = data[:, 1]
comp3 = data[:, 2]
comp4 = data[:, 3]

# 시간축 (100Hz)
time = np.arange(len(data)) * 0.01  # 초 단위
```

---

## 모델 상세

### Stage1StaticOffsetMLP
```
입력(6) → Dense(32) → ReLU → Dense(32) → ReLU → Dense(32) → ReLU → Dense(4)
```
- 파라미터: ~2,500개
- 역할: 자세 기반 정적 베이스라인 예측

### Stage2ResidualTCN
```
입력(K, 4) → TCNBlock(4→32) → TCNBlock(32→32) → Conv1d(32→4) → 출력(4)
```
- TCNBlock: Dilated Causal Convolution + Residual Connection
- Dilation: 1, 2, 4, ... (지수적 증가)
- 파라미터: ~10,000개
- 역할: 시간적 잔차 패턴 보정

### 정규화
- **Stage1 입력**: joint angles 정규화 (X_mean, X_std)
- **Stage1 출력**: 역정규화하여 원 스케일로 변환
- **Stage2 입력**: residual 정규화 (residual_mean, residual_std)
- **Stage2 출력**: 역정규화하여 원 스케일로 변환
- **모든 연산은 원 스케일 기준**

---

## 트러블슈팅

### 1. "No executable found"
```bash
# setup.cfg 파일 확인
cat src/self_detection/setup.cfg

# 없으면 생성
echo "[develop]
script_dir=\$base/lib/self_detection
[install]
install_scripts=\$base/lib/self_detection" > src/self_detection/setup.cfg

# 클린 빌드
rm -rf build/self_detection install/self_detection
colcon build --packages-select self_detection
source install/setup.bash
```

### 2. "PyTorch not available"
```bash
pip3 install torch
```

### 3. "Model not found"
모델 파일이 올바른 위치에 있는지 확인:
```bash
ls ~/rb10_Proximity/ml/self_detection/models/
# two_stage_config_*.pt, stage1_*.pth, stage2_*.pth 파일 필요
```

### 4. Stage2가 동작하지 않음
- Residual 버퍼가 `seq_len`만큼 차야 Stage2가 동작
- 시작 후 약 `seq_len * 0.01`초 후 Stage2 활성화
- 로그에 Stage1만 적용된 값이 먼저 기록됨

---

## 라이선스

MIT License

---

## 작성자

Robot Sensing Lab
