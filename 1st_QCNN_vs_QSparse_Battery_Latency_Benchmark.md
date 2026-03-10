# 세션 보고서: QCNN vs QSparse 모델 성능 측정 (Android)

**날짜**: 2026-03-10
**대상 기기**: Samsung Galaxy S24+ (SM-S926U, Snapdragon 8 Gen 3, Android 14)
**앱**: infer-android-main (PyTorch Lite 기반 추론 벤치마크)
**GitHub**: https://github.com/jayjayex8777/Performance_Test_on_Mobile

---

## 1. 프로젝트 개요

QCNN(Quantized CNN, T=20)과 QSparse(Quantized Sparse SNN, T=3, FR=05) 모델의 **지연 시간(Latency)**과 **배터리 소비(Battery Consumption)**를 Android 디바이스에서 비교 측정하는 프로젝트.

### 측정 대상 모델
| 모델 | 설명 | 크기 |
|------|------|------|
| QCNN (T=20) | Quantized CNN | Large/Medium/Small |
| QSparse (T=3, FR=05) | Quantized Sparse SNN | Large/Medium/Small |

---

## 2. 주요 파일 구조

```
infer-android-main/
├── app/src/main/java/com/example/infer/
│   └── MainActivity.kt          # 전체 앱 로직 (~1288줄)
├── app/src/main/res/
│   ├── layout/activity_main.xml  # UI 레이아웃 (143줄)
│   └── values/strings.xml        # 문자열 리소스
├── app/src/main/assets/
│   ├── data/                     # CSV 입력 데이터
│   └── *.ptl                     # PyTorch Lite 모델 파일
└── QCNN_vs_QSparse_Battery_Latency_Benchmark.md  # 이 문서
```

### MainActivity.kt 주요 함수

| 함수 | 줄 번호 | 기능 |
|------|---------|------|
| `startMeasure()` | L70-89 | Measure Start 버튼 핸들러 |
| `runInference()` | L102-286 | 기본 지연시간 측정 (warm-up 5, repeat 100) |
| `runAccuracy()` | L441-539 | 정확도 비교 (QCNN vs QSparse) |
| `runRebuttalInference()` | L707-947 | Rebuttal 스트리밍 측정 (TSNN, TCNN, CCNN) |
| `runRebuttalAccuracy()` | L949-1045 | Rebuttal 정확도 |
| `startAmpMeasure()` | L1049-1068 | Amplified Measure 버튼 핸들러 |
| `runAmpInference()` | L1091-1286 | **멀티스레드 차등 측정** (최종 버전) |

### UI 버튼 (activity_main.xml)

| 버튼 | ID | 기능 |
|------|-----|------|
| Measure Start | measureButton | 기본 지연시간 측정 |
| Accuracy 스타트 | accuracyButton | 정확도 비교 |
| Sparsity 스타트 | sparsityButton | Sparsity 측정 |
| Rebuttal Measure | rebuttalMeasureButton | Rebuttal 지연시간 |
| Rebuttal Accuracy | rebuttalAccuracyButton | Rebuttal 정확도 |
| Amplified Measure | ampMeasureButton | 멀티스레드 증폭 측정 |

---

## 3. 배터리 측정 방법론 진화 과정

배터리 소비 측정의 신뢰성을 높이기 위해 7단계에 걸쳐 방법론을 개선하였다.

### 3.1 단계별 진화

| 단계 | 방법 | 결과 |
|------|------|------|
| 1단계 | Warm-up 추가 + GC 제거 | 지연시간 정확도 향상 |
| 2단계 | Repeat 횟수 100으로 증가 | 샘플 수 증가 |
| 3단계 | **텐서 프리로딩** | I/O와 추론 분리 (amp ≤ 2에서 유효) |
| 4단계 | 화면 밝기 최소화 (`0.01f`) | 디스플레이 노이즈 감소 |
| 5단계 | Baseline 차감 + Coulomb Counter | **실패** (API 한계) |
| 6단계 | 단일스레드 차등 측정 (Differential) | **실패** (노이즈 > 신호) |
| 7단계 | **멀티스레드(8) 차등 측정** | 최종 방법 (미검증) |

### 3.2 실패한 방법들과 원인

#### Baseline 차감 실패 (5단계)
- `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW`는 시스템 전체 전류를 반환
- amp ≥ 4에서 배치 로딩 I/O의 idle 기간이 CPU 활동을 낮춰 평균 전류(150-500uA)가 baseline(290-380uA)보다 낮아짐
- `net_current = max(0, avg - baseline)` → 항상 0

#### Coulomb Counter 실패 (5단계)
- `BATTERY_PROPERTY_CHARGE_COUNTER`: Galaxy S24+에서 ~4797 uAh 단위로 보고
- 개별 모델 추론의 전력 차이 감지 불가 (해상도 부족)

#### 단일스레드 차등 측정 실패 (6단계)
- Phase 1 (N회) vs Phase 2 (2N회) 차이로 고정 시스템 전력 상쇄 시도
- 시스템 전류 노이즈 (~수백 uA 변동)가 추론 신호 (~수십 uW)를 압도
- energy_per_inference에 단조 증가 경향 없음

### 3.3 최종 방법: 멀티스레드 차등 측정 (7단계)

**핵심 아이디어**: 8개 스레드 × 8코어 Snapdragon 8 Gen 3으로 추론 전력 신호를 8배 증폭

```
┌─────────────────────────────────────────────┐
│           runAmpInference() 구조             │
├─────────────────────────────────────────────┤
│ ampFactors = [1, 2]  (프리로딩 경로만)        │
│ numThreads = 8                              │
│ N = 50 (Phase 1 반복), 2N = 100 (Phase 2)   │
│                                              │
│ 1. 텐서 프리로딩                              │
│    - CSV → Tensor 변환                       │
│    - amp > 1이면 repeatTensorRows()          │
│                                              │
│ 2. 모듈 8개 로딩                              │
│    - modules[0..7] = LiteModuleLoader.load() │
│                                              │
│ 3. 텐서 파티셔닝 (Method B)                   │
│    - 각 스레드가 exclusive 텐서 인덱스 접근     │
│    - Race condition 방지                     │
│                                              │
│ 4. 병렬 Warm-up (CountDownLatch)             │
│                                              │
│ 5. Phase 1: N=50 reps × 8 threads           │
│    - 배터리 샘플링 (전류 측정)                 │
│                                              │
│ 6. Phase 2: 2N=100 reps × 8 threads         │
│    - 배터리 샘플링 + 지연시간 측정             │
│                                              │
│ 7. 차등 계산                                  │
│    diffEnergy = max(0, energyLong - energyShort)│
│    energyPerInference = diffEnergy /          │
│                  (preloadedTensors.size × N)  │
└─────────────────────────────────────────────┘
```

#### Phase 1 vs Phase 2를 나눈 이유

| | Phase 1 (짧은 측정) | Phase 2 (긴 측정) |
|---|---|---|
| 반복 횟수 | N = 50회 | 2N = 100회 |
| 측정 에너지 | E_short = E_system + E_inference(N) | E_long = E_system + E_inference(2N) |

**차등 공식**:
```
diffEnergy = E_long - E_short
           = [E_system + E_inference(2N)] - [E_system + E_inference(N)]
           = E_inference(N)    ← 시스템 고정 전력 상쇄됨
```

Phase 2에서 2배로 늘린 이유: 동일 시간 단위의 시스템 전력을 상쇄하기 위해 추론 횟수만 2배 차이를 만듦.

#### 텐서 파티셔닝 (Method B)

```
Thread 0: tensors[0 .. chunkSize-1]
Thread 1: tensors[chunkSize .. 2*chunkSize-1]
...
Thread 7: tensors[7*chunkSize .. end]
```

- 각 스레드가 독립적인 텐서 범위에만 접근
- `IValue.from(tensor)` 동시 호출 시 race condition 방지
- Method A (공유 텐서 + 동기화)보다 안전하고 단순

---

## 4. 지연시간 측정 결과

### 4.1 기본 측정 (Measure Start)
텐서 프리로딩 + warm-up 5 + repeat 100 방식으로 안정적인 지연시간 측정 성공.

### 4.2 Amplified 측정 (단일스레드, amp 4x-16x)
- **지연시간**: 4x~16x amp에서 완벽한 모델 크기 순서의 단조 증가 확인
- **배터리**: 노이즈로 인해 의미 있는 모델 간 차이 미검출

### 4.3 8x QSparse Medium 이상값
- 672.597ms 지연시간 (정상의 ~200배)
- **원인 추정**: 측정 중 thermal throttling 발생

---

## 5. 발견된 문제와 해결

### 5.1 Samsung FreecessController 앱 동결
- **문제**: `screenBrightness = 0f`로 설정 시 Samsung이 앱을 freeze
- **해결**: `screenBrightness = 0.01f`로 변경 (startMeasure, startAmpMeasure 양쪽)

### 5.2 amp ≥ 4에서 OOM 방지
- 높은 amp factor에서 텐서를 한꺼번에 프리로딩하면 OOM 발생 가능
- **해결**: 배치 기반 텐서 로딩 (amp ≥ 4에서는 프리로딩 대신 배치 방식)
- 최종 코드에서는 amp = [1, 2]만 사용하여 이 문제 회피

### 5.3 미사용 변수 경고
- `var shortLastNs` 선언 후 미사용 → 멀티스레드 재작성 시 제거

---

## 6. 핵심 기술적 인사이트

### BatteryManager API의 한계
- `BATTERY_PROPERTY_CURRENT_NOW`: 시스템 전체 전류, 노이즈 수백 uA
- `BATTERY_PROPERTY_CHARGE_COUNTER`: Galaxy S24+에서 ~4797 uAh 해상도
- **결론**: 단일 모델 추론의 전력 차이(수십 uW)를 직접 측정하기에는 부적합

### 텐서 프리로딩의 영향
- I/O(CSV 로딩 + 텐서 변환)를 측정 구간에서 분리
- 순수 추론 시간/전력만 측정 가능
- amp ≤ 2에서는 메모리 부담 없이 전체 프리로딩 가능
- amp ≥ 4에서는 메모리 제한으로 배치 로딩 필요 → I/O idle이 전류 측정에 영향

### 멀티스레드 병렬 추론의 기대 효과
- 8코어 동시 추론으로 전력 신호 8배 증폭
- BatteryManager 노이즈 대비 SNR(Signal-to-Noise Ratio) 개선
- CountDownLatch로 스레드 동기화 → 모든 스레드가 동시에 시작/종료

---

## 7. 앱 UI 변경 사항

### 타이틀 변경
- **이전**: `모델 지연시간 측정`
- **이후**: `모델 지연시간 측정 (텐서프리로딩 + 모델 다중 실행)`
- 파일: `app/src/main/res/values/strings.xml` (latency_title)

---

## 8. Git 커밋 이력 (이번 세션)

| 커밋 | 내용 |
|------|------|
| `56831af` | 멀티스레드(8) 차등 측정 코드 (runAmpInference) — GitHub push 완료 |
| (미커밋) | 타이틀 문자열 변경 — 빌드 완료, 설치 대기 |

---

## 9. 남은 작업

- [ ] 타이틀 변경 커밋 및 GitHub push
- [ ] 앱 설치 (Galaxy S24+)
- [ ] 멀티스레드 Amplified Measure 실행 및 결과 분석
- [ ] 결과 CSV 파일 수집 (`amp_battery_*.csv`, `amp_latency_*.csv`)
- [ ] QCNN vs QSparse 배터리 소비 비교 최종 결론 도출

---

## 10. 사용된 기술 스택

| 항목 | 기술 |
|------|------|
| 언어 | Kotlin |
| 프레임워크 | Android (API 34), PyTorch Lite |
| 빌드 | Gradle |
| 병렬 처리 | `java.util.concurrent.Executors.newFixedThreadPool(8)`, `CountDownLatch` |
| 배터리 API | `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW`, `BATTERY_PROPERTY_CHARGE_COUNTER` |
| 대상 기기 | Galaxy S24+ (Snapdragon 8 Gen 3, 8코어) |
