# 2차 세션 보고서: QCNN vs QSparse 배터리/지연시간 벤치마크 개선

**날짜**: 2026-03-10
**대상 기기**: Samsung Galaxy S24+ (SM-S926U, Snapdragon 8 Gen 3, Android 14)
**앱**: infer-android-main (PyTorch Lite 기반 추론 벤치마크)
**GitHub**: https://github.com/jayjayex8777/Performance_Test_on_Mobile
**선행 세션**: `QCNN_vs_QSparse_Battery_Latency_Benchmark.md` 참조

---

## 1. 세션 목표

1차 세션에서 구현한 멀티스레드 차등 측정 코드(`runAmpInference`)를 실전 배포 가능하게 보완:
- 앱 타이틀 업데이트
- **화면 꺼짐/Suspend 방지** (PARTIAL_WAKE_LOCK)
- **측정 완료 시 화면 자동 복원** (FULL_WAKE_LOCK + ACQUIRE_CAUSES_WAKEUP)
- 세션 문서화 및 GitHub push
- 측정 소요 시간 분석
- `diff` / `per_infer` 결과값 해설

---

## 2. 수행 작업 요약

| 순번 | 작업 | 상태 |
|------|------|------|
| 1 | 앱 타이틀 변경 (strings.xml) | 완료 |
| 2 | 1차 세션 보고서 .md 파일 생성 | 완료 |
| 3 | Screen Timeout 시 앱 Suspend 문제 분석 | 완료 |
| 4 | PARTIAL_WAKE_LOCK 적용 (전 측정 버튼) | 완료 |
| 5 | 측정 완료 시 wakeUpScreen() 추가 | 완료 |
| 6 | 논리적 오류 검증 (2회) | 완료 |
| 7 | GitHub push | 완료 |
| 8 | 측정 소요 시간 분석 | 완료 |
| 9 | diff / per_infer 결과값 해설 | 완료 |

---

## 3. 문제 발견: 화면 꺼짐 시 앱 Suspend

### 3.1 문제 상황

- 앱 실행 후 측정 버튼을 누르면 화면 밝기가 최소(`0.01f`)로 설정됨
- Android Screen Timeout(10분) 이후 화면이 꺼짐
- **Samsung FreecessController**가 앱을 감지하여 **강제 freeze/suspend**
- CPU 활동이 중단되어 측정이 멈춤

### 3.2 원인 분석

| 요인 | 설명 |
|------|------|
| Android Doze 모드 | 화면 꺼짐 후 CPU idle 시 진입, coroutine 중단 |
| Samsung FreecessController | Samsung 전용 프로세스 관리, foreground가 아닌 앱 aggressive freeze |
| CoroutineScope | CPU가 sleep 상태면 `Dispatchers.IO` 스레드도 중단됨 |

### 3.3 해결: PARTIAL_WAKE_LOCK

`PowerManager.PARTIAL_WAKE_LOCK`을 사용하여 화면이 꺼져도 CPU가 활성 상태를 유지.

---

## 4. 코드 변경 사항

### 4.1 AndroidManifest.xml — WAKE_LOCK 퍼미션

```xml
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

### 4.2 MainActivity.kt — WakeLock 인프라 (L37-58)

```kotlin
private var wakeLock: PowerManager.WakeLock? = null

private fun acquireWakeLock() {
    val pm = getSystemService(POWER_SERVICE) as PowerManager
    wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "infer:measurement")
    wakeLock?.acquire(30 * 60 * 1000L) // 30분 타임아웃 safety net
}

private fun releaseWakeLock() {
    wakeLock?.let { if (it.isHeld) it.release() }
    wakeLock = null
}

@Suppress("DEPRECATION")
private fun wakeUpScreen() {
    val pm = getSystemService(POWER_SERVICE) as PowerManager
    val screenLock = pm.newWakeLock(
        PowerManager.FULL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP or PowerManager.ON_AFTER_RELEASE,
        "infer:screen_on"
    )
    screenLock.acquire(3000) // 3초 후 자동 해제
}
```

### 4.3 onDestroy() 안전 해제 (L88-92)

```kotlin
override fun onDestroy() {
    super.onDestroy()
    releaseWakeLock()
    coroutineScope.cancel()
}
```

### 4.4 5개 측정 함수에 WakeLock 적용

모든 측정 시작 함수에 동일한 패턴 적용:

```kotlin
private fun start*() {
    setButtonsEnabled(false)
    acquireWakeLock()         // ← 측정 시작 전 CPU wake lock 획득
    // [밝기 조절 — startMeasure, startAmpMeasure만 해당]
    coroutineScope.launch {
        val result = withContext(Dispatchers.IO) { run*() }
        // [밝기 복원]
        binding.resultText.text = result.message
        setButtonsEnabled(true)
        wakeUpScreen()        // ← 화면이 꺼져 있으면 다시 켜기
        releaseWakeLock()     // ← CPU wake lock 해제
    }
}
```

적용된 함수 목록:

| 함수 | 줄 번호 | 밝기 제어 |
|------|---------|-----------|
| `startMeasure()` | L94-116 | O (`0.01f`) |
| `startAccuracy()` | L453-468 | X |
| `startRebuttalMeasure()` | L590-604 | X |
| `startRebuttalAccuracy()` | L607-621 | X |
| `startAmpMeasure()` | L1085-1107 | O (`0.01f`) |

### 4.5 strings.xml — 앱 타이틀 변경

```xml
<string name="latency_title">모델 지연시간 측정 (텐서프리로딩 + 모델 다중 실행)</string>
```

---

## 5. 논리적 오류 검증 결과

2차에 걸쳐 전체 코드의 논리적 오류를 검증함. **오류 없음** 확인.

### 5.1 WakeLock 안전성

| 검증 항목 | 결과 |
|-----------|------|
| Double-acquire 가능성 | 없음 — 버튼이 disable되어 중복 호출 불가 |
| Wake lock 누수 | 없음 — 3단계 보호: ①정상 해제, ②onDestroy, ③30분 타임아웃 |
| 예외 경로 | 안전 — run*() 함수들이 모두 try/catch로 InferenceResult 반환 |
| isHeld 체크 | releaseWakeLock()에서 isHeld 확인 후 release |

### 5.2 wakeUpScreen 안전성

| 검증 항목 | 결과 |
|-----------|------|
| screenLock 누수 | 없음 — 로컬 변수 + 3초 자동 해제 타임아웃 |
| FULL_WAKE_LOCK deprecated | @Suppress("DEPRECATION") 추가, 기능적으로 동작함 |
| 여러 번 호출 | 안전 — 매번 새 로컬 변수, 이전 것과 독립 |

---

## 6. 측정 소요 시간 분석

### 6.1 Measure Start (기본 지연시간 측정)

| 단계 | 시간 |
|------|------|
| 모델 로드/warm-up | ~5-10초 × 10개 모델 |
| 추론 100회 × 텐서 N개 | ~20-40초 × 10개 모델 |
| 모델 간 GC/전환 | ~1-2초 × 10개 |
| **총 예상** | **약 5-10분** |

### 6.2 Amplified Measure (멀티스레드 차등 측정)

| 단계 | 시간 |
|------|------|
| 텐서 프리로딩 | ~2-5초 × 10개 모델 |
| 모듈 8개 로딩 | ~3-8초 × 10개 모델 |
| Warm-up (8 threads) | ~2-5초 × 10개 모델 |
| Phase 1 (N=50 × 8 threads) | ~15-30초 × 10개 모델 |
| Phase 2 (2N=100 × 8 threads) | ~30-60초 × 10개 모델 |
| 대기 (3초 × 3구간) | ~9초 × 10개 모델 |
| **amp=1 소계** | ~10-18분 |
| **amp=2 소계** | ~15-25분 (텐서 2배) |
| **총 예상** | **약 25-40분** |

### 6.3 기타 측정

| 버튼 | 예상 시간 |
|------|-----------|
| Accuracy | ~2-5분 |
| Rebuttal Measure | ~5-10분 |
| Rebuttal Accuracy | ~2-5분 |

---

## 7. Amplified Measure 결과값 해설

### 7.1 `diff` (diff_energy_uAs)

**Phase 1과 Phase 2의 에너지 차이** — 시스템 고정 전력을 상쇄한 순수 추론 에너지.

```
diff = E(Phase2) - E(Phase1)
     = [E_system + E_inference(2N)] - [E_system + E_inference(N)]
     = E_inference(N)    ← 시스템 고정 전력 상쇄
```

- 단위: `uA·s` (마이크로암페어·초)
- 음수인 경우 `max(0, ...)` 처리 → 노이즈 의미

### 7.2 `per_infer` (energy_per_inference_uAs)

**추론 1회당 에너지 소비 추정치**.

```
per_infer = diff / (preloadedTensors.size × N)
```

- `preloadedTensors.size` = 입력 텐서 수 (amp factor에 따라 변동)
- `N` = 50 (Phase 1 반복 횟수)
- 단위: `uA·s` (마이크로암페어·초)

### 7.3 결과 해석 기준

| 패턴 | 의미 |
|------|------|
| diff = 0 | Phase 2가 Phase 1보다 에너지를 덜 사용 → 노이즈 > 신호 (측정 무의미) |
| per_infer가 모델 크기 순 단조 증가 | 측정이 유효함 (Large > Medium > Small) |
| per_infer 값이 뒤죽박죽 | BatteryManager 노이즈가 신호를 압도 (신뢰 불가) |

### 7.4 CSV 출력 컬럼 (amp_battery_*.csv)

```
amp, group, model, method,
energy_short_uAs,        ← Phase 1 총 에너지 (avgCurrent × time)
energy_long_uAs,         ← Phase 2 총 에너지
diff_energy_uAs,         ← Phase 2 - Phase 1 (= diff)
energy_per_inference_uAs,← diff / (텐서 수 × N) (= per_infer)
avg_current_short_uA,    ← Phase 1 평균 전류
avg_current_long_uA,     ← Phase 2 평균 전류
time_short_s,            ← Phase 1 소요 시간 (초)
time_long_s,             ← Phase 2 소요 시간 (초)
samples_short,           ← Phase 1 전류 샘플 수
samples_long,            ← Phase 2 전류 샘플 수
threads                  ← 병렬 스레드 수 (8)
```

---

## 8. 파일 변경 목록

| 파일 | 변경 내용 |
|------|-----------|
| `app/src/main/AndroidManifest.xml` | WAKE_LOCK 퍼미션 추가 |
| `app/src/main/java/com/example/infer/MainActivity.kt` | WakeLock 3함수 추가, 5개 측정함수에 적용, onDestroy 안전 해제 |
| `app/src/main/res/values/strings.xml` | latency_title 텍스트 변경 |
| `QCNN_vs_QSparse_Battery_Latency_Benchmark.md` | 1차 세션 보고서 (신규 생성) |

---

## 9. Git 이력

| 커밋/작업 | 내용 |
|-----------|------|
| 이전 세션 | `56831af` — 멀티스레드 차등 측정 코드 최초 push |
| 이번 세션 | WAKE_LOCK + wakeUpScreen + 타이틀 변경 + 문서 → GitHub push 완료 |

---

## 10. 남은 작업

- [ ] Galaxy S24+에 앱 설치 및 Amplified Measure 실행
- [ ] 결과 CSV 수집 (`amp_battery_*.csv`, `amp_latency_*.csv`)
- [ ] diff / per_infer 값 분석 → QCNN vs QSparse 배터리 소비 최종 비교
- [ ] 다른 타겟 디바이스 측정 결과와 교차 검증
- [ ] 벤치마크 보고서 최종 작성 (MobiCom 논문용)

---

## 11. 기술 스택

| 항목 | 기술 |
|------|------|
| 언어 | Kotlin |
| 프레임워크 | Android (API 34), PyTorch Lite |
| 빌드 | Gradle |
| 병렬 처리 | `Executors.newFixedThreadPool(8)`, `CountDownLatch` |
| 전력 관리 | `PowerManager.PARTIAL_WAKE_LOCK`, `FULL_WAKE_LOCK` |
| 배터리 API | `BatteryManager.BATTERY_PROPERTY_CURRENT_NOW` |
| 대상 기기 | Galaxy S24+ (Snapdragon 8 Gen 3, 8코어) |
