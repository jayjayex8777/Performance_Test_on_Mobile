# CPU 주파수 고정 디버깅 가이드 (Galaxy S24+ / SM-S926B)

> 대상: OneUI 8.0, Android 16, 루팅 상태
> 목적: 앱에서 CPU 주파수 고정이 안 되는 원인 파악

---

## 1단계: SELinux 상태 확인

```bash
adb shell su -c getenforce
```

- **Enforcing** → sysfs 쓰기가 SELinux에 의해 차단될 수 있음. 2단계로 진행
- **Permissive** → SELinux는 문제 아님. 3단계로 건너뛰기

## 2단계: SELinux 임시 해제

```bash
adb shell su -c setenforce 0
```

확인:
```bash
adb shell su -c getenforce
```
→ `Permissive` 출력 확인

## 3단계: CPU 주파수 수동 쓰기 테스트

root shell 진입:
```bash
adb shell su
```

### 3-1. 현재 상태 확인

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### 3-2. governor를 performance로 변경

```bash
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

바로 확인:
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```
→ `performance` 출력되면 쓰기 성공

### 3-3. scaling_min_freq를 max_freq로 설정

```bash
# max_freq 값 확인
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq

# 위에서 나온 값을 그대로 사용 (예: 2025600)
echo 2025600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
```

바로 확인:
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
```

### 3-4. 5초 후 유지 여부 확인 (Samsung 데몬 override 체크)

```bash
sleep 5
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

## 4단계: 결과 판단

| 결과 | 의미 | 대응 |
|------|------|------|
| 3-2에서 performance가 안 써짐 | 쓰기 권한 문제 또는 SELinux 차단 | 2단계 SELinux 해제 후 재시도 |
| 3-2 성공, 3-4에서 되돌아감 | Samsung 전력 관리 데몬이 override | 5단계 진행 |
| 3-4까지 유지됨 | sysfs 쓰기는 정상, 앱 코드 문제 | Logcat에서 CPULock 태그 확인 |

## 5단계: Samsung 전력 관리 데몬 비활성화 (3-4에서 되돌아가는 경우)

### GOS (Game Optimizing Service) 비활성화
```bash
adb shell su -c "pm disable-user com.samsung.android.game.gos"
```

### Samsung Thermal 서비스 확인 및 중지
```bash
adb shell su -c "ps -A | grep thermal"
adb shell su -c "ps -A | grep perfman"
adb shell su -c "ps -A | grep power"
```

위에서 발견된 데몬 중지 (예시):
```bash
adb shell su -c "kill $(pidof vendor.samsung.hardware.thermal@2.0-service)"
```

### 전체 클러스터에 대해 governor 고정 (Little/Mid/Big)
```bash
adb shell su

# Little (cpu0-3)
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo <max값> > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

# Mid (cpu4-6)
echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq
echo <max값> > /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq

# Big (cpu7)
echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq
echo <max값> > /sys/devices/system/cpu/cpu7/cpufreq/scaling_min_freq
```

## 6단계: 원복 (측정 완료 후)

```bash
adb shell su

# SELinux 복원
setenforce 1

# Governor 원복 (보통 schedutil)
echo schedutil > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo schedutil > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
echo schedutil > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor

# GOS 재활성화
pm enable com.samsung.android.game.gos
```
