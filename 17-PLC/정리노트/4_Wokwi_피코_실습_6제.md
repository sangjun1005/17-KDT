# 4. Wokwi 피코 실습 6제 (4일차)

**폴더**: `17-PLC/project1` ~ `project6` — 각각 `main.py` · `diagram.json` · `wokwi-project.txt`
**환경**: Wokwi · Raspberry Pi Pico · MicroPython `micropython-20260406-v1.28.0`
**교재**: 34~75쪽 (아두이노 모터·서보·FND)을 **피코 + MicroPython으로 바꿔서** 한 실습

> **하루에 6개를 만들었고, 뒤로 갈수록 부품이 늘어난다.**
> 2개 → 6개 → 11개 → 22개. 마지막 6번은 **키패드 도어락**이다.
> 교재는 아두이노(C)로 가는데, 실습은 **피코(MicroPython)**로 갔다. 개념은 같고 문법만 다르다.

## 6개 한눈에

| # | 무엇 | 부품 | 연결 | 핵심 기법 |
| --- | --- | --- | --- | --- |
| 1 | 7세그먼트 **9→0 카운트다운** | 2종 | 9 | 숫자표(digits) |
| 2 | 초음파 + 온도 → **LED 조건 점등** | 6종 | 13 | **두 조건 AND** · NTC 온도 계산 |
| 3 | **전압·전류 계측반** | 22종 | **58** | **다이나믹 표시** · 트립 래치 · 부저 |
| 4 | 시리얼 입력 → **LED 5개** | 11종 | 15 | **stdin 폴링** — 손가락 인식의 수신부 |
| 5 | 스위치 3개로 **표시 모드 전환** | 15종 | 30 | 모드 상태 · 조도센서 |
| 6 | 키패드 **도어락 + 서보** | 7종 | 17 | **4×4 키 스캔** · 디바운스 · 상태기계 |

## 1 · 7세그먼트 카운트다운 (`project1`)

1자리 **공통 애노드**, GP0~GP7 → A~G·DP, COM을 3V3에 물렸다.
`digits` 표에서 **0이 점등**이다. `for i in range(9, -1, -1)`로 9→0, 1초 간격 무한 반복.

> 숫자표 10개를 7세그먼트 표준 모양과 **전부 대조했고 다 맞는다.**
> `range(9, -1, -1)`에서 끝값이 `-1`인 게 중요하다. `range(9, 0, -1)`로 쓰면 0이 안 나온다.

**이 하나만 저항이 없다.** 2번부터는 전부 330Ω이 붙어 있다 — 첫 실습 뒤에 배운 것으로 보인다.

## 2 · 초음파 + 온도 조건 점등 (`project2`)

| 부품 | 설정 |
| --- | --- |
| `wokwi-hc-sr04` 초음파 | 기본 거리 50 |
| `wokwi-ntc-temperature-sensor` | 24℃ · **beta 3950** |
| `wokwi-slide-potentiometer` | 임계값 조절용 |
| LED + 330Ω | 출력 |

```python
threshold = MIN_DISTANCE_CM + ratio * (MAX_DISTANCE_CM - MIN_DISTANCE_CM)   # 10~100cm
distance  = duration / 58.0
temperature = 1/(log(raw/(65535-raw))/BETA + 1/298.15) - 273.15             # 스타인하트-하트 B식
led.value(distance < threshold and temperature <= MAX_TEMPERATURE_C)        # 37.5℃
```

**두 조건을 `and`로 묶은 것**이 이 실습의 전부다 —
1장 **4절 연동 제어 회로**를 코드로 쓴 것과 같다. 둘 다 참이어야 출력이 난다.

세 가지를 잘 처리했다.

| 처리 | 코드 |
| --- | --- |
| 에코 신호 없음 | `if duration < 0:` → LED 끄고 `continue` |
| 센서 값 이상 | `raw <= 0 or raw >= 65535` → 끄고 `continue` |
| 임계값을 손으로 | 포텐쇼미터 비율을 10~100cm로 환산 |

> **`duration / 58.0`은 음속에서 나온 상수다.** 왕복 시간(µs)을 58로 나누면 cm다.
> 2로 나누고 0.034cm/µs를 곱한 것과 같다.

## 3 · 전압·전류 계측반 (`project3`) — 제일 큰 것

**부품 22개, 연결 58개.** 교재 75쪽 **"FND 다이나믹 표시"**가 여기서 나온다.

| 구성 | 내용 |
| --- | --- |
| 표시 | **4자리 FND 2개** (전압 빨강 · 전류 초록), 공통 캐소드 |
| 입력 | 포텐쇼미터 3개 (전압·전류·전류 제한) + 리셋 버튼 |
| 출력 | 릴레이 · 부저(PWM 1kHz) · LED 3개(빨강·노랑·파랑) |
| 저항 | 330Ω **12개** |

### 다이나믹 표시 — 핀 8개로 자리 8개를 켠다

```python
for index, digit_pin in enumerate(all_digits):   # 자리 8개를 돌면서
    pattern = FONT[display_values[index]]
    if index in (0, 4): pattern |= 0x80          # 첫 자리에 소수점
    for bit, pin in enumerate(segment_pins):     # 세그먼트 8개 세팅
        pin.value((pattern >> bit) & 1)
    digit_pin.value(0); time.sleep_us(1000); digit_pin.value(1)   # 1ms만 켠다
```

**한 번에 한 자리만 켠다.** 8자리를 1ms씩 돌면 한 바퀴가 8ms — 사람 눈에는 전부 켜져 보인다.
세그먼트 선 8개 + 자리 선 8개 = **16핀으로 8자리**를 쓴다. 개별 표시로 하면 64핀이 필요하다.

### 과전류 트립 — 한 번 걸리면 안 풀린다

```python
if current > current_limit: tripped = True      # 걸린다
...
if 버튼 눌림 and current <= current_limit:       # 조건을 만족해야만
    tripped = False                              # 풀린다
relay.value(0 if tripped else 1)
```

> **이게 1장 2절 자기 유지 회로다.** 한 번 참이 되면 계속 참이고(래치),
> **해제 조건을 따로 확인해야** 풀린다. 전류가 아직 높으면 리셋 버튼을 눌러도 안 풀린다.

| 신호 | 조건 |
| --- | --- |
| 빨강 LED | `tripped` |
| **노랑 LED** | 트립 전이고 **제한값의 90% 이상** — 미리 경고 |
| 부저 | 트립 중 `(now // 500) % 2` — **0.5초 간격 깜빡임** |
| 릴레이 | 트립이면 OFF |

**`finally` 블록에서 전부 끈다** — 릴레이·LED·부저·FND. 프로그램이 멈춰도 출력이 남지 않는다.
**PM 정리 3장의 "미해결 중대 위험은 진행 보류"와 같은 태도**다.

`encode_value`는 `value * 1000`으로 바꿔 4자리에 넣고 첫 자리에 소수점을 찍는다 → `3.300`.

## 4 · 시리얼 입력 → LED 5개 (`project4`) — **손가락 인식의 수신부**

LED 5개(빨강·노랑·초록·파랑·빨강) + 330Ω 5개, GP2~GP6.

```python
poll = select.poll(); poll.register(sys.stdin, select.POLLIN)
while True:
    if poll.poll(10):
        text = sys.stdin.readline().strip()
        if text in ("0","1","2","3","4","5"):
            show_leds(int(text))
```

> **교재 126쪽 "손가락 개수 만큼 램프동작"의 피코 쪽 절반이 이것이다.**
> 3장의 `gesture_doorlock.py`가 숫자를 보내고, 이 코드가 받아서 그만큼 켠다.
> `gesture_doorlock.py`의 `SERIAL` 모드가 바로 여기로 연결된다.

`poll.poll(10)`로 **기다리지 않고 확인만** 한다. `readline()`만 쓰면 입력이 올 때까지 멈춰서
다른 일을 못 한다. 0~5가 아니면 오류 메시지를 내고 LED는 그대로 둔다.

**발표자료가 따로 있다** — `Pico_LED_디자인.pptx` 11쪽. 배선표, 물리 핀 번호, 코드 2분할,
입력값별 점등표, 문제 해결까지 들어 있다. 핀 번호도 맞다 (GP2=4번, GP6=9번, GND=8번).

## 5 · 표시 모드 전환 (`project5`)

| 입력 | 동작 |
| --- | --- |
| SW1 (키 `1`) | 볼륨(포텐쇼) 값 표시 |
| SW2 (키 `2`) | 조도 값 표시 |
| SW3 (키 `3`) | 숫자 끄기 |
| 조도센서 | `light_raw > 32767` 이면 LED 점등 |

스위치는 전부 `Pin.PULL_UP` — **눌리면 0**이다. 부품 쪽 `bounce: 0`으로 채터링을 껐다.

`mode` 변수 하나로 표시 내용을 갈아 끼운다. **1장 3절 인터록**과 같은 구조 —
세 모드가 동시에 켜질 수 없다.

## 6 · 키패드 도어락 (`project6`) — 가장 완성도가 높다

멤브레인 키패드 4×4 + 서보 + LED 2개(LOCKED 빨강 / OPEN 초록).

### 키 스캔 — 핀 8개로 버튼 16개

```python
for r, row in enumerate(rows):
    row.init(Pin.OUT, value=0)          # 이 행만 0으로 내린다
    for c, col in enumerate(cols):
        if col.value() == 0:            # 0인 열이 눌린 자리
            found = keys[r][c]
    row.init(Pin.IN, Pin.PULL_UP)       # 원래대로 되돌린다
```

**행을 하나씩 내려 보면서 열을 읽는다.** 3번의 다이나믹 표시와 같은 발상이다 —
**한 번에 하나씩 빠르게 돌려서** 핀 수를 줄인다.

### 디바운스 — 25ms 안정화

```python
if raw != last_raw: last_raw = raw; changed_at = now
if ticks_diff(now, changed_at) >= 25 and raw != stable_key:
    stable_key = raw
    if stable_key is not None: handle_key(stable_key)
```

**값이 25ms 동안 안 바뀌어야 진짜 입력으로 받는다.**
3장 `gesture_doorlock.py`의 `HOLD_SEC = 1.0`과 같은 생각이고, 시간 단위만 다르다.

### 비밀번호 변경 — 상태 4개

```plain text
unlock  ──A──▶ verify ──기존 일치──▶ new ──입력──▶ confirm ──일치──▶ 변경 완료
   ▲                 │ 불일치                                │ 불일치
   └─────────────────┴────────────────────────────────────────┘
```

| 키 | 기능 |
| --- | --- |
| `#` | 확인 |
| `*` | 한 글자 지우기 |
| `B` | 전체 지우기 |
| `A` | 비밀번호 변경 시작 |
| `C` | 취소 |
| `D` | 즉시 닫기 |

**서보 각도를 3도씩 15ms 간격으로 움직인다** (`move_door`) — 한 번에 안 꺾고 천천히 연다.
열리면 **5초 뒤 자동으로 닫힌다**(`AUTO_CLOSE_MS`).

서보 펄스 계산도 표준대로다: `500 + 각도 * 2000 // 180` µs, 50Hz(주기 20000µs) 기준으로 듀티 환산.
0° → 500µs, 90° → 1500µs.

## 6개가 1장 시퀀스 회로와 맞물리는 곳

| 회로 (1장) | 어느 실습에서 |
| --- | --- |
| 1절 논리 회로 | 1 — 숫자 하나에서 세그먼트 8개 |
| **2절 자기 유지** | **3 — 과전류 트립 래치** |
| 3절 인터록 | 5 — 모드 하나만 |
| **4절 연동 제어** | **2 — 거리 AND 온도** |
| 5절 타이머 | 1 `sleep(1)` · 6 자동 닫힘 5초 · 6 디바운스 25ms |
| 6절 계수 | 1 카운트다운 · 4 LED 개수 |
| 7절 온도 제어 | 2 — NTC |
| 8~9절 접촉기·개폐기 | **3 — 릴레이 모듈** |
| 10절 무접점 | 전부 — 접점 없이 코드로 |

**1장에서 글로 배운 10개 회로가 6개 실습에 전부 들어 있다.**

## 확인 필요 / 주의

| 항목 | 내용 |
| --- | --- |
| **`project1`에 저항이 없다** | 시뮬레이터에서는 돌지만 **실물에서는 세그먼트마다 220~330Ω이 필요하다.** 2~6번에는 전부 붙어 있다 |
| **`project5`의 값이 뒤집혀 있다** | `pot_number = 9 - min(9, raw*10//65536)` — **raw가 클수록 작은 숫자**가 뜬다. 조도도 같다. 의도인지 확인 필요 |
| **`project5` LED 조건** | `led.value(light_raw > 32767)` — 값이 **클 때** 켜진다. 문서에는 "어두워지면 켜진다"고 되어 있어 Wokwi 조도센서 출력 방향 확인 필요 |
| **`project6` 비밀번호가 코드에 박혀 있다** | `password = "3152"`이고 시작할 때 **화면에 출력까지 한다.** 교육용이라 괜찮지만 실물에서는 안 된다 |
| **실행 결과가 없다** | 시뮬레이션 화면 캡처·로그가 폴더에 없다. 동작은 **코드와 회로 JSON을 읽고 판단**했다 |
| 발표자료 | `Pico_LED_디자인.pptx`는 `.gitignore`의 `*.pptx` 규칙에 걸려 저장소에서 제외된다 |
| 자료 출처 | 6개 모두 `wokwi-project.txt`에 `https://wokwi.com/projects/new/micropython-pi-pico` 템플릿에서 시작했다고 적혀 있다 |
