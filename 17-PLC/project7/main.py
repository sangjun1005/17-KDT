# 볼륨 / 조도센서 / FND 3모드 (Wokwi, Raspberry Pi Pico, MicroPython)
# SW1: 볼륨값을 9~0으로 표시
# SW2: 조도센서값을 0~9로 표시 (손바닥으로 가리면 숫자가 줄어듦)
# SW3: FND 모두 끄고, 조도센서가 가려지면(어두우면) LED ON
from machine import Pin, ADC
import time

# 세그먼트 핀  a  b  c  d  e  f  g  dp
seg = [Pin(p, Pin.OUT) for p in [2, 3, 4, 5, 6, 7, 8, 9]]

sw1 = Pin(10, Pin.IN, Pin.PULL_UP)   # 볼륨 모드 (누르면 0)
sw2 = Pin(11, Pin.IN, Pin.PULL_UP)   # 조도 모드
sw3 = Pin(12, Pin.IN, Pin.PULL_UP)   # 감지 모드
led = Pin(13, Pin.OUT)
pot = ADC(26)                        # 가변저항
cds = ADC(27)                        # 조도센서 모듈 AO (밝을수록 값 작아짐)

DARK_LEVEL = 5   # 조도 단계가 이 값보다 작으면 "가려짐"으로 인정

# 공통 캐소드: 1 = 켜짐   a b c d e f g
digits = [
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1],  # 9
]

mode = 0   # 0: 대기, 1: 볼륨, 2: 조도, 3: 감지

def show_digit(n):
    for i in range(7):
        seg[i].value(digits[n][i])
    seg[7].value(0)   # dp 끔

def clear_fnd():
    for s in seg:
        s.value(0)

clear_fnd()

while True:
    # 버튼으로 모드 선택
    if sw1.value() == 0:
        mode = 1
    if sw2.value() == 0:
        mode = 2
    if sw3.value() == 0:
        mode = 3

    cds_level = 9 - cds.read_u16() * 10 // 65536   # 밝음 → 9, 어두움 → 0

    if mode == 1:
        pot_level = 9 - pot.read_u16() * 10 // 65536   # 0~65535 → 9~0
        show_digit(pot_level)
        led.value(0)
    elif mode == 2:
        show_digit(cds_level)
        led.value(0)
    elif mode == 3:
        clear_fnd()
        if cds_level < DARK_LEVEL:
            led.value(1)   # 가려짐 → 동작 인정
        else:
            led.value(0)

    time.sleep(0.05)