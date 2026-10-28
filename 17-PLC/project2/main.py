from machine import Pin, ADC, time_pulse_us
import time
import math

trigger = Pin(15, Pin.OUT)
echo = Pin(14, Pin.IN)
led = Pin(3, Pin.OUT)
pot = ADC(Pin(28))
ntc = ADC(Pin(27))

trigger.value(0)
led.value(0)

MIN_DISTANCE_CM = 10.0
MAX_DISTANCE_CM = 100.0
MAX_TEMPERATURE_C = 37.5
BETA = 3950.0

while True:
    ratio = pot.read_u16() / 65535.0
    threshold = MIN_DISTANCE_CM + ratio * (
        MAX_DISTANCE_CM - MIN_DISTANCE_CM
    )

    trigger.value(0)
    time.sleep_us(2)
    trigger.value(1)
    time.sleep_us(10)
    trigger.value(0)

    duration = time_pulse_us(echo, 1, 30000)

    if duration < 0:
        led.value(0)
        print("Echo 신호를 받지 못했습니다.")
        time.sleep(0.1)
        continue

    distance = duration / 58.0

    raw = ntc.read_u16()

    if raw <= 0 or raw >= 65535:
        led.value(0)
        print("온도 센서 값을 확인해주세요.")
        time.sleep(0.1)
        continue

    resistance_ratio = raw / (65535.0 - raw)
    temperature = 1.0 / (
        math.log(resistance_ratio) / BETA + 1.0 / 298.15
    ) - 273.15

    led.value(
        distance < threshold
        and temperature <= MAX_TEMPERATURE_C
    )

    led_state = "ON" if led.value() else "OFF"

    print(
        "측정: {:.1f} cm, 임계: {:.1f} cm, "
        "온도: {:.1f} C → LED {}".format(
            distance, threshold, temperature, led_state
        )
    )

    time.sleep(0.1)