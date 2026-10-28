from machine import Pin, ADC
from time import sleep_ms

segments = [
    Pin(2, Pin.OUT),
    Pin(3, Pin.OUT),
    Pin(4, Pin.OUT),
    Pin(5, Pin.OUT),
    Pin(6, Pin.OUT),
    Pin(7, Pin.OUT),
    Pin(8, Pin.OUT)
]

sw1 = Pin(10, Pin.IN, Pin.PULL_UP)
sw2 = Pin(11, Pin.IN, Pin.PULL_UP)
sw3 = Pin(12, Pin.IN, Pin.PULL_UP)

led = Pin(13, Pin.OUT)
pot = ADC(Pin(26))
light_sensor = ADC(Pin(27))

FONT = [
    0x3F,
    0x06,
    0x5B,
    0x4F,
    0x66,
    0x6D,
    0x7D,
    0x07,
    0x7F,
    0x6F
]

mode = 1


def display_number(number):
    pattern = 0 if number is None else FONT[number]

    for i, segment in enumerate(segments):
        segment.value((pattern >> i) & 1)


display_number(None)
led.value(0)

print("SW1: 볼륨값 표시")
print("SW2: 조도값 표시")
print("SW3: 숫자 표시 끄기")
print("어두워지면 LED가 켜집니다.")

while True:
    if sw1.value() == 0:
        mode = 1
    elif sw2.value() == 0:
        mode = 2
    elif sw3.value() == 0:
        mode = 3

    pot_raw = pot.read_u16()
    light_raw = light_sensor.read_u16()

    pot_number = 9 - min(9, pot_raw * 10 // 65536)
    light_number = 9 - min(9, light_raw * 10 // 65536)

    if mode == 1:
        number = pot_number
    elif mode == 2:
        number = light_number
    else:
        number = None

    display_number(number)

    led.value(light_raw > 32767)

    print(
        "모드: {} | 볼륨: {} | 조도: {} | 표시: {} | LED: {}".format(
            mode,
            pot_number,
            light_number,
            "-" if number is None else number,
            "ON" if led.value() else "OFF"
        )
    )

    sleep_ms(50)