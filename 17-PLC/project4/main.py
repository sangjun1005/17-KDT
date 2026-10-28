from machine import Pin
import sys
import select
import time

leds = [
    Pin(2, Pin.OUT, value=0),
    Pin(3, Pin.OUT, value=0),
    Pin(4, Pin.OUT, value=0),
    Pin(5, Pin.OUT, value=0),
    Pin(6, Pin.OUT, value=0)
]


def show_leds(count):
    for i in range(5):
        leds[i].value(1 if i < count else 0)


show_leds(0)

print("==============================")
print("라즈베리파이 피코 LED 5개 제어")
print("0~5 숫자를 입력하고 Enter를 누르세요.")
print("==============================")

poll = select.poll()
poll.register(sys.stdin, select.POLLIN)

while True:
    if poll.poll(10):
        text = sys.stdin.readline().strip()

        if text in ("0", "1", "2", "3", "4", "5"):
            count = int(text)
            show_leds(count)
            print("입력: {} → LED {}개 켜짐".format(count, count))
        else:
            print("오류: 0부터 5까지 숫자를 입력하세요.")

    time.sleep_ms(10)