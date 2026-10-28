from machine import Pin, PWM
from time import sleep_ms, ticks_ms, ticks_diff

rows = [Pin(n, Pin.IN, Pin.PULL_UP) for n in (2, 3, 4, 5)]
cols = [Pin(n, Pin.IN, Pin.PULL_UP) for n in (6, 7, 8, 9)]

keys = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"]
]

servo = PWM(Pin(15))
servo.freq(50)

red_led = Pin(16, Pin.OUT)
green_led = Pin(17, Pin.OUT)

password = "3152"
entered = ""
new_password = ""
mode = "unlock"

door_open = False
opened_at = 0
angle = 0

AUTO_CLOSE_MS = 5000


def set_angle(degrees):
    pulse_us = 500 + degrees * 2000 // 180
    servo.duty_u16(pulse_us * 65535 // 20000)


def move_door(target):
    global angle

    step = 3 if target > angle else -3

    while angle != target:
        if step > 0:
            angle = min(target, angle + step)
        else:
            angle = max(target, angle + step)

        set_angle(angle)
        sleep_ms(15)


def close_door():
    global door_open, entered, mode, new_password

    move_door(0)
    door_open = False
    entered = ""
    new_password = ""
    mode = "unlock"

    red_led.value(1)
    green_led.value(0)

    print("\n[잠금] 문이 닫혔습니다.")
    print("비밀번호 4자리를 입력한 뒤 #을 누르세요.")


def open_door():
    global door_open, opened_at, entered

    entered = ""

    red_led.value(0)
    green_led.value(1)

    move_door(90)
    door_open = True
    opened_at = ticks_ms()

    print("\n[열림] 인증 성공! 5초 후 자동으로 닫힙니다.")


def scan_key():
    for r, row in enumerate(rows):
        row.init(Pin.OUT, value=0)

        found = None

        for c, col in enumerate(cols):
            if col.value() == 0:
                found = keys[r][c]
                break

        row.init(Pin.IN, Pin.PULL_UP)

        if found is not None:
            return found

    return None


def show_input():
    print("입력: " + "*" * len(entered))


def handle_key(key):
    global entered, password, new_password, mode

    if key == "D":
        close_door()
        return

    if door_open:
        return

    if key == "C":
        entered = ""
        new_password = ""
        mode = "unlock"
        print("\n취소했습니다. 비밀번호를 입력하세요.")
        return

    if key == "A":
        entered = ""
        new_password = ""
        mode = "verify"
        print("\n[비밀번호 변경] 기존 비밀번호 입력 후 #")
        return

    if key == "*":
        entered = entered[:-1]
        show_input()
        return

    if key == "B":
        entered = ""
        show_input()
        return

    if key.isdigit():
        if len(entered) < 4:
            entered += key
            show_input()
        return

    if key != "#":
        return

    if len(entered) != 4:
        print("비밀번호는 숫자 4자리입니다.")
        return

    value = entered
    entered = ""

    if mode == "unlock":
        if value == password:
            open_door()
        else:
            print("\n[오류] 비밀번호가 틀렸습니다. 다시 입력하세요.")

    elif mode == "verify":
        if value == password:
            mode = "new"
            print("새 비밀번호 4자리 입력 후 #")
        else:
            mode = "unlock"
            print("기존 비밀번호가 틀렸습니다. 변경을 취소합니다.")

    elif mode == "new":
        new_password = value
        mode = "confirm"
        print("새 비밀번호를 다시 입력한 뒤 #")

    elif mode == "confirm":
        if value == new_password:
            password = new_password
            print("비밀번호가 변경되었습니다.")
        else:
            print("두 입력이 달라 변경하지 않았습니다.")

        new_password = ""
        mode = "unlock"


set_angle(0)
red_led.value(1)
green_led.value(0)

print("=== 도어락 시뮬레이션 ===")
print("초기 비밀번호: 3152")
print("#: 확인 / *: 한 글자 삭제 / B: 전체 삭제")
print("A: 비밀번호 변경 / C: 취소 / D: 즉시 닫기")
print("비밀번호 입력 후 #을 누르세요.")

last_raw = None
stable_key = None
changed_at = ticks_ms()

while True:
    now = ticks_ms()

    if door_open and ticks_diff(now, opened_at) >= AUTO_CLOSE_MS:
        close_door()

    raw = scan_key()

    if raw != last_raw:
        last_raw = raw
        changed_at = now

    if ticks_diff(now, changed_at) >= 25 and raw != stable_key:
        stable_key = raw

        if stable_key is not None:
            handle_key(stable_key)

    sleep_ms(5)