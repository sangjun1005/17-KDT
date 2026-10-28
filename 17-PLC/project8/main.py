from machine import Pin, PWM, I2C
from time import ticks_ms, ticks_diff, sleep_us

# DS1307: SDA GP0, SCL GP1
i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=100000)
RTC_ADDR = 0x68

# 공통 애노드 표시기
digits = [
    Pin(n, Pin.OUT, value=0)
    for n in (2, 3, 4, 5)
]

segments = [
    Pin(n, Pin.OUT, value=1)
    for n in (6, 7, 8, 9, 10, 11, 12)
]

colon = Pin(13, Pin.OUT, value=1)

buttons = [
    Pin(16, Pin.IN, Pin.PULL_UP),
    Pin(17, Pin.IN, Pin.PULL_UP),
    Pin(18, Pin.IN, Pin.PULL_UP)
]

buzzer = PWM(Pin(19))
buzzer.freq(1000)
buzzer.duty_u16(0)

FONT = [
    0x3F, 0x06, 0x5B, 0x4F, 0x66,
    0x6D, 0x7D, 0x07, 0x7F, 0x6F
]


def from_bcd(value):
    return (value >> 4) * 10 + (value & 15)


def to_bcd(value):
    return (value // 10) * 16 + value % 10


def read_clock():
    data = i2c.readfrom_mem(RTC_ADDR, 0, 7)

    second = from_bcd(data[0] & 0x7F)
    minute = from_bcd(data[1] & 0x7F)

    if data[2] & 0x40:
        hour = from_bcd(data[2] & 0x1F) % 12
        if data[2] & 0x20:
            hour += 12
    else:
        hour = from_bcd(data[2] & 0x3F)

    date_key = (data[6], data[5], data[4])

    return hour, minute, second, date_key


def write_clock(hour, minute):
    i2c.writeto_mem(
        RTC_ADDR,
        0,
        bytes([
            0,
            to_bcd(minute),
            to_bcd(hour)
        ])
    )


def show_digit(position, number, colon_on):
    for digit in digits:
        digit.value(0)

    pattern = FONT[number]

    for i, segment in enumerate(segments):
        segment.value(0 if (pattern >> i) & 1 else 1)

    colon.value(0 if colon_on else 1)

    digits[position].value(1)
    sleep_us(1500)
    digits[position].value(0)


if RTC_ADDR not in i2c.scan():
    raise RuntimeError(
        "DS1307을 찾지 못했습니다. SDA GP0, SCL GP1 배선을 확인하세요."
    )

# 시계 정지 비트가 켜져 있다면 해제
seconds_register = i2c.readfrom_mem(RTC_ADDR, 0, 1)[0]
i2c.writeto_mem(
    RTC_ADDR,
    0,
    bytes([seconds_register & 0x7F])
)

hour, minute, second, date_key = read_clock()

alarm_hour = 7
alarm_minute = 0
alarm_enabled = False
setting_alarm = False

ringing = False
ring_started = 0
last_alarm_key = None

raw_states = [1, 1, 1]
stable_states = [1, 1, 1]
changed_at = [ticks_ms(), ticks_ms(), ticks_ms()]

alarm_pressed_at = None
last_clock_read = ticks_ms()
position = 0

print("=== 디지털 알람 시계 ===")
print("Hours: 현재 시각의 시 증가")
print("Minutes: 현재 시각의 분 증가")
print("Alarm 짧게: 알람 설정 시작 / 저장 및 활성화")
print("Alarm 1초 이상 누른 뒤 놓기: 알람 해제")
print("알람 설정 중 Hours / Minutes: 알람 시각 변경")
print("알람이 울릴 때 아무 버튼이나 누르면 정지")
print("초기 알람: 07:00, 비활성화")

while True:
    now = ticks_ms()

    if ticks_diff(now, last_clock_read) >= 200:
        hour, minute, second, date_key = read_clock()
        last_clock_read = now

    for i, button in enumerate(buttons):
        raw = button.value()

        if raw != raw_states[i]:
            raw_states[i] = raw
            changed_at[i] = now

        if (
            ticks_diff(now, changed_at[i]) >= 25
            and raw != stable_states[i]
        ):
            stable_states[i] = raw

            if raw == 0:
                if ringing:
                    ringing = False
                    buzzer.duty_u16(0)
                    alarm_pressed_at = None
                    print("알람을 정지했습니다.")
                    continue

                if i == 0:
                    if setting_alarm:
                        alarm_hour = (alarm_hour + 1) % 24
                    else:
                        hour = (hour + 1) % 24
                        second = 0
                        write_clock(hour, minute)

                elif i == 1:
                    if setting_alarm:
                        alarm_minute = (alarm_minute + 1) % 60
                    else:
                        minute = (minute + 1) % 60
                        second = 0
                        write_clock(hour, minute)

                elif i == 2:
                    alarm_pressed_at = now

                if i in (0, 1):
                    if setting_alarm:
                        print(
                            "알람 설정: {:02d}:{:02d}".format(
                                alarm_hour, alarm_minute
                            )
                        )
                    else:
                        print(
                            "현재 시각: {:02d}:{:02d}".format(
                                hour, minute
                            )
                        )

            elif i == 2 and alarm_pressed_at is not None:
                held_ms = ticks_diff(now, alarm_pressed_at)
                alarm_pressed_at = None

                if held_ms >= 1000:
                    setting_alarm = False
                    alarm_enabled = False
                    ringing = False
                    buzzer.duty_u16(0)
                    print("알람을 해제했습니다.")

                elif setting_alarm:
                    setting_alarm = False
                    alarm_enabled = True

                    print(
                        "알람 활성화: {:02d}:{:02d}".format(
                            alarm_hour, alarm_minute
                        )
                    )

                else:
                    setting_alarm = True

                    print(
                        "알람 설정 모드: {:02d}:{:02d}".format(
                            alarm_hour, alarm_minute
                        )
                    )

    alarm_key = (date_key, hour, minute)

    if (
        alarm_enabled
        and not setting_alarm
        and not ringing
        and hour == alarm_hour
        and minute == alarm_minute
        and alarm_key != last_alarm_key
    ):
        ringing = True
        ring_started = now
        last_alarm_key = alarm_key
        print("알람이 울립니다! 아무 버튼이나 눌러 정지하세요.")

    if ringing:
        elapsed = ticks_diff(now, ring_started)

        if elapsed >= 30000:
            ringing = False
            buzzer.duty_u16(0)
            print("30초가 지나 알람을 자동 정지했습니다.")
        else:
            buzzer.duty_u16(
                32768 if (elapsed // 250) % 2 == 0 else 0
            )
    else:
        buzzer.duty_u16(0)

    if setting_alarm:
        display_hour = alarm_hour
        display_minute = alarm_minute
        colon_on = True
    else:
        display_hour = hour
        display_minute = minute
        colon_on = second % 2 == 0

    numbers = [
        display_hour // 10,
        display_hour % 10,
        display_minute // 10,
        display_minute % 10
    ]

    show_digit(position, numbers[position], colon_on)
    position = (position + 1) % 4