from machine import Pin, ADC, PWM
import time

segment_pins = [
    Pin(n, Pin.OUT, value=0)
    for n in (2, 3, 4, 5, 6, 7, 8, 19)
]

voltage_digits = [
    Pin(n, Pin.OUT, value=1)
    for n in (9, 10, 11, 12)
]

current_digits = [
    Pin(n, Pin.OUT, value=1)
    for n in (13, 14, 15, 16)
]

all_digits = voltage_digits + current_digits

voltage_pot = ADC(Pin(26))
current_pot = ADC(Pin(27))
limit_pot = ADC(Pin(28))

button = Pin(0, Pin.IN, Pin.PULL_UP)
relay = Pin(1, Pin.OUT, value=0)
red_led = Pin(17, Pin.OUT, value=0)
yellow_led = Pin(18, Pin.OUT, value=0)

buzzer = PWM(Pin(20))
buzzer.freq(1000)
buzzer.duty_u16(0)

FONT = [
    0x3F, 0x06, 0x5B, 0x4F, 0x66,
    0x6D, 0x7D, 0x07, 0x7F, 0x6F
]

voltage = current = current_limit = 0.0
tripped = False
display_values = [0] * 8

last_sample = time.ticks_add(time.ticks_ms(), -100)
last_print = time.ticks_ms()
last_button = 1
last_reset = time.ticks_add(time.ticks_ms(), -200)


def encode_value(value):
    number = max(0, min(9999, int(value * 1000 + 0.5)))
    return [
        number // 1000,
        (number // 100) % 10,
        (number // 10) % 10,
        number % 10
    ]


def refresh_display():
    for index, digit_pin in enumerate(all_digits):
        pattern = FONT[display_values[index]]

        if index in (0, 4):
            pattern |= 0x80

        for bit, pin in enumerate(segment_pins):
            pin.value((pattern >> bit) & 1)

        digit_pin.value(0)
        time.sleep_us(1000)
        digit_pin.value(1)


try:
    while True:
        now = time.ticks_ms()

        if time.ticks_diff(now, last_sample) >= 100:
            last_sample = now

            voltage = voltage_pot.read_u16() / 65535.0 * 3.3
            current = current_pot.read_u16() / 65535.0 * 3.3
            current_limit = limit_pot.read_u16() / 65535.0 * 3.3

            if current > current_limit:
                tripped = True

            display_values = (
                encode_value(voltage) + encode_value(current)
            )

        button_now = button.value()

        if (
            last_button == 1
            and button_now == 0
            and time.ticks_diff(now, last_reset) >= 200
        ):
            last_reset = now

            if current <= current_limit:
                tripped = False

        last_button = button_now

        relay.value(0 if tripped else 1)
        red_led.value(tripped)

        yellow_led.value(
            not tripped
            and current_limit > 0
            and current >= current_limit * 0.9
        )

        buzzer.duty_u16(
            32768 if tripped and (now // 500) % 2 == 0 else 0
        )

        refresh_display()

        if time.ticks_diff(now, last_print) >= 500:
            last_print = now

            print(
                "Voltage: {:.3f}V, Current: {:.3f}A, "
                "Limit: {:.3f}A, Relay: {}".format(
                    voltage,
                    current,
                    current_limit,
                    "OFF" if tripped else "ON"
                )
            )

finally:
    relay.value(0)
    red_led.value(0)
    yellow_led.value(0)
    buzzer.duty_u16(0)
    buzzer.deinit()

    for digit in all_digits:
        digit.value(1)

    for segment in segment_pins:
        segment.value(0)