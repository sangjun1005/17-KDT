from machine import Pin
from utime import sleep

seg_A = Pin(0, Pin.OUT)
seg_B = Pin(1, Pin.OUT)
seg_C = Pin(2, Pin.OUT)
seg_D = Pin(3, Pin.OUT)
seg_E = Pin(4, Pin.OUT)
seg_F = Pin(5, Pin.OUT)
seg_G = Pin(6, Pin.OUT)
seg_DP = Pin(7, Pin.OUT)

digits = [
    [0, 0, 0, 0, 0, 0, 1, 1],  # 0
    [1, 0, 0, 1, 1, 1, 1, 1],  # 1
    [0, 0, 1, 0, 0, 1, 0, 1],  # 2
    [0, 0, 0, 0, 1, 1, 0, 1],  # 3
    [1, 0, 0, 1, 1, 0, 0, 1],  # 4
    [0, 1, 0, 0, 1, 0, 0, 1],  # 5
    [0, 1, 0, 0, 0, 0, 0, 1],  # 6
    [0, 0, 0, 1, 1, 1, 1, 1],  # 7
    [0, 0, 0, 0, 0, 0, 0, 1],  # 8
    [0, 0, 0, 0, 1, 0, 0, 1],  # 9
]

def display_digit(digit):
    state = digits[digit]

    seg_A.value(state[0])
    seg_B.value(state[1])
    seg_C.value(state[2])
    seg_D.value(state[3])
    seg_E.value(state[4])
    seg_F.value(state[5])
    seg_G.value(state[6])
    seg_DP.value(state[7])

while True:
    for i in range(9, -1, -1):
        display_digit(i)
        sleep(1)