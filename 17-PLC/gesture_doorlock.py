# -*- coding: utf-8 -*-
"""
손동작 도어락 - 웹캠 손가락 인식 프로그램

웹캠으로 손가락 개수(0~5)를 인식해서 숫자를 보냅니다.
  - TINKERCAD 모드: 팅커캐드 시리얼 모니터 입력칸에 숫자를 자동으로 입력하고 Enter
  - SERIAL 모드   : 실제 아두이노에 USB 시리얼로 숫자 전송

설치 (명령 프롬프트에서 한 번만):
  pip install opencv-python mediapipe pyautogui pyserial

사용법 (팅커캐드):
  1. 팅커캐드에서 '손동작 도어락' 시뮬레이션 시작 → 코드 창 아래 '시리얼 모니터' 열기
  2. 이 프로그램 실행:  python gesture_doorlock.py
  3. 팅커캐드 시리얼 모니터의 입력칸을 마우스로 한 번 클릭해 둔다 (커서가 깜빡이게)
  4. 카메라에 손가락을 펴서 1초 동안 유지 → 숫자 1개 입력됨
  5. 손을 화면 밖으로 뺐다가 다음 숫자 손동작 (같은 숫자 연속 입력도 가능)
  종료: 카메라 창을 클릭하고 q, 또는 명령 프롬프트에서 Ctrl+C
"""

import time
import cv2
import mediapipe as mp

# ───────── 설정 ─────────
MODE = "TINKERCAD"      # "TINKERCAD" 또는 "SERIAL"
SERIAL_PORT = "COM3"    # SERIAL 모드일 때 아두이노 포트 (장치 관리자에서 확인)
BAUD = 9600
CAMERA_INDEX = 0        # 웹캠 번호 (안 되면 1로)
HOLD_SEC = 1.0          # 같은 손동작을 이만큼 유지하면 입력
RELEASE_SEC = 0.5       # 입력 후 손을 이만큼 빼야 다음 입력 가능
# ────────────────────────

if MODE == "TINKERCAD":
    import pyautogui
    pyautogui.FAILSAFE = False
else:
    import serial
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
    time.sleep(2)  # 아두이노 리셋 대기


def send_digit(n):
    """숫자 하나를 팅커캐드 또는 아두이노로 보낸다."""
    if MODE == "TINKERCAD":
        pyautogui.write(str(n))
        pyautogui.press("enter")
    else:
        ser.write(str(n).encode())


def count_fingers(hand_landmarks, handedness_label):
    """펴진 손가락 개수(0~5)를 센다."""
    lm = hand_landmarks.landmark
    count = 0

    # 엄지: 손 방향에 따라 x 좌표로 판단 (화면을 좌우반전해서 보여주므로 라벨 기준)
    if handedness_label == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:
        if lm[4].x > lm[3].x:
            count += 1

    # 검지~새끼: 손가락 끝이 두 번째 마디보다 위에 있으면 펴진 것
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if lm[tip].y < lm[pip].y:
            count += 1
    return count


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. CAMERA_INDEX를 1로 바꿔 보세요.")
        return

    current = None          # 지금 보이는 손가락 개수
    since = 0.0             # 그 개수가 유지되기 시작한 시각
    armed = True            # True면 입력 가능 상태
    no_hand_since = time.time()
    entered = []            # 화면 표시용 입력 기록

    print("시작합니다. 팅커캐드 시리얼 모니터 입력칸을 클릭해 두세요.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # 거울처럼 좌우반전
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        now = time.time()

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            label = result.multi_handedness[0].classification[0].label
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            n = count_fingers(hand, label)
            no_hand_since = None

            if n != current:
                current, since = n, now

            held = now - since
            if armed and held >= HOLD_SEC:
                send_digit(current)
                entered.append(str(current))
                if len(entered) > 4:
                    entered = entered[-4:]
                print("입력:", current)
                armed = False   # 손을 뺄 때까지 다시 입력 안 함

            # 진행 바 (얼마나 유지했는지)
            ratio = min(held / HOLD_SEC, 1.0) if armed else 1.0
            cv2.rectangle(frame, (20, 90), (20 + int(200 * ratio), 110),
                          (0, 200, 0) if armed else (120, 120, 120), -1)
            cv2.putText(frame, f"Fingers: {current}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            if not armed:
                cv2.putText(frame, "Sent! Remove hand", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            current = None
            if no_hand_since is None:
                no_hand_since = now
            if not armed and now - no_hand_since >= RELEASE_SEC:
                armed = True    # 손을 뺐으니 다음 숫자 입력 가능
            cv2.putText(frame, "Show your hand", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame, "Entered: " + " ".join(entered), (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow("Gesture Door Lock (q: quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
