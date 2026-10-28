"""Local-only HTML preview and webcam bridge. Does not send keys or drive hardware."""
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def count_fingers(hand_landmarks, handedness_label):
    # Same rules as ../gesture_doorlock.py (mirrored image).
    lm = hand_landmarks.landmark
    count = 0
    if handedness_label == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:
        if lm[4].x > lm[3].x:
            count += 1
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if lm[tip].y < lm[pip].y:
            count += 1
    return count


class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.jpeg = None
        self.state = dict(status="off", message="카메라가 꺼져 있습니다.",
                          fingers=None, progress=0, armed=True, sequence=0, events=[])

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return {**self.state, "events": list(self.state["events"])}

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.update(status="starting", message="카메라를 연결하고 있습니다.")
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=3)
        if not self.worker or not self.worker.is_alive():
            self.update(status="off", message="카메라가 꺼져 있습니다.",
                        fingers=None, progress=0, armed=True)

    def run(self):
        cap = hands = None
        try:
            import cv2
            import mediapipe as mp
            hands = mp.solutions.hands.Hands(
                max_num_hands=1, min_detection_confidence=0.7,
                min_tracking_confidence=0.6)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                raise RuntimeError("웹캠을 열 수 없습니다. 다른 앱의 카메라 사용을 종료해 주세요.")
            current, since, armed, no_hand_since = None, 0.0, True, None
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("카메라 영상을 읽지 못했습니다. 다시 연결해 주세요.")
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now = time.monotonic()
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    label = result.multi_handedness[0].classification[0].label
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                    n = count_fingers(hand, label)
                    no_hand_since = None
                    if n != current:
                        current, since = n, now
                    held = now - since
                    if armed and held >= 1.0:
                        with self.lock:
                            self.state["sequence"] += 1
                            self.state["events"].append(
                                {"sequence": self.state["sequence"], "digit": current})
                            self.state["events"] = self.state["events"][-64:]
                        armed = False
                    self.update(status="on", fingers=n, armed=armed,
                                progress=min(held, 1.0) if armed else 1,
                                message="손동작을 유지해 주세요." if armed else "손을 화면 밖으로 빼 주세요.")
                else:
                    current = None
                    if no_hand_since is None:
                        no_hand_since = now
                    if not armed and now - no_hand_since >= 0.5:
                        armed = True
                    self.update(status="on", fingers=None, progress=0,
                                armed=armed, message="카메라에 손을 보여 주세요.")
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                self.stop_event.wait(0.03)
        except Exception as exc:
            self.update(status="error", message=str(exc), fingers=None, progress=0)
        finally:
            if cap is not None:
                cap.release()
            if hands is not None:
                hands.close()
            if self.stop_event.is_set():
                self.update(status="off", message="카메라가 꺼져 있습니다.", fingers=None, progress=0)


camera = Camera()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, *_):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            return self.json_response(camera.snapshot())
        if path == "/api/frame.jpg":
            with camera.lock:
                jpeg = camera.jpeg
            if jpeg is None:
                self.send_error(404, "Camera not ready")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return
        if path not in ("/", "/index.html", "/styles.css", "/app.js", "/logic.mjs"):
            self.send_error(404)
            return
        super().do_GET()

    def do_POST(self):
        # Only same-origin UI requests may change webcam state.
        if self.headers.get("Origin") != "http://" + self.headers.get("Host", ""):
            return self.json_response({"error": "Origin not allowed"}, 403)
        if self.path == "/api/camera/start":
            camera.start()
        elif self.path == "/api/camera/stop":
            camera.stop()
        else:
            return self.json_response({"error": "Not found"}, 404)
        self.json_response(camera.snapshot())


if __name__ == "__main__":
    import sys
    import webbrowser
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Doorlock UI: http://127.0.0.1:8765  (Ctrl+C to stop)", flush=True)
    if "--open" in sys.argv:
        webbrowser.open("http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        server.server_close()
