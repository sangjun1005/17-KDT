# 5. 손가락 도어락 웹 UI (4일차)

**폴더**: `17-PLC/doorlock-ui/` — 10개 파일
**실행**: `실행.cmd` → `python server.py --open` → `http://127.0.0.1:8765`

> **3장의 `gesture_doorlock.py`가 숫자를 "어디로 보낼까" 문제를 풀었다.**
> 팅커캐드에 키보드로 치거나(TINKERCAD), 아두이노에 USB로 보내거나(SERIAL).
> 여기서는 **세 번째 답**을 만들었다 — **브라우저 화면의 도어락**으로 보낸다.
>
> 그리고 `.diy`나 `.xgwx`와 달리, 이건 **테스트가 붙어 있다.** 이 폴더에서 제일 중요한 점이다.

## 파일

| 파일 | 크기 | 역할 |
| --- | --- | --- |
| `server.py` | 7,678 B | 로컬 HTTP 서버 + 카메라·MediaPipe 처리 |
| `index.html` | 12,281 B | 화면 |
| `styles.css` | 14,884 B | 스타일 |
| `app.js` | 10,245 B | 화면 동작 |
| **`logic.mjs`** | 2,777 B | **도어락 상태기계 — 화면과 분리된 순수 로직** |
| **`test_logic.mjs`** | 3,800 B | 로직 테스트 (Node) |
| **`test_server.py`** | 2,600 B | 서버 테스트 (unittest) |
| `사용방법.md` | 2,703 B | 사용 설명 |
| `실행.cmd` | 232 B | 더블클릭 실행 (UTF-8 코드페이지 설정 포함) |
| `*.log` | — | 서버 로그 (저장소 제외 — 사용자명 경로가 찍힌다) |

## 테스트가 실제로 통과한다 (직접 돌려 확인)

```plain text
$ node test_logic.mjs
PASS: PIN 3152, wrong PIN, 5-second auto-close, manual close, input guard, clear
PASS: delete last digit, correct partial PIN, edit denied PIN, clear partial/denied PIN, no edits during opening
PASS: password validation, current PIN, confirmation, save failure, new PIN, old PIN rejected, saved PIN reload, change guard

$ python -m unittest test_server -v
test_camera_not_started_by_page ............ ok
test_cross_origin_camera_request_rejected .. ok
test_original_finger_rules_preserved ....... ok
test_private_source_not_served ............. ok
test_static_and_state ...................... ok
Ran 5 tests — OK
```

**이번 과정 전체에서 테스트가 붙은 실습물은 드물다.** 특히 서버 테스트 5개 중 3개가
**보안·개인정보**에 대한 것이다.

| 테스트 | 무엇을 막는가 |
| --- | --- |
| `test_camera_not_started_by_page` | 페이지를 열기만 해도 카메라가 켜지는 일 |
| `test_cross_origin_camera_request_rejected` | 다른 사이트가 카메라를 켜는 일 |
| `test_private_source_not_served` | 소스 파일이 웹으로 노출되는 일 |
| `test_original_finger_rules_preserved` | **원본 판정 규칙이 바뀌는 일** |

> **마지막 것이 특히 좋다.** "원본 `gesture_doorlock.py`를 수정하지 않았다"를
> 주석이 아니라 **테스트로 보증한다.** 원본이 바뀌면 테스트가 깨진다.

## `logic.mjs` — 화면과 분리된 상태기계

```plain text
locked ──4자리 일치──▶ opening ──1.3초──▶ open ──5초──▶ closing ──1.3초──▶ locked
   │                                        │
   └──불일치──▶ denied ──1.4초──▶ locked     └── close() 수동 닫기
```

| 상태 | 들어오는 조건 | 머무는 시간 |
| --- | --- | --- |
| `locked` | 시작 · 닫힘 완료 · 거부 후 | — |
| `opening` | 4자리가 비밀번호와 일치 | 1,300 ms |
| `open` | 열림 완료 | **5,000 ms** (자동 닫힘) |
| `closing` | 자동/수동 닫기 | 1,300 ms |
| `denied` | 4자리 불일치 | 1,400 ms |

### 잘 만든 곳 — 입력을 받는 상태를 좁혔다

```javascript
input(digit) {
  if (this.phase !== 'locked' || !Number.isInteger(digit) || digit < 0 || digit > 5) return;
  ...
}
clear()     { if (!['locked','denied'].includes(this.phase)) return; ... }
backspace() { if (!['locked','denied'].includes(this.phase) || this.entered.length === 0) return; ... }
```

- 입력은 **`locked`에서만** 받는다 — 문이 열리는 중에 번호를 더 누를 수 없다
- 삭제는 **`locked`와 `denied`에서만** — 틀렸을 때 고칠 수는 있다
- 숫자는 **0~5만** — 손가락으로 만들 수 있는 범위다

**1장 3절 인터록**이다. "동시에 서로 반대되는 동작을 방지한다."

### 비밀번호 변경 — 거절 사유를 문장으로 돌려준다

```javascript
if (this.phase !== 'locked')        return '문이 잠긴 상태에서 변경해 주세요.';
if (current !== this.password)      return '현재 비밀번호가 일치하지 않습니다.';
if (!/^[0-5]{4}$/.test(next))       return '새 비밀번호는 0~5 숫자로 된 네 자리여야 합니다.';
if (next !== confirmation)          return '새 비밀번호와 확인 입력이 다릅니다.';
if (next === current)               return '현재와 다른 비밀번호를 입력해 주세요.';
try { save(next); } catch { return '브라우저에 저장하지 못했습니다...'; }
```

**성공이면 빈 문자열, 실패면 이유.** 그리고 **저장이 실패하면 비밀번호를 바꾸지 않는다** —
`save()`를 `try`로 감싸고, 던지면 그 자리에서 돌아간다. 테스트에도
`save failure` 항목이 있고 `settings.password`가 `'3152'` 그대로인지 확인한다.

생성자도 방어한다 — `password`가 `0~5` 네 자리가 아니면 기본값 `'3152'`로 되돌린다.
테스트가 `'broken'`과 숫자 `1234`를 넣어 확인한다.

## 문서가 한계를 먼저 적어 놨다

`사용방법.md`에 있는 문장들이다.

> - 변경한 번호는 **현재 브라우저의 로컬 저장소에만** 저장됩니다. 사이트 데이터를 지우면 기본 번호로 돌아갑니다.
> - **실제 보안용 저장 방식이 아니므로 다른 서비스의 비밀번호를 사용하지 마세요.**
> - 이 화면은 **교육·발표용 시연**입니다. **손가락 개수는 본인 확인 수단이 아니며 실제 문을 제어하지 않습니다.**
> - 카메라 영상은 **외부 서비스로 전송하거나 파일에 저장하지 않습니다.**
> - 서버는 **로컬 PC에서만** 접근할 수 있습니다.

**PM 정리에서 배운 태도가 여기 그대로 있다** — 할 수 있는 것과 없는 것을 먼저 나눠 적는다.
"손가락 개수는 본인 확인 수단이 아니다"는 **4일차 계약서 2조**
("과거 모의 버 기준 10μm는 자동 적용하지 않는다")와 같은 종류의 문장이다.

## 세 갈래가 여기서 만난다

```plain text
            gesture_doorlock.py  (3장)
            웹캠 → MediaPipe → 손가락 0~5
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  TINKERCAD        SERIAL        doorlock-ui  (이 장)
  키보드 자동입력   USB 시리얼     로컬 HTTP + 화면
                       │
                       ▼
                 project4  (4장)
                 피코 LED 5개
```

**같은 인식기 하나에 출력이 셋이다.** 손가락 판정 규칙은 건드리지 않고
**보내는 곳만 바꿨다** — 그래서 `test_original_finger_rules_preserved`가 의미를 가진다.

## 확인 필요 / 주의

| 항목 | 내용 |
| --- | --- |
| **비밀번호가 평문으로 로컬 저장소에** | 문서가 스스로 적어 뒀다. 교육용이므로 그대로 두되 **실제 용도로 쓰지 말 것** |
| **`server-error.log`에 사용자명 경로** | `C:\Users\(사용자명)\AppData\...` 가 MediaPipe 경고에 찍혀 있다. **저장소에서 제외했다** (`17-PLC/doorlock-ui/*.log`) |
| **카메라 충돌** | 문서에 적힌 대로 `gesture_doorlock.py`나 화상회의가 켜져 있으면 카메라를 못 잡는다 |
| **동작 화면이 없다** | 로그를 보면 MediaPipe가 **두 번 초기화**됐으므로 최소 2회 실행된 것은 맞다. 다만 화면 캡처가 없어 **실제 인식 성능은 알 수 없다** |
| 의존 환경 | 문서 기준 Python 3.12 · OpenCV · MediaPipe 0.10.14. `requirements.txt`는 없다 |
| 포트 고정 | `127.0.0.1:8765` 고정이다. 이미 쓰는 포트면 충돌한다 |
