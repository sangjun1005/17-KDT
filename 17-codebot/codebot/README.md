# 코드봇 · 스토리봇 웹서비스

FastAPI + React 서비스입니다. 기존 코드봇을 유지하고 회원가입·로그인·회원정보 수정, 스토리 생성 연결, 기록과 좋아요 기능을 추가했습니다. 화면은 네이버의 초록색·밝은 카드·회원 영역 배치를 참고했습니다.

## 현재 스토리봇 모델 상태

**기존 `../storybot/model_pretrain.pt`, `model_iter_500.pt`, `model_iter_5000.pt` 모두 39개 텐서에 NaN 또는 무한대 값이 있습니다. 첫 추론부터 정상 확률을 만들 수 없어 현재 실제 스토리 생성은 불가능합니다.** 원본 모델을 수정하거나 다른 모델로 대체하지 않았습니다.

스토리 생성 요청에는 503과 정상 모델이 필요하다는 안내를 반환하며 기록을 저장하지 않습니다. 정상 스토리봇 체크포인트 확보 또는 재학습은 사용자 결정 대기 사항입니다. 회원·기록 관리·좋아요·기존 코드봇 기능은 사용 가능합니다.

## 실행

PowerShell에서 저장소 루트 `17-codebot` 폴더를 기준으로 실행합니다. 현재 PC에는 프로젝트 전용 `.venv`와 프론트엔드 의존성이 설치되어 있습니다.

```powershell
npm.cmd --prefix codebot/frontend run build
.\.venv\Scripts\python.exe -m uvicorn codebot.backend.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 <http://127.0.0.1:8000>을 엽니다. React 빌드 결과를 FastAPI가 함께 제공합니다. 종료는 실행 터미널에서 `Ctrl+C`를 누릅니다. 생성 잠금과 모델을 공유하도록 **Uvicorn 작업자 1개**로 실행합니다. 공개 배포는 포함하지 않습니다.

다른 환경에서는 Python 3.12, Node.js 22.12 이상과 현재 프로젝트 및 형제 `storybot` 폴더가 필요합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r codebot/backend/requirements.txt
npm.cmd --prefix codebot/frontend ci
npm.cmd --prefix codebot/frontend run build
.\.venv\Scripts\python.exe -m uvicorn codebot.backend.main:app --host 127.0.0.1 --port 8000
```

현재 PC의 가상환경은 Codex에 포함된 Python 3.12와 CPU용 PyTorch를 사용합니다. 실행 장치는 기존 `get_device()`로 선택합니다.

프론트엔드 개발 시 백엔드를 실행한 상태에서 다른 터미널에 `npm.cmd --prefix codebot/frontend run dev`를 실행합니다. <http://127.0.0.1:5173>의 Vite 개발 화면이 `/api` 요청을 백엔드에 전달합니다.

## 이용 방법

1. 아이디·비밀번호·닉네임으로 회원가입합니다. 가입 후 로그인 상태가 됩니다. 아이디는 영문·숫자·밑줄 3~30자, 비밀번호는 8~128자, 닉네임은 1~30자입니다.
2. 로그인은 최대 7일 유지되며 새로고침이나 서버 재시작 후에도 유효한 토큰과 세션으로 복원됩니다. 로그아웃하면 해당 토큰은 서버에서도 무효화됩니다.
3. 회원정보 수정에서 닉네임을 변경할 수 있습니다. 비밀번호 변경에는 현재 비밀번호가 필요하며 다른 기기의 로그인도 종료됩니다. 아이디는 변경하지 않습니다.
4. 스토리봇은 시작 문장에서 이어지는 전체 이야기를 자동 저장하도록 연결되어 있습니다. 현재 모델 오류가 해결되어야 실제 생성·자동 저장을 사용할 수 있습니다.
5. 스토리 기록은 로그인한 전체 회원에게 공개하며 최신순으로 페이지당 10개씩 표시합니다. 내 기록에는 자신의 스토리만 표시합니다.
6. 자신의 스토리 제목·내용만 수정 또는 삭제할 수 있습니다. 서버에서도 작성자 권한을 확인합니다. 삭제하면 연결된 좋아요도 제거합니다.
7. 좋아요는 회원당 스토리 하나에 한 번씩 적용하며 다시 누르면 취소됩니다.
8. 코드봇은 기존과 같이 코드의 뒷부분만 생성·표시합니다. 코드 생성도 로그인한 회원만 사용하며 코드 기록은 저장하지 않습니다.

## 모델 재사용과 입력 한도

- 코드봇: 기존 `model.py`, `tokenizer.py`, `model_pretrain.pt`, `merge_rules.pkl`을 사용합니다.
- 스토리봇: `../storybot/model.py`, `tokenizer.py`, `model_pretrain.pt`, `merge_rules.pkl`을 직접 불러옵니다. 원본 파일을 복사·수정하지 않습니다.
- 스토리봇 원본 `utils.py`에는 `generated_ids.tolist` 호출 누락이 있습니다. 사용자 확인에 따라 코드봇의 기존 `utils.generate()`를 재사용합니다. 캐시 없는 추론으로 스토리봇의 RoPE 문맥도 256토큰 이내로 유지합니다.
- 각 봇의 기존 BPE 토크나이저로 입력을 검증합니다. 입력 최대 256토큰, 생성 최대 200토큰, `temperature=1.0`입니다. 문자 수 제한으로 대신하지 않습니다.
- 입력의 공백·들여쓰기·줄바꿈을 보존합니다. 생성이 길어지면 최근 256토큰의 문맥만 유지하며 종료 토큰이 먼저 나오면 짧게 종료합니다.
- 생성은 두 봇을 합쳐 한 번에 1건만 처리합니다. 다른 생성 요청에는 503을 반환합니다. 입력·생성 내용을 실행하지 않습니다.

## 데이터와 로그인

서버 시작 시 `backend/data/storybot.sqlite3`와 `backend/data/jwt.key`를 만듭니다. 회원·세션·스토리·좋아요는 SQLite에 저장하며 비밀번호는 Argon2 해시로 저장합니다. JWT 서명 키는 재시작 후에도 유지합니다. 이 폴더는 Git에서 제외되며 테스트는 별도 임시 DB를 사용합니다.

토큰은 `HttpOnly`, `SameSite=Strict`, 7일 만료 쿠키로 전달합니다. HTTPS에서는 `Secure`도 설정합니다. JWT 서명·만료·발급자·대상과 서버의 세션을 함께 확인합니다. 토큰을 브라우저 localStorage에 저장하지 않습니다. 다른 출처의 변경 요청은 거절합니다.

## API

요청·응답은 JSON입니다. 인증은 쿠키로 처리합니다.

| 메서드 | 경로 | 동작 |
| --- | --- | --- |
| POST | `/api/auth/signup` | `{username, password, nickname}` 회원가입·로그인 |
| POST | `/api/auth/login` | `{username, password}` 로그인 |
| GET | `/api/auth/me` | 로그인 복원·현재 회원 조회 |
| PATCH | `/api/auth/me` | `{nickname, current_password?, new_password?}` 회원정보 수정 |
| POST | `/api/auth/logout` | 해당 세션 무효화·쿠키 삭제 |
| POST | `/api/generate` | `{code}` → `{completion}` 코드 이어 쓰기 |
| POST | `/api/stories/generate` | `{prompt}` → 저장된 스토리, 성공 시 201 |
| GET | `/api/stories?page=1&mine=false` | `{items, page, page_size, total, total_pages}` 기록 조회 |
| PATCH | `/api/stories/{id}` | `{title, content}` 작성자만 수정 |
| DELETE | `/api/stories/{id}` | 작성자만 삭제 |
| PUT | `/api/stories/{id}/like` | 중복 없는 좋아요 추가 |
| DELETE | `/api/stories/{id}/like` | 현재 회원의 좋아요 취소 |

스토리에는 `id`, `user_id`, `author`, `title`, `prompt`, `content`, `created_at`, `updated_at`, `like_count`, `liked`, `is_owner`가 포함됩니다. 제목은 최초 입력 첫 줄에서 만들며 작성자가 수정할 수 있습니다. 비회원은 가입·로그인과 초기 화면만 이용합니다.

401은 로그인 필요·만료, 403은 권한·요청 출처 오류, 404는 없는 스토리, 409는 아이디 중복, 422는 입력 오류, 503은 생성 중 또는 비정상 스토리 모델, 500은 기타 생성 실패입니다. 토큰 초과 오류에는 실제 토큰 수와 허용 한도를 포함합니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
npm.cmd --prefix frontend run build
```

2026-10-01: **30개 테스트 중 29개 통과, 실제 스토리 생성 1개는 비정상 원본 가중치 때문에 건너뜀.** 기존 코드봇은 실제 모델 결과 일치를 확인했습니다. 스토리 저장·기록·권한·좋아요 테스트는 임시 DB와 명시적인 생성 대체 함수를 사용하므로 실제 스토리 생성 성공을 의미하지 않습니다.

회원가입·비밀번호 해시·중복 아이디·JWT 쿠키·로그인 복원·토큰 변조/만료/로그아웃 후 재사용 거절·회원정보 수정·비밀번호 변경 시 세션 무효화·서버 재시작 후 데이터 유지·256/257토큰 경계·실패 시 기록 미생성·페이지네이션·타인 수정/삭제 거절·좋아요 중복 방지/취소·삭제 시 좋아요 정리를 검증합니다.

설계 참고: [FastAPI JWT·비밀번호 해시](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/), [PyJWT 검증](https://pyjwt.readthedocs.io/en/latest/api.html), [Starlette 쿠키](https://starlette.dev/responses/), [Vite 프록시](https://vite.dev/config/server-options#server-proxy), [NAVER](https://www.naver.com/).
