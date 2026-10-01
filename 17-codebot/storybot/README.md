# 스토리봇 웹서비스

ch04에서 학습한 스토리봇(GPT)으로 이야기를 이어 쓰는 FastAPI + React 서비스입니다. 요구사항은 [prd.md](prd.md)에 있습니다.

- 회원가입·로그인·로그아웃·회원정보 수정 (JWT, HttpOnly 쿠키, 7일 유지)
- 스토리 생성 (입력 최대 56토큰, 생성 최대 200토큰)
- 스토리 기록 보기 (최신순, 페이지당 10개)
- 좋아요 / 내 스토리 제목·내용 수정·삭제
- 로그인하지 않으면 위 기능을 이용할 수 없음 (서버에서도 401)

## 실행

프로젝트 루트(`17-codebot`)의 `.venv`(CUDA PyTorch 포함)를 사용합니다. PowerShell에서 `storybot` 폴더 기준으로 실행합니다.

```powershell
cd storybot
npm.cmd --prefix frontend install      # 처음 한 번
npm.cmd --prefix frontend run build
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

브라우저에서 <http://127.0.0.1:8001> 을 엽니다. FastAPI가 React 빌드 결과를 함께 제공합니다. 생성은 한 번에 1건만 처리하므로 Uvicorn 작업자는 1개로 실행합니다. 기존 codebot 웹(8000)과 겹치지 않도록 8001 포트를 씁니다.

다른 PC에서는 Python 3.12와 Node.js 22 이상이 필요합니다. PyTorch를 설치한 뒤 `pip install -r backend/requirements.txt`로 나머지 패키지를 설치합니다.

프론트엔드 개발 중에는 백엔드를 띄운 상태에서 `npm.cmd --prefix frontend run dev`를 실행합니다(<http://127.0.0.1:5173>, `/api`는 8001로 전달).

## 기존 코드 사용

- `model.py`의 `GPT.load_from()`, `tokenizer.py`의 `BPETokenizer.load_from()`, `utils.py`의 `generate()`·`get_device()`를 그대로 import합니다.
- 모델 `model_pretrain.pt`, 토크나이저 `merge_rules.pkl`을 사용합니다.
- `utils.py`는 `generated_ids.tolist` → `generated_ids.tolist()` 괄호만 고쳤습니다.

## 입력 제한

`generate()`는 KV 캐시를 사용하고 RoPE 위치는 최대 256까지만 있습니다. 그래서 **입력 + 생성 ≤ 256토큰**이 되도록 입력 56토큰, 생성 200토큰으로 정했습니다. 입력 토큰 수는 서버가 실제 토크나이저로 계산합니다. 화면의 300자 제한은 대략적인 보조 장치입니다(영어는 약 4.1바이트/토큰).

## 데이터

처음 실행하면 `backend/data/storybot.sqlite3`(회원·세션·스토리·좋아요)와 `backend/data/jwt.key`(서명 키)가 만들어집니다. 비밀번호는 Argon2 해시로 저장합니다. 이 폴더는 Git에서 제외됩니다.

## 테스트

```powershell
cd storybot
..\.venv\Scripts\python.exe -m unittest backend.tests.test_api -v
```

임시 DB를 사용합니다. 실제 모델로 생성하는 테스트 1개 외에는 모델 호출을 대체해서 회원·권한·기록·좋아요·입력 제한을 검증합니다.

## API

| 메서드 | 경로 | 동작 |
| --- | --- | --- |
| POST | `/api/auth/signup` | `{username, password, nickname}` 회원가입·로그인 |
| POST | `/api/auth/login` | `{username, password}` 로그인 |
| GET / PATCH | `/api/auth/me` | 내 정보 조회 / `{nickname, current_password?, new_password?}` 수정 |
| POST | `/api/auth/logout` | 로그아웃 (서버 세션 삭제) |
| POST | `/api/stories/generate` | `{prompt}` → 생성 후 저장된 스토리 (201) |
| GET | `/api/stories?page=1` | `{items, page, page_size, total, total_pages}` |
| PATCH / DELETE | `/api/stories/{id}` | 작성자만 `{title, content}` 수정 / 삭제 |
| PUT / DELETE | `/api/stories/{id}/like` | 좋아요 / 취소 |
