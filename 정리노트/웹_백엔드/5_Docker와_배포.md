# 5. Docker와 배포

**실습 폴더**: `17-aws-docker`

만든 앱을 **컨테이너에 담아 nginx 뒤에 두는** 구성이다. 실어 나르는 앱은 천안시 읍면동 의료취약지역 지도(FastAPI + 정적 프런트엔드)다.

```
17-aws-docker/
├── docker-compose.yml     서비스 두 개 정의
├── .env                   DATABASE_URL (커밋되지 않음)
├── nginx/nginx.conf       리버스 프록시 설정
└── backend/
    ├── Dockerfile
    ├── pyproject.toml, uv.lock, requirements.txt
    ├── main.py            진입점
    ├── app/main.py        실제 FastAPI 앱
    ├── app/static/        빌드된 프런트엔드
    ├── frontend_source/   원본 React/Next 소스
    ├── build_frontend.js
    └── run.bat, run.sh
```

---

## 5-1. 담을 앱

```python
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="천안시 읍면동 의료취약지역 지도",
    description="기존 React 지도 화면과 기능을 그대로 제공하는 FastAPI 애플리케이션",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

**`FastAPI(title=..., description=..., version=...)`는 `/docs` 상단에 표시되는 메타데이터다.**

**`Path(__file__).resolve().parent`** — 실행 위치와 무관하게 이 파일 기준 경로를 얻는다. 컨테이너 안에서는 작업 디렉터리가 달라지므로 **상대 경로 대신 이 방식을 써야 한다.**

`app.mount()`는 **하위 앱을 통째로 붙이는 것**이다. `/static/...`으로 오는 요청은 FastAPI 라우팅을 거치지 않고 `StaticFiles`가 직접 파일을 내준다.

### SPA 라우팅 — 없는 경로는 전부 index.html

```python
@app.get("/api/health", tags=["system"])
def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok"})

@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/{path:path}", include_in_schema=False)
def serve_spa(path: str):
    requested = (STATIC_DIR / path).resolve()
    if STATIC_DIR.resolve() in requested.parents and requested.is_file():
        return FileResponse(requested)
    return FileResponse(STATIC_DIR / "index.html")
```

**`{path:path}`의 `:path`가 핵심이다.**

| 표기 | 매칭 |
| --- | --- |
| `{path}` | 슬래시가 **없는** 한 조각 (`/about`) |
| `{path:path}` | **슬래시 포함 전체** (`/a/b/c`) |

React 같은 SPA는 `/regions/cheonan` 같은 주소를 **브라우저 안에서** 처리한다. 사용자가 그 주소로 새로고침하면 서버에 요청이 가므로, 서버는 **뭐가 오든 `index.html`을 주고** 나머지는 프런트엔드에 맡긴다.

**`/{path:path}`는 반드시 맨 아래에 있어야 한다.** FastAPI는 등록 순서대로 검사하므로, 위에 있으면 `/api/health`까지 삼켜 버린다.

### 경로 탈출 막기

```python
requested = (STATIC_DIR / path).resolve()
if STATIC_DIR.resolve() in requested.parents and requested.is_file():
    return FileResponse(requested)
```

**이 `if` 문이 보안 장치다.**

`path`에 `../../etc/passwd`가 들어오면 `STATIC_DIR / path`는 static 바깥을 가리킨다. `resolve()`가 `..`을 실제로 계산해 정리하고, **`STATIC_DIR`이 그 조상 목록에 있는지 확인**해서 바깥이면 거부한다.

이 검사가 없으면 **경로 순회(path traversal)** 취약점이 된다. 서버의 아무 파일이나 읽힐 수 있다.

`include_in_schema=False`는 이 경로들을 `/docs`에서 감춘다. 정적 파일 서빙은 API 문서에 나올 필요가 없다.

### 진입점을 따로 둔 이유

```python
# backend/main.py
from app.main import app

__all__ = ["app"]
```

**세 글자짜리 파일이지만 역할이 있다.** `uvicorn main:app`이라는 관례적인 명령을 유지하면서, 실제 코드는 `app/` 패키지 안에 둘 수 있게 한다. Dockerfile·`run.sh`·README가 모두 `main:app`을 쓴다.

---

## 5-2. Dockerfile

```dockerfile
FROM python:3.13

# 공식 이미지에 uv가 없으므로 바이너리만 복사
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# 대문자 UV가 아니라 소문자 uv 명령 사용
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 한 줄씩

| 지시어 | 하는 일 |
| --- | --- |
| `FROM` | 기반 이미지 |
| `COPY --from=...` | **다른 이미지에서 파일만 꺼내 온다** (멀티스테이지) |
| `WORKDIR` | 이후 명령의 작업 디렉터리 (없으면 생성) |
| `COPY` | 호스트 → 이미지 |
| `ENV` | 환경변수 |
| `RUN` | **빌드할 때** 실행 |
| `CMD` | **컨테이너를 띄울 때** 실행 |

**`RUN`과 `CMD`의 차이가 가장 헷갈리는 지점이다.** `RUN`은 이미지를 만드는 도중 한 번, `CMD`는 만들어진 이미지를 실행할 때마다다.

### 레이어 캐시 — 이 파일의 진짜 요점

```dockerfile
COPY pyproject.toml uv.lock ./     ← 의존성 파일만 먼저
RUN uv sync --frozen ...            ← 설치
COPY . .                            ← 그다음 소스 전체
```

**왜 소스를 먼저 복사하지 않는가?**

Docker는 각 지시어의 결과를 레이어로 캐시하고, **입력이 바뀐 지점부터 다시 만든다.**

- `COPY . .`를 먼저 하면 → 소스 한 글자만 고쳐도 그 아래 `RUN uv sync`가 통째로 다시 돈다 (수 분)
- 지금 순서라면 → `pyproject.toml`이 그대로인 한 설치 레이어는 **캐시 재사용** (수 초)

**Dockerfile 작성에서 가장 효과가 큰 최적화다.** 언어와 무관하게 같은 원칙이 적용된다(`package.json` 먼저, `go.mod` 먼저 등).

### uv 관련

| | |
| --- | --- |
| `uv` | Rust로 만든 파이썬 패키지 관리자. pip보다 훨씬 빠르다 |
| `uv sync` | `uv.lock`대로 `.venv`를 만든다 |
| `--frozen` | **lock 파일을 갱신하지 않는다** — 빌드가 재현 가능해진다 |
| `--no-dev` | 개발용 의존성 제외 |
| `--no-install-project` | 프로젝트 자신은 빼고 **의존성만** — 캐시 전략과 짝 |
| `UV_COMPILE_BYTECODE=1` | `.pyc`를 미리 만들어 **첫 실행이 빨라진다** |
| `UV_LINK_MODE=copy` | 하드링크 대신 복사 — 도커 레이어 경계에서 안전 |

```dockerfile
ENV PATH="/app/.venv/bin:$PATH"
```

**`source .venv/bin/activate` 대신 쓰는 방법이다.** `RUN`은 매번 새 셸이라 activate가 유지되지 않는다. PATH 앞에 가상환경의 `bin`을 붙이면 `uvicorn`이 그 환경의 것으로 잡힌다.

### `--host 0.0.0.0`

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**컨테이너에서 가장 자주 나오는 실수가 이것이다.**

uvicorn의 기본 호스트는 `127.0.0.1`(자기 자신)이다. 컨테이너 안에서 `127.0.0.1`에 묶으면 **컨테이너 밖에서는 절대 접속할 수 없다.** `0.0.0.0`은 "모든 네트워크 인터페이스에서 받겠다"는 뜻이다.

`CMD`를 **리스트 형태**(exec form)로 쓴 것도 맞다. 문자열로 쓰면 셸을 거쳐 `docker stop`의 종료 신호가 프로세스에 제대로 전달되지 않는다.

**주의 — 파이썬 버전이 맞지 않는다**
`Dockerfile`은 `FROM python:3.13`인데 `pyproject.toml`은 이렇게 되어 있다.
```toml
requires-python = ">=3.14"
```
`uv sync --frozen`은 인터프리터가 요구 조건을 만족하지 않으면 실패한다. **`FROM python:3.14`로 올리거나 `requires-python`을 `>=3.13`으로 내려야** 빌드가 통과한다.
(`run.sh`/`requirements.txt` 경로로 로컬 실행할 때는 이 제약을 거치지 않아 드러나지 않는다.)

**참고 — 의존성 파일이 두 벌이다**
`pyproject.toml` + `uv.lock`(Docker용)과 `requirements.txt`(`run.sh`·README용)가 공존한다.
```
# requirements.txt
fastapi>=0.115,<1.0
uvicorn[standard]>=0.30,<1.0
# pyproject.toml
fastapi>=0.141.1
uvicorn[standard]>=0.52.1
```
**버전 범위가 서로 다르다.** 로컬에서 되던 게 컨테이너에서 안 될 수 있다. 하나를 기준으로 삼고 다른 하나는 거기서 생성하는 게 맞다.

---

## 5-3. docker compose — 두 컨테이너 묶기

```yaml
services:
  backend:
    build: ./backend
    env_file:
      - .env

  nginx:
    image: nginx
    ports:
      - 80:80
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
```

| 키 | 뜻 |
| --- | --- |
| `build: ./backend` | 그 폴더의 Dockerfile로 **직접 빌드** |
| `image: nginx` | Docker Hub에서 **받아다 쓴다** |
| `env_file` | `.env`의 값을 컨테이너 환경변수로 |
| `ports: 80:80` | `호스트:컨테이너` |
| `volumes` | 호스트 파일을 컨테이너 안에 **연결** |
| `depends_on` | 시작 **순서**만 보장 |

### 밖으로 열린 포트가 하나뿐이다

**`backend`에는 `ports`가 없다.** 의도된 것이다.

```
인터넷 ──80──> [nginx] ──backend:8000──> [backend]
                 ↑                          ↑
            유일한 입구              호스트에서 직접 접근 불가
```

compose가 만든 **내부 네트워크 안에서만** 8000번이 열려 있다. 공격 면적이 줄어든다.

### 서비스 이름이 곧 호스트 이름

```nginx
upstream fastapi {
    server backend:8000;
}
```

**`backend`는 IP가 아니라 compose 서비스 이름이다.** Docker의 내장 DNS가 이름을 컨테이너 IP로 바꿔 준다. 컨테이너가 재시작해 IP가 바뀌어도 이름은 그대로다.

### 설정 파일을 볼륨으로 넣기

```yaml
volumes:
  - ./nginx/nginx.conf:/etc/nginx/nginx.conf
```

**nginx 이미지를 다시 빌드하지 않고 설정만 갈아 끼운다.** 고친 뒤 `docker compose restart nginx`면 반영된다. 공식 이미지 + 설정 주입은 아주 흔한 패턴이다.

> **`depends_on`은 "준비될 때까지" 기다려 주지 않는다.**
> 컨테이너가 **시작**되면 다음으로 넘어간다. 앱이 아직 뜨는 중이면 nginx가 502를 낸다.
> 진짜로 기다리려면 `healthcheck` + `depends_on: condition: service_healthy`를 써야 한다.
> `app/main.py`에 있는 `/api/health`가 바로 그 용도의 엔드포인트다 — 연결만 하면 된다.

---

## 5-4. nginx 리버스 프록시

```nginx
user nginx;
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';
    access_log  /var/log/nginx/access.log  main;
    sendfile on;
    keepalive_timeout 65;

    upstream fastapi {
        server backend:8000;
    }

    server {
        listen 80;

        location / {
            proxy_pass         http://fastapi;
            proxy_redirect     off;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;
            proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header   X-Forwarded-Host $server_name;
        }
    }
}
```

### 왜 앞에 nginx를 두나

uvicorn만으로도 서비스는 된다. 그런데 앞에 nginx를 두면:

| 얻는 것 | 설명 |
| --- | --- |
| 정적 파일 | nginx가 훨씬 빠르게 내준다 |
| TLS 종료 | HTTPS 인증서를 여기서 처리 |
| 로드 밸런싱 | `upstream`에 서버를 여러 개 |
| 느린 클라이언트 흡수 | 애플리케이션 워커를 붙잡아 두지 않는다 |
| 접근 로그 · 요청 제한 | 한 군데서 |

### 프록시 헤더가 필요한 이유

프록시를 거치면 **백엔드가 보는 접속자는 nginx 컨테이너**다. 원래 정보를 헤더로 전달해 줘야 한다.

| 헤더 | 담기는 것 |
| --- | --- |
| `Host $host` | 사용자가 친 도메인 |
| `X-Real-IP $remote_addr` | 진짜 접속 IP |
| `X-Forwarded-For $proxy_add_x_forwarded_for` | **프록시 체인 전체** (기존 값에 이어 붙임) |
| `X-Forwarded-Host $server_name` | 원래 호스트 |

**`$proxy_add_x_forwarded_for`는 "기존 X-Forwarded-For + `,` + 현재 IP"다.** 프록시가 여러 단계여도 경로가 남는다.

**`X-Forwarded-Proto`가 빠져 있는 것**은 지금은 HTTP만 받으므로 괜찮지만, HTTPS를 붙이면 필요해진다. 없으면 앱이 리다이렉트 주소를 `http://`로 만들어 무한 루프가 생길 수 있다.

### 나머지 지시어

| 지시어 | 뜻 |
| --- | --- |
| `worker_processes auto` | CPU 코어 수만큼 워커 |
| `worker_connections 1024` | 워커당 동시 연결 (총합 = 워커 × 1024) |
| `include mime.types` | 확장자 → Content-Type 표 |
| `sendfile on` | **커널에서 바로 전송** — 사용자 공간 복사를 건너뛴다 (주석에 설명됨) |
| `keepalive_timeout 65` | 연결 재사용 시간 |
| `proxy_redirect off` | 백엔드가 준 Location 헤더를 고치지 않는다 |

**`upstream`으로 뺀 이유**는 서버를 늘리기 쉽기 때문이다.

```nginx
upstream fastapi {
    server backend1:8000;
    server backend2:8000;
}
```

이렇게만 바꾸면 라운드로빈 로드밸런싱이 된다.

---

## 5-5. 로컬 실행 경로

Docker 없이 돌릴 수단도 함께 들어 있다.

```bash
# uv
uv sync
uv run uvicorn main:app --reload

# 일반 python
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

```bash
# run.sh
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **`set -e`** — 한 줄이라도 실패하면 즉시 중단. 깨진 상태로 진행하지 않는다
- **`cd "$(dirname "$0")"`** — 어디서 실행하든 스크립트 위치로 이동
- **`python -m uvicorn`** — PATH의 `uvicorn`이 아니라 **이 파이썬의** uvicorn. 환경이 섞이지 않는다
- 여기서는 `--host 127.0.0.1` — 로컬이니 밖으로 열 이유가 없다 (컨테이너의 `0.0.0.0`과 대비)

윈도우용 `run.bat`과, 원본 TS/TSX에서 번들을 다시 만드는 `build_frontend.js`도 함께 있다. `frontend_source/`에는 원본 React/Next 소스가 통째로 들어 있다.

---

## 5-6. 배포 체크리스트

이 프로젝트를 실제로 올린다면 확인할 것들이다.

| 항목 | 현재 | 해야 할 것 |
| --- | --- | --- |
| 파이썬 버전 | Dockerfile 3.13 ↔ pyproject 3.14 | 맞추기 |
| 의존성 정의 | 두 벌이 서로 다름 | 하나로 통일 |
| `.env` | `.gitignore`에 있음 ✅ | 서버에는 별도 주입 |
| HTTPS | 없음 | 인증서 + `X-Forwarded-Proto` |
| 헬스체크 | `/api/health`는 있으나 미연결 | compose `healthcheck`에 연결 |
| 재시작 정책 | 없음 | `restart: unless-stopped` |
| 이미지 크기 | `python:3.13` (약 1GB) | `-slim` 태그 검토 |
| 실행 사용자 | root | 일반 사용자로 낮추기 |

---

## 이 장 정리

### 한 줄 요약

**Dockerfile은 "앱을 어떻게 이미지로 만드나", compose는 "그 이미지들을 어떻게 함께 띄우나", nginx는 "바깥 요청을 어떻게 받아 넘기나"를 담당한다.**

### 외워 둘 것

```dockerfile
FROM python:3.13
WORKDIR /app
COPY pyproject.toml uv.lock ./      # 의존성 먼저 (캐시!)
RUN uv sync --frozen --no-dev
COPY . .                            # 소스는 나중
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
```

```yaml
services:
  backend:
    build: ./backend      # 직접 빌드
  nginx:
    image: nginx          # 받아 쓰기
    ports: ["80:80"]      # 이것만 밖으로
```

```nginx
upstream fastapi { server backend:8000; }   # 서비스 이름 = 호스트 이름
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
```

### 명령어

```bash
docker compose up --build      # 빌드 후 실행
docker compose up -d           # 백그라운드
docker compose logs -f backend # 로그 추적
docker compose down            # 정지 + 정리
docker compose restart nginx   # 설정만 다시 읽히기
```

### 자주 틀리는 것

- `--host 0.0.0.0`을 빼먹는다 → **컨테이너 밖에서 접속 불가.** 1순위 실수
- `COPY . .`를 의존성 설치보다 먼저 한다 → 매번 전체 재빌드
- `depends_on`이 "준비 완료"를 보장한다고 믿는다 → 502
- 이미지 안에 `.env`를 넣는다 → 비밀이 이미지에 박힌다. `env_file`로 주입
- 프록시 뒤에서 `X-Forwarded-*`를 안 넘긴다 → 접속 IP가 전부 nginx로 기록
- `CMD`를 문자열로 쓴다 → 종료 신호가 전달되지 않아 `docker stop`이 느려진다
