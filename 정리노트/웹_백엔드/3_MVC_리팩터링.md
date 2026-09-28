# 3. MVC 리팩터링

**실습 폴더**: `17-todos/06_mvc`

2장의 `05_foreign`은 `main.py` 한 파일에 7,134바이트가 들어 있었다. 여기서는 **기능은 하나도 바꾸지 않고 파일만 나눈다.**

## 3-1. 나누기 전과 후

| | `05_foreign` | `06_mvc` |
| --- | --- | --- |
| 파일 수 | `main.py` 1개 | 6개 |
| `main.py` 크기 | 7,134 B | **229 B** |

```
06_mvc/
├── main.py          229 B   앱 조립만
├── database.py      252 B   engine, Base
├── models.py        559 B   SQLAlchemy 테이블
├── schemas.py       365 B   Pydantic 모델
├── dependencies.py  740 B   비밀번호 함수, get_db
├── controllers.py  5380 B   모든 라우트
└── templates/               (05_foreign과 동일)
```

**합계는 거의 같다.** 코드가 줄어든 게 아니라 **놓일 자리가 생긴 것**이다.

---

## 3-2. 파일 여섯 개

### `main.py` — 조립만

```python
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from controllers import router

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="secret-key")
app.include_router(router)
```

**진입점에는 "무엇을 조립하는지"만 남는다.** 앱이 아무리 커져도 이 파일은 이 크기를 유지한다.

### `database.py` — 연결

```python
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
Base = declarative_base()
```

**`engine`과 `Base`가 여기 하나만 있다는 것이 중요하다.** 여러 곳에서 각자 `create_engine`을 부르면 커넥션 풀이 여러 개 생긴다.

### `models.py` — 테이블

```python
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from database import Base

class Memo(Base):
    __tablename__ = "memos"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    content = Column(String)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
```

`from database import Base` — **DB 설정을 가져다 쓸 뿐 만들지 않는다.** 의존 방향이 한쪽이다.

### `schemas.py` — 요청 형태

```python
from pydantic import BaseModel
from typing import Optional

class MemoCreate(BaseModel):
    title: str
    content: str

class MemoUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str
```

**`models.py`와 `schemas.py`가 나뉜 것이 이 구조의 요점 중 하나다.**

| | `models.py` | `schemas.py` |
| --- | --- | --- |
| 무엇 | DB에 **저장되는** 모양 | 네트워크로 **오가는** 모양 |
| 라이브러리 | SQLAlchemy | Pydantic |
| 예 | `hashed_password` | `password` (평문) |

`UserCreate`에는 `password`가 있고 `User`에는 `hashed_password`가 있다. **모양이 다르기 때문에 파일도 다르다.**

### `dependencies.py` — 재사용 부품

```python
import bcrypt
from sqlalchemy.orm import Session
from database import engine

def get_password_hash(password: str): ...
def verify_password(plain_password: str, hashed_password: str): ...

def get_db():
    db = Session(bind=engine)
    try:
        yield db
    finally:
        db.close()
```

**어느 컨트롤러에서든 쓰이는 것들**이 모인다.

### `controllers.py` — 라우트 전부

```python
from fastapi import Request, Depends, HTTPException, APIRouter
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from schemas import UserCreate, UserLogin, MemoCreate, MemoUpdate
from sqlalchemy.orm import Session
from models import User, Memo
from dependencies import get_password_hash, verify_password, get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request, "home.html")
```

**`@app.get`이 `@router.get`으로 바뀐 것이 전부다.** 본문은 `05_foreign`과 동일하다.

---

## 3-3. `APIRouter` — 이 리팩터링을 가능하게 한 것

```python
# controllers.py
router = APIRouter()

@router.get("/")
def read_root(request: Request): ...
```

```python
# main.py
app.include_router(router)
```

**`APIRouter`는 "아직 앱에 붙지 않은 라우트 묶음"이다.**

`@app.get`을 쓰려면 `app` 객체가 필요하고, `app`은 `main.py`에 있다. 라우트를 다른 파일로 옮기려면 `main`을 import해야 하는데, `main`도 `controllers`를 import하므로 **순환 참조**가 된다.

`APIRouter`가 이 고리를 끊는다. `controllers.py`는 `main.py`를 몰라도 되고, 화살표가 한 방향으로만 흐른다.

```
main.py ──> controllers.py ──> schemas.py
                │                  
                ├──> models.py ──> database.py
                └──> dependencies.py ──> database.py
```

### `include_router`의 옵션

이 프로젝트에서는 쓰지 않지만, 실무에서 자주 붙는 인자들이다.

```python
app.include_router(router, prefix="/api/v1", tags=["memos"])
```

| 인자 | 효과 |
| --- | --- |
| `prefix` | 모든 경로 앞에 붙는다 (`/memos` → `/api/v1/memos`) |
| `tags` | `/docs`에서 그룹으로 묶인다 |
| `dependencies` | 이 라우터의 **모든** 경로에 공통 의존성 적용 |

**`dependencies=`를 쓰면 2장에서 본 "여섯 줄 반복"을 없앨 수 있다.** 로그인 확인을 의존성 함수 하나로 빼고 라우터에 걸면 된다.

---

## 3-4. MVC라는 이름

| MVC | 이 프로젝트 | 담당 |
| --- | --- | --- |
| **M**odel | `models.py` (+ `schemas.py`) | 데이터의 모양 |
| **V**iew | `templates/*.html` | 화면 |
| **C**ontroller | `controllers.py` | 요청을 받아 모델을 조작하고 뷰를 고름 |

`database.py`와 `dependencies.py`는 MVC 삼각형에 직접 들어가지 않는 **기반 계층**이다. 실무의 계층 구조는 보통 여기서 한 겹 더 나뉜다.

```
controllers (HTTP)  →  services (업무 로직)  →  repositories (DB 접근)  →  models
```

이 프로젝트는 `controllers.py`가 HTTP 처리와 DB 쿼리를 함께 한다. 규모가 커지면 중간에 `services`를 넣어 **"컨트롤러는 SQLAlchemy를 모르게"** 만든다.

---

## 3-5. 나눠서 얻은 것 / 잃은 것

| 얻은 것 | 잃은 것 |
| --- | --- |
| 고칠 곳을 파일 이름으로 찾는다 | 파일 사이를 오가야 한다 |
| 여러 사람이 다른 파일을 동시에 수정 → 충돌이 줄어든다 | import 구문이 길어진다 |
| 테스트에서 일부만 import 가능 | 처음 읽는 사람은 흐름 파악이 더 걸린다 |
| 라우터를 기능별로 더 쪼갤 수 있다 | |

**"파일 하나로 충분한 크기"에서는 나누는 게 손해다.** `01_temp`(260 B)를 6개로 쪼개면 우스운 일이 된다.
`05_foreign` 정도에서 슬슬 불편해지고, 거기서 나눈다 — **이 폴더들의 순서 자체가 그 판단 시점을 보여 준다.**

---

## 3-6. 옮기면서 남은 문제들

**기능을 그대로 옮겼으므로 2장의 버그도 그대로 왔다.**

> **주의 — 그대로 남은 것**
> - 로그인 실패 시 `return`이 없어 `null`이 200으로 나간다
> - `secret_key="secret-key"` 하드코딩
> - `signup`이 `hashed_password`가 든 `User` 객체를 그대로 반환
> - `print(request.session)` 디버깅 출력

> **주의 — `Base.metadata.create_all(bind=engine)`이 사라졌다**
> `05_foreign`에는 있었지만 `06_mvc` 어느 파일에도 없다.
> 앞 단계에서 이미 테이블이 만들어진 DB를 쓰면 문제없지만, **빈 DB에서 처음 실행하면 테이블이 없어 실패한다.**
> `main.py`에서 `models`를 import한 뒤 `Base.metadata.create_all(bind=engine)`을 불러 주면 된다. (import 순서 주의 — 모델 클래스가 먼저 정의돼야 `Base`가 테이블을 안다.)

> **주의 — `models.py`의 `create_engine` import**
> `from sqlalchemy import create_engine, Column, ...`에서 `create_engine`은 이 파일에서 쓰이지 않는다. 복사하면서 남은 것이다.

---

## 이 장 정리

### 한 줄 요약

**`APIRouter`가 라우트를 `main.py`에서 떼어낼 수 있게 해 주고, 그 덕분에 앱을 책임별 파일로 나눌 수 있다.**

### 파일 배치 기준

| 여기에 둔다 | 무엇 |
| --- | --- |
| `database.py` | engine, Base — **한 번만** |
| `models.py` | DB 테이블 = SQLAlchemy |
| `schemas.py` | 요청/응답 = Pydantic |
| `dependencies.py` | 여러 곳에서 쓰는 함수 |
| `controllers.py` | `@router.*` |
| `main.py` | 조립 |

### 의존 방향

```
main → controllers → {schemas, models, dependencies} → database
```

**화살표가 거꾸로 가면 순환 참조다.** `database.py`가 `models.py`를 import하면 안 된다.

### 자주 틀리는 것

- `controllers.py`에서 `@app.get`을 그대로 쓴다 → `app`이 없어 NameError
- `main.py`에서 `include_router`를 빠뜨린다 → 모든 경로가 404
- `models.py`를 아무 데서도 import하지 않는다 → `Base`가 테이블을 모른다
- 리팩터링하면서 기능도 같이 고친다 → **뭐가 깨졌는지 알 수 없어진다.** 구조 변경과 기능 변경은 따로
