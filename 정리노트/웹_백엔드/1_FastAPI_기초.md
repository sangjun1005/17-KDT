# 1. FastAPI 기초

**실습 폴더**: `17-fast-api` — 모듈 10개, 각각 `main.py` 하나짜리 최소 예제

각 모듈은 `uvicorn main:app --reload`로 따로 띄워 본다. 하나씩 문법 조각을 익히는 구성이다.

| 모듈 | 주제 |
| --- | --- |
| `fastapi` | 가장 작은 앱 |
| `routing` | 경로 매개변수 · 쿼리 매개변수 |
| `typehint` | 타입 힌트로 검증하기 |
| `http_method` | GET/POST/PUT/DELETE |
| `pydantic` | 요청 본문 모델 |
| `response_model` | 응답 클래스 |
| `exception` | HTTPException |
| `template`, `template2` | Jinja2 렌더링 |
| `quiz` | 라우팅 연습 |

---

## 1-1. 가장 작은 앱

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {
        'message':'hello world'
    }
```

이 세 줄이 전부다.

- `app = FastAPI()` — 애플리케이션 객체
- `@app.get('/')` — "GET `/` 요청이 오면 아래 함수를 실행해라"
- `return {...}` — **딕셔너리를 그대로 반환하면 JSON으로 직렬화된다**

실행은 `uvicorn main:app --reload`. `main`은 파일명(`main.py`), `app`은 그 안의 변수 이름이다.

> **`/docs`가 공짜로 따라온다.** `http://127.0.0.1:8000/docs`에 들어가면 Swagger UI가 떠 있다. 코드에 쓴 타입 힌트를 읽어 자동으로 만든 것이다.

---

## 1-2. 경로 매개변수와 쿼리 매개변수

`routing/main.py`에 둘이 나란히 나온다.

### 경로 매개변수 — URL의 일부

```python
@app.get('/items/{item_id}')
def read_item(item_id):
    return {
        'item_id':item_id
    }

@app.get('/users/{user_id}/items/{item_name}')
def read_user(user_id, item_name):
    return{
        'user_id':user_id,
        'item_name':item_name
    }
```

**`{중괄호}` 안의 이름과 함수 매개변수 이름이 같아야 한다.** 이름이 연결 고리다.

`/users/kim/items/book` → `{"user_id": "kim", "item_name": "book"}`

### 쿼리 매개변수 — `?` 뒤

```python
@app.get('/itemname')
def read_name(skip,limit):
    return {
        'skip':skip,
        'limit':limit
    }

@app.get('/products/')
def read_prodducts(skip=0, limit=10):
    return {
        'skip':skip,
        'limit':limit
    }
```

**경로에 없는 매개변수는 자동으로 쿼리 매개변수가 된다.** 규칙은 이것 하나다.

| 함수 정의 | 요청 | 결과 |
| --- | --- | --- |
| `read_name(skip, limit)` | `/itemname` | **422 에러** — 기본값이 없으니 필수 |
| `read_name(skip, limit)` | `/itemname?skip=0&limit=5` | OK |
| `read_prodducts(skip=0, limit=10)` | `/products/` | 기본값 사용 |

**기본값이 있으면 선택, 없으면 필수.** 파이썬 함수의 기본값 규칙이 그대로 HTTP로 올라온 것이다.

> **주의 — 타입 힌트가 없다**
> 이 파일의 함수들은 매개변수에 타입이 없다. FastAPI는 타입이 없으면 **전부 문자열로 받는다.**
> `/items/42` → `{"item_id": "42"}` (숫자 42가 아니라 문자열 `"42"`)
> `/products/?skip=3` → `{"skip": "3"}` (기본값 `0`은 int인데 넘어온 값은 str — 타입이 섞인다)
> 다음 절의 `typehint` 모듈이 바로 이 문제를 고친다.

---

## 1-3. 타입 힌트가 곧 검증이다

```python
from fastapi import FastAPI, Query
from typing import List, Dict

app = FastAPI()

@app.get('/items/{item_id}')
def read_item(item_id:int):
    return {
        'item_id':item_id
    }
```

**`item_id:int` 하나를 붙였을 뿐인데 세 가지가 생긴다.**

1. `/items/42` → `42` (문자열이 아니라 정수로 변환)
2. `/items/abc` → **422 Unprocessable Entity** — 자동 거부
3. `/docs`에 "integer" 라고 표시

**이것이 FastAPI의 핵심 아이디어다.** 파이썬 타입 힌트를 장식이 아니라 **실제 검증 규칙**으로 쓴다.

### 기본값 있는 쿼리

```python
@app.get('/getdata')
def read_items(data:str = 'funcoding'):
    return {
        'data':data
    }
```

`/getdata` → `{"data": "funcoding"}`, `/getdata?data=hello` → `{"data": "hello"}`

### 리스트 쿼리 — `Query([])`가 필요한 이유

```python
@app.get('/product')
def read_product(q: List[int] = Query([])):
    return{
        'q':q
    }
```

`/product?q=1&q=2&q=3` → `{"q": [1, 2, 3]}`

> **`Query([])`를 빼면 안 된다.**
> `q: List[int] = []`라고만 쓰면 FastAPI는 "리스트니까 **요청 본문**이겠네"라고 해석한다.
> `Query(...)`가 "이건 쿼리 매개변수다"라고 명시해 주는 역할이다.
> 같은 자리에 `Path(...)`, `Header(...)`, `Cookie(...)`, `Body(...)`도 쓸 수 있다.

### 딕셔너리 본문

```python
@app.post('/product-item')
def read_product_item(item: Dict[str,int]):
    return item
```

`{"a": 1, "b": 2}`는 통과, `{"a": "x"}`는 422다. **값의 타입까지 검사한다.**

---

## 1-4. HTTP 메서드 네 가지

`http_method/main.py`가 CRUD 한 벌을 보여 준다.

```python
@app.get('/items/{item_id}')
def read_item(item_id: int):
    return {'item_id':item_id}

@app.post('/items/')
def create_item(item:dict):
    return {'item':item}

@app.put('/items/{item_id}')
def update_item(item_id: int, item: dict):
    return {
        'item_id':item_id,
        'updated_item':item
    }

@app.delete('/items/{item_id}')
def delete_item(item_id:int):
    return{
        'message':f"{item_id} 삭제가 되었습니다."
    }
```

| 메서드 | 뜻 | 본문 | 데코레이터 |
| --- | --- | --- | --- |
| GET | 읽기 | 없음 | `@app.get` |
| POST | 만들기 | 있음 | `@app.post` |
| PUT | 통째로 바꾸기 | 있음 | `@app.put` |
| DELETE | 지우기 | 없음 | `@app.delete` |

**`update_item`을 잘 보자.** `item_id: int`는 경로에서, `item: dict`는 본문에서 온다.
**단순 타입(int/str/float)은 경로·쿼리, 복합 타입(dict/list/BaseModel)은 본문** — 이게 FastAPI가 출처를 정하는 기본 규칙이다.

브라우저 주소창으로는 GET밖에 못 보낸다. 나머지는 `/docs`의 "Try it out" 버튼이나 Postman을 쓴다. (`17-fast-api/postman/` 폴더에 컬렉션 파일이 들어 있다.)

---

## 1-5. Pydantic — 본문에 이름을 붙이기

```python
from pydantic import BaseModel, Field
from typing import List, Union

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.post('/items/')
def create_item(item: Item):
    return {'item':item}
```

`dict` 대신 **클래스**를 쓰면 필드 이름과 타입이 확정된다. `/docs`에도 그 구조가 그대로 나온다.

### 필드 제약조건

```python
class Item(BaseModel):
    name: str = Field(..., title='Item Name', min_length=2, max_length=50)
    description: str = Field(None, description='The Description of the item', max_length=300)
    price: float = Field(..., gt=0, description='0보다 커야한다.')
```

| 인자 | 뜻 |
| --- | --- |
| `...` (Ellipsis) | **필수** — 기본값이 없다는 표시 |
| `None` | 선택 |
| `min_length` / `max_length` | 문자열 길이 |
| `gt` / `ge` / `lt` / `le` | 숫자 범위 (greater than 등) |
| `title` / `description` | `/docs`에 표시될 설명 |

`price=0`으로 보내면 `gt=0` 때문에 422다. **검증 로직을 함수 안에 `if`로 쓰지 않고 모델 선언으로 끝낸다.**

### 살아 있는 모델

파일에서 주석이 아닌 채로 남아 실제로 동작하는 것은 이것이다.

```python
class Item(BaseModel):
    name:str
    tags:List[str]
    variant: Union[int, str]


@app.post('/items/')
def create_item(item: Item):
    return {
        'item':item
    }
```

- `List[str]` — 문자열 배열
- `Union[int, str]` — **정수든 문자열이든 받는다** (파이썬 3.10+에서는 `int | str`)

> **경로는 반드시 `/`로 시작한다.**
> 슬래시 없이 `'items/'`로 쓰면 정상적인 URL로 그 엔드포인트에 도달할 수 없다.
> FastAPI가 에러를 내지 않으므로 **조용히 안 잡히는** 종류의 문제다.

---

## 1-6. 응답의 종류 고르기

기본은 JSON이지만, 다른 것도 돌려줄 수 있다.

```python
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse

@app.get('/redirect')
def read_redirect():
    return RedirectResponse(url='/html')

@app.get('/html',response_class=HTMLResponse)
def read_html():
    return '<h1>This is HTML</h1>'

@app.get('/json',response_class=JSONResponse)
def read_json():
    return {'msg':'This is JSON'}

@app.get('/text',response_class=PlainTextResponse)
def read_text():
    return 'This is Plain Text'
```

| 클래스 | Content-Type | 쓰임 |
| --- | --- | --- |
| `JSONResponse` | `application/json` | 기본값 |
| `HTMLResponse` | `text/html` | 브라우저가 렌더링 |
| `PlainTextResponse` | `text/plain` | 그냥 글자 |
| `RedirectResponse` | — | 다른 주소로 보냄 (기본 307) |

**`response_class=`와 `return RedirectResponse(...)`는 쓰는 자리가 다르다.**
전자는 데코레이터에 미리 선언하는 것이고, 후자는 함수 안에서 그때그때 반환하는 것이다.

`HTMLResponse`를 빼고 `'<h1>...'`을 반환하면 화면에 태그가 **글자 그대로** 보인다. JSON 문자열로 직렬화되기 때문이다.

### response_model — 응답도 검증한다

```python
class Item(BaseModel):
    name:str
    description:str=None
    prices:float

@app.get('/items/{item_id}', response_model=Item)
def read_item(item_id:int):
    items = get_item_from_db(item_id)
    return items
```

**`response_model`은 필터 역할을 한다.** DB에서 가져온 딕셔너리에 `password` 같은 필드가 섞여 있어도, 모델에 없는 필드는 **응답에서 잘려 나간다.** 실수로 민감한 값을 내보내는 것을 막는 장치다.

---

## 1-7. 예외 처리

```python
from fastapi import FastAPI, HTTPException

@app.get('/items/{item_id}')
def read_item(item_id: int):
    if item_id == 42:
        raise HTTPException(status_code=404, detail="Item Not Found")
    return {
        'item_id':item_id
    }
```

`/items/42` → `{"detail": "Item Not Found"}` + **404 상태 코드**

**`return`이 아니라 `raise`다.** 그냥 `return {"error": "..."}` 하면 상태 코드가 200이 되어 버린다. 클라이언트는 상태 코드로 성공/실패를 판단하므로 이 차이가 중요하다.

### 파이썬 예외를 HTTP 예외로 바꾸기

```python
@app.get('/items/{item_id}')
def read_item(item_id: int):
    try:
        if item_id < 0:
            raise ValueError('음수는 허용되지 않습니다.')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**이 패턴이 실무의 기본형이다.** 안쪽 로직은 평범한 파이썬 예외를 던지고, 바깥 경계에서 HTTP 상태 코드로 번역한다.

| 코드 | 언제 |
| --- | --- |
| 400 Bad Request | 요청이 잘못됨 |
| 401 Unauthorized | 로그인 안 됨 |
| 403 Forbidden | 로그인했지만 권한 없음 |
| 404 Not Found | 없음 |
| 422 Unprocessable Entity | **FastAPI가 타입 검증 실패 시 자동으로** |

---

## 1-8. Jinja2 템플릿

JSON이 아니라 HTML 페이지를 돌려주는 방법이다.

```python
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory='templates')

@app.get('/')
def read_root(request: Request):
    return templates.TemplateResponse(request,'index.html',{
        'username':'Daegu'
    })
```

```html
<html>
    <head>
        <title>FastAPI Template</title>
    </head>
    <body>
        <h1>Hello, FastAPI {{username}}</h1>
    </body>
</html>
```

→ 화면에 `Hello, FastAPI Daegu`

**세 가지를 지켜야 한다.**

1. `Jinja2Templates(directory='templates')` — 폴더 이름 등록
2. 함수에 `request: Request` 매개변수가 **반드시** 있어야 한다
3. 세 번째 인자로 준 딕셔너리의 키가 템플릿의 `{{ }}` 이름이 된다

변수가 필요 없으면 생략해도 된다.

```python
@app.get('/')
def read_root(request: Request):
    print(request)
    return templates.TemplateResponse(request, "index.html")
```

`template2`는 여기에 Tailwind CSS를 CDN으로 얹었다.

```html
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
...
<h1 class='font-bold text-[#063B00] text-2xl'>hello fastapi</h1>
```

`text-[#063B00]`처럼 **대괄호 안에 임의의 값**을 넣는 것이 Tailwind의 arbitrary value 문법이다.

> **`TemplateResponse`의 인자 순서**
> 예전 FastAPI는 `TemplateResponse("index.html", {"request": request, ...})` 형태였다.
> 최신 버전은 `TemplateResponse(request, "index.html", {...})`처럼 **request가 첫 번째**다.
> 이 폴더는 새 방식을 쓴다. 인터넷 예제를 볼 때 순서가 다르면 버전 차이다.

---

## 1-9. 라우팅 연습 (quiz)

```python
@app.get('/')
def read_root():
    return {
        "번호별 명언 모음집"
    }

@app.get('/1st')
def first():
    return "바쁘다 바빠 현대사회"
```

`/1st` ~ `/10th`까지 열 개의 경로를 만드는 연습이다.

> **주의 — 이건 딕셔너리가 아니라 집합(set)이다**
> `return {"번호별 명언 모음집"}` — 콜론(`:`)이 없으므로 **집합 리터럴**이다.
> FastAPI는 집합을 리스트로 직렬화하므로 `["번호별 명언 모음집"]`이 응답된다.
> 딕셔너리로 만들려면 `{"title": "번호별 명언 모음집"}`처럼 **키:값** 쌍이 필요하다.

각 함수가 문자열을 그대로 반환하는 것은 유효하다. 문자열도 JSON 값이므로 `"바쁘다 바빠 현대사회"`(따옴표 포함)가 응답된다.

---

## 이 장 정리

### 한 줄 요약

**FastAPI는 파이썬 타입 힌트를 읽어 요청 검증·형 변환·API 문서를 자동으로 만든다.** 데코레이터로 경로를, 타입으로 규칙을 쓴다.

### 매개변수 출처 정리표

| 함수 시그니처 | 어디서 오나 |
| --- | --- |
| 경로 `{name}`에 있는 이름 | 경로 매개변수 |
| 단순 타입 + 경로에 없음 | 쿼리 매개변수 |
| `BaseModel` / `dict` / `list` | 요청 본문 |
| `= Query(...)` | 쿼리 (명시) |
| `request: Request` | 요청 객체 자체 |
| `= Depends(...)` | 의존성 주입 (2장) |

### 외워 둘 것

```
app = FastAPI()                                   # 앱
@app.get('/path/{id}')                            # 라우팅
def f(id: int, q: str = None): ...                # 경로 + 쿼리
def f(item: Item): ...                            # 본문
raise HTTPException(status_code=404, detail="")   # 에러
return templates.TemplateResponse(request, "x.html", {...})   # HTML
uvicorn main:app --reload                         # 실행
```

### 자주 틀리는 것

- 타입 힌트를 빼먹는다 → 전부 문자열로 들어온다
- 리스트 쿼리에 `Query([])`를 안 쓴다 → 본문으로 해석되어 422
- 에러를 `return`한다 → 상태 코드가 200이 되어 클라이언트가 성공으로 판단
- 템플릿 함수에 `request: Request`를 안 넣는다 → 에러
- 경로 문자열의 맨 앞 `/`를 빠뜨린다 → 접근 불가
