# 4. Flask로 만든 데이터 웹앱

**실습 폴더**: `17-apartment`(공공 API), `17-assignment`(크롤링 + 스케줄러)

1~3장의 FastAPI와 달리 여기서는 **Flask**를 쓴다. 두 프로젝트 모두 "바깥에서 데이터를 가져와 웹으로 보여 준다"는 같은 구조다.

---

## 4-1. Flask의 최소 형태

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
```

**FastAPI와 비교하면 차이가 분명하다.**

| | FastAPI | Flask |
| --- | --- | --- |
| 라우팅 | `@app.get('/')` | `@app.route('/')` (기본 GET) |
| 다른 메서드 | `@app.post`, `@app.put`... | `@app.route('/', methods=['POST'])` |
| 템플릿 | `Jinja2Templates(directory=...)` 등록 필요 | `render_template` **바로 사용** |
| 템플릿 폴더 | 직접 지정 | `templates/` **고정 규칙** |
| 실행 | `uvicorn main:app --reload` | `app.run(debug=True)` — 파일 안에서 |
| 요청 객체 | 함수 인자 `request: Request` | 전역 `request` import |

**`Flask(__name__)`의 `__name__`은 현재 모듈 이름**이다. Flask가 이걸 기준으로 `templates/`와 `static/` 폴더 위치를 찾는다.

**`debug=True`는 개발 전용이다.** 코드를 고치면 자동 재시작하고, 에러 화면에 스택 트레이스와 **대화형 디버거 콘솔**이 뜬다. 운영에 켜 두면 서버에서 임의 코드를 실행당할 수 있다.

---

## 4-2. `17-apartment` — 공공데이터 API 호출

### API 키를 환경변수로

```python
import requests
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

load_dotenv()
SERVICEKEY = os.getenv('SERVICEKEY')
```

2장의 `DATABASE_URL`과 같은 패턴이다. **공공데이터포털 인증키를 코드에 쓰지 않는다.**

### 요청과 XML 파싱

```python
def fetch_apt_trade(LAWD_CD, DEAL_YMD):
    serviceKey = SERVICEKEY

    URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"

    params = {
        'LAWD_CD': LAWD_CD,
        'DEAL_YMD': DEAL_YMD,
        'serviceKey': serviceKey
    }

    response = requests.get(URL, params=params)

    root = ET.fromstring(response.content)
    items = root.findall('./body/items/item')
    results = []

    for item in items:
        row = {}
        for child in item:
            row[child.tag] = child.text
        results.append(row)

    return results
```

| 매개변수 | 뜻 | 예 |
| --- | --- | --- |
| `LAWD_CD` | 법정동 코드 **앞 5자리** (시군구) | `27230` (대구 달서구) |
| `DEAL_YMD` | 계약 연월 `YYYYMM` | `202606` |
| `serviceKey` | 인증키 | `.env` |

**국토교통부 아파트 매매 실거래가 API**다. 주석에 예시 값이 남아 있다.

### XML 다루기 — `ElementTree`

이 과목에서 처음 나오는 형식이다. 판다스 과목에서 다룬 JSON·CSV·HTML과 달리 **공공데이터포털 API는 기본이 XML**이다.

```xml
<response>
  <body>
    <items>
      <item>
        <거래금액>52,000</거래금액>
        <아파트>OO아파트</아파트>
        ...
      </item>
```

| 코드 | 하는 일 |
| --- | --- |
| `ET.fromstring(response.content)` | 바이트열을 트리로 파싱 |
| `root.findall('./body/items/item')` | 경로에 맞는 노드를 **전부** |
| `for child in item` | 자식 태그를 순회 |
| `child.tag`, `child.text` | 태그 이름 / 안의 글자 |

**`'./body/items/item'`은 XPath다.** `.`은 현재 노드(root), `/`로 단계를 내려간다.

**`for child in item`으로 딕셔너리를 만드는 것이 이 코드의 좋은 점이다.** 필드 이름을 하드코딩하지 않으므로 API 응답에 항목이 늘어도 자동으로 따라간다. 결과가 `list[dict]`이므로 `pd.DataFrame(results)`에 바로 넣을 수 있다.

> **`response.content` vs `response.text`**
> `content`는 바이트, `text`는 헤더를 보고 디코딩한 문자열이다.
> **XML은 문서 첫 줄에 자기 인코딩을 써 두므로 바이트를 그대로 넘기는 것이 안전하다.** 여기서 `content`를 쓴 것이 맞다.

### 화면

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
```

```html
<h1 class="text-3xl text-blue-500 font-bold underline flex justify-center">
    17기 아파트 실거래 정보
</h1>
<div class="w-full h-60 bg-red-300">
    <input class="w-40 h-20 border border-red-500"
    type="text"
    placeholder="검색어를 입력하세요" />
    <button class="bg-[#95CCDD] text-white">조회하기</button>
</div>
```

> **주의 — 아직 연결되지 않았다**
> `app.py`는 `scrapper.py`를 **import하지 않는다.** 조회 버튼에도 동작이 없다.
> 화면과 데이터 수집 함수가 각각 만들어졌을 뿐 **아직 이어지지 않은 상태**다.
> 이어 붙이려면 `app.py`에서 `from scrapper import fetch_apt_trade`를 하고, 폼 값을 받아 `render_template("index.html", rows=fetch_apt_trade(...))`처럼 넘기면 된다.

**주의 — 템플릿의 따옴표가 깨져 있다**
```html
<div class=""w-5xl h-full mx-auto>
<div class="'w=full h-60 bg-red-300">
```
첫 줄은 `class=""`로 닫힌 뒤 `w-5xl...`이 속성 이름으로 해석되고 `>`가 없어 태그가 이어진다.
두 번째 줄은 `'`와 `=`가 섞였다. 브라우저가 알아서 복구하지만 스타일은 적용되지 않는다.
`class="w-5xl h-full mx-auto"`, `class="w-full h-60 bg-red-300"`가 맞다.

`main.py`는 `print("Hello from 17-apartment!")`만 있는 프로젝트 생성 시 기본 파일이다. 실행에는 쓰이지 않는다.

---

## 4-3. `17-assignment` — 노래방 차트 검색기

TJ미디어와 금영의 인기곡 차트를 모아 검색·다운로드하게 만든 것이다.

```
crawler_tj.py  →  songs_tj.csv  ┐
                                ├→  app.py  →  화면 / CSV 다운로드
crawler_ky.py  →  songs_ky.csv  ┘
```

**수집과 서빙이 CSV 파일로 분리되어 있다.** 웹 요청 때마다 크롤링하지 않으므로 화면이 빠르다.

```
flask
requests
beautifulsoup4
apscheduler
gunicorn
```

`requirements.txt`에 `gunicorn`이 있다 — **배포까지 염두에 둔 구성**이다.

### (1) `crawler_tj.py` — 숨은 API 호출

TJ는 HTML을 긁는 대신 **차트 페이지가 내부적으로 쓰는 API를 직접 부른다.**

```python
API_URL = "https://www.tjmedia.com/legacy/api/topAndHot100"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ... Chrome/124.0 Safari/537.36",
    "Referer": "https://www.tjmedia.com/chart/top100",
}

def fetch_chart(start_date, end_date):
    payload = {
        "chartType": "TOP",
        "searchStartDate": start_date,
        "searchEndDate": end_date,
        "strType": "",
    }
    res = requests.post(API_URL, data=payload, headers=HEADERS, timeout=10)
    data = res.json()
```

**HTML 파싱보다 훨씬 낫다.** 구조가 바뀌어도 잘 깨지지 않고, JSON이라 바로 쓸 수 있다.

| 헤더 | 왜 |
| --- | --- |
| `User-Agent` | 없으면 `python-requests/...`로 나가 차단될 수 있다 |
| `Referer` | "이 페이지에서 눌렀다"는 표시 — 없으면 거부하는 서버가 있다 |

`timeout=10`도 중요하다. **없으면 서버가 응답하지 않을 때 무한정 기다린다.**

### 재귀 탐색으로 응답 구조 무시하기

```python
def walk(obj):
    if isinstance(obj, dict):
        if "rank" in obj and "pro" in obj:
            results.append(obj)
        for v in obj.values():
            walk(v)
    elif isinstance(obj, list):
        for item in obj:
            walk(item)

walk(data)
```

**JSON 어디에 있든 `rank`와 `pro` 키를 동시에 가진 딕셔너리를 전부 찾아낸다.**

`data["result"]["list"]` 같은 경로를 몰라도 되고, API가 응답 구조를 바꿔도 동작한다. 중첩 자료구조를 다루는 전형적인 재귀 패턴이다.

- `dict`면 → 조건 검사 후 **모든 값**에 대해 재귀
- `list`면 → **모든 항목**에 대해 재귀
- 그 외(문자열·숫자)면 → 아무것도 안 함 (재귀 종료)

### 날짜 구간을 거꾸로 쪼개기

```python
def get_date_windows(months_back=24):
    windows = []
    today = date.today()
    cursor = today
    for i in range(months_back * 2):
        end = cursor
        start = end - timedelta(days=14)
        windows.append((start.isoformat(), end.isoformat()))
        cursor = start - timedelta(days=1)
    return windows
```

**한 번에 2년치를 요청하면 상위 100곡만 오므로, 2주 단위로 48번 나눠 부른다.** 오늘부터 과거로 거슬러 간다.

`cursor = start - timedelta(days=1)`에서 **하루를 빼는 것**이 구간이 겹치지 않게 하는 처리다.

### 중복 제거와 안전장치

```python
song_list = []
seen = set()

for start, end in windows:
    rows = fetch_chart(start, end)
    new_count = 0
    for row in rows:
        song_no = str(row.get("pro", "")).strip()
        if song_no == "" or song_no in seen:
            continue
        song_list.append({
            "곡번호": song_no,
            "제목": row.get("indexTitle", ""),
            "가수": row.get("indexSong", ""),
            "브랜드": "TJ",
        })
        seen.add(song_no)
        new_count += 1
    print(start + " ~ " + end + " 새곡 " + str(new_count) + "개 누적 " + str(len(song_list)))
    time.sleep(0.5)
    if len(song_list) >= 500:
        break
```

| 장치 | 왜 |
| --- | --- |
| `seen = set()` | 기간이 겹쳐 같은 곡이 여러 번 나온다. **집합은 조회가 O(1)** |
| `row.get(key, "")` | 키가 없어도 `KeyError`가 안 난다 |
| `time.sleep(0.5)` | **서버에 부담 주지 않기.** 크롤링 예의이자 차단 회피 |
| `if len(...) >= 500: break` | 목표치를 채우면 48번을 다 돌지 않고 멈춘다 |
| `print(...)` 진행 상황 | 오래 걸리는 작업이라 어디까지 갔는지 보여 준다 |

**`song_list`와 `seen`을 같이 관리하는 것**이 이 패턴의 핵심이다. 리스트는 순서를, 집합은 중복 검사를 맡는다.

### (2) `crawler_ky.py` — HTML 파싱

금영은 API가 없어 **BeautifulSoup으로 화면을 긁는다.**

```python
def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip() != ""]
```

**`soup.get_text()`로 태그를 전부 버리고 글자만 남긴다.** `select()`로 CSS 선택자를 쓰는 대신 택한 방식이다. HTML 구조가 바뀌어도 살아남지만, **순서에 의존**하게 된다.

```python
for i, line in enumerate(lines):
    if re.fullmatch(r"\d{4,7}", line):
        song_no = line
        if song_no in seen:
            continue

        rest_lines = []
        for j in range(i + 1, min(i + 8, len(lines))):
            candidate = lines[j]
            if re.fullmatch(r"[▲▼-]\s*\d*", candidate):    # 순위 변동
                continue
            if re.fullmatch(r"\d{1,3}", candidate):        # 순위 번호
                continue
            if re.fullmatch(r"\d{4}\.\d", candidate):      # 2024.1 형태
                continue
            if re.fullmatch(r"\d{4,7}", candidate):        # 다음 곡번호
                break
            rest_lines.append(candidate)
            if len(rest_lines) >= 2:
                break

        if len(rest_lines) < 2:
            continue

        title = rest_lines[0]
        singer = rest_lines[1]
```

**"곡번호처럼 생긴 줄(4~7자리 숫자)을 찾고, 그다음 줄들에서 노이즈를 걸러 제목과 가수를 집는다."**

| 정규식 | 거르는 것 |
| --- | --- |
| `\d{4,7}` | 곡번호 — **기준점** |
| `[▲▼-]\s*\d*` | `▲3`, `▼1`, `-` 같은 순위 변동 |
| `\d{1,3}` | 1~3자리 = 순위 번호 |
| `\d{4}\.\d` | `2024.1` 같은 발매 시기 |

- `min(i + 8, len(lines))` — **리스트 끝을 넘어가지 않는 안전장치**
- 다음 곡번호를 만나면 `break` — 이 곡의 정보 구간이 끝났다는 신호
- `if len(rest_lines) < 2: continue` — 둘을 못 채우면 **버린다.** 잘못된 데이터를 넣는 것보다 낫다

`re.fullmatch`는 **문자열 전체**가 맞아야 한다. `re.match`(앞부분만)나 `re.search`(어디든)를 쓰면 `12345번`도 곡번호로 잡힌다. 여기서는 `fullmatch`가 정확한 선택이다.

### 2페이지 넘어가기

```python
def find_next_page_url(soup):
    links = soup.find_all("a")
    for link in links:
        text = link.get_text(strip=True)
        if "51" in text and "100" in text:
            href = link.get("href")
            if href:
                return urljoin(BASE_URL, href)
    return None
```

**"51"과 "100"이 모두 든 링크 글자를 찾는다** — `51~100위` 버튼이다.

`urljoin(BASE_URL, href)`은 상대 경로(`/popular/?page=2`)를 절대 URL로 만든다. `href`가 이미 절대 URL이면 그대로 둔다. **문자열 이어붙이기보다 항상 안전하다.**

```python
res = requests.get(BASE_URL, headers=HEADERS, timeout=10)
res.encoding = "utf-8"
```

**`res.encoding`을 직접 지정한 것**에 주의. 서버가 인코딩을 잘못 알려 주면 한글이 깨지므로 강제한 것이다.

### (3) `app.py` — 검색과 다운로드

```python
def load_csv(path):
    songs = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append(row)
    except FileNotFoundError:
        pass
    return songs
```

**`utf-8-sig`가 중요하다.** 크롤러가 이 인코딩으로 저장했다. BOM(파일 앞 3바이트)을 붙여 **엑셀이 한글 CSV를 깨지 않게** 하는 것인데, 읽을 때 `utf-8`로 읽으면 첫 컬럼 이름이 `﻿곡번호`가 되어 조회가 실패한다.

**`except FileNotFoundError: pass`** — 크롤러를 아직 돌리지 않았어도 서버는 뜬다.

```python
def get_songs(keyword, brand, chart):
    if chart == "TJ":
        return load_csv("songs_tj.csv")
    if chart == "금영":
        return load_csv("songs_ky.csv")
    if keyword == "":
        return []

    songs = load_csv("songs_tj.csv") + load_csv("songs_ky.csv")

    if brand != "전체":
        songs = [s for s in songs if s["브랜드"] == brand]

    songs = [s for s in songs if keyword in s["제목"] or keyword in s["가수"]]
    return songs
```

**`if keyword == "": return []`이 있어 첫 화면에 500곡이 쏟아지지 않는다.**
필터를 리스트 컴프리헨션으로 차례로 좁혀 간다. `keyword in s["제목"]`은 부분 일치 검색이다.

```python
@app.route("/")
def index():
    keyword = request.args.get("keyword", "").strip()
    brand = request.args.get("brand", "전체")
    chart = request.args.get("chart", "")

    show_results = keyword != "" or chart != ""
    songs = get_songs(keyword, brand, chart)

    return render_template(
        "index.html",
        songs=songs, keyword=keyword, brand=brand, chart=chart,
        show_results=show_results,
        total=len(songs),
        tj_count=len(load_csv("songs_tj.csv")),
        ky_count=len(load_csv("songs_ky.csv")),
    )
```

**`request.args`가 Flask의 쿼리 매개변수다.** FastAPI가 함수 인자로 받던 것을 여기서는 직접 꺼낸다.

`keyword=keyword`를 템플릿에 되돌려 주는 것은 **검색 후에도 입력창에 검색어가 남아 있게** 하기 위함이다.

### 메모리에서 CSV 만들어 내려주기

```python
@app.route("/download")
def download():
    songs = get_songs(keyword, brand, chart)

    output = io.StringIO()
    fieldnames = ["곡번호", "제목", "가수", "브랜드"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for s in songs:
        writer.writerow(s)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)

    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="my_songs.csv")
```

**디스크에 임시 파일을 만들지 않는다.**

| 단계 | 무엇 |
| --- | --- |
| `io.StringIO()` | 메모리 위의 **문자열** 파일 — `csv` 모듈이 여기에 쓴다 |
| `.encode("utf-8-sig")` | 문자열 → 바이트 (BOM 포함) |
| `io.BytesIO()` | 메모리 위의 **바이트** 파일 — `send_file`이 요구 |
| `mem.seek(0)` | **읽기 위치를 처음으로** — 빠뜨리면 빈 파일이 내려간다 |
| `as_attachment=True` | 브라우저에서 열지 말고 **다운로드** |

**`seek(0)`이 가장 자주 빠뜨리는 부분이다.** 쓰고 나면 커서가 끝에 있어 그 상태로 읽으면 아무것도 없다.

### 새벽 4시 자동 갱신

```python
from apscheduler.schedulers.background import BackgroundScheduler

def refresh_charts():
    print("차트 자동 갱신 시작")
    crawler_tj.main()
    crawler_ky.main()
    print("차트 자동 갱신 끝")

scheduler = BackgroundScheduler()
scheduler.add_job(refresh_charts, "cron", hour=4, minute=0)
scheduler.start()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
```

**`BackgroundScheduler`는 별도 스레드에서 돈다.** 웹 요청 처리를 막지 않는다.

`"cron"` 트리거는 리눅스 crontab과 같은 방식이다. `hour=4, minute=0` → 매일 04:00.

> **`use_reloader=False`가 꼭 필요하다.**
> `debug=True`는 코드 변경 감시를 위해 **프로세스를 두 개 띄운다.** 그대로 두면 스케줄러도 두 개가 돌아 크롤링이 **하루에 두 번** 실행된다.
> `use_reloader=False`로 재시작 기능을 끄면 프로세스가 하나가 된다. **디버그 모드와 백그라운드 작업을 함께 쓸 때의 정석 처리다.**

> **주의 — 여러 워커로 배포하면 다시 문제가 된다**
> `gunicorn -w 4`로 띄우면 워커 4개가 각각 스케줄러를 갖는다. 실무에서는 스케줄러를 **별도 프로세스로 분리**하거나 워커 1개에서만 돌도록 잠금을 건다.

---

## 이 장 정리

### 한 줄 요약

**바깥 데이터를 가져오는 방법은 세 가지다** — 공개 API(XML/JSON), 페이지가 쓰는 숨은 API, HTML 파싱. **위에서부터 우선한다.**

### 수집 방법 비교

| 방법 | 예 | 안정성 | 난이도 |
| --- | --- | --- | --- |
| 공개 API | 공공데이터포털 (XML) | 높음 | 낮음 (키 발급 필요) |
| 내부 API | TJ `topAndHot100` (JSON) | 중간 | 중간 (개발자도구로 찾아야) |
| HTML 파싱 | 금영 (BeautifulSoup) | **낮음** | 높음 |

### 기억할 코드

```python
ET.fromstring(res.content)                  # XML
root.findall('./body/items/item')           # XPath
requests.post(url, data=payload, headers=HEADERS, timeout=10)
soup.get_text("\n", strip=True)             # 태그 제거
re.fullmatch(r"\d{4,7}", line)              # 전체 일치
urljoin(BASE_URL, href)                     # 상대 → 절대
open(path, encoding="utf-8-sig")            # 엑셀 호환 CSV
mem.seek(0)                                 # 읽기 전 필수
scheduler.add_job(fn, "cron", hour=4, minute=0)
app.run(debug=True, use_reloader=False)     # 스케줄러와 함께 쓸 때
```

### 자주 틀리는 것

- `timeout`을 안 준다 → 응답 없는 서버에 영원히 매달린다
- `time.sleep`을 안 넣는다 → 차단당한다
- `utf-8-sig`로 쓰고 `utf-8`로 읽는다 → 첫 컬럼 이름에 BOM이 붙는다
- `send_file` 전에 `seek(0)`을 빠뜨린다 → 빈 파일
- `debug=True`와 스케줄러를 같이 쓰면서 `use_reloader=False`를 안 준다 → **작업이 두 번 실행**
- HTML 구조에 의존하는 선택자를 쓴다 → 사이트 개편 때 조용히 깨진다
