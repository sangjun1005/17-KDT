# 17-job-scrapper — 채용공고 검색 웹앱

인크루트 채용공고를 긁어와 웹 페이지로 보여 주는 **Flask 연습 프로젝트**입니다.
웹 스크래핑과 간단한 서버 만들기를 익히려고 만들었습니다.

## 동작 방식

```
브라우저  →  Flask (app.py)  →  scrapper.py  →  인크루트 검색 페이지
                  ↑                                    │
                  └──────── 회사명·공고 제목 목록 ◄──────┘
```

| 파일 | 역할 |
|---|---|
| `app.py` | Flask 라우팅 — `/`(검색 폼), `/search`(결과 목록) |
| `scrapper.py` | `requests`로 페이지를 받아 `BeautifulSoup`으로 공고를 뽑아냄 |
| `templates/index.html` | 시작 화면 |
| `templates/search.html` | 검색 결과 화면 |

## 실행 방법

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

브라우저에서 `http://127.0.0.1:5000` 으로 접속합니다.

## 알아 둘 점

- **스크래핑은 사이트 구조에 의존합니다.** 인크루트가 HTML 클래스 이름
  (`c_col`, `cpname`, `cell_mid`, `cl_top`)을 바꾸면 결과가 비어서 나옵니다.
  그럴 때는 `scrapper.py`의 선택자를 실제 페이지에 맞춰 고쳐야 합니다.
- 검색어가 `scrapper.py` 안에 고정되어 있습니다. 폼에서 받은 값을 넘기도록 바꾸면
  실제 검색 기능이 됩니다.
- 요청을 짧은 시간에 많이 보내지 않도록 주의하세요. 학습 목적의 프로젝트입니다.
