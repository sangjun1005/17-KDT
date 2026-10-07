# 3. 로컬 LLM — Ollama 로 내 노트북에서 돌린다

**실습 파일**: `17-KDT/17-RAG/TotalAI.ipynb` (셀 20~26)

2장까지는 분류·임베딩이었다. 3장은 **대화형 LLM** 이다.
그런데 API 를 쓰지 않는다 — **내 노트북에서 직접 돌린다.**

> **2026-10-06 — 전부 실행됐다**
> Ollama 서버 자동 기동, `gemma3:4b` 로 대화, 시스템 프롬프트·멀티턴·스트리밍,
> **JSON 강제 출력**, 후기 4개 일괄 분석까지. 4장 RAG 의 "생성" 부분이 여기서 준비된다.

| 절 | 내용 |
| --- | --- |
| 3-1 | 왜 로컬 LLM 인가 |
| 3-2 | 서버를 코드로 띄운다 — `ensure_ollama` |
| 3-3 | `chat` 함수 하나로 끝낸다 |
| 3-4 | 시스템 프롬프트 — 성격을 준다 |
| 3-5 | 멀티턴 — 기억은 모델이 아니라 내가 들고 있다 |
| 3-6 | 스트리밍 — 기다림을 없앤다 |
| 3-7 | JSON 강제 출력 — LLM 을 함수처럼 쓴다 |
| 3-8 | 일괄 분석 — 그리고 드러난 한계 |
| 3-9 | 고칠 것 |

---

## 3-1. 왜 로컬 LLM 인가

```python
OLLAMA_URL = 'http://localhost:11434'
LLM_MODEL = 'gemma3:4b'
```

**`localhost` 다.** 인터넷 밖으로 나가는 요청이 없다.

| | **API** (OpenAI·Claude 등) | **로컬** (Ollama) |
| --- | --- | --- |
| 데이터 | 외부로 나간다 | **내 PC 안에만** |
| 비용 | 토큰당 과금 | **0원** (전기값) |
| 인터넷 | 필요 | **한 번 받으면 불필요** |
| 속도 | 빠르다 | GPU 에 따라 다르다 |
| 성능 | 최상급 | **작은 모델 수준** |
| 모델 교체 | 이름만 바꾼다 | 받아야 한다 |

**이 수업이 로컬을 고른 이유는 셋이다.**

1. **동아리 내부 문서를 다룬다** (4장). 회비·임원진·연락처가 외부로 안 나간다
2. **키가 필요 없다.** 수강생 전원이 각자 돌릴 수 있다
3. **API 호출이 아니라 "모델을 돌린다"는 감각**을 준다

**`gemma3:4b`** — 구글의 Gemma 3, 40억 파라미터.

| 모델 | 파라미터 | 비교 |
| --- | --- | --- |
| 코드 생성 LM 3장 `codebot` | **1,112만** | 직접 만든 것 |
| 코드 생성 LM 4장 `storybot` | **2,270만** | 〃 |
| e5-small (2장) | **1억 1,766만** | 임베딩 |
| **`gemma3:4b`** | **약 40억** | **직접 만든 것의 360배** |
| GPT-4 급 | 수천억~ | 비교 불가 |

**40억도 "작은 모델"이다.** 그런데 한국어로 자연스럽게 대화한다 —
코드 생성 LM 3장의 1,112만 모델이 "문법은 맞고 논리는 틀린" 코드를 쓴 것과 비교하면
**규모가 무엇을 바꾸는지**가 한눈에 보인다.

---

## 3-2. 서버를 코드로 띄운다 — `ensure_ollama`

**Ollama 는 백그라운드 서버다.** HTTP 로 요청을 받는다. 그래서 먼저 살아 있는지 봐야 한다.

```python
def ollama_alive():
    try:
        return requests.get(f'{OLLAMA_URL}/api/tags', timeout=2).ok
    except requests.RequestException:
        return False
```

**`try/except` 가 필수다.** 서버가 없으면 `requests` 는 `False` 를 주지 않고 **예외를 던진다**
(`ConnectionError`). 잡지 않으면 노트북이 거기서 멈춘다.

```python
def ensure_ollama(model=LLM_MODEL):
    global _ollama_proc
    if not ollama_alive():
        if shutil.which('ollama') is None:
            raise RuntimeError('올라마가 설치되어 있지 않습니다.')
        print('올라마 서버를 켜는 중')
        log = open(WORK / 'ollama_server.log', 'w')
        _ollama_proc = subprocess.Popen(['ollama', 'serve'],
                                        stdout=log, stderr=log, start_new_session=True)
        for _ in range(60):
            if ollama_alive():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('올라마 서버가 꺼져있습니다. log 확인을 해주세요.')
```

**네 가지를 제대로 했다.**

| 코드 | 왜 |
| --- | --- |
| `shutil.which('ollama')` | **설치 여부를 먼저 본다.** 없으면 바로 알려 준다 |
| `stdout=log, stderr=log` | 서버 로그를 파일로. 안 하면 노트북 출력이 더럽혀진다 |
| `start_new_session=True` | **커널이 죽어도 서버가 안 따라 죽는다** |
| `for ... else: raise` | **최대 30초(60 × 0.5s) 기다리고 포기한다.** 무한 대기가 아니다 |

**`for ... else` 를 쓴 것이 파이썬다운 코드다.** `break` 없이 루프가 끝났을 때만 `else` 가 돈다 —
"60번 다 돌았는데도 안 살아났다"를 플래그 변수 없이 표현한다 (파이썬 기초 3장).

### 모델이 없으면 받는다

```python
have = [m['name'] for m in requests.get(f'{OLLAMA_URL}/api/tags').json().get('models', [])]
if model not in have and f'{model}:latest' not in have:
    print(f'모델 {model} 다운 받는 중')
    last = -1
    with requests.post(f'{OLLAMA_URL}/api/pull', json={'model': model}, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if d.get('total') and d.get('completed'):
                pct = int(d['completed'] / d['total'] * 100)
                if pct // 10 != last // 10:        # 10% 단위로만 찍는다
                    print(f'{pct}%', end=" ", flush=True)
                    last = pct
    print('다운로드 완료')
print(f'서버준비완료 {model}')
```

**`stream=True` + `iter_lines()` 가 NDJSON 패턴이다.**
Ollama 는 진행 상황을 **한 줄에 JSON 하나**씩 계속 보낸다. 끝까지 모아서 파싱하는 게 아니라
**한 줄씩 받아 즉시 처리한다.** 3-6의 스트리밍도 같은 구조다.

**`pct // 10 != last // 10` 로 10% 단위만 찍는다.** 안 그러면 수천 줄이 쏟아진다.
`if not line: continue` 는 keep-alive 빈 줄을 건너뛰는 것이다.

**실측**: `서버준비완료 gemma3:4b` — **이미 켜져 있고 모델도 있었다.**
다운로드 진행률은 안 찍혔다(두 번째 실행이라서).

> **`model not in have and f'{model}:latest' not in have`** —
> Ollama 가 태그를 `gemma3:4b` 로도 `gemma3:4b:latest` 로도 보고할 수 있어 둘 다 본다.
> 실무에서 걸리는 종류의 함정이다.

---

## 3-3. `chat` 함수 하나로 끝낸다

```python
def chat(messages, temperature=0.7, model=LLM_MODEL,
         fmt=None, max_tokens=None, seed=None):
    options = {'temperature': temperature}
    if max_tokens:
        options['num_predict'] = max_tokens
    if seed is not None:
        options['seed'] = seed

    payload = {'model': model, 'messages': messages, 'stream': False,
               'options': options, 'keep_alive': '30m'}
    if fmt:
        payload['format'] = fmt

    r = requests.post(f'{OLLAMA_URL}/api/chat', json=payload, timeout=600)
    r.raise_for_status()
    return r.json()['message']['content']
```

```python
print(chat([{'role': 'user', 'content': '안녕하세요. 당신은 누구입니까? 한 문장으로 답해주세요.'}]))
# 저는 Google에서 훈련한 대규모 언어 모델입니다.
```

**설계가 좋은 점 다섯.**

| 코드 | 왜 |
| --- | --- |
| `messages` 를 받는다 | **역할(role) 구조를 그대로 노출한다.** 멀티턴·시스템 프롬프트가 공짜로 된다 |
| `if max_tokens:` / `if seed is not None:` | **준 것만 보낸다.** 기본값을 덮어쓰지 않는다 |
| `keep_alive='30m'` | **모델을 30분간 GPU 에 남긴다.** 매 호출마다 로딩하면 수 초씩 날아간다 |
| `timeout=600` | 로컬 생성은 느릴 수 있다. 기본 타임아웃으로는 끊긴다 |
| `raise_for_status()` | **HTTP 오류를 조용히 넘기지 않는다** |

> **`if seed is not None` 과 `if max_tokens` 의 차이에 주의.**
> `seed=0` 은 유효한 시드인데 `if seed:` 로 쓰면 **0이 거짓이라 무시된다.**
> `max_tokens=0` 은 의미가 없으니 `if max_tokens:` 로 충분하다. **구분해 쓴 것이 정확하다.**

**`options` 안에 들어가는 이름이 OpenAI 와 다르다.**

| 뜻 | OpenAI | **Ollama** |
| --- | --- | --- |
| 최대 생성 길이 | `max_tokens` | **`num_predict`** |
| 온도 | `temperature` | `temperature` |
| 응답 형식 | `response_format` | **`format`** (최상위) |

**`format` 은 `options` 안이 아니라 `payload` 최상위다.** 3-7에서 쓴다.

---

## 3-4. 시스템 프롬프트 — 성격을 준다

```python
MOA_SYSTEM = ("너는 대학 AI 입문 동아리, '모아'의 친절한 AI비서 '모아'야."
              "항상 한국어로, 존댓말로, 3문장 이내로 짧게 따뜻하게 대답해줘.")

q = '동아리에 처음 왔는데, 너무 떨려요. 어떡하죠?'

print('System 없이 : ', chat([{'role': 'user', 'content': q}], max_tokens=120))
print('성격 부여 후 : ', chat([{'role': 'system', 'content': MOA_SYSTEM},
                               {'role': 'user', 'content': q}], max_tokens=120))
```

### 실측 — 같은 질문, 같은 모델

**System 없이** (120토큰에서 잘렸다)

```
동아리에 처음 오셨다니, 정말 떨리는 마음 충분히 이해해요! 새로운 환경에 대한 기대감과
함께 긴장되는 건 당연하니까 너무 걱정하지 마세요.

**1. 떨리는 마음 다스리기:**
*   **심호흡:** 잠시 멈춰서 심호흡을 몇 번 해보세요. ...
*   **긍정적인 생각:** "나는
```

**성격 부여 후**

```
안녕하세요, 처음 오시셔서 긴장하신다니 조금 떨리는 마음 충분히 이해합니다.
모아는 여러분의 즐거운 동아리 생활을 위해 항상 곁에서 응원할게요.
편안하게 다가가셔서 많은 활약을 보여주시길 바랍니다!
```

**세 가지가 바뀌었다.**

| | System 없이 | 성격 부여 후 |
| --- | --- | --- |
| 형식 | **마크다운 목록·굵게** | 평문 3문장 |
| 길이 | **120토큰에서 중간에 잘렸다** | 깔끔히 끝났다 |
| 정체성 | 일반 조언자 | **"모아"로 자신을 부른다** |

**가장 중요한 변화는 "길이"다.**
`max_tokens=120` 이 같은데 한쪽은 잘리고 한쪽은 안 잘렸다 —
**"3문장 이내"라는 지시가 생성을 실제로 줄였다.** 자르기(`num_predict`)와 지시는 다른 수단이다.
**자르기는 문장을 망가뜨리고, 지시는 짧게 완성시킨다.**

**LLM 이 기본적으로 마크다운 목록을 쓰려 한다는 점도 실무에서 중요하다.**
채팅창에는 좋지만 **문자 메시지·음성 안내·DB 필드**에 넣으면 `**` 와 `*` 가 그대로 보인다.
시스템 프롬프트로 막는 것이 첫 번째 방어다.

> **파이썬은 나란히 쓴 문자열을 그냥 이어 붙인다.** 줄을 나눠 쓸 때
> **앞 문자열 끝에 공백 하나**를 넣지 않으면 두 문장이 붙어 버린다.
> 동작에는 문제가 없지만 모델이 읽는 지시문이 지저분해진다. (코드는 그 뒤 고쳤다.)

---

## 3-5. 멀티턴 — 기억은 모델이 아니라 내가 들고 있다

```python
class MoaChat:
    def __init__(self, system=MOA_SYSTEM, temperature=0.7):
        self.messages = [{'role': 'system', 'content': system}]
        self.temperature = temperature

    def say(self, text):
        self.messages.append({'role': 'user', 'content': text})
        answer = chat(self.messages, temperature=self.temperature)
        self.messages.append({'role': 'assistant', 'content': answer})
        return answer
```

**`say` 한 번에 리스트가 2개 늘어난다** — 내 말과 모델의 답.
**다음 호출은 지금까지의 전부를 다시 보낸다.**

```
1번째 호출  [system, user1]
2번째 호출  [system, user1, assistant1, user2]        ← 전부 다시
3번째 호출  [system, user1, assistant1, user2, assistant2, user3]
```

### 실측 — 기억이 있을 때와 없을 때

```python
bot = MoaChat()
print(bot.say('저는 권상준이고, 파이썬을 처음 배워요.'))
print(bot.say('제 이름이 뭐였죠?'))
print(chat([{'role': 'system', 'content': MOA_SYSTEM},
            {'role': 'user', 'content': '제 이름이 뭐였죠?'}]))
```

```
① 안녕하세요, 권상준님! 파이썬을 처음 배우신다니 정말 멋지네요. ...

② 저에게 다시 알려주셔서 감사합니다, 권상준님!
   당신의 이름은 권상준이 맞습니다.

③ 안녕하세요, 만나뵙게 되어 정말 기쁩니다. 혹시 제가 기억력이 부족한 것 같아 죄송합니다.
   성함을 다시 한번 여쭤부릴 수 있을까요?
```

**②와 ③이 같은 질문에 다른 답을 한다.**

| 호출 | 보낸 것 | 결과 |
| --- | --- | --- |
| ② `bot.say` | `[system, user1, assistant1, user2]` | **이름을 안다** |
| ③ 생 `chat` | `[system, user2]` | **모른다** |

**LLM 은 상태가 없다(stateless).** 모델 쪽에 "대화"라는 것은 없고,
**매 요청이 완전히 독립이다.** 기억처럼 보이는 것은 전부 **내가 리스트를 들고 다시 보내기 때문**이다.

**이것이 두 가지 실무 결과를 낳는다.**

**① 비용과 속도가 대화 길이에 비례해 늘어난다.** 10턴이면 10턴 전부를 매번 다시 읽는다.
**② 문맥 길이를 넘기면 앞이 잘린다.** 코드 생성 LM 3-6의 슬라이딩 윈도우와 같은 문제다 —
거기서는 `ids[:, -256:]` 로 직접 잘랐고, 여기서는 Ollama 가 알아서 자른다(그래서 **조용히** 잊는다).

> **긴 대화에서는 요약으로 압축한다.** 앞부분을 모델에게 요약시켜 `system` 에 넣고
> 최근 몇 턴만 그대로 두는 것이 일반적인 수법이다. 이 노트북에는 없다.

**셀 23에 사용자 본인 이름이 들어 있다.** 저장소가 Public 이므로 알아 둘 것.

---

## 3-6. 스트리밍 — 기다림을 없앤다

```python
def chat_stream(messages, temperature=0.7, model=LLM_MODEL):
    payload = {'model': model, 'messages': messages,
               'stream': True, 'options': {'temperature': temperature}}

    with requests.post(f'{OLLAMA_URL}/api/chat', json=payload,
                       stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                print(chunk['message']['content'], end="", flush=True)
```

**3-3의 `chat` 과 딱 두 군데가 다르다.**

| | `chat` | `chat_stream` |
| --- | --- | --- |
| payload | `'stream': False` | **`'stream': True`** |
| 받기 | `r.json()['message']['content']` | **`for line in r.iter_lines()`** |

**`stream=True` 를 두 곳에 줘야 한다.** payload 안(Ollama 에게 "나눠 보내라")과
`requests.post` 인자(requests 에게 "다 받기 전에 넘겨라"). **하나만 주면 효과가 없다.**

**`end=""` + `flush=True` 가 짝이다.**
`end=""` 는 줄바꿈을 막고, `flush=True` 는 **버퍼에 모아 두지 말고 즉시 화면에 쓰라**는 뜻이다.
`flush` 를 빼면 조각이 모여 있다가 한꺼번에 나와 **스트리밍처럼 안 보인다.**

**`with` 로 감싼 것이 중요하다.** 스트리밍 응답은 연결을 열어 둔다 — 안 닫으면 소켓이 샌다.

**실측** — 마스코트 이름 아이디어를 받았다. 3-4의 "3문장 이내" 지시가 지켜졌다.

```
안녕하세요, 회원님! '모아'입니다. 동아리 마스코트 이름은 동아리의 정체성을 잘 나타내는 것이 중요하죠.
몇 가지 아이디어를 드려볼까요? '모아'처럼 함께 모여 배우는 느낌을 주는 '알파', '데이터', '코드' 등도 좋을 것 같습니다.
혹시 특별히 생각하시는 컨셉이나 키워드가 있으신가요?
```

> **이 함수는 값을 돌려주지 않는다 (`return` 이 없다).** 화면에 찍고 끝난다.
> `MoaChat` 에 끼워 쓰려면 조각을 모아 `return` 해야 한다 —
> `parts = []` 에 모으고 `''.join(parts)` 를 돌려주면 된다. **3-9에 적었다.**

**스트리밍은 성능이 아니라 체감의 문제다.** 전체 생성 시간은 같다.
첫 글자가 0.3초에 나오는 것과 8초에 전부 나오는 것의 차이다 —
코드 생성 LM 6장의 웹서비스가 `generate` 를 동기로 호출해 요청을 오래 붙잡은 것과 대비된다.

---

## 3-7. JSON 강제 출력 — LLM 을 함수처럼 쓴다

**지금까지는 사람이 읽는 글이었다. 프로그램이 쓸 데이터가 필요하면?**

```python
review = '해커톤 때 간식은 맛있었는데, 와이파이가 자꾸 끊겨서 프로젝트 제출을 못 할 뻔했어요. 그래도 팀원 덕분에 즐거웠습니다.'

sys_json = ('후기를 읽고 JSON으로만 답해. 키: sentiment (긍정/부정/중립), '
            'topics (문자열 배열), complaint (불만 한 줄, 없으면 빈 문자열)')

raw = chat([{'role': 'system', 'content': sys_json},
            {'role': 'user', 'content': review}],
           temperature=0, fmt='json')

parsed = json.loads(raw)
```

### 실측

```json
{
  "sentiment": "중립",
  "topics": ["간식", "와이파이", "프로젝트", "팀원"],
  "complaint": "와이파이 끊김으로 프로젝트 제출 못할 뻔"
}
```

```python
print(parsed['sentiment'], parsed['topics'])
# 중립 ['간식', '와이파이', '프로젝트', '팀원']
```

**세 가지가 함께 동작해야 한다.**

| 장치 | 역할 | 빼면 |
| --- | --- | --- |
| **`fmt='json'`** | Ollama 가 **문법적으로 유효한 JSON만** 생성하게 제약한다 | ```` ```json ```` 같은 코드펜스가 섞여 `json.loads` 가 터진다 |
| **`temperature=0`** | 같은 입력에 같은 출력 | 키 이름·값이 호출마다 흔들린다 |
| **시스템 프롬프트의 키 명세** | 어떤 키를 쓸지 지정 | 모델이 제멋대로 스키마를 만든다 |

**`fmt='json'` 이 하는 일은 "부탁"이 아니라 "제약"이다.**
생성 단계에서 **JSON 문법을 깨는 토큰의 확률을 0으로 만든다**(constrained decoding).
프롬프트로 "JSON으로만 답해"라고 부탁하는 것과 **원리가 다르다** — 이쪽은 실패할 수 없다.

**`temperature=0` 은 1장 1-7·코드 생성 LM 3-10에서 본 그 설정이다.**
`argmax` 라 재현 가능하다. **구조화 출력에 다양성은 해롭다.**

### 결과를 읽으면

| 키 | 값 | 평가 |
| --- | --- | --- |
| `sentiment` | `"중립"` | **타당하다.** 칭찬·불만이 섞여 있다 |
| `topics` | `간식 / 와이파이 / 프로젝트 / 팀원` | 네 개를 정확히 뽑았다 |
| `complaint` | `"와이파이 끊김으로 프로젝트 제출 못할 뻔"` | **원문을 요약했다.** 그대로 복사가 아니다 |

**이것이 "LLM 을 함수처럼 쓴다"는 뜻이다.**
`def analyze(review) -> dict` 를 **규칙 없이** 얻은 것이다.
2장의 감정 분류 모델은 **긍정/부정 둘만** 낼 수 있었다. 여기는 `topics` 추출까지 같이 한다 —
**분류 모델로는 못 하고, 규칙 기반으로는 끝없이 예외가 생기는 일**이다.

> **스키마는 받은 뒤에 검증해야 한다.**
> `fmt='json'` 은 **문법**만 보장한다 — `sentiment` 가 `"매우 긍정"` 으로 와도 유효한 JSON 이다.
> Pydantic 으로 키와 값 범위를 검사하는 것이 실무 패턴이다
> (코드 생성 LM 6장의 `StringConstraints(strict=True)` 와 같은 발상).
> 이 셀에는 검증이 없다.

---

## 3-8. 일괄 분석 — 그리고 드러난 한계

```python
ALL_REVIEWS = ['와이파이가 느려서 실습 중간에 계속 멈췄어요.',
               '회장님이 질문마다 끝까지 설명해줘서 감동받았습니다.',
               '모임 장소가 좁아서 사람이 많은 날은 앉을 자리가 없어요.',
               '마스코트 스티커가 너무 귀여워요! 노트북에 붙였어요.']

numbered = '\n'.join(f'{i + 1}. {r}' for i, r in enumerate(ALL_REVIEWS))

report = chat([
    {'role': 'system', 'content': '너는 동아리 운영을 돕는 분석가야. 한국어로 답해줘.'},
    {'role': 'user', 'content': f'아래는 동아리 행사 후기 모음이야.\n'
                                f'잘한 점 3가지와 개선할 점 3가지를 각각 한 줄로 정리해줘.\n\n{numbered}'}
], temperature=0.3)
```

### 실측

```
## 행사 후기 분석 결과

**잘한 점 (3가지)**
1. 회장님의 꼼꼼한 설명으로 참여자들의 이해도를 높였다.
2. 마스코트 스티커를 활용하여 동아리 분위기를 활성화했다.
3. (후기 내용이 부족하여 3가지 잘한 점을 모두 제시할 수 없음)

**개선할 점 (3가지)**
1. 와이파이 속도 문제로 실습 진행에 차질이 발생하여 개선이 필요하다.
2. 모임 장소의 좁은 공간으로 인해 많은 인원을 수용하기 어렵다는 점을 보완해야 한다.
3. (후기 내용이 부족하여 3가지 개선점을 모두 제시할 수 없음)
```

**3번이 둘 다 "내용이 부족하다"다 — 이것이 이 셀의 가장 중요한 결과다.**

**후기 4개 중 칭찬이 2개, 불만이 2개다.** "각각 3가지"를 요구했는데 **근거가 2개뿐이다.**
모델은 **지어내지 않고 "없다"고 말했다.**

| 가능한 반응 | 이 모델이 한 것 |
| --- | --- |
| 억지로 3개를 채운다 (환각) | ✗ |
| **근거가 없다고 말한다** | **✓** |

**이것이 좋은 동작이다.** 그리고 **4장에서 같은 모델이 정반대로 행동한다** —
RAG 에서 "회장님 나이"를 물었을 때 **20살이라고 지어낸다.**
차이는 모델이 아니라 **프롬프트가 제대로 전달됐는지**였다 (4-6).

**읽어 낼 것 둘.**

**① 요청한 개수를 억지로 맞추게 하면 환각이 생긴다.**
"3가지"가 아니라 **"있는 만큼"** 이라고 쓰는 것이 안전하다.

**② 시스템 프롬프트에 형식 제약이 없어 마크다운이 그대로 나왔다.**
3-4의 `MOA_SYSTEM` 대신 `'너는 ... 분석가야. 한국어로 답해줘.'` 만 줬다.
리포트라면 마크다운이 괜찮지만, **3-7처럼 프로그램이 쓸 거라면 `fmt='json'` 이 맞다.**

**`temperature=0.3`** — 3-7의 0(완전 결정적)과 3-4의 0.7(대화) 사이다.
**분석 리포트는 안정적이어야 하지만 문장은 자연스러워야 한다.** 타당한 선택이다.

**`enumerate` 로 번호를 붙인 것도 의도적이다.** 모델이 몇 개를 봤는지 세기 쉬워지고,
답에서 "3번 후기에서..."처럼 가리킬 수 있다.

---

## 3-9. 고칠 것

| 위치 | 내용 | 수정안 |
| --- | --- | --- |
| `chat_stream` | **`return` 이 없다.** 화면에만 찍고 값을 안 준다 | 조각을 모아 `''.join(parts)` 를 돌려준다 |
| `chat_stream` | `MoaChat` 과 합쳐 쓸 수 없다 | 위를 고치면 `say` 에 끼울 수 있다 |
| 셀 25 | `fmt='json'` 은 **문법만** 보장한다. 스키마 검증이 없다 | Pydantic 모델로 키·값 검사 |
| 셀 26 | "3가지"를 강제해 **2개는 "내용 부족"으로 채워졌다** | "있는 만큼"으로 바꾼다 |
| 셀 26 | 시스템 프롬프트에 형식 제약이 없어 마크다운이 나온다 | 평문 지시 또는 `fmt='json'` |
| `MoaChat` | **문맥 길이 관리가 없다.** 길어지면 Ollama 가 조용히 앞을 자른다 | 턴 수 제한 또는 요약 압축 |
| `MoaChat` | `messages` 가 공개 속성이라 밖에서 바꿀 수 있다 | 되감기·초기화 메서드를 두는 편이 낫다 |
| `ensure_ollama` | `log` 파일 핸들을 **닫지 않는다** | 프로세스 수명과 같이 두려는 의도로 보이나 참조 보관이 필요하다 |
| `ensure_ollama` | `_ollama_proc` 를 띄우고 **끄는 코드가 없다** | 노트북 끝에 `terminate()` |
| 셀 23 | 사용자 **본인 이름**이 출력에 남는다. 저장소가 Public | 예시 이름으로 바꾸거나 출력을 지운다 |
| 공통 | `seed` 인자를 만들어 두고 **한 번도 안 쓴다** | 재현이 필요한 셀에 `seed=0` |
| 공통 | 모델 응답을 캐싱하지 않아 재실행마다 답이 바뀐다 | `temperature=0` + `seed` 로 고정 |

---

## 이 장 정리

### 한 줄 요약

**LLM 은 `localhost:11434` 에 JSON 을 POST 하는 HTTP 서버다.**
`messages` 리스트를 내가 들고 다니면 멀티턴이 되고, `format='json'` 을 주면 **함수처럼** 쓸 수 있다.

### 전체 흐름

```
ensure_ollama()                     서버 확인 -> 없으면 띄우고 모델 받는다
   |
chat(messages, ...)                 POST /api/chat  (stream=False)
   |  + system 역할                 성격·형식·제약을 준다
   |  + messages 누적               멀티턴 (모델은 기억하지 않는다)
   |  + stream=True                 조각을 받아 즉시 찍는다
   |  + format='json'               문법적으로 유효한 JSON 만 생성
   v
4장: 검색 결과를 messages 에 끼워 넣는다 -> RAG
```

### 숫자 한눈에

| 항목 | 값 |
| --- | --- |
| 서버 | `http://localhost:11434` |
| 모델 | **`gemma3:4b`** (약 40억 파라미터) |
| 엔드포인트 | `/api/tags` · `/api/pull` · `/api/chat` |
| 기동 대기 | 최대 **30초** (60회 × 0.5초) |
| `keep_alive` | **30분** |
| `timeout` | **600초** |
| 최대 생성 길이 키 | **`num_predict`** (OpenAI 의 `max_tokens`) |
| 온도 사용 | 대화 0.7 · 리포트 0.3 · **구조화 출력 0** |
| 후기 일괄 분석 | 4개 → 잘한 점 2개 + **"내용 부족" 1개** |

### 외워 둘 코드

```python
# 서버 살아 있는지 — 예외를 반드시 잡는다
try:
    requests.get(f'{URL}/api/tags', timeout=2).ok
except requests.RequestException:
    ...

# 커널이 죽어도 서버는 살려 둔다
subprocess.Popen(['ollama', 'serve'], stdout=log, stderr=log, start_new_session=True)

# 최대 N번 기다리고 포기 — for...else
for _ in range(60):
    if alive(): break
    time.sleep(0.5)
else:
    raise RuntimeError(...)

# 호출 한 방
requests.post(f'{URL}/api/chat',
              json={'model': M, 'messages': msgs, 'stream': False,
                    'options': {'temperature': 0.7, 'num_predict': 120},
                    'keep_alive': '30m'},
              timeout=600).json()['message']['content']

# 멀티턴 — 내가 들고 있는다
messages.append({'role': 'user', 'content': text})
messages.append({'role': 'assistant', 'content': answer})

# 스트리밍 — stream 을 두 곳에
with requests.post(..., json={... 'stream': True}, stream=True) as r:
    for line in r.iter_lines():
        if line:
            print(json.loads(line)['message']['content'], end="", flush=True)

# JSON 강제 — format 은 payload 최상위
payload['format'] = 'json'      # + temperature=0 + 키 명세
```

### 자주 틀리는 것

- 서버 확인에 `try/except` 를 안 쓴다 → `ConnectionError` 로 노트북이 멈춘다
- `if seed:` 로 쓴다 → **`seed=0` 이 무시된다.** `is not None` 을 써라
- `max_tokens` 를 그대로 보낸다 → Ollama 는 **`num_predict`** 다
- `format` 을 `options` 안에 넣는다 → **payload 최상위**여야 한다
- `fmt='json'` 만 믿는다 → **문법만 보장한다.** 스키마는 따로 검증
- 구조화 출력에 온도를 준다 → 키 이름이 흔들린다. **0으로**
- 스트리밍에서 `flush=True` 를 뺀다 → 조각이 모였다 한꺼번에 나온다
- `stream=True` 를 한 곳에만 준다 → 스트리밍이 안 된다
- 모델이 대화를 기억한다고 생각한다 → **매 요청이 독립이다.** 리스트를 다시 보내야 한다
- 멀티턴을 무한정 쌓는다 → 문맥을 넘기면 **조용히** 앞이 잘린다
- `keep_alive` 를 안 준다 → 호출마다 모델 로딩에 수 초
- "N가지"를 강제한다 → 근거가 없으면 지어낸다. **"있는 만큼"으로**
- 시스템 프롬프트 없이 쓴다 → 마크다운 목록이 섞여 나온다
