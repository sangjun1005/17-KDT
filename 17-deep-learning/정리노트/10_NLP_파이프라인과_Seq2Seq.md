# 10. NLP 파이프라인과 Seq2Seq

**실습 파일**: `10.ipynb`

7장에서 LSTM으로 감성 분석을 했고, 8장에서 BERT를 봤다.
10장은 **자연어 처리를 실무 관점에서** 다시 본다.
규칙 기반이 어디까지 되는지, 그리고 번역·요약에 쓰이는 Seq2Seq 구조.

| 절 | 내용 |
|---|---|
| 10-1 | NLP 파이프라인의 다섯 단계 |
| 10-2 | 규칙 기반 감성 분석 — 어디까지 되나 |
| 10-3 | LSTM 감성 분류 |
| 10-4 | Seq2Seq — 인코더·디코더와 티처 포싱 |

---

## 10-1. NLP 파이프라인의 다섯 단계

```python
def nlp_pipeline_demo():
    raw_text = "이 영화는 정말 재미있었어요! 배우들 연기가 훌륭했습니다."

    tokens = ["이", "영화는", "정말", "재미있었어요", "!", "배우들", "연기가", "훌륭했습니다", "."]
    cleaned = ["영화", "정말", "재미있었어요", "배우들", "연기", "훌륭했습니다"]
    vectorized = [45, 123, 892, 156, 234, 567]

    prediction = {"감정": "긍정", "확률": 0.94}
    print(prediction)
```

```
{'감정': '긍정', '확률': 0.94}
```

**이 셀은 모델이 아니라 "지도"다.** 결과를 손으로 적어 둔 것이고,
파이프라인의 단계를 한눈에 보여 주는 게 목적이다.

<table fit-page-width="true" header-row="true">
<tr><td>단계</td><td>하는 일</td><td>한국어의 어려움</td></tr>
<tr><td>1. 원문</td><td>입력</td><td>—</td></tr>
<tr><td>2. 토큰화</td><td>문장을 토큰으로 쪼갬</td><td>띄어쓰기가 불규칙, 교착어</td></tr>
<tr><td>3. 정제</td><td>불용어·기호 제거, 어간 추출</td><td>조사·어미가 붙어 있음</td></tr>
<tr><td>4. 벡터화</td><td>토큰 → 숫자</td><td>어휘 폭발</td></tr>
<tr><td>5. 모델</td><td>분류·생성</td><td>—</td></tr>
</table>

**한국어가 영어보다 까다로운 이유가 3단계에 있다.**

```
영어  : "movies" → "movie"                   (어미 몇 개)
한국어: "영화는", "영화가", "영화를", "영화에서"  (조사가 무한정 붙는다)
```

`"영화는"`에서 `"영화"`를 뽑으려면 **형태소 분석기**가 필요하다.
(KoNLPy의 Okt, Mecab, Kiwi 등)

7장에서는 영어 문장을 `text.split()`으로 잘랐다. 한국어에는 그 방법이 통하지 않는다.

**요즘은 4단계가 형태소 분석을 대신하기도 한다.**
BERT 계열의 서브워드 토크나이저(WordPiece, SentencePiece)는
`"영화는"`을 `["영화", "##는"]`처럼 통계적으로 쪼갠다. 형태소 분석기 없이도 동작한다.

---

## 10-2. 규칙 기반 감성 분석 — 어디까지 되나

### 사전을 만들고 세어 본다

```python
class BusinessSentimentAnalyzer:
    def __init__(self):
        self.positive_keywords = ['좋', '훌륭', '최고', '대박', '강추', '꿀템', '만족']
        self.negative_keywords = ['나쁘', '별로', '짜증', '실망', '최악', '환불']

    def analyze_sentiment(self, text):
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        negative_count = sum(1 for word in self.negative_keywords if word in text)

        if positive_count > negative_count:
            sentiment, confidence = "긍정", min(0.9, 0.6 + positive_count * 0.1)
        elif negative_count > positive_count:
            sentiment, confidence = "부정", min(0.9, 0.6 + negative_count * 0.1)
        else:
            sentiment, confidence = "중립", 0.5
        ...
```

**학습이 없다.** 키워드가 몇 개 들어 있는지만 센다.

**어간만 사전에 넣은 게 요령이다.** `'좋'` 하나로
`"좋네요"`, `"좋아요"`, `"좋습니다"`를 전부 잡는다. 한국어 활용을 우회하는 값싼 방법이다.

### 감정에서 행동으로

```python
def _get_negative_action(self, text):
    if any(word in text for word in ['배송', '늦어요']):
        return "물류팀 우선 처리 (배송 개선)"
    elif any(word in text for word in ['품질', '별로']):
        return "품질관리팀 검토 필요"
    elif any(word in text for word in ['서비스', '직원']):
        return "고객서비스팀 교육 강화"
    ...

def _get_priority(self, sentiment, confidence):
    if sentiment == "부정" and confidence > 0.8:  return "긴급"
    elif sentiment == "부정":                     return "높음"
    elif sentiment == "긍정" and confidence > 0.8: return "활용"
    else:                                         return "모니터링"
```

**"긍정/부정"에서 멈추지 않고 담당 부서와 우선순위까지 낸다.**
실무에서 감성 분석이 실제로 쓰이는 모습이다. 분류는 수단이고 **행동이 목적**이다.

### 결과

```
리뷰                                   판정   확신   우선순위
배송이 너무 늦어요. 짜증납니다.            부정   0.7   높음
제품 품질이 생각보다 좋네요!               긍정   0.7   모니터링
그냥 그래요. 딱히 나쁘지도 좋지도 않아요.   중립   0.5   모니터링
와! 이거 완전 꿀템이다 강추강추!!!         긍정   0.8   모니터링
고객서비스 직원이 너무 불친절해요.         중립   0.5   모니터링
빠른 배송에 만족합니다. 품질도 훌륭해요!    긍정   0.8   모니터링

긍정 3, 부정 1, 중립 2
긴급 0건, 높음 1건
```

**5번 리뷰를 보자.**

```
"고객서비스 직원이 너무 불친절해요."  →  중립 (0.5)
```

명백히 부정인데 중립으로 나왔다. **`'불친절'`이 사전에 없기 때문이다.**

이게 규칙 기반의 근본적인 한계다.

<table fit-page-width="true" header-row="true">
<tr><td>못 잡는 것</td><td>예</td></tr>
<tr><td>사전에 없는 단어</td><td>"불친절", "엉성", "허술"</td></tr>
<tr><td>부정 표현</td><td>"좋지 <strong>않</strong>다" → '좋'이 있으니 긍정</td></tr>
<tr><td>반어·비꼬기</td><td>"참 잘하는 짓이다"</td></tr>
<tr><td>문맥</td><td>"싸구려 같지만 <strong>가격 대비</strong> 좋다"</td></tr>
</table>

**7장의 LSTM, 8장의 BERT가 필요한 이유가 이 표다.**
학습 기반 모델은 사전에 없는 단어도 **주변 문맥으로** 판단한다.

그렇다고 규칙 기반이 쓸모없는 건 아니다.

```
장점 : 학습 데이터가 필요 없다 / 왜 그렇게 판정했는지 설명된다 / 빠르다
용도 : 프로토타입, 학습 데이터가 전혀 없는 초기 단계, 명확한 키워드 알림
```

**항상 이런 단순 기준선을 먼저 만들고, 학습 모델이 그걸 넘는지 확인한다.**
(7장에서 주가 예측에 나이브 기준선이 필요하다고 한 것과 같은 이야기다.)

### 주의 — "활용"이 절대 안 나온다

```
elif sentiment == "긍정" and confidence > 0.8:  return "활용"
```

confidence는 `min(0.9, 0.6 + count * 0.1)`로 계산된다.

```
키워드 1개 → 0.7
키워드 2개 → 0.8     ← 0.8 > 0.8 은 거짓
키워드 3개 → 0.9     ← 여기서야 참
```

**0.8은 "0.8 초과"가 아니다.** 실행 결과에서 확신 0.8짜리 긍정 리뷰 두 개가
전부 "모니터링"으로 빠진 게 그 때문이다. `>=`로 고치거나 경계값을 0.75로 내려야 한다.

같은 이유로 **"긴급"도 한 번도 안 나온다.** 부정 키워드 3개가 한 문장에 있어야 한다.

### 주의 — 키워드 종류만 센다

```python
positive_count = sum(1 for word in self.positive_keywords if word in text)
```

`"강추강추!!!"`처럼 같은 단어가 두 번 나와도 **1로 센다.**
등장 횟수를 세려면 `text.count(word)`를 써야 한다.

### 주의 — 실시간 데모의 `time.sleep(1)`

```python
for review in streaming_reviews:
    ...
    time.sleep(1)
```

스트리밍처럼 보이게 하려고 1초씩 쉰다. 5건이면 5초다.
실제 처리 속도와는 아무 상관이 없다. **연출이다.**

실행 결과가 전부 "일반"인 것도 사전 문제다.
`"포장이 너무 엉성해요"`, `"교환 요청했는데 처리가 늦네요"` 모두
부정 키워드가 하나도 안 걸린다.

---

## 10-3. LSTM 감성 분류

### 패딩

```python
x_train_seq = [torch.tensor([1, 5, 23, 67, 89, 34]), torch.tensor([45, 78, 12, 34]),
               torch.tensor([1, 5, 23, 67]), torch.tensor([45, 78, 12, 34, 90])]
y_train = torch.tensor([1, 0, 1, 0])

x_train = pad_sequence(x_train_seq, batch_first=True, padding_value=0)
print(x_train.shape)      # torch.Size([4, 6])
```

길이가 6, 4, 4, 5인 네 문장을 **가장 긴 6에 맞춰** 0으로 채웠다. 7장과 같다.

### 모델

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_feature = hidden[-1]           # 마지막 층의 마지막 시점
        return self.fc(final_feature)
```

```
SentimentLSTM(
  (embedding): Embedding(5000, 100, padding_idx=0)
  (lstm): LSTM(100, 128, num_layers=2, batch_first=True, dropout=0.3)
  (fc): Linear(in_features=128, out_features=2, bias=True)
  (dropout): Dropout(p=0.3, inplace=False)
)
```

**7장의 `AttentionRNN`과 비교하면 차이가 분명하다.**

<table fit-page-width="true" header-row="true">
<tr><td></td><td>7장 AttentionRNN</td><td>10장 SentimentLSTM</td></tr>
<tr><td>출력 차원</td><td>1 (BCEWithLogitsLoss)</td><td>2 (CrossEntropyLoss)</td></tr>
<tr><td>요약 방식</td><td>어텐션 가중 평균</td><td><code>hidden[-1]</code></td></tr>
<tr><td>패딩 처리</td><td>pack + 마스킹</td><td><strong>없음</strong></td></tr>
<tr><td>양방향</td><td>O</td><td>X</td></tr>
</table>

**이진 분류를 출력 2개 + CrossEntropy로 푸는 것도 맞다.**
출력 1개 + BCE와 수학적으로 거의 같다. 다중 클래스로 확장하기 쉬운 쪽이 이 방식이다.

### 주의 — 패딩을 무시하지 않는다

```python
lstm_out, (hidden, cell) = self.lstm(embedded)      # pack을 안 쓴다
final_feature = hidden[-1]
```

`pack_padded_sequence`가 없으므로 **LSTM이 패딩 0까지 끝까지 읽는다.**
길이 4짜리 문장은 마지막 2스텝이 패딩인데, `hidden[-1]`이 그 시점의 상태다.

`padding_idx=0`이 있어서 임베딩은 0 벡터지만, **LSTM은 0 입력도 상태를 갱신한다.**
문장이 짧을수록 원래 내용이 희석된다.

7장에서 배운 대로 고치면 이렇다. (원본은 그대로 두었다.)

```python
lengths = (x != 0).sum(dim=1)
packed = pack_padded_sequence(embedded, lengths.cpu(),
                              batch_first=True, enforce_sorted=False)
packed_out, (hidden, cell) = self.lstm(packed)
```

### 학습 — 숫자를 믿으면 안 된다

```python
for epoch in range(5):
    for batch_x, batch_y in [(x_train, y_train)]:      # 배치가 하나
        ...
```

```
0.5750
0.5408
0.4854
0.4165
0.3722

정확도 100.0
```

**샘플이 4개다.** 그리고 **학습에 쓴 그 4개로 정확도를 쟀다.**

```
학습 데이터 4개, 검증 0개, 테스트 0개
```

정확도 100%는 "네 개를 외웠다"는 뜻이다. 성능 지표로 읽으면 안 된다.
(7장의 감성 분석 10문장·100에폭과 같은 상황이다.)

손실이 0.69(= `ln 2`)에서 시작해 0.37까지 내려간 것만 확인하면 된다.
**연결이 제대로 돼 있고 역전파가 돈다**는 증거다. 그 이상은 아니다.

### 주의 — 그래프의 숫자가 가짜다

```python
epochs = [1, 2, 3, 4, 5]
train_loss = [1.2, 0.9, 0.7, 0.6, 0.5]  # 가상 데이터
train_accuracy = [70, 75, 80, 85, 88]     # 가상 데이터
```

주석에 "가상 데이터"라고 적혀 있다. **바로 위 셀에서 나온 실제 손실과 다르다.**

```
실제 : 0.5750 → 0.3722
그림 : 1.2    → 0.5
```

그래프 그리는 방법을 보여 주려는 셀이지만,
**손실 곡선을 볼 때는 그 숫자가 실제 실행값인지 먼저 확인해야 한다.**
실제 `history`를 쓰려면 학습 루프에서 `losses.append(avg_loss)`로 모아 두면 된다.

---

## 10-4. Seq2Seq — 인코더·디코더와 티처 포싱

### 무엇이 달라지나

지금까지는 **입력이 시퀀스, 출력이 하나**였다. (many-to-one)
번역·요약·챗봇은 **입력도 출력도 시퀀스**다. (many-to-many, 길이가 다름)

```
분류    : "이 영화 재밌다"        → 긍정
번역    : "이 영화 재밌다"        → "This movie is fun"
                                   ↑ 길이가 다르다
```

### 인코더

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)
```

**인코더가 남기는 건 `(hidden, cell)`이다.** 입력 문장 전체가 벡터 한 쌍으로 압축된다.
이걸 **문맥 벡터(context vector)**라고 부른다.

7장의 LSTM 오토인코더와 구조가 같다. 거기서는 같은 시퀀스를 복원했고,
여기서는 **다른 언어의 시퀀스**를 만든다.

### 디코더 — 한 스텝씩

```python
class Decoder(nn.Module):
    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(1)                   # (B,) → (B, 1)
        embedded = self.embedding(trg)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell
```

**디코더는 한 번에 토큰 하나만 처리한다.** 그래서 `unsqueeze(1)`로 시점 축을 만든다.

**`(hidden, cell)`을 인자로 받고 다시 돌려주는 게 핵심이다.**
호출자가 상태를 들고 다니면서 루프를 돈다.

### 티처 포싱

```python
def forward(self, src, trg, teacher_forcing_ratio=0.5):
    outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
    enc_outputs, (hidden, cell) = self.encoder(src)

    input = trg[:, 0]                              # <sos> 토큰
    for t in range(1, trg_len):
        output, hidden, cell = self.decoder(input, hidden, cell)
        outputs[:, t, :] = output

        teacher_force = torch.rand(1).item() < teacher_forcing_ratio
        top1 = output.argmax(1)
        input = trg[:, t] if teacher_force else top1
    return outputs
```

**다음 입력을 무엇으로 줄 것인가 — 이게 티처 포싱이다.**

<table fit-page-width="true" header-row="true">
<tr><td></td><td>다음 입력</td><td>장점</td><td>단점</td></tr>
<tr><td>티처 포싱</td><td><strong>정답</strong> <code>trg[:, t]</code></td><td>학습이 빠르고 안정적</td><td>추론과 조건이 다르다</td></tr>
<tr><td>자유 실행</td><td><strong>자기 예측</strong> <code>top1</code></td><td>추론과 같은 조건</td><td>초반에 발산한다</td></tr>
</table>

**노출 편향(exposure bias)이 문제다.**
티처 포싱만 쓰면 모델은 "항상 올바른 앞부분"만 보고 학습한다.
그런데 추론할 때는 자기 예측을 먹으므로, 한 번 틀리면 계속 어긋난다.

`teacher_forcing_ratio=0.5`는 **절반씩 섞는** 타협이다.

**`torch.rand(1)`을 시점마다 뽑으므로 배치 전체가 같은 선택을 한다.**
샘플마다 다르게 하려면 `torch.rand(batch_size)`로 뽑아 마스크를 써야 한다.

**`outputs[:, 0, :]`는 0으로 남는다.** `<sos>` 자리라 예측이 없다.
그래서 손실을 계산할 때 잘라낸다.

```python
output_seq = output_seq[:, 1:, :].reshape(-1, output_dim)
trg_seq = trg_example[:, 1:].reshape(-1)
```

**`reshape(-1, output_dim)`로 펴는 게 관용구다.**
`CrossEntropyLoss`는 `(N, C)`와 `(N,)`을 받으므로 배치와 시점을 하나로 합친다.

```
(32, 11, 5000) → (352, 5000)
(32, 11)       → (352,)
```

### `ignore_index=0`

```python
criterion_seq2seq = nn.CrossEntropyLoss(ignore_index=0)
```

**패딩 토큰(0)은 손실에서 뺀다.** 6장에서 `trainId`의 255를 뺀 것과 같은 장치다.
안 그러면 모델이 "패딩을 잘 맞히는 법"을 배운다.

### 결과 — 딱 무작위다

```python
loss_seq.backward()
print(loss_seq.item())     # 8.520575523376465
```

```
ln(5000) = 8.5172
실제      = 8.5206
```

**어휘 5000개를 균등하게 찍었을 때의 손실과 일치한다.**

당연한 결과다. 입력이 난수이고 정답도 난수다.

```python
src_example = torch.randint(1, SRC_VOCAB_SIZE, (32, 10))
trg_example = torch.randint(1, TRG_VOCAB_SIZE, (32, 12))
```

**이 셀은 순전파·역전파가 도는지, 모양이 맞는지만 확인한다.**
9장의 멀티태스크 예제와 같은 성격이다.

**`ln(어휘 크기)`가 언어 모델의 기준선**이라는 걸 기억해 두면 좋다.
실제 번역 모델을 학습시켰는데 손실이 8.5 근처에 머물면 **아무것도 못 배우고 있는 것**이다.

### Seq2Seq의 한계와 그다음

```
인코더가 문장 전체를 벡터 한 쌍으로 압축한다
  → 문장이 길수록 앞부분 정보가 사라진다  (정보 병목)
```

**그래서 어텐션이 나왔다.**

```
Seq2Seq (10장)       : 디코더가 문맥 벡터 하나만 본다
Seq2Seq + 어텐션     : 디코더가 매 시점 인코더의 모든 출력을 본다
트랜스포머 (8장)      : LSTM을 아예 버리고 어텐션만 쓴다
```

이 노트북의 `Encoder`는 `outputs`도 돌려주는데 `Seq2Seq.forward`에서
`enc_outputs`를 받아 놓고 **쓰지 않는다.** 어텐션을 붙이려면 그게 필요하다.
자리는 마련되어 있는 셈이다.

---

## 이 장 정리

### 흐름

```
파이프라인    10-1   원문 → 토큰화 → 정제 → 벡터화 → 모델
   ↓
규칙 기반     10-2   사전 + 카운트. 빠르지만 사전에 없으면 못 잡는다
   ↓
LSTM 분류    10-3   임베딩 → LSTM → hidden[-1] → Linear
   ↓
Seq2Seq     10-4   인코더 문맥 벡터 → 디코더 한 스텝씩 + 티처 포싱
```

### Seq2Seq 뼈대

```python
enc_outputs, (hidden, cell) = encoder(src)
input = trg[:, 0]                                  # <sos>

for t in range(1, trg_len):
    output, hidden, cell = decoder(input, hidden, cell)
    outputs[:, t, :] = output
    input = trg[:, t] if teacher_force else output.argmax(1)
```

### 손실 계산 시 모양

```python
output = output[:, 1:, :].reshape(-1, vocab_size)   # <sos> 제외
target = trg[:, 1:].reshape(-1)
loss = nn.CrossEntropyLoss(ignore_index=PAD)(output, target)
```

### 학습 전 손실의 기댓값 (누적)

<table fit-page-width="true" header-row="true">
<tr><td>과제</td><td>기댓값</td><td>이 장의 실제값</td></tr>
<tr><td>이진 분류</td><td>ln(2) = 0.693</td><td>0.575 (조금 학습됨)</td></tr>
<tr><td>어휘 5000 생성</td><td>ln(5000) = 8.517</td><td><strong>8.521</strong></td></tr>
</table>

### 규칙 기반 vs 학습 기반

<table fit-page-width="true" header-row="true">
<tr><td></td><td>규칙 기반 (10-2)</td><td>LSTM (10-3)</td><td>BERT (8장)</td></tr>
<tr><td>학습 데이터</td><td>필요 없음</td><td>수천 건</td><td>수백 건 (미세조정)</td></tr>
<tr><td>사전에 없는 말</td><td><strong>못 잡음</strong></td><td>문맥으로 추정</td><td>잘 추정</td></tr>
<tr><td>부정·반어</td><td>못 잡음</td><td>어느 정도</td><td>잘 잡음</td></tr>
<tr><td>설명 가능성</td><td><strong>완전</strong></td><td>어텐션으로 일부</td><td>어텐션으로 일부</td></tr>
<tr><td>속도</td><td>매우 빠름</td><td>빠름</td><td>느림</td></tr>
</table>

### 자주 틀리는 것

- 한국어를 `split()`으로 토큰화한다 (조사가 붙어 있다)
- 규칙 기반의 **경계값 비교**를 `>`로 쓴다 (0.8 > 0.8 은 거짓)
- 키워드 **종류**만 세고 횟수를 안 센다
- 학습 데이터로 정확도를 재고 성능이라 부른다
- **가상 데이터로 그린 손실 곡선**을 실제 결과로 읽는다
- 패딩을 `pack`이나 마스크로 걸러내지 않는다
- 디코더 손실에서 `<sos>` 자리를 안 잘라낸다
- `ignore_index`로 패딩을 안 뺀다
- 티처 포싱만 100%로 쓴다 (추론 때 무너진다)
