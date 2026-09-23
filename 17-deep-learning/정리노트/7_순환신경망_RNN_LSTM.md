# 7. 순환 신경망 — 시퀀스를 다루는 법

**실습 파일**: `7.ipynb` (09/09 ~ 09/10 수업)

1~6장은 전부 **이미지**였다. 입력 순서가 없다.
7장부터는 **순서가 의미를 갖는 데이터**를 다룬다. 주가, 문장, 센서 신호.

| 절 | 내용 |
|---|---|
| 7-1 | 기울기 클리핑 — RNN 전에 먼저 나오는 장치 |
| 7-2 | RNN과 LSTM의 입출력 모양 |
| 7-3 | 길이가 다른 시퀀스 — pack / pad |
| 7-4 | 주가 예측 (LSTM 회귀) |
| 7-5 | 감성 분석 + 어텐션 |
| 7-6 | 하이퍼파라미터 탐색 |
| 7-7 | LSTM 오토인코더로 이상치 탐지 |

---

## 7-1. 기울기 클리핑

### 왜 RNN 이야기의 시작이 클리핑인가

RNN은 같은 가중치를 시퀀스 길이만큼 **반복해서 곱한다.**
길이가 100이면 역전파에서 같은 행렬이 100번 곱해진다.

```
가중치 크기가 1보다 조금 크면  →  1.1^100 ≈ 13,780     (기울기 폭발)
가중치 크기가 1보다 조금 작으면 →  0.9^100 ≈ 0.000027  (기울기 소실)
```

**폭발은 클리핑으로 막고, 소실은 LSTM 구조로 막는다.** 7장의 두 축이 이것이다.

### 코드

```python
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**`backward()` 다음, `step()` 앞**에 넣는다. 이 순서가 아니면 의미가 없다.
기울기가 계산된 뒤에 잘라야 하고, 자른 값으로 갱신해야 한다.

**`max_norm`은 개별 기울기가 아니라 전체 노름 기준이다.**

```
total_norm = 모든 파라미터 기울기를 한 줄로 이어 붙인 벡터의 L2 노름
total_norm > max_norm 이면  →  전부 (max_norm / total_norm) 배
```

**방향은 그대로 두고 길이만 줄인다.** 이게 값마다 자르는 `clip_grad_value_`와 다른 점이다.

<table fit-page-width="true" header-row="true">
<tr><td>함수</td><td>기준</td><td>방향</td></tr>
<tr><td>clip_grad_norm_</td><td>전체 L2 노름</td><td>유지된다</td></tr>
<tr><td>clip_grad_value_</td><td>원소별 절댓값</td><td>바뀔 수 있다</td></tr>
</table>

**끝의 밑줄(`_`)이 in-place라는 뜻이다.** 밑줄 없는 `clip_grad_norm`은 폐기 예정이다.
(7-4의 노트북 출력에 그 `FutureWarning`이 실제로 찍혀 있다.)

### 확인용 미니 예제

```python
X = torch.randn(1000, 10)
y = torch.sum(X[:, :5], dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1
```

**정답이 앞 5개 열의 합이다.** 뒤 5개는 잡음이다.
모델이 제대로 학습하면 뒤 5개 가중치를 0에 가깝게 만들어야 한다.

```
1 5.2650
2 4.7585
3 4.2960
4 3.7122
5 3.0941
학습완료
```

**5에폭으로는 한참 모자라다.** 5개 표준정규의 합은 분산이 5라
아무것도 학습 못 한 모델의 MSE가 약 5.0이다. 3.09면 절반쯤 온 것이다.
`lr=0.01`의 SGD라 느리다. Adam이었으면 훨씬 빨랐을 것이다.

---

## 7-2. RNN과 LSTM의 입출력 모양

### 모양부터 외운다

```python
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

x = torch.randn(32, 10, input_size)      # (배치, 시점, 특징)
output, (h_n, c_n) = lstm(x)

print(output.shape)   # torch.Size([32, 10, 20])
print(h_n.shape)      # torch.Size([1, 32, 20])
print(c_n.shape)      # torch.Size([1, 32, 20])
```

<table fit-page-width="true" header-row="true">
<tr><td>이름</td><td>모양</td><td>뜻</td></tr>
<tr><td>output</td><td>(배치, 시점, 은닉)</td><td><strong>모든 시점</strong>의 마지막 층 은닉 상태</td></tr>
<tr><td>h_n</td><td>(층×방향, 배치, 은닉)</td><td><strong>마지막 시점</strong>의 모든 층 은닉 상태</td></tr>
<tr><td>c_n</td><td>(층×방향, 배치, 은닉)</td><td>마지막 시점의 셀 상태 (LSTM만)</td></tr>
</table>

**`output`과 `h_n`은 축 순서가 다르다.** 여기서 제일 많이 헷갈린다.
`output`은 배치가 앞(batch_first 때문), `h_n`은 **항상 층이 앞**이다.
`batch_first=True`를 줘도 `h_n`의 순서는 안 바뀐다.

**1층 단방향이면 `output[:, -1, :]`와 `h_n[-1]`이 같은 값이다.**

```python
torch.allclose(output[:, -1, :], h_n[-1])   # True (단방향·마지막 층 기준)
```

### 어느 쪽을 쓰나

```
문장 전체를 한 벡터로 요약 → h_n[-1]  또는 output[:, -1, :]   (분류, 회귀)
시점마다 답이 필요          → output                          (품사 태깅, 시퀀스 생성)
어텐션을 걸 것              → output                          (7-5)
```

### RNN vs LSTM vs GRU

<table fit-page-width="true" header-row="true">
<tr><td></td><td>게이트</td><td>상태</td><td>장기 의존성</td></tr>
<tr><td>nn.RNN</td><td>없음</td><td>h</td><td>약함 (기울기 소실)</td></tr>
<tr><td>nn.LSTM</td><td>입력·망각·출력</td><td>h, c</td><td>강함</td></tr>
<tr><td>nn.GRU</td><td>리셋·업데이트</td><td>h</td><td>LSTM과 비슷, 더 가벼움</td></tr>
</table>

**LSTM의 핵심은 셀 상태 `c`다.** `c`는 곱셈이 아니라 **덧셈으로 갱신**되기 때문에
기울기가 시점을 거슬러 갈 때 지수적으로 줄지 않는다.

```
RNN : h_t = tanh(W h_{t-1} + U x_t)        ← W가 계속 곱해진다
LSTM: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t      ← 망각 게이트가 1이면 그대로 흐른다
```

**망각 게이트 `f_t`가 1에 가까우면 정보가 손실 없이 지나간다.**
ResNet의 잔차 연결과 같은 아이디어를 시간축에 쓴 것이다.

### 주의 — 오타 하나로 안 도는 클래스

```python
self.rnn = nn.RNN(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  bathc_first = True)        # ← batch_first 오타
```

`nn.RNN`은 모르는 키워드를 받으면 `TypeError`를 낸다.
이 셀이 에러 없이 실행된 건 **클래스를 정의만 하고 한 번도 만들지 않았기 때문**이다.

`RNNModel(...)`을 호출하는 순간 터진다. 이 노트북에서는 끝까지 안 쓰인다.
(원본은 그대로 두었다. `batch_first=True`로 고치면 된다.)

**클래스를 정의한 셀이 에러 없이 돌았다고 그 클래스가 맞는 건 아니다.**
정의 시점에는 몸통이 실행되지 않는다. 최소한 한 번은 인스턴스를 만들어 봐야 한다.

---

## 7-3. 길이가 다른 시퀀스 — pack과 pad

### 문제

문장 3개의 길이가 5, 3, 7이라고 하자. 배치로 묶으려면 길이를 맞춰야 한다.

```python
data = [torch.rand(5, 10), torch.randn(3, 10), torch.randn(7, 10)]
lengths = [5, 3, 7]

padded_seq = pad_sequence(data, batch_first=True)
print(padded_seq.shape)      # torch.Size([3, 7, 10])
```

가장 긴 것(7)에 맞춰 0으로 채운다. **그런데 그 0도 RNN이 읽는다.**
짧은 문장의 마지막 은닉 상태가 0 입력으로 계속 덮여 망가진다.

### 해법 — PackedSequence

```python
packed_seq = pack_padded_sequence(padded_seq, lengths,
                                  batch_first=True,
                                  enforce_sorted=False)
```

```
PackedSequence(
  data=tensor([...]),
  batch_sizes=tensor([3, 3, 3, 2, 2, 1, 1]),
  sorted_indices=tensor([2, 0, 1]),
  unsorted_indices=tensor([1, 2, 0]))
```

**`batch_sizes`를 읽으면 구조가 보인다.**

```
시점  1  2  3  4  5  6  7
살아있는 문장 수
      3  3  3  2  2  1  1
         ↑        ↑     ↑
      전부 살아있음  |     길이 7짜리만
                길이 3짜리 끝남
```

RNN은 시점마다 `batch_sizes[t]`개만 계산한다. **패딩을 아예 안 읽는다.**

**`enforce_sorted=False`가 실용적으로 중요하다.**
원래 이 API는 길이 내림차순 정렬을 요구한다.
`False`를 주면 내부에서 정렬하고(`sorted_indices`),
되돌릴 순서(`unsorted_indices`)도 같이 들고 다닌다. 직접 정렬할 필요가 없다.

### 다시 펴기

```python
output, hidden = rnn(packed_seq)         # 입력이 packed면 출력도 packed
unpacked_output, unpacked_lengths = pad_packed_sequence(output, batch_first=True)

print(unpacked_output.shape)   # torch.Size([3, 7, 20])
print(unpacked_lengths)        # tensor([5, 3, 7])
```

**원래 순서와 원래 길이가 복원된다.** `lengths`가 `[5, 3, 7]` 그대로 나온다.

정리하면 이 흐름이다.

```
리스트 → pad_sequence → pack_padded_sequence → RNN → pad_packed_sequence → 텐서
       (길이 맞춤)      (패딩 무시)                    (다시 펴기)
```

**`h_n`은 pack을 쓰면 자동으로 "각 문장의 진짜 마지막 시점"이 된다.**
이게 pack을 쓰는 가장 큰 이유다. 안 쓰면 `output[:, -1, :]`이 패딩 자리를 가리킨다.

### 임베딩과 패딩

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
```

**`padding_idx=0`은 0번 토큰의 임베딩을 0 벡터로 고정하고 학습에서 제외한다.**
`<pad>`를 0번에 두는 관례가 여기서 나온다.

```python
mask = (padded_seq != 0).float()
```

마스크는 "여기가 진짜 토큰"을 표시하는 0/1 텐서다. 7-5의 어텐션에서 쓴다.

### 주의 — 첫 번째 `build_vocab`은 첫 문장만 본다

```python
def build_vocab(tokenized_texts, min_freq=1):
    word_counts = {}
    for text in tokenized_texts:
        for token in text:
            word_counts[token] = word_counts.get(token, 0) + 1
        vocab = {'<pad>': 0, '<unk>': 1}     # ← for 안
        ...
        return vocab                          # ← for 안
```

`vocab` 생성과 `return`이 **바깥 for문 안**에 들어가 있다.
첫 문장을 처리하자마자 돌려주므로 **나머지 문장의 단어가 전부 `<unk>`**가 된다.

09/10 수업 셀(22번)에서 들여쓰기를 고쳐 다시 정의한다.
실제로 쓰인 건 고친 쪽이라 결과에는 영향이 없다.

### 주의 — `pack_sequence`와 `pack_padded_sequence`는 다른 함수다

```python
packed_embedded = nn.utils.rnn.pack_sequence(embedded, text_length.cpu(),
                                             batch_first=True,
                                             enforce_sorted=False)
```

`pack_sequence`는 **패딩 안 된 텐서들의 리스트**를 받는다. 길이 인자가 없다.
여기 필요한 건 `pack_padded_sequence`다. 호출하면 `TypeError`가 난다.

이 `SentimentRNN`도 정의만 되고 쓰이지 않았다.
실제로 쓰이는 7-5의 `AttentionRNN`은 `pack_padded_sequence`로 제대로 되어 있다.

---

## 7-4. 주가 예측 — LSTM 회귀

### 데이터

```python
ticker = '005930.KS'                 # 삼성전자
stock_data = yf.download(ticker, start='2018-01-01', end='2023-01-01')
df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
```

`yfinance`로 5년치 일봉을 받는다. 특징 5개(시가·고가·저가·종가·거래량).

### 슬라이딩 윈도우

```python
def create_sequence(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['Close']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

X, y = create_sequence(df_scaled, seq_length=20)
print(X.shape, y.shape)
```

```
(1210, 20, 5) (1210, 1)
```

**20일치를 보고 21일째 종가를 맞힌다.** 시계열을 지도학습으로 바꾸는 표준 방법이다.

```
X[0] = 1~20일  →  y[0] = 21일 종가
X[1] = 2~21일  →  y[1] = 22일 종가
```

### 분할은 반드시 시간 순서대로

```python
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]                                   # 앞 70%
X_val   = X[train_size:train_size + val_size]              # 다음 15%
X_test  = X[train_size + val_size:]                        # 뒤 15%
```

**`shuffle=True`로 나누면 안 된다.** 미래 데이터로 학습해 과거를 맞히는 꼴이 된다.
이미지 분류에서 늘 하던 랜덤 분할이 시계열에서는 치명적이다.

### 모델

```python
class StockModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]      # 마지막 시점만
        out = self.dropout(last_time_step)
        return self.fc(out)
```

**`lstm_out[:, -1, :]`가 "20일을 다 읽은 뒤의 요약"이다.** 그걸 하나의 수로 바꾼다.

**`dropout if num_layers > 1 else 0`이 관용구다.**
`nn.LSTM`의 `dropout`은 **층 사이**에 걸리므로 1층이면 걸 데가 없다.
1층에 0이 아닌 값을 주면 경고가 뜬다. 그래서 이렇게 방어한다.

여기서는 `num_layers=2, hidden_size=64`.

### 학습과 평가

```
1   train 0.1714  val 0.0362
2   train 0.0256  val 0.0841
3   train 0.0027  val 0.0073
...
99  train 0.00047 val 0.00100
100 train 0.00076 val 0.00180
```

```
MSE  1111586.51
RMSE 1054.32
MAE   843.83
R²      0.9083
```

**스케일을 되돌린 뒤에 지표를 잰다.** 0~1로 정규화된 상태의 MSE는 해석이 안 된다.

```python
def inverse_transform_close(data):
    dummy = np.zeros((len(data), 5))
    dummy[:, 3] = data.flatten()          # Close가 3번 열
    return scaler.inverse_transform(dummy)[:, 3]
```

**`MinMaxScaler`는 5개 열을 한꺼번에 학습했으므로 역변환도 5열을 요구한다.**
종가 하나만 되돌리려고 나머지를 0으로 채운 더미를 만든다. 흔히 쓰는 우회법이다.
(깔끔하게 하려면 종가 전용 스케일러를 따로 두면 된다.)

**RMSE 1,054원**이 실감나는 숫자다. 삼성전자 주가가 5~8만 원대이므로 약 1.5~2%다.

### 이 결과를 어떻게 읽어야 하나

**R² 0.908은 좋아 보이지만 그대로 믿으면 안 된다.** 이유가 셋이다.

**(1) 스케일러가 전체 데이터로 학습됐다.**

```python
df_scaled = pd.DataFrame(scaler.fit_transform(df), ...)    # ← 분할 전
X, y = create_sequence(df_scaled, seq_length)
train_size = int(len(X) * 0.7)                             # ← 분할은 그다음
```

`fit_transform`이 **테스트 구간의 최댓값·최솟값까지 보고** 정규화 기준을 정했다.
미래 정보가 훈련 데이터로 새어 들어간 것이다. (데이터 누수)

올바른 순서는 이렇다. (원본은 그대로 두었다.)

```python
scaler.fit(df.iloc[:train_end])          # 훈련 구간으로만 fit
df_scaled = scaler.transform(df)         # 전체에 적용
```

**(2) 기준선과 비교하지 않았다.**

주가 예측에서 "오늘 종가 = 어제 종가"라고 찍는 것(나이브 예측)만으로도
R²가 0.95 이상 나오는 경우가 흔하다. 주가가 연속적이기 때문이다.

**R² 0.908은 나이브 예측보다 못할 수도 있는 값이다.**
반드시 기준선을 같이 계산해서 비교해야 한다.

```python
naive = y_test_orig[:-1]                 # 어제 값을 그대로
print(r2_score(y_test_orig[1:], naive))  # ← 이걸 넘겨야 의미가 있다
```

**(3) 예측 그래프가 한 칸 밀려 보인다면 그게 증거다.**
모델이 "직전 값을 따라 하기"를 학습하면 그래프가 실제보다 하루 늦게 움직인다.

**시계열 회귀에서는 R²보다 방향 정확도(오를지 내릴지)를 보는 게 정직하다.**

### 주의 — 폐기된 함수

```python
nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
```

```
FutureWarning: `torch.nn.utils.clip_grad_norm` is now deprecated
in favor of `torch.nn.utils.clip_grad_norm_`.
```

7-1에서 쓴 밑줄 버전이 맞다.

### 주의 — 미니배치가 섞이지 않는다

```python
for i in range(0, len(X_train), batch_size):
    batch_X = X_train[i:i + batch_size]
```

`DataLoader` 없이 슬라이싱으로 배치를 만든다.
**에폭마다 같은 순서, 같은 묶음**이라 셔플 효과가 없다.
시계열이라 순서를 지키는 게 맞는 경우도 있지만,
윈도우로 잘라낸 시점에서 각 샘플은 독립이므로 섞어도 된다.

---

## 7-5. 감성 분석 + 어텐션

### 데이터

영화 리뷰 10개, 긍정/부정 라벨. 토큰화는 `text.split()`으로 공백 분리.

```python
vocab = build_vocab(tokenized_texts)
print(len(vocab))       # 78
```

**단어 78개짜리 장난감 데이터다.** 구조를 배우는 게 목적이지 성능이 목적이 아니다.

### 어텐션 층

```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, attention_dim)
        self.context = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x, mask=None):
        attention = torch.tanh(self.attention(x))            # (B, T, A)
        attention = self.context(attention).squeeze(2)       # (B, T)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention_weights = torch.softmax(attention, dim=1)  # (B, T)
        weighted_output = torch.bmm(
            attention_weights.unsqueeze(1), x).squeeze(1)    # (B, H)

        return weighted_output, attention_weights
```

**이게 이 장에서 가장 중요한 30줄이다.** 모양을 따라가 보자.

```
x                          (B, T, H)   LSTM의 모든 시점 출력
  ↓ Linear(H → A), tanh
                           (B, T, A)
  ↓ Linear(A → 1), squeeze
attention                  (B, T)      시점마다 점수 하나
  ↓ softmax(dim=1)
attention_weights          (B, T)      합이 1인 가중치
  ↓ bmm
weighted_output            (B, H)      시점들의 가중 평균
```

**`squeeze(1)` 대신 `mean(dim=1)`을 쓰면 그냥 평균이다.**
어텐션은 **평균의 가중치를 데이터가 정하게 만든 것**이다. 그게 전부다.

**`torch.bmm`은 배치 행렬곱이다.** `(B, 1, T) @ (B, T, H) = (B, 1, H)`.
배치마다 독립적으로 행렬곱을 한다. `torch.matmul`도 되지만 의도가 분명해진다.

**`masked_fill(mask == 0, -1e10)`이 패딩 처리의 핵심이다.**

```
패딩 자리에 -1e10을 넣는다
   ↓ softmax
exp(-1e10) ≈ 0  →  가중치가 0이 된다
```

**softmax 전에 넣어야 한다.** 후에 0을 곱하면 합이 1이 아니게 된다.
`-1e10` 대신 `float('-inf')`를 쓰면 한 행이 전부 패딩일 때 nan이 난다.
그래서 아주 작은 유한값을 쓰는 게 관행이다.

### 양방향 LSTM과 결합

```python
self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
self.attention = AttentionLayer(hidden_dim * 2 if bidirectional else hidden_dim, attention_dim)
```

**양방향이면 은닉 크기가 두 배다.** 정방향 출력과 역방향 출력이 이어 붙는다.
`hidden_dim=128`이면 `output`이 256채널. 뒤의 `Linear`도 전부 256을 받아야 한다.

```
AttentionRNN(
  (embedding): Embedding(78, 100, padding_idx=0)
  (rnn): LSTM(100, 128, batch_first=True, bidirectional=True)
  (attention): AttentionLayer(...)
  (fc): Linear(in_features=256, out_features=1, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
)
```

`fc`의 `in_features=256`이 그 결과다.

### forward 전체 흐름

```python
def forward(self, text, text_lengths):
    embedded = self.embedding(text)
    packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(),
                                           batch_first=True, enforce_sorted=False)
    packed_output, _ = self.rnn(packed_embedded)
    output, _ = pad_packed_sequence(packed_output, batch_first=True)

    mask = (text != 0)
    weighted_output, attention_weights = self.attention(output, mask)
    weighted_output = self.dropout(weighted_output)
    return self.fc(weighted_output), attention_weights
```

**`text_lengths.cpu()`가 필요한 이유** — `pack_padded_sequence`의 길이 인자는
반드시 CPU 텐서여야 한다. GPU에 있으면 에러가 난다. 자주 걸리는 함정이다.

**`h_n`이 아니라 `output` 전체를 쓴다.** 어텐션은 모든 시점을 봐야 하니까.

### 학습 결과

```
10  0.1974
20  0.0093
...
100 0.0012
```

```
Text: This movie is truly a masterpiece with excellent directing and standout acting
Actual sentiment: Positive
Predicted sentiment: Positive (Confidence: 0.9998)
```

**10문장을 100에폭 돌렸으니 외운 것이다.** 손실 0.0012는 완전 암기의 신호다.
검증셋도 없다. 성능 수치로 볼 값이 아니다.

**여기서 볼 것은 어텐션 가중치다.**

```python
attention = attention.squeeze().numpy()
```

문장의 각 단어에 가중치가 하나씩 붙는다. 합은 1이다.
`plot_attention`이 그 가중치로 단어의 배경색을 칠한다. 진할수록 많이 본 단어다.

**이게 어텐션의 진짜 값어치다 — 설명 가능성.**
5장의 Grad-CAM이 이미지에서 한 일을 텍스트에서 하는 것이다.

<table fit-page-width="true" header-row="true">
<tr><td></td><td>Grad-CAM (5장)</td><td>어텐션 가중치 (7장)</td></tr>
<tr><td>대상</td><td>이미지 픽셀</td><td>문장의 단어</td></tr>
<tr><td>얻는 법</td><td>역전파 기울기 (사후)</td><td>순전파에서 그냥 나옴</td></tr>
<tr><td>학습에 쓰이나</td><td>아니다</td><td><strong>그렇다</strong> — 모델의 일부</td></tr>
</table>

**어텐션은 공짜로 해석을 준다.** 따로 계산할 필요 없이 `forward`가 돌려준다.

---

## 7-6. 하이퍼파라미터 탐색

### 탐색 공간

```python
hyperparameters = {
    'hidden_size':   [64, 128, 256, 512],
    'num_layers':    [1, 2, 3, 4],
    'dropout':       [0.0, 0.1, 0.2, 0.3, 0.5],
    'bidirectional': [True, False],
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'batch_size':    [16, 32, 64, 128],
    # ...
}
```

전부 곱하면 조합이 수십만 개다. **전수 탐색은 불가능하다.**

<table fit-page-width="true" header-row="true">
<tr><td>방법</td><td>원리</td><td>언제</td></tr>
<tr><td>Grid Search</td><td>격자를 전부 시도</td><td>후보가 적을 때</td></tr>
<tr><td>Random Search</td><td>무작위로 N개</td><td>대부분의 경우 이게 낫다</td></tr>
<tr><td>Bayesian Optimization</td><td>지금까지 결과로 다음을 예측</td><td>1회 평가가 비쌀 때</td></tr>
</table>

**중요한 파라미터가 소수일 때는 랜덤 탐색이 격자 탐색보다 효율적이다.**
격자는 덜 중요한 축에도 같은 수의 점을 쓰기 때문이다.

### 랜덤 샘플링을 곁들인 격자 탐색

```python
from sklearn.model_selection import ParameterGrid

param_combinations = list(ParameterGrid(param_grid))
if len(param_combinations) > max_trials:
    param_combinations = random.sample(param_combinations, max_trials)
```

**`ParameterGrid`가 딕셔너리를 조합 리스트로 펼쳐 준다.**
너무 많으면 `random.sample`로 추려서 사실상 랜덤 탐색이 된다.

### 베이지안 최적화

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

self.space = [Integer(64, 512, name='hidden_size'),
              Integer(1, 4, name='num_layers'),
              Real(0.0, 0.5, name='dropout'),
              Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
              Categorical([16, 32, 64, 128], name='batch_size'),
              Real(0, 5.0, name='grad_clip')]

res = gp_minimize(self.objective, self.space, n_calls=50, random_state=42)
```

**원리:** 지금까지의 (설정, 점수) 쌍으로 가우시안 과정을 학습해
"다음에 어디를 시도하면 가장 이득일까"를 예측한다.
좋아 보이는 곳(활용)과 안 가 본 곳(탐험)을 저울질한다.

**`gp_minimize`는 최소화 함수다.** 정확도처럼 클수록 좋은 지표는 부호를 뒤집어 넘긴다.

**`prior='log-uniform'`이 학습률에서 특히 중요하다.**
학습률은 1e-4와 1e-3의 차이가 1e-3과 1e-2의 차이와 같은 크기다.
선형 균등으로 뽑으면 큰 값 쪽만 잔뜩 뽑힌다.

### 주의 — 이 절의 코드는 뼈대만이다

```python
score = -sum(params.values())            # 가짜 점수
time.sleep(0.1)                          # 학습하는 척
```

**실제로 모델을 학습시키지 않는다.** `_train_and_evaluate`가 비어 있다.
탐색 루프의 구조를 보여 주는 예제다. 두 셀 모두 출력이 없다(정의만 실행).

여기에 오타와 논리 오류가 몇 개 있다. (원본은 그대로 두었다.)

<table fit-page-width="true" header-row="true">
<tr><td>위치</td><td>문제</td><td>결과</td></tr>
<tr><td>grid_search</td><td><code>best_score = float('inf')</code>인데 <code>if score &gt; best_score</code></td><td>조건이 절대 참이 안 됨 → best_params가 None</td></tr>
<tr><td>grid_search</td><td><code>raise NotImplemented</code></td><td>NotImplemented는 예외가 아님 → TypeError</td></tr>
<tr><td>space</td><td><code>prior='log-unifrom'</code></td><td>오타 → skopt가 값을 거부</td></tr>
<tr><td>optimize</td><td><code>best_params.item()</code></td><td><code>.items()</code>여야 함 → AttributeError</td></tr>
<tr><td>hyperparameters</td><td><code>'sequence_lenght'</code></td><td>오타 (length)</td></tr>
</table>

**최소화/최대화 방향을 헷갈리는 건 탐색 코드에서 가장 흔한 버그다.**
`float('inf')`로 시작했으면 `<`, `float('-inf')`로 시작했으면 `>`를 쓴다.

### 주의 — 스케줄러의 들여쓰기

```python
def step(self, current_score):
    self.current_epoch += 1
    if self.current_epoch <= self.warmup_epochs:
        lr_scale = self.current_epoch / self.warmup_epochs
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
            return                                    # ← for 안
        is_improvement = (...)                        # ← if 안
        ...
```

문제가 둘이다.

```
1. return이 for 루프 안에 있다   → 파라미터 그룹이 여럿이면 첫 번째만 조정된다
2. 개선 판정 블록이 if 안에 있다  → 워밍업이 끝나면 step()이 아무것도 안 한다
```

`param_group['initial_lr']`도 기본 옵티마이저에는 없는 키다.
직접 넣어 주거나 `optimizer.defaults['lr']`을 써야 `KeyError`가 안 난다.

**들여쓰기 한 칸이 기능을 통째로 없앤다.** 파이썬에서 특히 조심할 부분이다.

---

## 7-7. LSTM 오토인코더로 이상치 탐지

### 발상

**"정상 데이터만 잘 복원하도록 학습시키면, 복원이 안 되는 지점이 이상치다."**

라벨이 없어도 된다. 이상치가 무엇인지 몰라도 된다. 비지도 학습이다.

```
정상 구간  →  잘 복원됨  →  오차 작음
이상 구간  →  못 복원함  →  오차 큼  →  임계값 넘으면 이상치
```

### 인공 시계열

```python
def generate_time_series(n_samples=1000, anomaly_positions=[200, 400, 600, 800]):
    t = np.linspace(0, 10, n_samples)
    series = 0.8 * np.sin(t) + 0.2 * np.sin(5 * t) + 0.1 * np.random.randn(n_samples)
    series += 0.005 * np.arange(n_samples)          # 추세

    for pos in anomaly_positions:
        series[pos] += 1.5 * np.random.rand() + 0.5
        if np.random.rand() > 0.5:
            series[pos + 1] -= 1.2 * np.random.rand() + 0.3
    return series
```

**주기 둘 + 잡음 + 추세**로 현실적인 신호를 만들고, 정해진 4곳에 튀는 값을 심는다.
정답을 알고 만든 데이터라 탐지 결과를 검증할 수 있다.

### 구조

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        decoder_input = hidden.permute(1, 0, 2).repeat(1, self.sequence_length, 1)
        output, _ = self.decoder(decoder_input)
        return self.fc(output)
```

**병목이 시간축에 있다.**

```
입력       (B, 20, 1)   20시점
  ↓ encoder — 마지막 은닉만 남긴다
hidden     (1, B, 32)   ← 시간이 사라졌다. 20시점이 벡터 하나로 압축됨
  ↓ permute(1,0,2) → (B, 1, 32),  repeat → (B, 20, 32)
  ↓ decoder
출력       (B, 20, 1)   다시 20시점
```

**`hidden.permute(1, 0, 2)`가 필요한 이유** — `h_n`은 `(층, 배치, 은닉)`인데
`batch_first=True`인 디코더는 `(배치, 시점, 특징)`을 원한다. 축을 바꿔 준다.
7-2에서 "`output`과 `h_n`은 축 순서가 다르다"고 한 게 여기서 걸린다.

**`repeat(1, sequence_length, 1)`은 같은 벡터를 20번 복제한다.**
디코더는 매 시점 같은 요약 벡터를 받고, 자기 은닉 상태만으로 순서를 복원한다.
가장 단순한 시퀀스 디코딩 방식이다.

### 학습 곡선이 특이하다

```
10   0.001943
20   0.001876
...
100  0.001896        ← 90에폭 동안 제자리
110  0.001389        ← 여기서 빠져나옴
120  0.000229
130  0.000119
...
200  0.0000388
```

**100에폭 가까이 0.0019에서 멈춰 있다가 갑자기 내려간다.**

이건 **평평한 지역(plateau)에 갇혀 있다가 탈출한** 전형적인 모양이다.
초기에는 "전부 평균값을 출력"하는 안전한 해에 머문다.
그 상태의 손실이 데이터 분산 정도인데, 거기서 벗어나야 실제 파형을 그리기 시작한다.

**50에폭에서 멈췄다면 "학습이 안 된다"고 결론 냈을 것이다.**
손실이 평평하다고 바로 포기하면 안 되는 사례다.
(반대로 `ln(클래스 수)` 근처에 고정된 분류 손실은 진짜 안 되는 것이다. 구분해야 한다.)

### 임계값

```python
def set_threshold(errors, z_score=3.0):
    return np.mean(errors) + z_score * np.std(errors)
```

```
threshold = 0.04613
탐지된 이상치 6개
```

**평균 + 3σ가 통계의 기본 규칙이다.** 정규분포라면 0.3%만 이 밖에 있다.
1,000개 중 3개 정도가 기대값인데 6개가 나왔다.

**심은 이상치는 4곳이다.** 6개가 나온 이유를 생각해 보면 이렇다.

```
- 이상치 하나가 윈도우 20개에 걸쳐 영향을 준다 → 인접 지점이 같이 잡힐 수 있다
- pos+1 에도 튐을 넣은 경우가 있다 (np.random.rand() > 0.5)
- 복원 오차 분포는 정규분포가 아니다 → 3σ가 정확히 0.3%가 아니다
```

**z_score는 조정 가능한 손잡이다.** 5장의 임계값 이야기와 완전히 같다.

```
z를 낮춘다 → 더 많이 잡는다 → 놓치는 건 줄고 오탐이 는다
z를 높인다 → 확실한 것만    → 오탐은 줄고 놓치는 게 는다
```

### 실무에서 달라지는 점

**이 예제는 이상치가 섞인 데이터로 학습했다.**
제대로 하려면 **정상 구간만으로 학습**해야 한다.
이상치까지 복원할 수 있게 되면 오차가 안 커져서 탐지가 안 된다.

지금은 이상치가 1,000개 중 4개라 영향이 미미해서 동작한 것이다.

### 주의 — 오차를 두 번 계산한다

```python
errors = [...]                                    # 1,000번 추론
threshold = set_threshold(errors, z_score=3.0)
anomalies = detect_anomalies(model, data_scaled, window_size, threshold)  # 또 1,000번
```

`detect_anomalies`가 같은 추론을 반복한다. `errors`를 재사용하면 절반으로 준다.

```python
anomalies = [(i + window_size - 1, e) for i, e in enumerate(errors) if e > threshold]
```

### 주의 — 작은 실수들

```python
plt.title(title, fontsize=6)        # 제목이 글씨 6pt — 안 보인다
plt.annotate(f'{score:3f}', ...)    # '{score:.3f}'여야 한다
```

`f'{score:3f}'`는 **최소 폭 3, 소수점 6자리**로 해석된다.
`0.046125`처럼 길게 찍힌다. 점 하나가 빠졌다.

---

## 이 장 정리

### 흐름

```
클리핑          7-1   backward 다음, step 앞. 노름 기준
   ↓
RNN/LSTM 모양   7-2   output (B,T,H) / h_n (층,B,H) — 축 순서가 다르다
   ↓
pack/pad       7-3   길이가 다른 시퀀스. enforce_sorted=False
   ↓
주가 회귀       7-4   윈도우 → LSTM → 마지막 시점 → Linear(1)
   ↓
어텐션          7-5   시점 가중 평균 + 마스크. 해석까지 공짜로
   ↓
탐색           7-6   랜덤/베이지안. 최소화 방향을 헷갈리지 말 것
   ↓
오토인코더      7-7   복원 오차로 이상치. 평균+3σ
```

### 모양 요약

```
# (배치, 시점, 특징) — batch_first=True 기준
output, (h_n, c_n) = lstm(x)

output  (B, T, H)          모든 시점, 마지막 층
h_n     (L*D, B, H)        마지막 시점, 모든 층   ← batch_first와 무관
c_n     (L*D, B, H)        LSTM만

# 양방향이면 output의 H가 2배, h_n의 L*D가 2배
```

### 시퀀스 배치 만들기

```python
padded = pad_sequence(list_of_tensors, batch_first=True)
packed = pack_padded_sequence(padded, lengths.cpu(),
                              batch_first=True, enforce_sorted=False)
out, h = rnn(packed)
out, lens = pad_packed_sequence(out, batch_first=True)
```

### 어텐션 3줄 요약

```python
scores = self.context(torch.tanh(self.attention(x))).squeeze(2)  # (B, T)
scores = scores.masked_fill(mask == 0, -1e10)                    # 패딩 제거
w = torch.softmax(scores, dim=1)                                 # 합이 1
out = torch.bmm(w.unsqueeze(1), x).squeeze(1)                    # 가중 평균
```

### 시계열에서 특히 조심할 것

<table fit-page-width="true" header-row="true">
<tr><td>실수</td><td>왜 위험한가</td></tr>
<tr><td>랜덤 분할</td><td>미래로 과거를 맞히게 된다</td></tr>
<tr><td>전체 데이터로 스케일러 fit</td><td>미래의 min/max가 새어 들어간다</td></tr>
<tr><td>기준선 없이 R²만 본다</td><td>나이브 예측만 못할 수 있다</td></tr>
<tr><td>정확도만 본다</td><td>방향(상승/하락)을 못 맞히면 쓸모없다</td></tr>
</table>

### 자주 틀리는 것

- `output`과 `h_n`의 축 순서를 헷갈린다 (`batch_first`는 `h_n`에 안 걸린다)
- `pack_padded_sequence`에 GPU 텐서를 넘긴다 → **`.cpu()`** 필요
- `pack_sequence`와 `pack_padded_sequence`를 혼동한다
- 패딩을 마스킹하지 않고 어텐션을 건다
- 마스킹을 **softmax 뒤에** 한다 (앞에 해야 한다)
- `nn.LSTM(dropout=...)`을 1층에 준다 (층 사이에만 걸린다)
- 시계열을 랜덤으로 분할한다
- 스케일러를 분할 **전에** `fit`한다
- 탐색 코드에서 최소화/최대화 방향을 반대로 쓴다
- `raise NotImplemented` (→ `NotImplementedError`)
- 손실이 평평하다고 바로 학습을 멈춘다 (7-7은 100에폭 뒤에 내려갔다)
