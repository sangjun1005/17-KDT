# 3. GPT 만들고 학습시키기

**실습 파일**: `17-codebot/ch03.ipynb` + `codebot/` 패키지 3개 파일

1장에서 토크나이저를, 2장에서 어텐션을 만들었다. **이제 조립한다.**

> **2026-09-28 — 끝까지 실행됐다**
> 코드 셀 17개 중 11개에 출력이 남았고 `model_pretrain.pt`·`model_sft.pt`·`loss.pretrain.png`·`loss_sft.png`가 전부 생겼다.
> 사전학습 20,000회(40분 15초, 최종 배치 손실 **0.6441**), SFT 500회(57초), 장치는 **CUDA**.
> **실측 결과는 3-9에 모았다.** 이전 판에서 "계산값"으로만 적었던 파라미터 수 11,121,640은 노트북 출력과 **정확히 일치**했다.

| 절 | 내용 |
| --- | --- |
| 3-1 | 노트북에서 패키지로 |
| 3-2 | BPE 를 진짜 말뭉치에 (어휘 1000) |
| 3-3 | 말뭉치를 `.bin` 으로 |
| 3-4 | GPT 구조 — `model.py` |
| 3-5 | 사전학습 루프 |
| 3-6 | 생성 — 온도와 샘플링 |
| 3-7 | 지도 미세조정(SFT) |
| 3-8 | 챗봇 |
| 3-9 | **실측 결과 (2026-09-28 실행)** |

---

## 3-1. 노트북에서 패키지로

```
codebot/
├── tokenizer.py   pretokenize · count_pairs · merge · train_bpe · BPETokenizer
├── model.py       MultiHeadAttention · LayerNorm · GELU · FFN · Block · GPT
└── utils.py       generate · get_device
```

```python
import sys
sys.path.append(".")
from codebot.tokenizer import train_bpe
from codebot.model import GPT
from codebot.utils import get_device, generate
```

**1·2장에서 노트북에 썼던 코드가 그대로 여기로 들어갔다.**

**왜 옮기는가**

| 노트북에 둘 때 | 모듈로 뺐을 때 |
| --- | --- |
| 셀을 위에서부터 다시 실행해야 쓸 수 있다 | `import` 한 줄 |
| 같은 클래스가 여러 노트북에 복사된다 | 한 군데만 고치면 된다 |
| 고쳤는데 어느 셀이 최신인지 헷갈린다 | 파일이 정답 |

**경계는 "실험이냐 확정이냐"다.** 2장에서 `Attention` 클래스를 네 번 고쳐 쓴 것은 실험이고,
확정된 `MultiHeadAttention`은 `model.py`로 간다.

> `sys.path.append(".")`가 `import` **뒤에** 있는 셀이 여럿이다. 노트북 실행 디렉터리가
> 이미 프로젝트 루트라 동작하지만, 순서로 보면 `append`가 먼저여야 맞다.

### `get_device()`

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
```

**`mps`가 들어 있다** — 애플 실리콘 맥의 GPU다. 수업 1장의 `cuda`/`cpu` 2단 분기에 한 줄 더 붙인 것이다.

---

## 3-2. BPE 를 진짜 말뭉치에 — 어휘 1000

```python
vocab_size = 1000
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()
merge_rules = train_bpe(text, vocab_size)

with open("codebot/merge_rules.pkl", "wb") as f:
    pickle.dump(merge_rules, f)
```

1장에서는 짧은 문장으로 `vocab_size=260`을 돌려 봤다. 여기서는 **6.5MB 전체**다.

### 어휘 크기 계산 — 실측 확인

```python
num_merges = vocab_size - 256 - 1      # 1000 - 256 - 1 = 743
```

`merge_rules.pkl`을 직접 열어 확인한 값이다.

| 항목 | 값 |
| --- | --- |
| 병합 규칙 수 | **743** |
| 새 토큰 id 범위 | 256 ~ **998** |
| `end_token_id` | **999** |
| 어휘 크기 | 256 + 743 + 1 = **1000** |

**`- 256`은 바이트 256개, `- 1`은 `<|endoftext|>` 자리다.** 이 두 개를 빼야 딱 맞는다.

### 학습된 토큰을 들여다보면

**처음 학습된 8개** (가장 빈번한 쌍부터)

```
256  '  '     (공백 2칸)
257  '    '   (공백 4칸)
258  'in'
259  '   '    (공백 3칸)
260  're'
261  'or'
262  ' ='
263  'st'
```

**마지막 학습된 8개**

```
991  " ''"    992  'atter'   993  'St'    994  'thod'
995  ' on'    996  'cle'     997  ' R'    998  'ery'
```

**가장 먼저 병합된 것이 공백 2칸·4칸·3칸이다.**
파이썬 코드 말뭉치라 **들여쓰기가 가장 흔한 패턴**이기 때문이다.
같은 알고리즘을 영어 산문에 돌리면 `the`, `ing` 같은 것이 먼저 나온다.

**BPE가 데이터에서 배운다는 뜻이 이것이다.** 규칙을 사람이 넣지 않았다.

`' ='`, `'thod'`(→ `method`), `'St'` 같은 것도 코드 말뭉치다운 결과다.

---

## 3-3. 말뭉치를 `.bin` 으로

```python
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
text = open("codebot/tiny_codes.txt", encoding="utf-8").read()
ids = tokenizer.encode(text, show_progress=True)

ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile("codebot/tiny_codes.bin")
```

**파일에서 직접 잰 결과**

| 항목 | 값 |
| --- | --- |
| `tiny_codes.txt` | 6,487,033 바이트 |
| `tiny_codes.bin` | 5,354,796 바이트 |
| 토큰 수 (uint16) | **2,677,398개** |
| 압축률 | **2.4229 바이트/토큰** |

**왜 `uint16`인가**

어휘가 1000이므로 토큰 id의 최댓값이 999다. `uint16`은 0~65,535를 담으니 충분하다.

| 타입 | 바이트/토큰 | 이 말뭉치 크기 |
| --- | --- | --- |
| `int64` (기본) | 8 | 21.4 MB |
| `int32` | 4 | 10.7 MB |
| **`uint16`** | **2** | **5.35 MB** |

**`int64`로 두면 4배를 쓴다.** 어휘가 65,536을 넘으면 `uint16`을 못 쓰니
`vocab_size`를 정할 때 같이 생각할 일이다.

**왜 미리 저장하는가**

`encode`는 743개 병합 규칙을 순서대로 전부 적용한다. 6.5MB에 대해 매번 돌리면 학습할 때마다 몇 분이 날아간다.
**한 번 해서 파일로 두면 이후에는 `np.fromfile` 한 줄이다.**

```python
ids = np.fromfile(data_path, dtype=np.uint16)
```

> **`dtype`을 틀리면 조용히 망가진다.** `uint16`으로 저장하고 `uint8`로 읽으면
> 에러 없이 **토큰 수가 두 배인 쓰레기**가 나온다. 저장·로드 타입을 같이 관리해야 한다.

---

## 3-4. GPT 구조 — `model.py`

2장의 조각들이 여기서 합쳐진다.

### 설정

```python
context_len   = 256
vocab_size    = 1000
batch_size    = 32
learning_rate = 3e-4
max_iters     = 20000
embed_dim     = 384
n_head        = 6
n_layer       = 6
ff_dim        = 4 * embed_dim     # 1536
dropout_rate  = 0.1
```

`head_dim = embed_dim // n_head = 384 // 6 = 64`

**`ff_dim = 4 × embed_dim`은 트랜스포머의 관례다.** 원 논문부터 GPT 계열까지 거의 그대로 쓴다.
**`head_dim = 64`도 관례다.** GPT-2도 모델 크기와 무관하게 헤드 차원을 64로 둔다
(헤드 수를 늘려 차원을 맞춘다).

### 블록

```python
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, dropout_rate=0.1):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

**두 줄이 전부다. 그런데 이 두 줄에 세 가지가 들어 있다.**

| 요소 | 코드 | 왜 |
| --- | --- | --- |
| 잔차 연결 | `x + ...` | 기울기가 층을 건너 바로 흐른다 |
| **Pre-LN** | `attn(self.norm1(x))` | 정규화를 **안쪽**에 |
| 서브층 2개 | 어텐션 → FFN | 섞고, 각자 처리하고 |

**Pre-LN이 중요하다.** 원 트랜스포머 논문은 `norm(x + attn(x))`(Post-LN)였는데,
지금은 거의 전부 `x + attn(norm(x))`(Pre-LN)를 쓴다.
**잔차 경로에 정규화가 끼지 않아** 깊게 쌓아도 학습이 안정적이다.
Post-LN은 층이 깊어지면 워밍업 없이는 발산하기 쉽다.

**어텐션은 토큰끼리 섞고, FFN은 토큰마다 따로 처리한다.** 역할이 다르다.

### FFN

```python
class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),   # 384 → 1536
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),   # 1536 → 384
            nn.Dropout(dropout_rate)
        )
```

**넓혔다가 다시 좁힌다.** 4배로 펼친 공간에서 비선형 변환을 하고 돌아온다.
**파라미터의 3분의 2가 여기 있다** (블록당 1,181,568 / 1,772,928).

### GPT 전체

```python
self.embed     = nn.Embedding(vocab_size, embed_dim)
self.pos_embed = nn.Embedding(max_context_len, embed_dim)
self.dropout   = nn.Dropout(dropout_rate)
self.blocks    = nn.ModuleList([Block(...) for _ in range(n_layer)])
self.norm      = nn.LayerNorm(embed_dim)
self.unembed   = nn.Linear(embed_dim, vocab_size)
self.embed.weight = self.unembed.weight        # 가중치 공유
self.apply(self._init_weights)
```

```python
def forward(self, ids):
    B, C = ids.shape
    pos = torch.arange(0, C, dtype=torch.long, device=ids.device)
    emb = self.embed(ids)
    pos_emb = self.pos_embed(pos)
    x = self.dropout(emb + pos_emb)
    for block in self.blocks:
        x = block(x)
    x = self.norm(x)
    logits = self.unembed(x)
    return logits
```

**위치 임베딩이 `nn.Embedding`이다.** 파이토치 NLP 4-1의 sin/cos 고정식이 아니라 **학습되는** 방식이다.

| | sin/cos (고정) | `nn.Embedding` (학습) |
| --- | --- | --- |
| 파라미터 | 0 | `max_context_len × embed_dim` |
| 학습 때 안 본 길이 | 외삽 가능 | **불가** |
| 쓰는 곳 | 원 트랜스포머 | **GPT-2 계열** |

**`context_len`을 넘는 위치는 아예 없다.** 그래서 `generate()`에서 잘라 내야 한다.

### 가중치 공유 (weight tying)

```python
self.embed.weight = self.unembed.weight
```

**입력 임베딩과 출력 언임베딩이 같은 행렬을 쓴다.**
`nn.Embedding`의 weight는 `(vocab, embed)`, `nn.Linear(embed, vocab)`의 weight도 `(vocab, embed)`라 모양이 맞는다.

**직관**: "토큰 → 벡터"와 "벡터 → 토큰 점수"는 같은 대응의 양방향이다.

**효과**: `1000 × 384 = 384,000` 파라미터를 아낀다 (전체의 3.5%).
어휘가 5만 개쯤 되면 절약이 훨씬 커진다.

**`self.apply(self._init_weights)`를 공유 **뒤에** 부른 것이 맞다.**
먼저 초기화하고 나중에 묶으면 한쪽 초기화가 버려진다.

### 파라미터 수 — 계산값과 실측이 일치했다

아래 표는 실행 전에 **구조에서 직접 계산**한 값이다.
2026-09-28 실행 결과 노트북이 찍은 `파라미터 수: 11121640`과 **한 자리도 틀리지 않고 같았다.**

| 구성 | 수 |
| --- | --- |
| 토큰 임베딩 (공유) | 384,000 |
| 위치 임베딩 | 98,304 |
| 블록 1개 | 1,772,928 |
| 블록 6개 | 10,637,568 |
| 최종 LayerNorm | 768 |
| `unembed` 편향 | 1,000 |
| **합계** | **11,121,640** |

블록 내역: `norm1` 768 · `attn` 589,824 · `norm2` 768 · `ffn` 1,181,568

**약 1,112만 개다.** GPT-2 small(1.24억)의 1/11 규모다.
가중치 공유를 안 했다면 11,505,640개가 된다.

> 셀 5의 `print("파라미터수 :", total_params)`를 실행하면 이 값이 나올 것이다.
> 계산으로 먼저 구해 두면 실행 뒤 숫자가 다를 때 **어디가 틀렸는지**를 바로 좁힐 수 있다.

### `model.py`에서 눈에 띄는 것

> **정의해 놓고 쓰지 않는 클래스가 둘 있다**
> `LayerNorm`(직접 구현)과 `GELU`(tanh 근사)를 만들어 뒀지만,
> `Block`은 `nn.LayerNorm`을, `FFN`은 `nn.GELU()`를 쓴다.
> **교육용으로 "안이 이렇게 생겼다"를 보여 주려는 의도로 보인다.** 동작에는 문제가 없다.
> 직접 구현한 쪽을 쓰려면 `nn.LayerNorm(embed_dim)` → `LayerNorm(embed_dim)`,
> `nn.GELU()` → `GELU()`로 바꾸면 된다.

**마스크를 `forward`마다 새로 만든다**
```python
mask = torch.tril(torch.ones(C, C, device=scores.device))
```
매 스텝 `256×256` 행렬을 새로 만들고 GPU로 올린다.
`register_buffer`로 한 번 만들어 두면 반복 비용이 사라진다.
```python
self.register_buffer("mask", torch.tril(torch.ones(max_context_len, max_context_len)))
# forward 안에서: mask = self.mask[:C, :C]
```
결과는 같고 속도만 다르다.

**`_init_weights`의 `elif` 들여쓰기**
```
elif isinstance(module, nn.Embedding):
              torch.nn.init.normal_(...)
```
들여쓰기가 과하게 깊지만 문법상 유효하고 동작도 맞다.

---

## 3-5. 사전학습 루프

### 데이터셋 — 한 칸 밀기

```python
class TokenDataset(Dataset):
    def __init__(self, tokens, context_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.context_len]
        y = self.tokens[idx + 1: idx + self.context_len + 1]
        return x, y
```

**`y`는 `x`를 한 칸 민 것이다.** 이게 "다음 토큰 맞히기"의 전부다.

```
tokens:  [ A  B  C  D  E  F ]
x     =  [ A  B  C  D ]
y     =  [ B  C  D  E ]
          ↑A를보고B    ↑ABC를보고D
```

**라벨을 따로 만들지 않는다.** 텍스트 자체가 정답이다 — 자기지도학습(self-supervised).

**`__len__`에서 `- context_len`** 하는 이유는 마지막 구간에서 `y`가 범위를 넘지 않게 하기 위함이다.
2,677,398 토큰에서 샘플 2,677,142개가 나온다. 시작 위치가 한 칸씩 밀리며 **거의 전부 겹친다.**

### 학습 루프

```python
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)                                   # (B, C, V)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           batch_y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
```

**수업 2-5의 학습 루프 다섯 줄이 그대로다.** 달라진 것은 두 가지뿐이다.

**① 에폭이 아니라 반복(iteration) 기준**

```
data_iter = cycle(dataloader)   # itertools.cycle — 끝나면 처음부터
for i in range(max_iters):      # 20,000번
```

샘플이 267만 개라 1에폭도 못 돈다. **"몇 바퀴"가 아니라 "몇 스텝"으로 센다.**
`20,000 × 32 = 640,000` 샘플 — 전체의 24% 정도다.

**② `view(-1, ...)`로 펴기**

```
logits  (32, 256, 1000)  →  view(-1, 1000)  →  (8192, 1000)
batch_y (32, 256)        →  view(-1)        →  (8192,)
```

`cross_entropy`는 `(N, C)`와 `(N,)`을 받는다. **배치와 시퀀스를 한 줄로 편다.**
한 스텝에 8,192개의 "다음 토큰 맞히기" 문제를 푸는 셈이다.

### 기대 손실

**학습 전 손실은 `ln(vocab_size)`여야 한다.**

```
ln(1000) = 6.9078
```

**첫 손실이 6.9 근처면 정상, 아니면 초기화나 데이터가 잘못된 것이다.**
딥러닝 10-4에서 Seq2Seq 손실이 `ln(5000) = 8.5172`였던 것과 같은 점검이다.

**실행 결과**: 20,000회를 40분 15초(8.28 it/s)에 돌았고 마지막 배치 손실은 **0.6441**이었다.
시작 기댓값 `ln(1000) = 6.9078`에서 0.6441까지 내려왔으니 **학습은 정상으로 진행됐다.**
곡선은 `loss.pretrain.png`에 저장된다 — 파일명이 `loss_pretrain`이 아니라 **`loss.pretrain`**(점)이다.

> **검증셋이 없다.** 손실 곡선이 학습 손실뿐이라 과적합을 볼 수 없다.
> 토큰을 90:10으로 나눠 검증 손실을 같이 찍으면 "언제 멈출지"가 보인다.

> **`AdamW`에 `weight_decay`를 안 줬다.** 기본값 0.01이 적용된다.
> GPT 계열은 보통 LayerNorm과 편향에는 decay를 빼고 나머지에만 건다.

> **학습률 스케줄러가 없다.** `3e-4` 고정이다. 워밍업 + 코사인 감쇠가 관례다.

### 저장

```python
def save(self, file_path):
    checkpoint = {
        'model_state_dict': self.state_dict(),
        'vocab_size': ..., 'max_context_len': ..., 'embed_dim': ...,
        'n_head': ..., 'n_layer': ..., 'ff_dim': ..., 'dropout_rate': ...,
    }
    torch.save(checkpoint, file_path)
```

**가중치와 함께 구조 설정도 저장한다.** 이래야 불러올 때 하이퍼파라미터를 다시 안 적어도 된다.

```
@classmethod
def load_from(cls, file_path, device='cpu'):
    checkpoint = torch.load(file_path, map_location=device, weights_only=True)
    model = cls(vocab_size=checkpoint['vocab_size'], ...)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model
```

**`weights_only=True`가 좋다.** 체크포인트에 든 임의 코드 실행을 막는다
(파이토치 2.6부터 기본값이지만 명시하는 편이 안전하다).

**`map_location=device`**는 GPU에서 저장한 것을 CPU에서 열 때 필요하다.

---

## 3-6. 생성 — 온도와 샘플링

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated_ids = ids.clone()

    for _ in range(max_new_tokens):
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]          # 창을 민다

        logits = model(ids)[:, -1, :]                       # 마지막 위치만

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.end_token_id:
            break

        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    return tokenizer.decode(generated_ids[0].tolist())
```

**`logits[:, -1, :]`가 핵심이다.** 모델은 모든 위치의 예측을 내놓지만 **마지막 것만** 쓴다.

**`@torch.no_grad()`와 `model.eval()`** — 수업 2-8에서 배운 짝이다.
`eval()`은 드롭아웃을 끈다. 안 끄면 같은 입력에 매번 다른 결과가 나온다.

### 온도

```python
probs = F.softmax(logits / temperature, dim=-1)
```

| `temperature` | 효과 |
| --- | --- |
| `0` | `argmax` — 항상 같은 결과 (greedy) |
| `< 1` | 분포를 뾰족하게 — 안전하지만 반복적 |
| `1.0` | 모델이 낸 확률 그대로 |
| `> 1` | 평평하게 — 다양하지만 엉뚱해진다 |

**2-4에서 본 `/√d`와 수학적으로 같은 일이다.** 나누면 분포가 평평해진다.
스케일링은 학습 안정을 위해, 온도는 생성 다양성을 위해 쓴다는 점만 다르다.

**`temperature == 0`을 따로 분기한 것이 맞다.** 0으로 나누면 `inf`가 된다.

### 슬라이딩 윈도우

```python
if ids.size(1) > model.max_context_len:
    ids = ids[:, -model.max_context_len:]
```

**위치 임베딩이 `max_context_len`까지만 있어서 반드시 필요하다.**
넘으면 `nn.Embedding`이 인덱스 범위 초과로 죽는다.

**`ids`는 자르고 `generated_ids`는 안 자른다.** 모델 입력은 최근 256개, 최종 출력은 전체다.

> **매 토큰마다 전체 시퀀스를 다시 계산한다.** KV 캐시가 없어 길이에 제곱으로 느려진다.
> 200 토큰 정도면 체감이 안 되지만, 실제 서비스에서는 캐시가 필수다.

> **`generated_ids`에 프롬프트도 포함된다.** 그래서 3-8에서 `### Response:`로 잘라 낸다.

> **`temperature=1.0`에 top-k/top-p가 없다.** 꼬리의 저확률 토큰까지 뽑힐 수 있어
> 작은 모델에서는 자주 헛소리가 된다. `top_k=40`쯤 얹으면 크게 안정된다.

---

## 3-7. 지도 미세조정 (SFT)

**사전학습 모델은 "다음에 올 법한 말"을 이어 쓸 뿐 지시를 따르지 않는다.**
`def`를 주면 함수를 이어 쓰지만, "피보나치 함수 만들어 줘"라고 하면 그 문장을 이어 쓴다.

### 알파카 포맷

```python
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
```

```
### Instruction:
Hello

### Response:
Hello. What can I help you with?<|endoftext|>
```

**형식이 곧 규칙이다.** `### Response:\n` 다음에는 답이 온다는 것을 모델이 배운다.
`<|endoftext|>`는 "여기서 멈춰라" 신호이고, `generate()`가 이 id를 만나면 `break` 한다.

**`<|endoftext|>`가 1장에서 특수 토큰으로 만들어 둔 그것이다.** 여기서 쓰임이 드러난다.

### 데이터 — 실측

`tiny_codes_sft.json`을 직접 열어 잰 값이다.

| 항목 | 값 |
| --- | --- |
| 샘플 수 | **1,092** |
| 키 | `instruction`, `response` |
| instruction 길이 | 평균 19.1자 · 중앙 18 · 최대 118 |
| response 길이 | 평균 45.3자 · 중앙 41 · 최대 331 |

사전학습 말뭉치(268만 토큰)에 비하면 **아주 적다.** SFT는 원래 그렇다.
새 지식을 넣는 게 아니라 **형식을 가르치는 것**이라 조금이면 된다.

### 프롬프트 마스킹 — 이 절의 핵심

```python
def _create_sample(self, instruction, response):
    prompt   = f"### Instruction\n{instruction}\n\n### Response:\n"   # 콜론이 없다
    response = f"{response}<|endoftext|>"

    prompt_ids   = self.tokenizer.encode(prompt)
    response_ids = self.tokenizer.encode(response)

    ids    = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids     # ← 프롬프트를 가린다

    ids    = ids[:-1]
    labels = labels[1:]
    ...
```

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       batch_y.view(-1),
                       ignore_index=-100)
```

**`-100`은 파이토치 `cross_entropy`의 기본 `ignore_index`다.** 그 위치는 손실 계산에서 빠진다.

```
ids    :  ### Instruction: \n Hello \n\n ### Response: \n   Hello.  What  can ...
labels :  -100 -100 -100 -100 -100 -100 -100 -100 -100     Hello.  What  can ...
          └────────── 질문 부분: 학습 안 함 ──────────┘  └── 답 부분만 학습 ──┘
```

**질문을 생성하는 법은 배울 필요가 없다.** 사용자가 주는 것이다.
안 가리면 모델이 **질문을 지어내는 것도 같이 배워** 혼자 묻고 답하기 시작한다.

**`ids[:-1]` / `labels[1:]`가 한 칸 밀기다.** 3-5의 `TokenDataset`과 같은 일을 미리 해 두는 것이다.

### 설정

```python
learning_rate = 3e-4
max_iters = 500          # 사전학습 20,000의 1/40
```

```python
model = GPT.load_from(pretrain_model_path, device=device)
```

**사전학습 가중치에서 이어서 학습한다.** 처음부터가 아니다.

> **학습률이 사전학습과 같은 `3e-4`다.** SFT는 보통 **10분의 1 이하**(`1e-5`~`3e-5`)로 낮춘다.
> 높으면 사전학습으로 쌓은 것을 지워 버릴 수 있다(catastrophic forgetting).
> 다만 이 모델은 1,112만 파라미터로 작고 SFT 데이터도 1,092개뿐이라 영향은 제한적일 수 있다.
> **실제로 돌려 보고 생성 품질을 비교해 볼 일이다.**

> **여기도 검증셋이 없다.** 1,092개 중 일부를 떼어 두면 과적합 시점을 볼 수 있다.
>
> **학습 프롬프트에 콜론이 빠져 있다.** `### Instruction`으로 학습하는데
> 3-8의 `format_prompt`는 `### Instruction:`으로 묻는다. 3-9에 자세히 적었다.

---

## 3-8. 챗봇

```python
def format_prompt(user_message):
    return f"### Instruction:\n{user_message}\n\n### Response:\n"

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue

    prompt = format_prompt(user_input)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)

    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    ...
```

**학습할 때 쓴 형식을 추론할 때도 똑같이 만들어 준다.** 이 한 줄이 SFT와 짝이다.
형식이 조금이라도 다르면 모델이 못 알아본다.

**`generate()`가 프롬프트까지 돌려주므로 `### Response:`로 잘라 낸다.**
`split(...)[-1]`로 마지막 조각을 쓰는 것은, 모델이 응답 안에 또 `### Response:`를 만들어 낼 수 있어서다.

> **`while True`에 종료 조건이 없다.** `quit`·`exit` 입력을 받거나 `KeyboardInterrupt`를 잡아야
> 노트북 셀을 강제 중단하지 않고 빠져나올 수 있다.

> **`input()`은 노트북에서는 되지만 스크립트로 돌릴 때만 자연스럽다.**

> **파이썬 3.12 이상이 필요하다.** 셀 9에 `f"...{item["instruction"]}..."`처럼
> **f-string 안에 같은 따옴표**를 쓴 곳이 있다. PEP 701로 3.12부터 허용된 문법이라
> 3.11 이하에서는 `SyntaxError`다. (`__pycache__`가 `cpython-312`라 현재 환경은 3.12다.)

---

## 3-9. 실측 결과 — 2026-10-02 재실행

노트북을 처음부터 끝까지 돌린 결과다. **아래 숫자는 전부 노트북 출력에서 그대로 가져왔다.**

### 실행 환경과 시간

| 단계 | 값 |
| --- | --- |
| 장치 | `cuda` |
| 토큰화 | 43,671묶음 · **4분 12초** (173.05 it/s) |
| 사전학습 | 20,000회 · **40분 14초** (8.28 it/s) |
| SFT | 500회 · **57초** (8.65 it/s) |
| `model_pretrain.pt` | 44,516,353 바이트 |
| `model_sft.pt` | 44,515,938 바이트 |

파라미터가 1,112만 개인데 파일이 44.5MB인 이유 — `float32`는 파라미터당 4바이트다.
11,121,640 × 4 = 44,486,560 바이트. 나머지 3만 바이트가 메타데이터다.

### 손실

| 지점 | 값 |
| --- | --- |
| 이론상 시작값 | `ln(1000)` = **6.9078** |
| 사전학습 마지막 배치 | **0.6080** (첫 실행 0.6441) |
| SFT 마지막 배치 | **기록되지 않았다** (아래 주의) |

> **주의 — SFT 손실이 노트북에 남지 않았다**
> 진행 표시줄이 손실 값을 찍지 못해 **SFT 구간의 손실 곡선이 기록에 없다.**
> 학습 자체는 정상으로 돌았고 `model_sft.pt` 도 생겼다 — **값만 없다.**
> 코드는 그 뒤 `f'{loss.item():.4f}'` 로 고쳤으니, **다시 돌리면 손실이 남는다.**

### BPE 가 실제로 배운 것

첫 10개와 마지막 10개다.

| 순서 | 토큰 |
| --- | --- |
| 256~259 | 공백 2칸 · 공백 4칸 · `in` · 공백 3칸 |
| 260~265 | `re` · `or` · ` =` · `st` · `te` · 줄바꿈+공백 |
| 990~994 | `Error` · `' '` · `atter` · `St` · `thod` |
| 995~999 | ` on` · `cle` · ` R` · `ery` · 특수 토큰 `endoftext` |

**앞쪽이 전부 들여쓰기다.** 말뭉치가 파이썬 코드라서 가장 자주 붙어 나오는 쌍이 공백이었다.
마지막 999번이 `<|endoftext|>`인 것도 설계대로다 — 특수 토큰은 병합 뒤에 마지막으로 붙인다.

### 압축률 — 두 값이 다른 이유

| 대상 | 바이트 | 토큰 | 압축률 |
| --- | --- | --- | --- |
| 앞부분 10,000자 (노트북 출력) | 10,000 | 4,709 | **2.1236** |
| 파일 전체 (직접 계산) | 6,487,033 | 2,677,398 | **2.4229** |

앞 10,000자는 짧은 예제가 몰려 있어 긴 병합 토큰이 덜 쓰인다.
**같은 토크나이저라도 어떤 글을 재느냐에 따라 압축률이 달라진다.**

### 사전학습만 마친 모델이 쓴 코드

`prompt="def"`, `temperature=1.0`으로 5번 생성했다.

```python
# 1번
def remove_vowels(string):
  vowels = ["a", "e", "i", "o", "u"]
  return string.count(char)          # char 가 정의돼 있지 않다

# 2번
def detectMax(x):
    def __init__(self, x, y):        # 함수 안에 __init__ 이 들어갔다
        ...
    def __sub__(self, other): ...    # 같은 메서드를 두 번 썼다
    def __sub__(self, other): ...

# 3번
def reverse_list(list):
 half = 0
 while temp >= temp:                 # 항상 참 — 무한 루프
```

**읽어 낼 것 세 가지.**

1. **모양은 완벽하다.** `def`·들여쓰기·`return`·`__init__`·독스트링 위치가 전부 파이썬이다. 1,112만 파라미터로 40분 돌린 모델이 문법을 통째로 익혔다.
2. **의미는 없다.** 정의 안 된 변수(`char`), 같은 메서드 중복, `temp >= temp` 같은 항상 참인 조건. 다음 토큰만 맞히도록 배웠으니 **문장은 맞고 논리는 틀린다.**
3. **이름과 본문이 따로 논다.** `reverse_list`인데 본문은 자릿수를 뒤집고 있다. 함수 이름을 "의도"로 읽지 못한다는 뜻이다.

이 단계가 **지시를 못 따르는 이유**를 그대로 보여 준다. 그래서 SFT 가 필요하다.

### SFT 후 챗봇

```
You: Hello
Bot: Hello. What can I help you with?
```

학습 데이터 첫 줄이 `{'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}`다.
**답을 통째로 외워서 돌려준 것에 가깝다.** 1,092쌍을 500회(배치 32 = 약 16,000샘플, 14.6에폭) 돌렸으니 외울 만하다.
지시 따르기의 "형식"은 붙었지만, 일반화됐다고 말하려면 **학습에 없던 질문**으로 확인해야 한다.

> **주의 — 학습과 추론의 프롬프트 형식이 다르다**
> `SFTDataset`은 `### Instruction\n`(콜론 없음)으로 학습하고,
> 챗봇의 `format_prompt`는 `### Instruction:\n`(콜론 있음)으로 묻는다.
> 모델이 한 번도 본 적 없는 형식으로 물어보는 셈이다.
> 그런데도 답이 나온 것은 외운 문장이 강했기 때문으로 보인다. **둘 중 하나로 맞춰야 한다.**

### 실행 순서가 섞여 있다

실행 번호가 `2,3,4,5,8,9,10,11,12,13` 다음에 다시 `3,10,11,12,14,18`로 돌아간다.
**사전학습과 SFT 사이에 커널을 다시 시작했다.**
셀 순서대로 위에서 아래로 읽으면 안 되고, `.pt` 파일이 이어 주는 흐름으로 읽어야 한다.

### 그림이 깨졌다

`loss.pretrain.png` 저장에서 한글 폰트 경고가 났다.

```
UserWarning: Glyph 48152 (HANGUL SYLLABLE BAN) missing from font(s) DejaVu Sans.
```

`반복`·`손실` 축 라벨이 **네모로 나온다.** 아래 두 줄을 맨 위에 넣으면 된다.

```python
plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
plt.rcParams['axes.unicode_minus'] = False
```

SFT 쪽은 축 라벨을 영어(`Iteration`, `loss`)로 써서 깨지지 않았다.

---

### 재실행으로 바뀐 것

같은 노트북을 두 번 돌린 결과를 나란히 두면 **무엇이 재현되고 무엇이 안 되는지**가 보인다.

| 항목 | 2026-09-28 | 2026-10-02 | 재현되나 |
| --- | --- | --- | --- |
| BPE 병합 규칙 | 743개 | 743개 | **같다** (결정적) |
| 토큰 수 | 2,677,398 | 2,677,398 | **같다** (결정적) |
| 파라미터 | 11,121,640 | 11,121,640 | **같다** (구조가 정한다) |
| BPE 학습 시간 | — | 10분 43초 (1.15 it/s) | 기계 상태에 따라 다르다 |
| 토큰화 시간 | 4분 12초 | 3분 29초 (208.17 it/s) | 다르다 |
| 사전학습 시간 | 40분 15초 | 40분 14초 | 거의 같다 |
| **사전학습 최종 손실** | **0.6441** | **0.6080** | **다르다** |
| 생성 품질 | 문법만 맞음 | **돌아가는 함수가 나온다** | 다르다 |

**토크나이저는 결정적이고 학습은 아니다.** BPE 는 같은 입력에 같은 규칙을 만들지만(동점 처리를 바이트 값으로 고정해 둔 덕이다 — 1장),
학습은 `torch.randint` 로 뽑는 배치 순서와 드롭아웃이 매번 달라 **시드를 고정하지 않으면 손실이 다르게 끝난다.**

> **시드를 고정하지 않았다.** `torch.manual_seed(...)` 가 어디에도 없다.
> 두 실행의 손실 차이 0.036 이 "개선"인지 "운"인지 구분할 방법이 없다.
> 설정을 비교하려면 시드를 박아 두고 여러 번 돌려야 한다 (5장의 `seed_offset` 참고).

### 두 번째 실행의 생성 결과 — 눈에 띄게 좋아졌다

`prompt="def"`, `temperature=1.0`, 5회 생성.

```python
# 0번 — 거의 맞는다
def fibonacci_series(n):
    f1 = 0
    a = 1
    b = 1
    if (n < 0):
        print("Incorrect input")
    elif (n > 0):          # ← n == 0 이어야 한다
        return 0
    elif (n == 1):
        return 1
    else:
        for i in range(2, n):
            c = a + b
            a = b
            b = c
        return b

# 4번 — 그대로 돌아간다
def reverse_sentence(sentence):
    words = sentence.split(' ')
    reversed_sentence = words[::-1]
    reversed_sentence = ' '.join(reversed_sentence)
    return reversed_sentence

sentence = 'Hello world'
reverse_sentence = reverse_sentence(sentence)
print(reverse_sentence)
```

**0번은 피보나치 알고리즘이 실제로 맞다.** `a, b` 를 굴리는 루프와 `return b` 가 정확하다.
틀린 곳은 `elif (n > 0): return 0` 한 줄 — `n == 0` 이어야 한다. **논리의 뼈대는 서고 조건 하나가 어긋났다.**

**4번은 복사해 붙여도 돌아간다.** 다만 마지막 두 줄에서 **함수 이름을 결과로 덮어쓴다**(`reverse_sentence = reverse_sentence(...)`).
한 번은 동작하고 두 번째 호출에서 `TypeError` 가 난다 — 모델이 흔히 내는 실수다.

```python
# 1번 — 이름과 동작이 다르다
def capitalize(s):
    s = s.lower()      # 대문자로 만드는 게 아니라 소문자로 만든다
    return s

# 2번 — 한 줄에서 끊겼다
def

# 3번 — 완전히 망가졌다
def LargestCommonSubsequence(list1, list2):
    ...
    if list1[i + j] == list2[j] and is_anagram(...)   # 콜론 없음, 괄호 안 맞음
    longest = tappend(list1[i + 1][j], list2[j - dp[i][j - 1])
```

**5개 중 2개가 쓸 만하고, 1개는 이름이 틀리고, 2개는 망가졌다.**
첫 실행에서는 **쓸 만한 것이 하나도 없었다**(정의 안 된 변수, `temp >= temp` 같은 항상 참인 조건).
같은 설정에서 이만큼 벌어지니, **생성 품질을 두 눈으로 비교하려면 샘플을 수십 개 봐야 한다.**

### 챗봇 응답도 바뀌었다

```
You: Hello
Bot: hex = 'hello'
```

첫 실행은 학습 데이터 1번 항목을 그대로 돌려줬다(`Hello. What can I help you with?`).
이번에는 **코드로 답했다.** SFT 데이터가 1,092쌍뿐이고 사전학습 말뭉치는 코드 268만 토큰이라,
**조금만 흔들려도 사전학습 쪽 성향이 튀어나온다.** 3-7에서 "SFT 학습률이 사전학습과 같다"고 지적한 것이 여기서 드러난다.

---

## 3-10. GRPO — 강화학습으로 덧셈을 가르친다

SFT 는 "정답을 베껴 쓰게" 가르쳤고, 4장의 DPO 는 "둘 중 나은 쪽"을 가르쳤다.
GRPO 는 **"직접 풀어 보고 맞으면 그 방향으로"** 가르친다.

> **GRPO (Group Relative Policy Optimization)** 는 DeepSeek 이 쓴 기법이다.
> PPO 에서 **가치망(critic)을 없애고**, 대신 같은 문제를 여러 번 풀어 **그 그룹의 평균을 기준선**으로 쓴다.

### 과제 — 한 자리 수 덧셈

```python
class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.data = []
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = f'### Instruction:\n{i}+{j}=\n\n### Response:\n'
                ground_truth = i + j
                self.data.append((prompt, ground_truth))
```

**9 × 9 = 81문제가 전부다.** 1+1 부터 9+9 까지.

**왜 덧셈인가.** 강화학습에는 **맞았는지 기계적으로 판정할 수 있는 과제**가 필요하다.
"이야기가 좋은가"는 사람이나 LLM 이 봐야 하지만(4장), 덧셈은 숫자만 비교하면 끝난다.

```python
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0
```

**응답에서 마지막 숫자를 뽑아 정답과 비교한다.** 맞으면 1, 틀리면 0. 중간 과정은 보지 않는다.
`matches[-1]` 로 **마지막** 숫자를 쓰는 것이 포인트다 — 모델이 `1+2= 3` 처럼 문제를 되풀이해도 답만 읽는다.

### 그룹을 만들어 평균을 뺀다 — 이게 GRPO 의 핵심

```python
def generate_group(model, tokenizer, prompts, gts, group_size):
    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):                      # 같은 문제를 8번 푼다
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            responses.append(full_text[len(prompt):])

        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        advantages = rewards - rewards.mean()            # ← 그룹 평균이 기준선
```

```
3+5= 를 8번 풀어 본다
응답:      8      7      8      8      3      8      8      9
보상:      1      0      1      1      0      1      1      0
평균:                        0.625
이득:  +0.375 -0.625 +0.375 +0.375 -0.625 +0.375 +0.375 -0.625
        ↑ 더 자주 하게          ↑ 덜 하게
```

**`rewards - rewards.mean()` 이 한 줄이 가치망을 대체한다.**

| | PPO | **GRPO** |
| --- | --- | --- |
| 기준선 | **가치망**(critic)이 "이 상태의 기대 보상"을 예측 | **같은 문제 8번의 평균** |
| 필요한 모델 | 정책 + 가치망 + 참조 | **정책 + 이전 정책** |
| 추가 학습 | 가치망도 같이 학습해야 한다 | 없다 |
| 비용 | 모델 하나 더 | **생성 8배** |

**"기준선"이 왜 필요한가.** 보상만 보고 밀면 **모든 응답의 확률이 다 올라간다**(보상이 전부 양수니까).
평균을 빼면 **평균보다 잘한 것만** 올라가고 못한 것은 내려간다. 이게 **상대 비교**다.
8개가 전부 맞거나 전부 틀리면 이득이 모두 0이 되어 **그 문제에서는 아무것도 배우지 않는다** — 자연스러운 커리큘럼이 생긴다.

### 클리핑 — PPO 에서 가져온 안전장치

```python
def compute_probs(model, ids):
    logits = model(ids)
    probs = F.softmax(logits[:, :-1, :], dim=-1)
    labels = ids[:, 1:]
    token_probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return token_probs

def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2):
    probs = compute_probs(model, ids)
    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)

    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    mask = mask[:, 1:]
    token_objective = torch.min(unclipped, clipped) * mask
    return -token_objective.sum() / ids.size(0)
```

**`ratio` 는 "지금 모델이 이 토큰을 예전보다 몇 배 더 좋아하는가"다.**

```
ratio = 1.0   → 안 바뀌었다
ratio = 1.5   → 1.5배 더 좋아하게 됐다
ratio = 0.5   → 절반으로 줄었다
```

`torch.clamp(ratio, 0.8, 1.2)` 로 **한 번에 ±20% 이상 못 움직이게 묶는다.**
`torch.min(unclipped, clipped)` 은 **둘 중 작은 쪽**을 고른다 — 이득이 양수든 음수든 **보수적인 쪽**으로 간다.

**왜 묶는가.** 강화학습은 자기가 만든 데이터로 자기를 학습시킨다. 한 번에 크게 움직이면
다음 생성이 망가지고, 망가진 생성으로 또 학습해 **되돌릴 수 없게 무너진다**(policy collapse).

```python
mask = mask[:, 1:]
token_objective = torch.min(unclipped, clipped) * mask
```

**마스크가 또 나온다.** 3-7의 `-100`, 4장 DPO 의 0/1 마스크와 같은 역할이다 —
**프롬프트 부분은 점수에 넣지 않는다.** 모델이 만든 응답 토큰만 센다.

**`-...sum() / n_samples`** — 목적함수를 **최대화**해야 하므로 손실로 쓸 때 부호를 뒤집는다.

### 학습 루프

```python
learning_rate = 7e-6
max_iters = 100
n_update_per_generation = 2
epsilon = 0.2
group_size = 8
batch_size = 32
eval_interval = 10

model = GPT.load_from(sft_model_path, device=device)        # SFT 모델에서 출발
old_model = GPT.load_from(sft_model_path, device=device)
old_model.eval()
```

```python
for i in pbar:
    prompts, gts = next(data_iter)
    old_model.load_state_dict(model.state_dict())           # ① 사본을 떠 둔다
    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts, gts, group_size)      # ② 사본으로 생성

    ids, mask = dataset.get_batch(all_prompts, all_responses, device)

    for _ in range(n_update_per_generation):                 # ③ 같은 데이터로 2번 갱신
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

**세 단계가 강화학습의 모양을 그대로 보여 준다.**

| 단계 | 하는 일 | 왜 |
| --- | --- | --- |
| ① `old_model.load_state_dict(...)` | 매 반복마다 현재 모델을 복사 | `ratio` 의 기준이 **방금 전 모델**이어야 한다 |
| ② `generate_group(old_model, ...)` | **사본으로** 생성 | 학습 중인 모델로 생성하면 기준이 흔들린다 |
| ③ 같은 데이터로 2번 갱신 | 생성이 비싸니 재활용 | 그래서 클리핑이 필요하다 |

**생성이 병목이다.** 한 반복에 `batch_size 32 × group_size 8 = 256`번 생성한다.
그래서 100회에 **4분 55초**(2.96 s/it)가 걸렸다 — 사전학습은 8.28 it/s 였다. **25배 느리다.**

`learning_rate = 7e-6` 은 사전학습(`3e-4`)의 **1/43** 이다. SFT 가 낮추지 않았던 학습률을 여기서는 제대로 낮췄다.

### 평가 — 81문제 전부를 `temperature=0` 으로

```python
if i % eval_interval == 0:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for prompt, gt in dataset.data:
            response = generate(model, tokenizer, prompt, temperature=0)
            correct += calculate_reward(gt, response) > 0
            total += 1
    model.train()
    current_accuracy = correct / total * 100
```

**학습은 `temperature=1.0`, 평가는 `temperature=0`.**
학습에는 다양한 시도가 필요하고(안 그러면 탐색이 안 된다), 평가는 **재현 가능한 한 가지 답**이어야 한다.
`temperature=0` 은 `argmax` 라서 같은 입력에 항상 같은 출력이 나온다 (3-6).

**`model.eval()` / `model.train()` 을 짝으로 감싼 것도 맞다.** 평가 중에 드롭아웃이 켜져 있으면 점수가 흔들린다.

### 실측 결과

| 항목 | 값 |
| --- | --- |
| 출발 모델 | `codebot/model_sft.pt` |
| 반복 | 100회 |
| 소요 시간 | **4분 55초** (2.96 s/it) |
| 반복당 생성 | 32 × 8 = **256회** |
| 최종 손실 | **0.2154** |
| **최종 정확도** | **69.1%** (81문제 중 56개) |
| 저장 | `codebot/model_grpo.pt` |

**무작위로 맞힐 확률은 거의 0 이다.** 답이 2~18 사이의 수이고 모델은 자유 텍스트를 생성한다.
SFT 모델 상태에서 몇 %였는지는 출력에 없지만, 1,092쌍짜리 SFT 데이터에 덧셈 문제가 있었을 가능성은 낮다.

> **주의 — 이 69.1% 를 어떻게 읽을까**
> **81문제를 외운 것일 수도 있다.** 학습과 평가가 **완전히 같은 81문제**다. 검증 집합이 없다.
> 두 자리 수 덧셈(`12+34=`)이나 뺄셈으로 물어보면 **일반화됐는지** 알 수 있다.
>
> **시작점이 없다.** 학습 전 정확도를 찍지 않았다(`eval_interval` 평가가 `i=0`에서 돌지만 그때 이미 1회 갱신 뒤다).
> "GRPO 가 몇 %p 올렸는가"를 말하려면 SFT 모델의 정확도를 먼저 재야 한다.
>
> **정확도 곡선은 그렸지만 숫자가 남지 않았다.** `accuracies` 리스트를 `plt.plot` 만 하고 출력하지 않았다.
> `print(accuracies)` 한 줄이면 10회마다의 값이 기록된다.

### 세 가지 정렬 기법을 나란히

| | SFT (3-7) | DPO (4-8) | **GRPO (3-10)** |
| --- | --- | --- | --- |
| 데이터 | 지시-응답 쌍 | 선호 쌍 (chosen/rejected) | **문제 + 정답만** |
| 응답을 누가 만드나 | 사람이 미리 | 사람이 미리 | **모델이 직접** |
| 신호 | 정답을 베껴라 | 이쪽이 낫다 | **맞았다 / 틀렸다** |
| 필요한 모델 | 1개 | 2개 (학습 + 기준) | **2개 (학습 + 직전 사본)** |
| 기준선 | — | 기준 모델의 로그확률 | **그룹 평균 보상** |
| 학습률 | `3e-4` | `5e-6` | **`7e-6`** |
| 쓸 수 있는 과제 | 전부 | 선호를 매길 수 있는 것 | **정답을 판정할 수 있는 것** |
| 비용 | 싸다 | 중간 | **비싸다 (생성 × group_size)** |

**GRPO 가 수학·코딩에서 강한 이유가 표에 다 있다.** 정답 판정이 기계적이고, 응답을 사람이 안 만들어도 되고,
"틀렸다"는 신호가 "덜 좋다"보다 훨씬 선명하다. 반대로 **번역·요약·글쓰기에는 쓰기 어렵다** — 정답이 하나가 아니다.

### 이 절에서 고칠 것

| 위치 | 내용 | 수정안 |
| --- | --- | --- |
| 평가 | 학습·평가가 **같은 81문제**다. 검증 집합이 없다 | 두 자리 수나 뺄셈으로 일반화 확인 |
| 평가 | **학습 전 정확도를 재지 않았다** | GRPO 전에 한 번 평가 |
| 로그 | `accuracies` 를 그리고 출력하지 않는다 | `print(accuracies)` |
| `calculate_reward` | `except:` 로 전부 삼킨다 | `except (ValueError, TypeError)` |
| `generate_group` | `max_new_tokens` 를 안 넘겨 기본값(1000)을 쓴다 | 덧셈 답에는 10~20 이면 충분하다. **생성이 병목이라 효과가 크다** |
| `grpo_loss` | 참조 모델(KL 벌점)이 없다 | 원 GRPO 는 `beta × KL(정책‖참조)` 를 더한다. 없으면 사전학습 실력을 잃을 수 있다 |
| 전체 | 시드를 고정하지 않았다 | 강화학습은 분산이 커서 특히 필요하다 |

---

## 이 장 정리

### 한 줄 요약

**GPT = 임베딩 + (어텐션 + FFN) × N + 언임베딩.**
학습은 "다음 토큰 맞히기" 하나이고, 챗봇은 거기에 형식을 조금 더 가르친 것이다.

### 전체 파이프라인

```
tiny_codes.txt 6.5MB
   │  train_bpe(vocab_size=1000)
   ↓
merge_rules.pkl  743개 병합 규칙
   │  tokenizer.encode(전체)
   ↓
tiny_codes.bin   2,677,398 토큰 (uint16, 5.35MB)
   │  TokenDataset  x=[i:i+256]  y=[i+1:i+257]
   ↓
GPT 사전학습  20,000 iter × batch 32 → model_pretrain.pt
   │  generate(prompt="def")
   ↓
이어 쓰기는 되지만 지시는 못 따른다
   │  SFT 1,092쌍, 알파카 포맷, 프롬프트 마스킹, 500 iter
   ↓
model_sft.pt → 챗봇
```

### 외워 둘 숫자

| 항목 | 값 | 출처 |
| --- | --- | --- |
| 어휘 크기 | 1000 = 256 + 743 + 1 | 파일 실측 |
| 토큰 수 | 2,677,398 | 파일 실측 |
| 압축률 | 2.4229 바이트/토큰 | 파일 실측 |
| SFT 샘플 | 1,092쌍 | 파일 실측 |
| 파라미터 | 11,121,640 | 계산값 = **노트북 실측** |
| 사전학습 최종 손실 | 0.6080 | 노트북 실측 (재실행) |
| 사전학습 시간 | 40분 14초 (20,000회) | 노트북 실측 |
| 압축률(앞 10,000자) | 2.1236 바이트/토큰 | 노트북 실측 |
| 학습 전 기대 손실 | `ln(1000)` = 6.9078 | 이론값 |
| `head_dim` | 384 / 6 = 64 | 설정 |
| `ff_dim` | 4 × 384 = 1536 | 설정 |
| GRPO 정확도 | 69.1% (81문제) | 노트북 실측 |
| GRPO 그룹 크기 | 8 | 설정 |
| GRPO 학습률 | `7e-6` (사전학습의 1/43) | 설정 |

### 외워 둘 코드

```python
x = tokens[i : i+C]                      # 입력
y = tokens[i+1 : i+C+1]                  # 한 칸 밀기 = 정답
loss = F.cross_entropy(logits.view(-1, V), y.view(-1))

x = x + self.attn(self.norm1(x))         # Pre-LN + 잔차
x = x + self.ffn(self.norm2(x))

self.embed.weight = self.unembed.weight  # 가중치 공유
logits = model(ids)[:, -1, :]            # 생성: 마지막 위치만
probs = F.softmax(logits / temperature, dim=-1)
labels = [-100]*len(prompt_ids) + response_ids   # SFT: 프롬프트 마스킹
```

### 이 노트북에서 고칠 것

| 위치 | 내용 |
| --- | --- |
| SFT 셀 | 진행 표시줄이 손실을 찍지 못해 **SFT 손실 곡선이 노트북에 없다** (코드는 고쳐졌다) |
| `SFTDataset` | 학습은 `### Instruction`, 추론은 `### Instruction:` → **형식 불일치** |
| 사전학습 셀 | matplotlib 한글 폰트 미지정 → 축 라벨이 네모로 나온다 |
| `model.py` | `LayerNorm`·`GELU`를 만들어 두고 `nn.LayerNorm`·`nn.GELU`를 쓴다 |
| `model.py` | 마스크를 `forward`마다 새로 만든다 → `register_buffer` |
| 사전학습 | 검증셋이 없다 → 과적합 시점을 볼 수 없다 |
| 사전학습 | 학습률 스케줄러가 없다 → 워밍업 + 코사인 감쇠 |
| SFT | 학습률이 사전학습과 같은 `3e-4` → 보통 10분의 1 이하 |
| `generate` | top-k/top-p가 없어 저확률 토큰이 뽑힌다 |
| `generate` | KV 캐시가 없어 길이에 제곱으로 느려진다 |
| 챗봇 | `while True`에 종료 조건이 없다 |
| 셀 9 | f-string 중첩 따옴표 → **파이썬 3.12 이상 전용** |
| 여러 셀 | `sys.path.append(".")`가 `import` 뒤에 있다 |
| GRPO 평가 | 학습·평가가 같은 81문제다. 검증 집합이 없다 |
| GRPO | 학습 전 정확도를 재지 않았다. 참조 모델(KL 벌점)도 없다 |
| 전체 | 시드를 고정하지 않아 재실행마다 손실이 다르다 (0.6441 → 0.6080) |

### 자주 틀리는 것

- `y`를 한 칸 밀지 않는다 → 자기 자신을 맞히는 문제가 된다
- `.bin` 저장·로드의 `dtype`이 다르다 → **에러 없이** 쓰레기가 나온다
- 생성에서 `model.eval()`을 안 한다 → 드롭아웃이 켜져 결과가 흔들린다
- `logits` 전체를 쓴다 → **마지막 위치만** 필요하다
- `context_len`을 넘겨 넣는다 → 위치 임베딩 인덱스 초과
- SFT에서 프롬프트를 안 가린다 → 모델이 질문을 지어내기 시작한다
- 추론 프롬프트 형식이 학습 때와 다르다 → 모델이 못 알아본다
- 가중치 공유 **뒤에** 초기화하지 않는다 → 한쪽 초기화가 버려진다
