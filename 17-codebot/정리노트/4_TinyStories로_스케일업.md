# 4. TinyStories 로 스케일업 — RoPE · SwiGLU · DPO

**실습 파일**: `17-codebot/ch04.ipynb` + `storybot/` 패키지

3장은 6.5MB 코드 말뭉치에 1,112만 파라미터였다. 4장은 **5.4억 토큰**에 **2,270만 파라미터**다.
크기만 커진 게 아니라 **구조가 통째로 바뀐다.**

> **2026-10-01 — 끝까지 실행됐다**
> 사전학습 10,000회, 최종 검증 손실 **1.9645**. GPT 심사로 체크포인트 3개를 비교했고 DPO 까지 돌렸다.
> 실측값은 전부 노트북 출력에서 가져왔다. **4-9에 모아 두었다.**

| 절 | 내용 |
| --- | --- |
| 4-1 | 3장에서 무엇이 바뀌었나 |
| 4-2 | TinyStories 와 멀티프로세스 BPE |
| 4-3 | RoPE — 회전으로 위치를 넣는다 |
| 4-4 | RMSNorm 과 SwiGLU |
| 4-5 | KV 캐시 |
| 4-6 | 학습 루프 — 스케줄러 · bf16 · 클리핑 · 검증 |
| 4-7 | LLM 으로 LLM 을 채점한다 |
| 4-8 | DPO — 좋은 답을 직접 가르친다 |
| 4-9 | **실측 결과 (2026-10-01 실행)** |

---

## 4-1. 3장에서 무엇이 바뀌었나

| 항목 | 3장 `codebot/` | 4장 `storybot/` | 왜 |
| --- | --- | --- | --- |
| 위치 정보 | `nn.Embedding` (학습) | **RoPE** (회전, 학습 안 함) | 상대 위치를 자연히 담고 길이에 덜 묶인다 |
| 정규화 | `nn.LayerNorm` | **`nn.RMSNorm`** | 평균을 안 빼서 더 싸다 |
| FFN | Linear → GELU → Linear | **SwiGLU** (행렬 3개) | 같은 파라미터로 성능이 더 낫다 |
| 추론 | 매번 전체 재계산 | **KV 캐시** | 토큰당 비용이 일정해진다 |
| 편향 | `nn.Linear` 기본 (있음) | **전부 `bias=False`** | 효과가 거의 없고 수만 늘린다 |
| 가중치 공유 | `embed.weight = unembed.weight` | **안 한다** | 어휘가 1만이라 공유 제약이 더 아프다 |
| 드롭아웃 | 0.1 | **없음** | 데이터가 5.4억 토큰이라 과적합 걱정이 작다 |
| BPE 학습 | 단일 프로세스, 매 단계 전체 재계산 | **멀티프로세스 + 역인덱스** | 1만 어휘를 36초에 |
| 학습률 | `3e-4` 고정 | **워밍업 200 + 선형 감쇠** | 3장 정리본에서 지적한 것 |
| 정밀도 | fp32 | **bf16 autocast** | 메모리 절반, 속도 약 2배 |
| 기울기 | 손대지 않음 | **`clip_grad_norm_(1.0)`** | 폭발 방지 |
| 검증 | 없음 | **valid 분할 + 주기 평가** | 3장 정리본에서 지적한 것 |
| 배치 | `DataLoader(shuffle=True)` | **`np.memmap` + 무작위 인덱스** | 1GB 파일을 메모리에 안 올린다 |
| 정렬 | SFT (지시-응답) | **DPO** (선호 쌍) | 좋은 답/나쁜 답을 직접 비교해 가르친다 |

**3장 정리본에서 "고칠 것"으로 적었던 네 가지가 4장에서 실제로 고쳐졌다** — 스케줄러, 검증셋, KV 캐시, top-k 대신 쓸 수 있는 정렬 기법.

---

## 4-2. TinyStories 와 멀티프로세스 BPE

### 데이터

**TinyStories** — GPT 가 만든 "네 살 아이가 이해할 어휘로 쓴 짧은 이야기" 모음이다.
문법은 온전하고 어휘는 좁다. **작은 모델로도 "말이 되는 글"에 도달하는 것**이 이 데이터의 목적이다.

| 파일 | 크기 | 토큰 |
| --- | --- | --- |
| `tiny_stories_train.bin` | 1,082,458,446 바이트 | **541,229,223** (uint16) |
| `tiny_stories_valid.bin` | 10,931,746 바이트 | **5,465,873** (노트북 출력) |
| `tiny_stories_valid.txt` | 22,502,601 바이트 | — |

**학습:검증 = 99:1 이다.** 3장의 268만 토큰에서 **202배**로 늘었다.

### 어휘 1만, 병합 9,743개

```python
vocab_size = 10000
merge_rules = train_bpe(file_path, vocab_size, num_processes=8)
```

```
num_merges = vocab_size - 256 - 1 = 9743
```

3장과 **똑같은 공식**이다. 다만 구현이 두 군데 달라졌다.

**① 파일을 8조각으로 나눠 8개 프로세스가 동시에 프리토크나이즈**

```python
def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    ...
    for bi in range(1, len(chunk_boundaries) - 1):
        file.seek(chunk_position)
        while True:
            buffer = file.read(4096)
            end_position = buffer.find(byte_end_token)
            if end_position != -1:
                chunk_boundaries[bi] = chunk_position + end_position
                break
            chunk_position += 4096
```

**단순히 파일 크기를 8로 나누지 않는다.** 나눈 자리에서 가장 가까운 `<|endoftext|>`까지 밀어 준다.
**이야기가 중간에서 잘리면 없던 바이트쌍이 생긴다.** 경계를 문서 끝에 맞춰 그걸 막는다.

```python
with Pool(processes=num_processes) as pool:
    all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list), ...))
```

CPU 코어 8개가 각자 한 조각을 읽어 **단어별 빈도 표**를 만들고, 메인이 합친다.

**② 병합할 때 전체를 다시 세지 않는다 — 역인덱스**

```python
pair_to_ids = defaultdict(set)      # 쌍 → 그 쌍이 들어 있는 단어들

for step in range(num_merges):
    best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p[0], p[1]))
    affected_ids = pair_to_ids[best_pair]        # ← 영향받는 단어만
    for ids in affected_ids:
        new_ids = merge(ids, best_pair, new_id)
        for pair, count in count_pairs(ids).items():
            pair_counts[pair] -= count * ids_count       # 빼고
        for pair, count in count_pairs(new_ids).items():
            pair_counts[pair] += count * ids_count       # 더한다
```

1장·3장 방식은 매 단계 말뭉치 전체의 쌍을 다시 셌다. **9,743번 × 전체**는 끝나지 않는다.
여기서는 **그 쌍을 포함한 단어만** 건드리고 차이만 가감한다. 그래서 **36초**에 끝났다.

> **동점 처리가 결정적이다**
> `key=(pair_counts[p], p[0], p[1])` — 빈도가 같으면 **바이트 값이 큰 쪽**을 고른다.
> 이 한 줄이 없으면 딕셔너리 순서에 따라 결과가 달라져 **같은 데이터로 다른 토크나이저**가 나온다.

### 적용 순서도 고쳐졌다

```python
def _encode_text(self, text):
    ids = list(text.encode("utf-8"))
    while len(ids) > 1:
        counts = count_pairs(ids)
        best_pair = min(counts, key=lambda p: self.merge_rules.get(p, float("inf")))
        if best_pair not in self.merge_rules:
            break
        ids = merge(ids, best_pair, self.merge_rules[best_pair])
    return ids
```

**현재 남아 있는 쌍 중 "가장 먼저 배운" 쌍부터 적용한다.** `min`의 기준이 병합 규칙의 id다.
규칙을 만든 순서와 쓰는 순서가 반드시 같아야 하는데, 이 코드는 그걸 **id 비교로 보장한다.**

### 토큰화 — 조각 캐시 후 이어 붙이기

```python
def encode_file(self, file_path, output_file, num_processes=4, num_chunks=64, cache_dir="bpe_cache"):
    try:
        ...
        with Pool(processes=num_processes) as pool:
            cache_results = list(tqdm(pool.imap(self._encode_chunk, chunk_info_list), ...))
        total_tokens = sum(r[1] for r in cache_results)
        arr = np.memmap(output_file, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        idx = 0
        for cache_file in cache_files:
            chunk_data = np.fromfile(cache_file, dtype=np.uint16)
            arr[idx: idx + len(chunk_data)] = chunk_data
            idx += len(chunk_data)
        arr.flush()
    finally:
        shutil.rmtree(cache_dir)
```

**64조각을 각각 파일로 떨어뜨린 뒤 memmap 으로 이어 붙인다.**
5.4억 토큰을 파이썬 리스트로 들고 있으면 메모리가 수십 GB 필요하다. 이렇게 하면 **1GB 짜리 결과 파일만** 있으면 된다.
`finally`로 캐시 폴더를 반드시 지우는 것도 눈여겨볼 점이다.

> **참고 두 가지**
> `process_single_chunk`가 `pretoken_chunk`와 **내용이 똑같은데 쓰이지 않는다.** 복사 흔적으로 보인다.
> 캐시 파일 이름이 `chunk_00000.npy`인데 `tofile`로 **raw 바이트**를 쓴다. 진짜 `.npy` 형식이 아니다.
> `np.fromfile`로 읽으니 동작은 맞지만, 확장자를 `.bin`으로 두는 편이 덜 헷갈린다.

---

## 4-3. RoPE — 회전으로 위치를 넣는다

3장은 위치마다 벡터를 하나씩 **학습**했다. 4장은 **각도를 돌린다.**

```python
class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        assert key_dim % 2 == 0
        half = key_dim // 2
        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ((2.0 * half_ids) / key_dim))
        positions = torch.arange(max_context_len)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cache", torch.cos(angles))
        self.register_buffer("sin_cache", torch.sin(angles))
```

```python
    def forward(self, x, offset=0):
        cos = self.cos_cache[offset:offset + context_len]
        sin = self.sin_cache[offset:offset + context_len]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        return out.reshape(batch_size, num_head, context_len, key_dim).to(input_dtype)
```

**벡터를 2개씩 묶어 (짝수 자리, 홀수 자리) 2차원 평면으로 보고, 위치에 비례해 돌린다.**
위 두 줄은 고등학교에서 배운 회전 행렬 그대로다.

```
[x'] = [cos  -sin] [x]
[y']   [sin   cos] [y]
```

### 왜 이게 위치 정보가 되나

**핵심은 내적이다.** 어텐션은 `Q·K`를 계산한다.
위치 `m`의 Q 를 `m` 만큼, 위치 `n`의 K 를 `n` 만큼 돌려 놓으면, 내적에는 **`m - n`만 남는다.**
즉 **상대 거리**가 자동으로 들어간다. 더하지도 않고 학습하지도 않았는데 생긴다.

| | `nn.Embedding` (3장) | RoPE (4장) |
| --- | --- | --- |
| 파라미터 | `256 × 384` = 98,304 | **0** |
| 담기는 정보 | 절대 위치 | **상대 거리** |
| 적용 위치 | 임베딩에 더함 (1회) | **Q·K 에 곱함 (층마다)** |
| 안 배운 길이 | 불가 | 이론상 외삽 가능 |
| 쓰는 곳 | GPT-2 | **LLaMA · Qwen · Mistral 등 현재 대부분** |

**`theta = 10000`** 이 주기를 정한다. 차원마다 다른 속도로 돌아서, 빠른 차원은 가까운 거리를, 느린 차원은 먼 거리를 구분한다.
(긴 문맥 모델이 `theta`를 50만 이상으로 올리는 이유가 이것이다.)

**`x.float()`로 올려 계산하고 `.to(input_dtype)`로 되돌린다.** bf16 으로 삼각함수를 돌리면 각도 오차가 누적된다.

> **주의 — 문맥을 넘으면 조용히 틀린다**
> ```python
> if offset + context_len > max_context_len:
>     offset = max_context_len - context_len
> ```
> 256 을 넘으면 **에러가 아니라 offset 을 고정한다.** 그 뒤 토큰들이 전부 같은 위치로 계산된다.
> 웹서비스에서 입력 56 + 생성 200 = 256 으로 묶어 둔 이유가 바로 이 줄이다 (6장).

---

## 4-4. RMSNorm 과 SwiGLU

### RMSNorm

```python
self.norm1 = nn.RMSNorm(embed_dim)
```

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| 식 | `(x - 평균) / 표준편차 × γ + β` | `x / √(평균(x²)) × γ` |
| 파라미터 | γ, β (2E개) | **γ 만 (E개)** |
| 평균 빼기 | 한다 | **안 한다** |

**평균을 빼지 않고 크기만 맞춘다.** 실제 성능 차이가 거의 없다는 것이 알려져서,
LLaMA 이후 거의 전부 RMSNorm 을 쓴다. `nn.RMSNorm`은 **파이토치 2.4 이상**에 있다.

### SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, x_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = int(x_dim * 8 / 3)
        self.W = nn.Linear(x_dim, hidden_dim, bias=False)
        self.V = nn.Linear(x_dim, hidden_dim, bias=False)
        self.O = nn.Linear(hidden_dim, x_dim, bias=False)

    def forward(self, x):
        gated = F.silu(self.W(x)) * self.V(x)
        return self.O(gated)
```

```
3장 FFN :  x → [Linear] → GELU → [Linear] → out        행렬 2개
4장 SwiGLU: x → [W] → SiLU ──┐
            x → [V] ─────────× → [O] → out              행렬 3개
```

**두 갈래로 보낸 뒤 하나가 다른 하나의 "문" 역할을 한다.**
`F.silu(a) * b` — `a`쪽이 0 에 가까우면 `b`가 지나가지 못한다. 이게 게이팅(gating)이다.

`SiLU(x) = x · sigmoid(x)` 는 GELU 와 모양이 거의 같고 계산이 더 싸다.

**행렬이 3개라 `8/3` 배를 기본 은닉 차원으로 쓴다.** 2개짜리 FFN 의 4배와 파라미터를 맞추려는 값이다
(`3 × 8/3 = 8 = 2 × 4`). 이 노트북은 `ff_dim = 1344`를 직접 넘겼다 (`512 × 8/3 = 1365`에 가까운 값).

### 블록

```python
class Block(nn.Module):
    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.ffn(self.norm2(x))
        return x
```

**Pre-LN + 잔차 구조는 3장과 똑같다.** 안에 든 부품만 바뀌었다.
"구조는 유지하고 부품을 교체한다" — 트랜스포머가 지난 8년간 발전한 방식이 이것이다.

---

## 4-5. KV 캐시

3장 정리본에서 "KV 캐시가 없어 길이에 제곱으로 느려진다"고 적었다. 여기서 붙었다.

```python
    def forward(self, x, use_cache=False):
        Q, K, V = ...                       # 이번에 들어온 토큰만
        if use_cache:
            is_first_call = (self.k_cache is None)
            if is_first_call:
                self.k_cache, self.v_cache = K, V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)
            self.cache_offset += C
            K, V = self.k_cache, self.v_cache
```

**K 와 V 는 과거 토큰이 바뀌지 않는다.** 한 번 계산한 것을 쌓아 두고, 새 토큰의 Q 하나만 가지고 전체와 비교한다.

```
캐시 없음 (3장)                      캐시 있음 (4장)
100번째 토큰 → 99개 전부 재계산      100번째 토큰 → K,V 1개만 계산 후 붙임
전체 비용 O(n²)                      전체 비용 O(n)
```

`generate()`도 그에 맞춰 바뀌었다.

```python
model.clear_cache()
...
logits = model(next_id, use_cache=True)[:, -1, :]     # 전체가 아니라 새 토큰 1개
```

**첫 호출에는 프롬프트 전체를 넣고(마스크 적용), 이후에는 토큰 1개씩 넣는다.**

> **주의 — 이 캐시에는 조건이 붙어 있다**
> ```python
> if not use_cache or (use_cache and is_first_call):
>     mask = torch.tril(...)
> ```
> **두 번째 호출부터는 인과 마스크를 걸지 않는다.** 토큰을 1개씩만 넣으니 가릴 것이 없어서 맞다.
> 하지만 **한 번에 2개 이상 넣으면 그 안에서 미래를 본다.** 토큰 1개 전제가 깨지면 조용히 틀린다.
>
> **`generate()`의 슬라이딩 윈도우가 무효가 됐다.**
> ```python
> if ids.size(1) > model.max_context_len:
>     ids = ids[:, -model.max_context_len:]     # ids 는 모델에 안 들어간다
> ```
> 모델에 들어가는 것은 `next_id`뿐이라 **이 줄은 아무 일도 하지 않는다.** 캐시는 계속 자란다.
> 256 을 넘기면 4-3의 RoPE offset 고정이 발동해 문맥이 깨진다. (5장 `webbot`에서는 캐시를 잘라 낸다.)

---

## 4-6. 학습 루프 — 스케줄러 · bf16 · 클리핑 · 검증

### 학습률 스케줄

```python
def get_lr(it, max_lr, warmup_iters, max_iters):
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)          # 0 → max 로 선형 상승
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)             # max → 0 으로 선형 하강
    return 0.0
```

```
lr
0.001 ┤      ╱‾╲___
      │     ╱      ‾‾‾╲___
      │    ╱              ‾‾‾╲___
    0 ┼───╱──────────────────────╲──→ iter
      0  200                   10000
```

**왜 워밍업이 필요한가.** 초기 가중치는 난수다. 처음부터 큰 학습률로 밀면 엉뚱한 방향으로 크게 간다.
**왜 감쇠가 필요한가.** 끝으로 갈수록 작게 움직여야 최저점에 안착한다.

`param_group["lr"]`을 매 스텝 직접 바꿔 넣는다. `torch.optim.lr_scheduler` 없이 손으로 구현한 것이다.

### bf16 자동 혼합정밀도

```python
with autocast(device_type=device.type, dtype=torch.bfloat16):
    logits = model(batch_x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
loss.backward()
```

**순전파만 bf16, 역전파와 가중치 갱신은 fp32 로 남는다.** `autocast`가 연산별로 알아서 고른다.

| | fp32 | fp16 | **bf16** |
| --- | --- | --- | --- |
| 지수부 | 8비트 | 5비트 | **8비트** |
| 정밀도 | 높음 | 중간 | 낮음 |
| `GradScaler` | 불필요 | **필요** | **불필요** |

**bf16 은 fp32 와 표현 범위가 같아서** 손실 스케일링 없이 그냥 쓸 수 있다.
fp16 은 범위가 좁아 기울기가 0 으로 가라앉기 때문에 `GradScaler`가 필요하다.
(Ampere 이상 NVIDIA GPU 에서 쓸 수 있다.)

### 기울기 클리핑

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**전체 기울기의 노름이 1.0 을 넘으면 비례 축소한다.** 방향은 유지하고 크기만 줄인다.
딥러닝 7장에서 RNN 에 쓴 것과 같은 함수다. 트랜스포머에도 그대로 쓴다.

### 배치 — memmap 과 무작위 인덱스

```python
train_data = np.memmap(data_path, dtype=np.uint16, mode="r")

def get_batch(data, context_len, batch_size, device, random=True, offset=0):
    if random:
        ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        ix = torch.arange(offset, offset + batch_size * context_len, context_len)
        ix = ix[ix + context_len + 1 < len(data)]
    x = torch.stack([torch.from_numpy(data[i:i + context_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_len+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
```

**`np.memmap`은 파일을 메모리에 안 올린다.** OS 가 필요한 부분만 그때그때 읽어 온다. 1GB 파일을 이렇게 다룬다.

| | 3장 | 4장 |
| --- | --- | --- |
| 방식 | `Dataset` + `DataLoader(shuffle=True)` | **무작위 시작 위치 32개** |
| 데이터 로드 | 전체를 텐서로 | **memmap** |
| 에폭 | `itertools.cycle` | 개념 없음 |

`random=False` 분기는 **검증용**이다. 검증은 매번 같은 구간을 같은 순서로 봐야 값을 비교할 수 있다.
`offset`을 `context_len`씩 띄워 **겹치지 않게** 자른다 (학습은 1칸씩 겹쳤다).

### 검증과 체크포인트

```python
eval_iters = 500
save_iters = [500, 5000]

if i in save_iters:
    model.save(f"storybot/model_iter_{i}.pt")

if (i % eval_iters) == 0 or i == max_iters - 1:
    val_loss = evaluate(model, valid_data, context_len, batch_size, device)
```

**500 스텝마다 검증, 500·5000 스텝에서 체크포인트 저장.**
3장에는 둘 다 없었다. 체크포인트 3개가 있어서 **4-7의 "학습이 진행되며 얼마나 나아지는가"** 실험이 가능해졌다.

---

## 4-7. LLM 으로 LLM 을 채점한다

손실 1.9645 는 "얼마나 잘 쓰는지"를 알려 주지 않는다. **GPT 에게 채점을 맡긴다.**

```python
client = OpenAI(api_key="")

def evaluate_story(client, story):
    evaluation_prompt = f"""다음 어린이용 스토리를 두 가지 관점에서 1 ~ 5 점으로 평가해줘.

스토리 :
{story}

평가관점 :
1. Coherence(일관성) : 논리적으로 이어지는가, 이야기로서 앞뒤가 맞는가
2. Grammar(문법) : 문법적으로 올바른 영어인가

다음 JSON 형식으로 대답해줘.
{{"coherence": <1~5 범위의 정수>, "grammar": <1~5 범위의 정수>, "comment": <간단한 평>}}"""
```

```python
model_paths = {500:   "storybot/model_iter_500.pt",
               5000:  "storybot/model_iter_5000.pt",
               10000: "storybot/model_pretrain.pt"}
num_samples = 10
```

**체크포인트 3개 × 샘플 10개 = 30편을 채점한다.**

이 방식의 이름이 **LLM-as-a-judge** 다. 사람이 채점하면 정확하지만 느리고 비싸다.
자동 지표(BLEU·ROUGE)는 정답 문장이 있어야 쓴다. **정답이 없는 생성 과제**에서는 강한 모델에게 묻는 것이 현실적인 타협이다.

**`JSON 형식으로 대답해줘`가 중요하다.** 자유 문장으로 받으면 점수를 뽑아낼 수 없다.
`statistics`로 평균·표준편차를 내려면 숫자가 구조화돼 있어야 한다.

> **이 평가를 읽을 때 조심할 것**
> **샘플이 10개다.** 4-9의 표준편차가 0.5~0.85 다. 체크포인트 간 0.6점 차이를 "확실히 좋아졌다"고 말하기엔 표본이 적다.
> **채점자도 모델이다.** GPT 의 취향이 점수에 섞인다. 사람 채점과 얼마나 맞는지는 확인하지 않았다.
> **`temperature=1.0`으로 생성했다.** 같은 체크포인트도 돌릴 때마다 다른 글이 나온다.

> **`api_key = ""` 는 실수가 아니라 잘한 일이다**
> 저장된 노트북에 키가 **빈 문자열**로 남아 있다. 실행할 때 넣었다가 지운 것으로 보인다.
> 공개 저장소에 올라가도 키가 새지 않는다. 다만 매번 손으로 지우는 것은 잊기 쉬우니
> `os.environ["OPENAI_API_KEY"]`를 읽는 편이 안전하다 (`OpenAI()`는 인자 없이도 이 변수를 자동으로 읽는다).

---

## 4-8. DPO — 좋은 답을 직접 가르친다

3장은 SFT 였다. "이런 질문엔 이렇게 답해라"를 **정답 하나**로 가르쳤다.
DPO 는 **둘을 나란히 놓고** "이쪽이 더 좋다"를 가르친다.

```python
data_path = "storybot/tiny_stories_dpo.json"      # {prompt, chosen, rejected}
learning_rate = 5e-6                              # 사전학습의 1/200
beta = 0.1
max_iters = 1000
batch_size = 8
```

```python
class DPODataset(Dataset):
    def _pad_and_mask(self, ids, prompt_len):
        mask = [0] * prompt_len + [1] * (len(ids) - prompt_len)
        if len(ids) > self.context_len:
            ids, mask = ids[:self.context_len], mask[:self.context_len]
        else:
            pad_len = self.context_len - len(ids)
            ids, mask = ids + [0] * pad_len, mask + [0] * pad_len
        return ids, mask
```

**마스크가 3장의 `-100`과 같은 역할이다.** 프롬프트 부분(0)은 점수 계산에서 빼고, 응답 부분(1)만 센다.

```python
loss = compute_dpo_loss(model, ref_model, chosen_ids, chosen_mask,
                        rejected_ids, rejected_mask, beta)
```

### DPO 가 하는 일

```
학습 중 모델 π :  chosen 의 확률 ↑   rejected 의 확률 ↓
기준 모델 ref  :  고정 (사전학습 그대로)

손실 ≈ -log σ( β × [ (logπ(chosen) - logref(chosen)) - (logπ(rejected) - logref(rejected)) ] )
```

**"기준 모델보다 chosen 을 얼마나 더 좋아하게 됐는지"에서 "rejected 를 얼마나 더 좋아하게 됐는지"를 뺀다.**
이 차이가 커지도록 밀어 준다.

**`ref_model`이 왜 필요한가.** 기준 없이 chosen 확률만 올리면 모델이 그 문장만 외우고 망가진다.
기준 모델과의 **차이**를 보기 때문에, 원래 실력에서 크게 벗어나지 않으면서 선호만 옮긴다.

**`beta = 0.1`이 그 고삐다.** 크면 기준에 더 묶이고, 작으면 더 자유롭게 움직인다.

| | SFT (3장) | DPO (4장) |
| --- | --- | --- |
| 데이터 | `{instruction, response}` | `{prompt, chosen, rejected}` |
| 가르치는 것 | "정답은 이것" | **"이쪽이 저쪽보다 낫다"** |
| 필요한 모델 | 1개 | **2개 (학습용 + 기준)** |
| 학습률 | `3e-4` (3장은 안 낮췄다) | **`5e-6`** |
| 보상 모델 | — | **필요 없다** |

**DPO 의 의의는 "보상 모델이 필요 없다"는 점이다.**
전통적인 RLHF 는 ① 선호 데이터로 보상 모델 학습 → ② PPO 로 강화학습, 두 단계였다.
DPO 는 수식 정리로 **보상 모델을 건너뛰고 한 단계**로 만들었다.

### 평가도 비교로 한다

```python
def compare_stories(client, story_a, story_b):
    evaluation_prompt = f"""다음의 두 어린이용 스토리를 비교하여, 어느 쪽이 더 해피엔딩인지 판단해줘.
    [Story A] {story_a}
    [Story B] {story_b}
    JSON형식으로 답변 : {{"winner": "A" or "B" or "tie", ...}}"""
num_comparisons = 100
```

**100회 일대일 비교.** 4-9에 결과가 있다.

> **주의 — 측정한 것이 학습시킨 것과 같다**
> DPO 데이터의 `chosen`이 해피엔딩이고, 심사 질문도 "어느 쪽이 더 해피엔딩이냐"다.
> **70.7% 는 "DPO 가 의도대로 작동했다"는 뜻이고, "이야기가 전반적으로 좋아졌다"는 뜻은 아니다.**
> 일반 품질을 보려면 4-7처럼 일관성·문법을 따로 재야 한다.
>
> **A/B 자리 편향을 통제하지 않았다.** 어느 쪽이 Story A 인지 무작위로 섞지 않으면
> 심사 모델이 앞쪽(또는 뒤쪽)을 선호하는 경향이 결과에 섞인다.
>
> **최종 손실 0.0026 은 거의 0 이다.** 1,092쌍 규모에 1,000스텝을 돌려 선호 쌍을 사실상 외운 상태다.
> 검증 쌍을 떼어 두고 재 보면 과적합 여부를 확인할 수 있다.

---

## 4-9. 실측 결과 — 2026-10-01 실행

**아래 숫자는 전부 노트북 출력에서 그대로 가져왔다.**

### 시간

| 단계 | 값 |
| --- | --- |
| 장치 | `cuda` |
| BPE 프리토크나이즈 | 8조각 · **36초** (4.52 s/it) |
| BPE 병합 9,743회 | **36초** (267.69 it/s) |
| 학습 데이터 토큰화 | 64조각 · **9분 17초** (8.71 s/it) |
| 검증 데이터 토큰화 | 64조각 · **4초** (13.79 it/s) |
| 검증 1회 | 668배치 · **약 59초** (11.3 it/s) |
| DPO 1,000회 | **2분 04초** (8.00 it/s) |

### 파라미터 — 계산과 실측이 일치했다

노트북 출력은 `22696448`이다. 구조에서 직접 계산해 맞춰 보았다.

| 구성 | 계산 | 수 |
| --- | --- | --- |
| 토큰 임베딩 | 10000 × 512 | 5,120,000 |
| 언임베딩 (공유 안 함) | 512 × 10000 | 5,120,000 |
| `RMSNorm` × 2 | 512 × 2 | 1,024 |
| 어텐션 `W_q·W_k·W_v·W_o` | 512 × 512 × 4 | 1,048,576 |
| SwiGLU `W·V·O` | 512×1344 × 2 + 1344×512 | 2,064,384 |
| **블록 1개** | | **3,113,984** |
| 블록 4개 | | 12,455,936 |
| 최종 `RMSNorm` | | 512 |
| **합계** | | **22,696,448** |

**한 자리도 틀리지 않았다.** `bias=False`와 가중치 공유를 하지 않은 것까지 반영한 결과다.
파일 크기도 맞는다 — 22,696,448 × 4바이트 = 90,785,792, 실제 `model_pretrain.pt`는 90,835,441 바이트다.

3장과 비교하면 **파라미터는 2.04배, 데이터는 202배**다. 데이터가 훨씬 빠르게 늘었다.

| 항목 | 3장 | 4장 |
| --- | --- | --- |
| 어휘 | 1,000 | 10,000 |
| 학습 토큰 | 2,677,398 | **541,229,223** |
| 파라미터 | 11,121,640 | **22,696,448** |
| 층 | 6 | 4 |
| 임베딩 차원 | 384 | 512 |
| 헤드 | 6 (head_dim 64) | **16 (head_dim 32)** |

### 검증 손실

| 스텝 | 검증 손실 |
| --- | --- |
| 이론상 시작값 | `ln(10000)` = **9.2103** |
| 약 1,000 | 2.4350 |
| 약 5,000 | 2.1112 |
| **약 10,000 (마지막)** | **1.9645** |

**감소 폭이 줄고 있다.** 2.44 → 2.11 (-0.32), 2.11 → 1.96 (-0.15).
`max_iters = 40000`이 주석으로 남아 있는데, 이 추세라면 4배를 더 돌려도 1.7 근처일 것으로 보인다 — **확인된 값이 아니라 추정이다.**

### GPT 채점 (샘플 10편 × 체크포인트 3개)

| 스텝 | Coherence (일관성) | Grammar (문법) |
| --- | --- | --- |
| 500 | 2.00 ± 0.00 | 2.00 ± 0.47 |
| 5,000 | 2.60 ± 0.52 | 2.80 ± 0.79 |
| **10,000** | **3.20 ± 0.63** | **3.40 ± 0.84** |

**학습이 진행될수록 점수가 오른다 — 손실 감소가 품질 향상으로 이어졌다.**
500 스텝의 일관성이 10편 모두 2점(표준편차 0)인 것이 눈에 띈다. **"전부 똑같이 엉망"** 이었다는 뜻이다.

실제 500 스텝 생성물:

```
John was having lots of fun playing. He looked through his bed, he needed to be
a rare one of what had happened. All his friends talked to their bucket and
immediately said he was very friendly...
```

GPT 의 평: *"The story lacks logical flow and clarity, making it difficult to follow."*

### 10,000 스텝 생성물

`prompt` 를 종료 토큰 하나로 주어 아무 조건 없이 처음부터 쓰게 했다.

```
One day, a soft frog named Fifi lived in a small pond. Fifi was sad and miserable.
She could not find any food to eat.
A duck named Ducky saw Froggy looking for food. Ducky asked, "Why are you so sad?"
Froggy said, "I miss any food, can you show me how to find food?"
Ducky and Froggy went to a new pond with lots of fish. They found yummy fish,
tasty fish, and a big tree. Ducky and Froggy were so happy.
```

**읽어 낼 것 세 가지.**

1. **이야기의 틀이 생겼다.** 등장인물 소개 → 문제 발생 → 조력자 등장 → 대화 → 해결. TinyStories 의 전형적 구조를 익혔다.
2. **대화 형식이 정확하다.** `Ducky asked, "Why are you so sad?"` — 인용 부호, 쉼표 위치, 말한 사람 표기가 전부 맞다.
3. **그런데 이름이 흔들린다.** `Fifi` → `Freddy` → `Froggy` 로 같은 개구리의 이름이 세 번 바뀐다.
   **긴 범위의 일관성이 아직 없다.** 3장의 `reverse_list` 문제와 같은 종류다 — 국소적으로는 맞고 전체적으로는 틀린다.

다른 샘플에는 문법은 맞고 뜻이 안 통하는 문장도 섞여 있다 (`these men are chefs ever messy about knives`).

### DPO — 일대일 비교 100회

| 승자 | 횟수 |
| --- | --- |
| 사전학습 모델 | 29 (29.0%) |
| **DPO 모델** | **70 (70.0%)** |
| 무승부 | 1 (1.0%) |
| **DPO 승률 (무승부 제외)** | **70.7%** |

DPO 최종 손실은 **0.0026** 이다.

**"더 해피엔딩인 쪽"을 묻는 심사에서 DPO 모델이 70.7% 이겼다.** DPO 가 의도대로 작동했다.
다만 4-8에 적은 대로, 이 수치는 **학습 목표를 그대로 측정한 것**이다.

### 체크포인트가 한때 전부 NaN 이었다

웹서비스를 만들며 발견한 사건이다 (`storybot/prd.md` 10.1).

| 날짜 | 내용 |
| --- | --- |
| 2026-10-01 | `model_pretrain.pt`·`model_iter_500.pt`·`model_iter_5000.pt` 세 파일의 **학습 가중치 39개 텐서 전체가 NaN** 이었다 |
| 조치 | ch04 설정으로 재학습 시작 (iter 2000 에서 val_loss 1.88 확인) |
| 결과 | 재학습 중 `storybot/` 폴더가 정상 체크포인트로 교체됨. 4개 파일 모두 NaN·무한대 없음 확인 |
| 현재 | `"Once upon a time"` 입력에 자연스러운 영어 이야기가 나온다 |

**교훈은 "저장된 체크포인트를 믿지 말고 열어 보라"는 것이다.** NaN 이 든 모델도 `load_state_dict`는 성공한다.
웹서비스 쪽은 그래서 **서버 시작 시 가중치를 검사한다.**

```python
app.state.model_error = (
    "스토리봇 모델 가중치에 NaN 또는 무한대 값이 있어 생성할 수 없습니다."
    if any(not p.isfinite().all().item() for p in model.parameters()) else None
)
```

> **주의 — 이 장의 실측값은 노트북 출력 기준이다**
> 위 표의 검증 손실·채점·DPO 수치는 **노트북에 남은 출력**이다. NaN 체크포인트가 그 뒤에 교체됐으므로,
> 현재 폴더의 `.pt` 파일이 이 출력을 낸 그 파일과 같다고 단정할 수는 없다.
> **숫자를 다시 쓰려면 한 번 더 돌려 확인하는 편이 안전하다.**

### 실행 상태에 남은 흔적

| 항목 | 내용 |
| --- | --- |
| 셀 5·7·11·12 | 출력은 있는데 `execution_count`가 `None` 이다. 저장 시점에 실행 번호가 지워졌다 |
| 셀 0~4 | `ec = 1~5` 로 순서대로 남아 있다 |
| `max_iters` | `40000`이 주석 처리되고 `10000`으로 실행됐다 |

---

## 이 장 정리

### 한 줄 요약

**3장이 "GPT 를 만든다"였다면 4장은 "현대 LLM 의 부품으로 갈아 끼우고 정렬까지 한다"다.**
RoPE · RMSNorm · SwiGLU · KV 캐시 · bf16 · DPO — 2026년의 모델이 실제로 쓰는 것들이다.

### 전체 파이프라인

```
TinyStories 원문
   │  train_bpe(vocab_size=10000, num_processes=8)   — 36초 + 36초
   ↓
merge_rules.pkl  병합 규칙 9,743개
   │  encode_file(num_chunks=64)                     — 9분 17초
   ↓
tiny_stories_train.bin  541,229,223 토큰 (uint16, 1.08GB)
tiny_stories_valid.bin    5,465,873 토큰
   │  memmap + 무작위 시작 위치 32개
   ↓
GPT 2,270만 (RoPE·RMSNorm·SwiGLU, 4층 × 16헤드 × 512)
   │  10,000 iter · bf16 · 워밍업 200 + 선형감쇠 · clip 1.0
   ↓  검증 손실 9.2103(이론) → 1.9645
model_pretrain.pt  +  model_iter_500.pt  model_iter_5000.pt
   │  GPT 채점: 일관성 2.00 → 2.60 → 3.20
   ↓
   │  DPO 1,000 iter · lr 5e-6 · beta 0.1             — 2분 04초
   ↓
model_dpo.pt   해피엔딩 심사 70.7% 승
```

### 외워 둘 숫자

| 항목 | 값 | 출처 |
| --- | --- | --- |
| 어휘 크기 | 10,000 = 256 + 9,743 + 1 | 공식 |
| 학습 토큰 | 541,229,223 | 파일 실측 |
| 검증 토큰 | 5,465,873 | 노트북 출력 |
| 파라미터 | 22,696,448 | 노트북 출력 = 계산값 |
| 최종 검증 손실 | 1.9645 | 노트북 출력 |
| 학습 전 기대 손실 | `ln(10000)` = 9.2103 | 이론값 |
| 일관성 점수 (10,000스텝) | 3.20 ± 0.63 | 노트북 출력 |
| DPO 승률 | 70.7% | 노트북 출력 |
| `head_dim` | 512 / 16 = 32 | 설정 |
| SwiGLU 기본 은닉 | `x_dim × 8/3` | 코드 |

### 외워 둘 코드

```python
# RoPE — 2개씩 묶어 회전
x_rot_even = x_even * cos - x_odd * sin
x_rot_odd  = x_even * sin + x_odd * cos

# SwiGLU — 한쪽이 다른 쪽의 문
gated = F.silu(self.W(x)) * self.V(x)

# KV 캐시 — 과거는 다시 계산하지 않는다
self.k_cache = torch.cat([self.k_cache, K], dim=2)

# 워밍업 + 선형 감쇠
lr = max_lr * (it / warmup) if it < warmup else max_lr * (1 - progress)

# bf16 — GradScaler 없이
with autocast(device_type=device.type, dtype=torch.bfloat16):

# memmap — 1GB 를 메모리에 안 올린다
train_data = np.memmap(path, dtype=np.uint16, mode="r")
```

### 이 노트북·패키지에서 고칠 것

| 위치 | 내용 |
| --- | --- |
| `utils.generate` | `ids[:, -256:]` 가 KV 캐시에 영향이 없어 **슬라이딩 윈도우가 무효**다 |
| `RoPE.forward` | 256 초과 시 에러 대신 offset 고정 → **조용히 문맥이 깨진다** |
| `MultiHeadAttention` | 캐시 2회 호출 이후 마스크 미적용. **토큰 1개 전제가 깨지면 미래를 본다** |
| `GPT.load_from` | 3장에 있던 `weights_only=True` 가 빠졌다 |
| `tokenizer.py` | `process_single_chunk` 가 `pretoken_chunk` 와 중복인데 쓰이지 않는다 |
| `tokenizer.py` | 캐시 파일 확장자가 `.npy` 인데 `tofile` raw 저장이다 |
| DPO 평가 | A/B 자리를 무작위로 섞지 않아 **위치 편향**이 섞인다 |
| DPO | 검증 쌍이 없다. 최종 손실 0.0026 은 과적합 신호다 |
| GPT 채점 | 샘플 10개는 표준편차 대비 적다. 30~50개면 차이를 말할 수 있다 |
| `api_key` | 빈 문자열을 손으로 넣는 대신 `os.environ` 을 읽는 편이 안전하다 |
| 노트북 | 셀 5·7·11·12 의 실행 번호가 비어 있다 |

### 자주 틀리는 것

- RoPE 의 `key_dim`이 홀수 → `assert`에서 걸린다. 짝수여야 2개씩 묶을 수 있다
- RoPE 를 임베딩에 **더한다** → Q·K 에 **곱해야** 한다 (층마다)
- bf16 에 `GradScaler`를 붙인다 → 불필요하다. fp16 에만 필요하다
- SwiGLU 은닉 차원을 4배로 둔다 → 행렬이 3개라 파라미터가 1.5배가 된다. `8/3` 배가 맞다
- KV 캐시를 비우지 않고 다시 생성한다 → `model.clear_cache()` 를 먼저 부른다
- 검증에 무작위 배치를 쓴다 → 매번 값이 달라 비교가 안 된다. `random=False`
- `memmap` 데이터를 `.astype(np.int64)` 없이 텐서로 → `uint16`은 파이토치가 받지 않는다
- 체크포인트를 열어 보지 않고 쓴다 → **NaN 이어도 `load_state_dict`는 성공한다**
