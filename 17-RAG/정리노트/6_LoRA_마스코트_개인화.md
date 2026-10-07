# 6. LoRA — 모델에게 우리 마스코트를 가르친다

**실습 파일**: `17-KDT/17-RAG/TotalAI.ipynb` (셀 41~48)

5장의 Stable Diffusion 은 **세상에 있는 것**을 그린다. 우리 동아리 마스코트는 모른다.
6장은 **사진 21장으로 모델에게 그 캐릭터를 가르친다.**

> **2026-10-07 — 돌았다. 단, 학습은 생략됐다**
> `work/models/moa-mascot-lora/pytorch_lora_weights.safetensors` (**6.4MB**) 가 이미 있어서
> `RETRAIN = False` 조건에 걸려 **학습 셀이 건너뛰어졌다.**
> **학습 손실 곡선과 학습 시간은 이 노트북에 없다** — 적용 결과(셀 45~47)만 있다.

| 절 | 내용 |
| --- | --- |
| 6-1 | LoRA 가 무엇인가 — 숫자로 먼저 |
| 6-2 | 학습 데이터 — 사진 21장과 트리거 단어 |
| 6-3 | 학습 루프 — 확산 모델을 직접 돌린다 |
| 6-4 | 꽂고 빼기 — `load_lora_weights` 와 세기 조절 |
| 6-5 | 고칠 것 |

---

## 6-1. LoRA 가 무엇인가 — 숫자로 먼저

```python
d, r = 768, 8                      # d: 원래 표의 한 변 / r: LoRA의 '랭크'
W = torch.randn(d, d)              # 원래 가중치 (고정)
A = torch.randn(d, r) * 0.01       # 보조 표 1
B = torch.zeros(r, d)              # 보조 표 2 (처음엔 0)
x = torch.randn(1, d)

y_original = x @ W
y_lora = x @ W + x @ A @ B
print("처음엔 LoRA가 결과를 바꾸지 않음 →", torch.allclose(y_original, y_lora))
```

```plain text
처음엔 LoRA가 결과를 바꾸지 않음 → True

원래 표 W        :   589,824개 숫자
LoRA (A + B)     :    12,288개 숫자  → 원래의 2.1% 만 학습
```

**이 네 줄이 LoRA 전부다.**

```plain text
원래:  y = x W            W 를 직접 고친다 → 589,824개를 다 움직여야 한다
LoRA:  y = x W + x A B    W 는 얼리고 A·B 만 움직인다 → 12,288개
```

**`B = torch.zeros(...)` 가 설계의 핵심이다.**
`A @ B` 가 처음에 **0 행렬**이므로 **학습 시작 시점의 모델은 원래 모델과 완전히 같다.**
`torch.allclose(y_original, y_lora) == True` 가 그것을 보여 준다.

| 만약 | 결과 |
| --- | --- |
| `A`·`B` 둘 다 난수 | 학습 시작부터 모델이 **망가진 상태**에서 출발한다 |
| **`A` 난수 + `B` 0** | **원래 모델에서 출발해 조금씩 벗어난다** |
| `A` 0 + `B` 난수 | 기울기가 0이라 **영원히 안 움직인다** (`A` 가 0이면 `B` 의 기울기도 0) |

**한쪽만 0이어야 하고, 그 한쪽은 ****`B`**** 여야 한다.**

### 왜 저차원으로 충분한가

`A @ B` 는 `(768, 8) @ (8, 768) = (768, 768)` 이지만 **랭크가 최대 8이다.**
768×768 행렬이 가질 수 있는 랭크는 768인데, 그중 8차원만 쓴다.

**"과제에 맞추는 변화는 저차원이면 된다"는 것이 LoRA 논문의 주장이고**,
실제로 잘 동작한다. 랭크 `r` 이 그 "두께"다.

| `r` | 학습 파라미터 | 쓰는 곳 |
| --- | --- | --- |
| 4 | 1% 미만 | 화풍만 살짝 |
| **8** | **2.1%** | **이 노트북. 캐릭터 한 명** |
| 16~32 | 4~8% | 복잡한 개념, 여러 요소 |
| 64+ | — | 과적합 위험. 데이터가 많아야 한다 |

### 2장의 프롬프트 튜닝과 나란히

| | 프롬프트 튜닝 (2-9) | **LoRA (6장)** |
| --- | --- | --- |
| 무엇을 더하나 | **입력 앞에 벡터 20개** | **가중치에 저차원 보정** |
| 모델 내부를 건드리나 | 아니다 | **그렇다** (어텐션 층) |
| 학습 파라미터 | 8,450 (0.0072%) | UNet 어텐션의 약 2% |
| 저장 크기 | 30KB | **6.4MB** |
| 표현력 | 입력 재해석만 | **더 깊은 변화** |

**둘 다 "원래 가중치를 얼리고 작은 것만 학습한다"는 PEFT 의 발상이다.**
프롬프트 튜닝은 **입구에서**, LoRA 는 **안에서** 바꾼다.

---

## 6-2. 학습 데이터 — 사진 21장과 트리거 단어

```python
MASCOT_DIR = ASSETS / "mascot"
TRIGGER = "moa_character"          # 평소엔 안 쓰는 낯선 단어여야 한다

captions = {}
for line in open(MASCOT_DIR / "metadata.jsonl", encoding="utf-8"):
    row = json.loads(line)
    captions.setdefault(row["file_name"],
        row["prompt"].replace("(lora-misato-token)", TRIGGER).strip())   # 중복 줄은 첫 번째만
train_files = [f for f in captions if (MASCOT_DIR / f).exists()]
print(f"학습 사진 {len(train_files)}장")
```

```plain text
학습 사진 21장
```

### `TRIGGER` 가 LoRA 의 "이름표"다

```plain text
학습:  "moa_character 가 웃고 있는 정면 사진"  +  실제 사진
   -> 모델은 moa_character 라는 토큰과 그 외형을 묶어 외운다

생성:  "portrait of moa_character as an astronaut"
   -> 외운 외형을 우주복 맥락에 끼워 넣는다
```

**낯선 단어여야 하는 이유** — `"robot"` 을 트리거로 쓰면
모델이 이미 알고 있는 "로봇" 개념 전체가 **마스코트로 오염된다.**
`moa_character` 처럼 **학습 데이터에 없던 조합**이면 깨끗한 새 자리를 차지한다.

**`metadata.jsonl` 의 `(lora-misato-token)` 을 바꿔 쓴 것에 주의** —
받아 온 데이터셋이 다른 트리거를 쓰고 있었고, 그것을 내 트리거로 **치환**했다.
남이 만든 LoRA 데이터셋을 쓸 때는 **그 데이터셋의 트리거가 무엇인지 먼저 확인해야 한다.**

**`captions.setdefault(...)` 가 중복 방어다.**
`metadata.jsonl` 에 같은 파일이 여러 줄 있으면 **첫 줄만 쓴다.**
`dict[key] = value` 로 썼다면 마지막 줄이 이겼을 것이다.

**`[f for f in captions if (MASCOT_DIR / f).exists()]`**
메타데이터에 적혀 있지만 **실제로 없는 파일을 걸러낸다.** 없는 파일을 열면 학습 루프 중간에 죽는다.

### 21장이 적은가

| 기법 | 보통 필요한 장수 |
| --- | --- |
| DreamBooth | 3~5장 (단, 전체 UNet 학습) |
| **LoRA** | **10~30장** |
| 처음부터 학습 | 수십만 장 |

**21장은 LoRA 로 캐릭터 하나를 가르치기에 적당한 양이다.**
다만 **각도·표정이 다양해야** 한다 — 파일 이름(`front-portrait1`, `rightprofile-smile`,
`front-full3`, `34-5-smile`)을 보면 **정면·측면·3/4 각도·전신·표정**이 섞여 있다.
같은 각도만 21장이면 그 각도만 외운다.

---

## 6-3. 학습 루프 — 확산 모델을 직접 돌린다

```python
LORA_DIR = WORK / "models" / "moa-mascot-lora"
RETRAIN = False                             # True면 저장된 LoRA 가 있어도 다시 학습
LORA_STEPS = 300 if QUICK else 1200
BATCH, RANK, LR = 2, 8, 1e-4

need_train = RETRAIN or not (LORA_DIR / "pytorch_lora_weights.safetensors").exists()
```

```plain text
✅ 이미 학습된 LoRA가 있어서 학습을 건너뜁니다
```

> **이 셀의 설계가 좋다.** 파일이 있으면 학습을 건너뛰므로 **노트북을 다시 돌려도 30분을 또 안 쓴다.**
> 대신 **이번 실행에는 학습 기록이 남지 않았다** — 6-5 참조.

### 부품을 직접 조립한다

```python
tokenizer_sd = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder",
                                             variant="fp16", torch_dtype=torch.float32).to(DEVICE).eval()
vae          = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae", ...).to(DEVICE).eval()
unet         = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet", ...).to(DEVICE)
noise_sched  = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
```

**`StableDiffusionPipeline` 을 쓰지 않는다.** 학습에는 **부품 네 개를 따로** 들고 와야 한다.

| 부품 | 역할 | 학습하나 |
| --- | --- | --- |
| `CLIPTokenizer` + `CLIPTextModel` | 캡션 → 임베딩 | **아니다** (얼림) |
| `AutoencoderKL` (VAE) | 사진 ↔ latent | **아니다** (얼림) |
| **`UNet2DConditionModel`** | 노이즈 예측 | **LoRA 부분만** |
| `DDPMScheduler` | **노이즈를 섞는** 일정 | 학습 안 함 |

**`DPMSolverMultistepScheduler`(5-3) 가 아니라 ****`DDPMScheduler`**** 다.**

```plain text
생성용 스케줄러 : 노이즈를 어떻게 빼낼까   (DPMSolver, 20스텝)
학습용 스케줄러 : 노이즈를 어떻게 섞을까   (DDPM, 1000 timestep)
```

**방향이 반대다.** 학습은 `add_noise` 를 쓰고, 생성은 `step` 을 쓴다.

**`torch_dtype=torch.float32` 로 올린 것이 중요하다.**
5장에서는 추론이라 `fp16` 이었지만, **학습에서 fp16 은 기울기가 0으로 죽는다**(언더플로).
`variant="fp16"` 으로 **파일은 작게 받고**, `torch_dtype=float32` 로 **메모리에는 fp32 로 올린다** —
5-2에서 본 "두 인자는 다르다"가 여기서 실질적인 차이를 만든다.

### 어텐션 층에만 LoRA 를 붙인다

```python
text_encoder.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(False)
unet.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights="gaussian",
                            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
lora_params = [p for p in unet.parameters() if p.requires_grad]
```

| 인자 | 뜻 |
| --- | --- |
| `r=8` | 6-1의 랭크 |
| `lora_alpha=8` | 스케일. `r` 과 같게 두면 배율 1 |
| `init_lora_weights="gaussian"` | `A` 를 정규분포로 (`B` 는 라이브러리가 0으로) |
| **`target_modules=["to_k","to_q","to_v","to_out.0"]`** | **어텐션의 Q·K·V·출력 투영만** |

**왜 어텐션만인가.** 크로스 어텐션이 **"텍스트와 이미지를 잇는 자리"**다.
`moa_character` 라는 토큰이 어떤 모양에 붙는지를 정하는 곳이 거기다.
합성곱 층까지 건드리면 파라미터가 훨씬 늘고 **과적합이 쉬워진다.**

**`peft` 라이브러리를 여기서 처음 쓴다** — 1장 `requirements` 에 있었지만
2장에서는 프롬프트 튜닝을 직접 구현했고, 6장에서는 **라이브러리를 쓴다.**

### 미리 계산해 두고 학습한다

```python
to_tensor = T.Compose([T.Resize(512), T.CenterCrop(512), T.ToTensor(), T.Normalize([0.5], [0.5])])
with torch.no_grad():
    latents = torch.cat([vae.encode(to_tensor(img).unsqueeze(0).to(DEVICE))
                           .latent_dist.sample() * vae.config.scaling_factor
                         for f in train_files])
    ids = tokenizer_sd([captions[f] for f in train_files], padding="max_length", ...).input_ids.to(DEVICE)
    text_embeds = text_encoder(ids)[0]
del text_encoder, vae; free_memory()
```

**21장뿐이니 전부 미리 변환해 메모리에 들고 있는다.**
그러면 **학습 루프에서 VAE·텍스트 인코더를 한 번도 부르지 않는다** — 그래서 `del` 로 바로 지운다.

**`* vae.config.scaling_factor`** — 5-7에서 꺼낼 때 나눴던 그 상수를 **넣을 때는 곱한다.**
**`Normalize([0.5], [0.5])`** 는 `[0,1]` → `[-1,1]` 로 보낸다. VAE 가 그 범위를 기대한다.

### 루프는 세 줄이다

```python
for step in range(1, LORA_STEPS + 1):
    idx = torch.randint(0, len(train_files), (BATCH,))
    x0 = latents[idx]                                              # 깨끗한 사진(압축본)
    noise = torch.randn_like(x0)                                   # 섞을 잡음
    t = torch.randint(0, noise_sched.config.num_train_timesteps, (BATCH,), device=DEVICE)
    x_noisy = noise_sched.add_noise(x0, noise, t)                   # 1. 잡음 섞기
    pred = unet(x_noisy, t, encoder_hidden_states=text_embeds[idx]).sample   # 2. "섞인 잡음이 뭐게?"
    loss = torch.nn.functional.mse_loss(pred.float(), noise.float())         # 3. 정답과의 차이
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step(); optimizer.zero_grad()
```

**5-1에서 글로 적은 학습 목표가 그대로 코드가 됐다.**

```plain text
깨끗한 latent  --add_noise(t)-->  흐린 latent
                                      |  unet(흐린 것, t, 캡션 임베딩)
                                      v
                                  예측한 노이즈
                                      |  MSE
                                      v
                                  실제 섞은 노이즈
```

**정답이 "그림"이 아니라 "섞은 노이즈"다.** 그래서 라벨을 사람이 만들 필요가 없다 —
자기지도학습이다(코드 생성 LM 3-5의 "텍스트 자체가 정답"과 같은 발상).

**`t` 를 매 스텝 무작위로 뽑는다.** 0~999 중 아무 세기나 섞어 보고 맞히게 한다.
특정 세기만 학습하면 **생성할 때 그 구간만 잘 빠진다.**

| 설정 | 값 | 비고 |
| --- | --- | --- |
| `LORA_STEPS` | **300** (`QUICK`) / 1200 | 21장이니 300스텝 = 약 28에폭 |
| `BATCH` | **2** | 512×512 fp32 UNet 이라 VRAM 이 빡빡하다 |
| `LR` | **1e-4** | LoRA 의 관례. 2장 프롬프트 튜닝(`2e-2`)보다 200배 작다 |
| `clip_grad_norm_` | 1.0 | 7장 RNN 과 같은 안전장치 |

**저장은 LoRA 만 한다.**

```python
StableDiffusionPipeline.save_lora_weights(
    str(LORA_DIR),
    unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
    safe_serialization=True)
```

| 저장한 것 | 크기 |
| --- | --- |
| SD1.5 전체 (fp16) | 약 2GB |
| **LoRA 만** | **6.4MB** |

**0.3% 다.** 캐릭터 10명이면 LoRA 10개(64MB)를 들고 다니면 된다 —
2장에서 프롬프트 튜닝으로 본 "과제마다 모델 전체를 저장하지 않는다"의 같은 이득이다.

---

## 6-4. 꽂고 빼기 — `load_lora_weights` 와 세기 조절

```python
pipe = StableDiffusionPipeline.from_pretrained(SD_ID, torch_dtype=IMG_DTYPE, variant="fp16", ...)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(str(LORA_DIR), adapter_name="moa")        # 포스트잇 꽂기

def draw_lora(prompt, use_lora=True, scale=1.0, **kw):
    if use_lora:
        pipe.enable_lora(); pipe.set_adapters(["moa"], adapter_weights=[scale])
    else:
        pipe.disable_lora()
    return draw(prompt, **kw)
```

**추론에서는 다시 ****`fp16`**** + ****`DPMSolverMultistep`**** 이다.** 학습 설정(fp32·DDPM)과 다르다.

**`enable_lora()` / `disable_lora()` 로 같은 파이프라인에서 켜고 끌 수 있다** —
비교 실험을 하려고 모델을 두 번 올릴 필요가 없다.

```plain text
No LoRA keys associated to CLIPTextModel found with the prefix='text_encoder'.
```

**이 경고는 정상이다.** UNet 에만 LoRA 를 붙였으니 텍스트 인코더용 키가 없는 것이 맞다.
(2-8의 `classifier.weight MISSING` 과 같은 종류의 "정상 경고"다.)

### 실측 ① LoRA 없이 / 있이

```python
prompt = f"a portrait photo of {TRIGGER}, high quality, natural light"
before = draw_lora(prompt, use_lora=False, seed=3)
after  = draw_lora(prompt, use_lora=True,  seed=3)
```

**같은 프롬프트, 같은 시드(3)로 셋을 나란히 놓았다** — LoRA 없음 / LoRA 적용 / 실제 마스코트 사진.

**LoRA 없이는 `moa_character` 가 의미 없는 토큰이다.** 모델은 그 자리를 아무렇게나 채운다.
**LoRA 를 켜면 마스코트가 나온다.** 이것이 "모델에게 새 개념을 가르쳤다"는 증거다.

**시드를 고정한 것이 이 비교를 성립시킨다.** 시드가 다르면
"LoRA 때문인지 난수 때문인지" 구분할 수 없다 — 비전 트랜스포머 2장의 통제 비교와 같은 원칙이다.

### 실측 ② LoRA 세기

```python
scene = f"portrait of {TRIGGER} as an astronaut, wearing a space suit, face visible, no helmet, moon background"
imgs = [draw_lora(scene, use_lora=s > 0, scale=max(s, 0.01), seed=8) for s in (0.0, 0.5, 1.0)]
```

| `scale` | 효과 |
| --- | --- |
| **0.0** | LoRA 끔. 일반 우주인 |
| **0.5** | 마스코트 특징이 섞인다 |
| **1.0** | 학습한 외형에 가장 충실 |

**`adapter_weights` 가 "포스트잇을 얼마나 세게 누를까"다.** `A @ B` 에 곱하는 배율이다.

**세기를 1 보다 올릴 수도 있다**(1.2~1.5). 다만 **학습 데이터의 구도까지 끌려와**
"우주복"처럼 새로 요구한 요소가 무시되기 시작한다 — **5-6의 guidance 와 같은 모양의 거래**다.

**`max(s, 0.01)` 이 방어 코드다.** `scale=0` 을 넘기면 라이브러리가 어댑터를 어떻게 다룰지
애매해지므로, **끄는 것은 `use_lora=False`(`disable_lora`) 로 하고** scale 은 0 을 피한다.

### 실측 ③ 네거티브 프롬프트가 더 중요해진다

```python
bad = ("low quality, grainy, blurry, deformed, disfigured, bad anatomy, "
       "extra limbs, mutated hands, watermark, text")
show_images([draw_lora(scene, seed=8, negative=""),
             draw_lora(scene, seed=8, negative=bad)], ...)
```

**5-3의 `NEGATIVE` 보다 길어졌다.** `bad anatomy`, `extra limbs`, `mutated hands` 가 추가됐다.

**캐릭터 생성에서는 손·팔다리가 가장 잘 망가진다.** 21장으로 외운 외형을
새 포즈(우주복)에 끼워 넣다 보면 해부학이 어긋난다. **그 방향을 미리 밀어내는 것**이다.

---

## 6-5. 고칠 것

| 위치 | 내용 | 수정안 |
| --- | --- | --- |
| 셀 43·44 | **학습이 생략됐다** (`RETRAIN=False` + 파일 존재) → **손실 곡선·학습 시간이 노트북에 없다** | 한 번은 `RETRAIN=True` 로 돌려 기록을 남긴다 |
| 셀 43 | 저장된 LoRA 가 **300스텝인지 1200스텝인지 알 수 없다** | 저장할 때 설정을 JSON 으로 함께 남긴다 |
| 셀 44 | **검증이 없다.** 21장을 300스텝 돌리면 28에폭이다 | 사진 2~3장을 떼어 두거나, 중간 생성으로 눈으로 확인 |
| 셀 44 | `del unet, optimizer, ...` 가 **학습 분기 안에만** 있다 | 생략 분기에서도 메모리 상태를 같게 맞춘다 |
| 셀 45 | LoRA 를 **UNet 에만** 붙였다 | 텍스트 인코더에도 붙이면 트리거 학습이 더 잘 되는 경우가 있다 |
| 셀 46 | `scale` 3개만 봤다. **1.0 초과를 보지 않았다** | 1.2·1.5 를 함께 보면 한계가 보인다 |
| 셀 46·47 | **생성 시간을 재지 않는다** | `time.time()` |
| 공통 | 마스코트 사진의 **출처·라이선스가 노트북에 없다** | `metadata.jsonl` 의 원 트리거가 `(lora-misato-token)` 이다. 받아 온 데이터셋이면 출처를 적는다 |
| `work/models` | `__MACOSX/` 폴더가 생겼다 (맥 압축 잔여물) | 삭제. `.gitignore` 가 이미 잡는다 |

---

## 이 장 정리

### 한 줄 요약

**LoRA 는 원래 가중치를 얼리고, 어텐션 층에 저차원 보정 행렬 두 개(`A`·`B`)만 학습한다.**
사진 21장과 낯선 트리거 단어로 **6.4MB 파일 하나**를 만들어, 모델이 모르던 캐릭터를 그리게 했다.

### 전체 흐름

```plain text
마스코트 사진 21장 + metadata.jsonl 캡션
   |  캡션의 (lora-misato-token) -> moa_character 로 치환
   |  VAE 로 latent 미리 계산 · CLIP 으로 캡션 임베딩 미리 계산
   v
UNet (fp32, 얼림) + LoRA 어댑터 (r=8, to_q/to_k/to_v/to_out.0)
   |  for 300 steps:
   |     t 무작위 -> DDPM add_noise -> UNet 이 노이즈 예측 -> MSE
   v
pytorch_lora_weights.safetensors  6.4MB
   |  추론 파이프라인에 load_lora_weights(adapter_name="moa")
   |  set_adapters(["moa"], adapter_weights=[scale])
   v
"portrait of moa_character as an astronaut" -> 마스코트 우주인
```

### 숫자 한눈에

| 항목 | 값 |
| --- | --- |
| 학습 사진 | **21장** |
| 트리거 단어 | `moa_character` |
| 랭크 `r` / `lora_alpha` | **8 / 8** |
| 대상 층 | `to_q` · `to_k` · `to_v` · `to_out.0` (크로스 어텐션) |
| 배치 / 학습률 | **2 / `1e-4`** |
| 학습 스텝 | **300** (`QUICK`) · 1200 (전체) |
| 개념 설명 예제 | `768×768` = 589,824 → LoRA 12,288 (**2.1%**) |
| 저장 파일 | `pytorch_lora_weights.safetensors` **6.4MB** |
| 학습 정밀도 | **fp32** (추론은 fp16) |
| 학습 스케줄러 | `DDPMScheduler` (추론은 `DPMSolverMultistep`) |
| 비교 실험 | LoRA 유/무 (시드 3) · `scale` 0·0.5·1.0 (시드 8) · 네거티브 유/무 |
| **학습 손실·시간** | **기록 없음** (이번 실행에서 학습이 생략됐다) |

### 외워 둘 코드

```python
# LoRA 의 전부
y = x @ W + x @ A @ B          # W 얼림, A 난수, B 는 0 에서 출발

# 어텐션 층에만 어댑터를 붙인다
unet.requires_grad_(False)
unet.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
                            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
lora_params = [p for p in unet.parameters() if p.requires_grad]

# 학습 한 스텝 = 잡음 섞고, 맞히고, 차이를 줄인다
t = torch.randint(0, noise_sched.config.num_train_timesteps, (B,), device=DEVICE)
x_noisy = noise_sched.add_noise(x0, noise, t)
pred = unet(x_noisy, t, encoder_hidden_states=text_embeds[idx]).sample
loss = F.mse_loss(pred.float(), noise.float())

# 저장은 LoRA 만
StableDiffusionPipeline.save_lora_weights(
    path, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))

# 꽂고 세기 조절
pipe.load_lora_weights(path, adapter_name="moa")
pipe.enable_lora(); pipe.set_adapters(["moa"], adapter_weights=[1.0])
pipe.disable_lora()
```

### 자주 틀리는 것

- `A`·`B` 둘 다 난수로 초기화한다 → **망가진 모델에서 출발한다.** `B` 는 0
- `A` 를 0으로 둔다 → 기울기가 0이라 **영원히 안 움직인다**
- 학습을 `fp16` 으로 한다 → **기울기가 언더플로로 죽는다.** `variant='fp16'` + `dtype=float32`
- 학습에 `DPMSolverMultistep` 을 쓴다 → 학습은 `DDPMScheduler`(`add_noise`)다
- 트리거로 흔한 단어를 쓴다 → 모델이 알던 개념이 **오염된다**
- `scaling_factor` 를 곱하지 않고 latent 를 만든다 → 분포가 어긋나 학습이 안 된다
- 사진을 한 각도로만 모은다 → **그 각도만 외운다**
- 전체 UNet 을 학습한다 → 2GB 를 저장해야 하고 과적합도 쉽다
- LoRA 를 켠 채로 다른 그림을 그린다 → `disable_lora()` 를 잊으면 **모든 그림에 캐릭터가 섞인다**
- 학습 셀을 건너뛰고 "학습했다"고 적는다 → **파일이 있다고 이번에 학습한 것은 아니다**
