# 5. Stable Diffusion — 글로 그림을 만든다

**실습 파일**: `17-KDT/17-RAG/TotlaAI.ipynb` (셀 33~36)

4장까지는 **글을 읽고 글을 쓰는** 모델이었다. 5장은 **글을 읽고 그림을 그린다.**
그리고 3장의 LLM 을 **프롬프트 번역기**로 다시 쓴다.

> **2026-10-06 — 끝까지 돌았다**
> `stable-diffusion-v1-5` 를 `fp16` 으로 올리고, `DPMSolverMultistep` 으로 스케줄러를 교체,
> 20스텝 512×512 이미지 생성. **LLM 이 한국어 요청을 영어 프롬프트로 바꿔** 3장을 뽑고,
> **스텝(2·5·10·25)과 guidance(1·4·7.5·15)를 각각 4장씩** 비교했다.
>
> **다만 생성 시간·이미지 품질 평가 수치는 출력에 없다.** 그림만 남아 있다.

| 절 | 내용 |
| --- | --- |
| 5-1 | 확산 모델은 무엇을 하나 |
| 5-2 | 파이프라인 올리기 — 네 가지 선택 |
| 5-3 | 스케줄러 교체 — 50스텝을 20스텝으로 |
| 5-4 | `draw` 함수 — 시드로 고정한다 |
| 5-5 | LLM 을 프롬프트 번역기로 |
| 5-6 | 스텝과 guidance — 노브 두 개 |
| 5-7 | 고칠 것 |

---

## 5-1. 확산 모델은 무엇을 하나

**언어모델은 "다음 토큰"을 맞혔다. 확산 모델은 "노이즈를 뺀 그림"을 맞힌다.**

```
학습:   깨끗한 그림 --노이즈를 조금 넣는다--> 흐린 그림
        모델에게 "넣은 노이즈가 무엇이었나"를 맞히게 한다

생성:   완전한 노이즈에서 시작
        --노이즈를 조금 뺀다--> --조금 뺀다--> ... (N스텝)
        --> 그림
```

| | 코드 생성 LM (3·4장) | **확산 모델 (5장)** |
| --- | --- | --- |
| 학습 목표 | 다음 토큰 맞히기 | **넣은 노이즈 맞히기** |
| 생성 | 토큰을 하나씩 **앞으로** | 노이즈를 조금씩 **빼면서** |
| 반복 횟수 | 생성 토큰 수 | **스텝 수** (5-6) |
| 조건 | 프롬프트 토큰 | **프롬프트 임베딩** (CLIP) |

**"조건부 생성"이라는 점은 같다.** 둘 다 텍스트를 받아 뭔가를 만든다.
다만 확산 모델에는 **텍스트 인코더가 따로 붙어 있다** — 보통 CLIP 이다.
**1장 1-6에서 "멀티모달 RAG 는 CLIP 같은 모델로 텍스트와 이미지를 같은 공간에 넣는다"**고 적은
그 모델이 여기서 쓰인다.

```python
SD_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
SD_INPAINT_ID = 'stable-diffusion-v1-5/stable-diffusion-inpainting'

from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                       StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler)
```

| 파이프라인 | 입력 | 하는 일 |
| --- | --- | --- |
| **`StableDiffusionPipeline`** | 글 | **글 → 그림** (이 노트북이 쓰는 것) |
| `StableDiffusionImg2ImgPipeline` | 글 + 그림 | 그림을 글 방향으로 **변형** |
| `StableDiffusionInpaintPipeline` | 글 + 그림 + 마스크 | 마스크 부분만 **다시 그린다** |

**셋을 다 import 했지만 실제로 쓴 것은 첫 번째뿐이다.**
`assets/assets/puppymask.png` 가 폴더에 있는 것을 보면 **inpaint 실습이 예정돼 있었다** —
아직 셀이 없다.

---

## 5-2. 파이프라인 올리기 — 네 가지 선택

```python
pipe = StableDiffusionPipeline.from_pretrained(
    SD_ID,
    torch_dtype=IMG_DTYPE,
    variant='fp16',
    safety_checker=None,
    requires_safety_checker=False
).to(DEVICE)
```

| 인자 | 값 | 왜 |
| --- | --- | --- |
| `torch_dtype=IMG_DTYPE` | `float16` (cuda/mps) | **메모리 절반.** 1장 셀 1에서 분기해 둔 그 값 |
| `variant='fp16'` | — | **fp16 가중치 파일을 받는다.** 다운로드도 절반 |
| `safety_checker=None` | 끈다 | NSFW 필터 모델(약 1.2GB)을 **안 올린다** |
| `requires_safety_checker=False` | — | 위를 끄면 경고가 나오는데 그걸 막는다 |

**`torch_dtype` 과 `variant` 는 다르다.**

```
variant='fp16'    어떤 파일을 받을까      (디스크·다운로드)
torch_dtype       메모리에 어떻게 올릴까  (VRAM·연산)
```

**둘 다 줘야 완전히 절약된다.** `variant` 만 주고 `torch_dtype` 을 빼면
fp16 파일을 받아 **fp32 로 변환해 올린다** — 다운로드만 이득이다.

**`IMG_DTYPE` 이 1장의 그 변수다.**

```python
IMG_DTYPE = torch.float16 if DEVICE in ('cuda', 'mps') else torch.float32
```

**CPU 에서는 `float32` 로 돌아간다** — 1장 1-2에서 적은 대로
CPU 는 반정밀도를 하드웨어로 지원하지 않아 `float16` 이 더 느리거나 에러가 난다.
**이 한 줄 덕에 같은 노트북이 세 장치에서 돈다.**

> **`safety_checker=None` 은 실습 환경의 선택이다.**
> 모델 1.2GB 와 추론 시간을 아끼고, 오탐으로 검은 이미지가 나오는 것을 막는다.
> **공개 서비스에서는 켜야 한다.** 코드 생성 LM 6장이 웹서비스에 입력 검증을 붙인 것과 같은 맥락 —
> 실습과 배포의 기준이 다르다.

```
FutureWarning: `torch_dtype` is deprecated and will be removed in version 1.0.0.
Please use `dtype` instead.
```

**`diffusers` 가 인자 이름을 `dtype` 으로 바꾸는 중이다.** 아직 동작하지만 고쳐 두는 편이 좋다.
비전 트랜스포머의 `evaluation_strategy` → `eval_strategy`, `tokenizer=` → `processing_class=`
와 같은 종류의 변화다.

---

## 5-3. 스케줄러 교체 — 50스텝을 20스텝으로

```python
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

NEGATIVE = 'low quality, blurry, deformed, ugly, text, watermark'
```

**스케줄러가 "노이즈를 얼마나 어떻게 빼나"를 정한다.** 모델은 그대로 두고 **풀이 방식만 바꾼다.**

| 스케줄러 | 필요한 스텝 | 특징 |
| --- | --- | --- |
| `PNDM` (SD1.5 기본) | **50** | 안정적, 느리다 |
| **`DPMSolverMultistep`** | **20** | 품질을 거의 유지하며 **2.5배 빠르다** |
| `DDIM` | 50 | 결정적 |
| `Euler a` | 20~30 | 스텝마다 결과가 크게 바뀐다 |

**`from_config(pipe.scheduler.config)` 로 기존 설정을 물려받는다.**
`beta_start`·`beta_end`·`num_train_timesteps` 같은 **학습 때 쓴 노이즈 일정**은 그대로 두고
**풀이법만 교체**한다. 새로 만들면 학습과 안 맞아 결과가 망가진다.

**이것이 미분방정식 수치해법 교체와 같다.** 같은 방정식을 오일러법으로 50번 푸는 것과
고차 해법으로 20번 푸는 것의 차이다 — **DPM-Solver 는 다단계(multistep) 고차 해법이다.**

**`set_progress_bar_config(disable=True)`** — 그림을 10여 장 뽑으므로
진행 표시줄이 출력을 뒤덮는 것을 막는다. 셀 20의 `pct // 10` 과 같은 발상이다.

### 부정 프롬프트

```python
NEGATIVE = 'low quality, blurry, deformed, ugly, text, watermark'
```

**"이런 건 그리지 마"를 따로 준다.**

```
guidance 방향 = (프롬프트 쪽) - (부정 프롬프트 쪽)
```

**부정 프롬프트는 "무시"가 아니라 "반대로 밀기"다.**
빈 문자열 대신 `low quality` 를 넣으면 **그 반대 방향으로 더 세게 민다.**

| 단어 | 막는 것 |
| --- | --- |
| `low quality`, `blurry` | 흐릿함 |
| `deformed`, `ugly` | 망가진 형태 (특히 손·얼굴) |
| **`text`, `watermark`** | **글자·워터마크** |

**`text`·`watermark` 가 실무에서 가장 효과가 크다.**
학습 데이터(LAION)에 스톡 이미지가 많아 **워터마크와 글자를 자주 그린다.**
일러스트를 뽑는 데 글자가 끼면 쓸 수 없으니 미리 막는 것이다.

---

## 5-4. `draw` 함수 — 시드로 고정한다

```python
def draw(prompt, negative=NEGATIVE, steps=20, guidance=7.5, seed=0,
         width=512, height=512, pipeline=None, **kw):
    gen = torch.Generator('cpu').manual_seed(seed)
    return (pipeline or pipe)(prompt, negative_prompt=negative,
                              num_inference_steps=steps, guidance_scale=guidance,
                              width=width, height=height, generator=gen, **kw).images[0]

test_img = draw('a cute robot studying with a laptop, flat illustration, pastel colors', seed=1)
```

**`torch.Generator('cpu')` — GPU 가 아니라 CPU 다.**

**이게 의도적이다.** 시작 노이즈를 CPU 에서 만들면
**같은 시드가 CUDA·MPS·CPU 에서 같은 그림을 준다.** GPU 제너레이터를 쓰면
드라이버·장치마다 난수가 달라져 재현이 안 된다.

> **1장에서 지적한 `set_seed` 의 빈틈을 여기서는 피했다.**
> `set_seed` 에 `torch.cuda.manual_seed_all` 이 없어 GPU 난수가 안 고정되는 문제가 있었는데,
> `draw` 는 **제너레이터를 직접 넘겨** 전역 시드에 의존하지 않는다. **더 견고한 방식이다.**

| 기본값 | 값 | 뜻 |
| --- | --- | --- |
| `steps=20` | 20 | 5-3의 스케줄러 교체로 가능해진 값 |
| `guidance=7.5` | 7.5 | SD1.5 의 관례적 기본값 |
| `seed=0` | 0 | **기본이 고정 시드다** — 같은 프롬프트면 같은 그림 |
| `width/height=512` | 512 | SD1.5 의 **학습 해상도** |

**512×512 를 벗어나면 품질이 떨어진다.** SD1.5 는 512로 학습됐다 —
비전 트랜스포머 1장에서 "ViT 는 224를 벗어나면 위치 임베딩을 보간해야 한다"고 한 것과 같은 제약이다.
**학습 해상도를 지키는 것이 기본이다.**

**`pipeline=None` + `(pipeline or pipe)`** — img2img·inpaint 를 끼울 자리를 비워 뒀다.
**`**kw`** 로 `image`·`mask_image`·`strength` 같은 파이프라인별 인자를 통과시킨다.
**셋을 import 해 둔 것과 이어지는 설계다** — 아직 쓰이지 않았다.

**`.images[0]`** — 파이프라인은 항상 리스트를 준다 (`num_images_per_prompt` 로 여러 장 가능).

```
<PIL.Image.Image image mode=RGB size=512x512>
```

**PIL 이미지가 나온다.** 1장 1-6에서 `Image.open(...)` 으로 읽었던 그 타입이다 —
`.save()`·`.resize()` 가 바로 되고, 주피터에서 셀 마지막에 두면 그림으로 보인다.

---

## 5-5. LLM 을 프롬프트 번역기로

**Stable Diffusion 은 영어로 학습됐다. 한국어 프롬프트는 거의 안 통한다.**

```python
def make_prompt(korean_request):
    return chat([
        {'role': 'system',
         'content': "You write prompts for Stable Diffusion, Convert the user's Korean "
                    "request into ONE English prompt: comma-seperated keywords, "
                    "vivid style words, max 35 words. Output only the prompt."},
        {'role': 'user', 'content': korean_request}
    ], temperature=0.4, max_tokens=100).strip().strip('"')

request = '동아리 홍보에 쓸, 노트북으로 공부하는 귀여운 로봇 일러스트, 파스텔 색감.'
en_prompt = make_prompt(request)
show_images([draw(en_prompt, seed=s) for s in (3, 4, 5)])
```

**3장의 `chat` 이 여기서 재사용된다 — 모델 두 개를 이어 붙인 것이다.**

```
한국어 요청
   | gemma3:4b (3장)        "한국어 -> SD용 영어 프롬프트"
   v
영어 프롬프트
   | Stable Diffusion (5장)  "영어 프롬프트 -> 그림"
   v
그림 3장
```

**시스템 프롬프트의 세 가지 제약이 중요하다.**

| 제약 | 왜 |
| --- | --- |
| `comma-separated keywords` | SD 는 **문장보다 키워드 나열**에 더 잘 반응한다 |
| `vivid style words` | `flat illustration`·`pastel` 같은 **화풍 단어**를 넣게 한다 |
| `max 35 words` | CLIP 텍스트 인코더가 **77토큰**에서 자른다 |
| `Output only the prompt` | "물론이죠! 프롬프트는..." 같은 머리말을 막는다 |

**`.strip().strip('"')` 가 방어 코드다.**
LLM 이 프롬프트를 따옴표로 감싸 주는 일이 흔하다 — 그러면 SD 가 `"` 를 토큰으로 먹는다.
3장 3-7의 `fmt='json'` 과 같은 문제에 대한 **가벼운 해법**이다.

**`temperature=0.4`** — 프롬프트마다 조금씩 달라도 되지만 지시를 지켜야 한다. 중간값이다.

**`seed=(3, 4, 5)` 로 3장을 뽑았다.**
**프롬프트는 같고 시작 노이즈만 다르다** — 같은 지시로 어느 정도 폭이 생기는지 보는 것이다.
비전 트랜스포머 2장에서 "시드 1개로는 결론을 못 낸다"고 한 것과 같은 발상을 **생성에 적용한 것**이다.

> **번역이 아니라 "다시 쓰기"다.**
> `'동아리 홍보에 쓸'` 같은 **용도 설명**은 그림에 그릴 수 없다.
> LLM 이 알아서 버리고 그릴 수 있는 요소만 남긴다 — 기계 번역으로는 안 되는 일이다.
>
> **다만 `en_prompt` 를 `print` 하지 않았다.** 어떤 프롬프트가 나왔는지 기록에 없다.
> `make_prompt` 의 품질을 평가할 수 없다. **한 줄 추가가 필요하다** (5-7).
>
> 시스템 프롬프트 자체에 오타도 있다 — `comma-seperated`(→ `separated`),
> `"You write prompts for Stable Diffusion, Convert"`(콤마 → 마침표).

---

## 5-6. 스텝과 guidance — 노브 두 개

```python
base_prompt = 'a friendly robot mascot holding a laptop, simple flat illustration, pastel colors'

show_images([draw(base_prompt, steps=s, seed=1) for s in (2, 5, 10, 25)])
show_images([draw(base_prompt, guidance=g, seed=1) for g in (1, 4, 7.5, 15)])
```

**`seed=1` 로 고정하고 한 번에 하나만 바꿨다.**
**비전 트랜스포머 2장의 통제 비교와 같은 설계다** — 차이를 그 변수 탓으로 돌릴 수 있다.

### ① 스텝 수 — 노이즈를 몇 번에 걸쳐 뺄까

| 스텝 | 결과 | 시간 |
| --- | --- | --- |
| **2** | 형태가 안 잡힌다. 색 덩어리 | 1배 |
| **5** | 윤곽이 보이기 시작 | 2.5배 |
| **10** | 거의 완성. 세부가 거칠다 | 5배 |
| **25** | 세부까지 정리 | 12.5배 |

**스텝과 시간이 정비례한다.** 스텝 1회 = 모델 1회 추론(정확히는 guidance 때문에 2회)이다.

**`DPMSolverMultistep` 에서 20이 기본값인 이유가 여기 있다.**
10~20 사이에서 품질이 거의 포화하고, 25 이상은 **시간만 쓴다.**
`PNDM` 기본값이 50인 것과 비교하면 **스케줄러 교체의 이득**이 그림으로 보인다.

> **비전 트랜스포머 1장의 에폭별 F1(0.8950 → 0.9191 → 0.9218 → 0.9246 → 0.9269)과 같은 모양이다.**
> **초반에 대부분을 얻고 뒤로 갈수록 수익이 줄어든다.** 학습이든 생성이든 같다.

### ② guidance scale — 프롬프트를 얼마나 세게 따를까

```
최종 방향 = 무조건 예측 + guidance × (프롬프트 예측 - 무조건 예측)
```

| guidance | 결과 |
| --- | --- |
| **1** | 프롬프트를 거의 무시. 일반적인 그림 (`guidance=1` 은 CFG 를 끈 것) |
| **4** | 프롬프트를 따르되 자유롭다 |
| **7.5** | **균형** (SD1.5 기본값) |
| **15** | 프롬프트에 과하게 충실. **색이 타고 대비가 세진다** |

**이것이 1장 1-7의 온도와 같은 축이다 — 방향은 반대다.**

| | 낮으면 | 높으면 |
| --- | --- | --- |
| **온도** (LLM) | 안전·반복적 | 창의적·엉뚱 |
| **guidance** (SD) | **자유·프롬프트 무시** | **충실·과포화** |

**온도는 "얼마나 무작위로"이고 guidance 는 "조건을 얼마나 세게"다.**
둘 다 **극단으로 가면 망가진다**는 점이 같다.

**`guidance > 1` 이면 추론이 2배 든다.**
무조건 예측과 프롬프트 예측을 **둘 다** 계산해야 해서다(classifier-free guidance).
`guidance=1` 로 두면 절반으로 줄지만 프롬프트가 안 먹는다.

> **주의 — 이 절에 숫자가 없다.**
> 그림 8장이 출력에 남아 있지만 **생성 시간·품질 점수가 없다.**
> "스텝이 늘면 느리다"는 **이론이고, 이 PC 에서 몇 초인지는 측정되지 않았다.**
> `time.time()` 으로 재면 "20스텝 512×512 가 몇 초"라는 쓸 수 있는 숫자가 남는다.
> 2장 2-8이 `학습시간 15.17초`를 남긴 것과 대비된다.

---

## 5-7. 고칠 것

| 위치 | 내용 | 수정안 |
| --- | --- | --- |
| 셀 35 | **`en_prompt` 를 `print` 하지 않는다** → 어떤 프롬프트였는지 기록에 없다 | `print(en_prompt)` |
| 셀 35 | 시스템 프롬프트 오타 `comma-seperated` | `comma-separated` |
| 셀 35 | 시스템 프롬프트 `"...Stable Diffusion, Convert..."` — 콤마로 문장을 이었다 | 마침표 |
| 셀 36 | **생성 시간을 재지 않는다** | `t0=time.time()` 으로 스텝별 시간을 남긴다 |
| 셀 36 | `show_images` 에 **`titles` 를 안 준다** → 어느 그림이 몇 스텝인지 모른다 | `titles=[f'{s} steps' for s in (2,5,10,25)]` |
| 셀 33 | `torch_dtype` 는 deprecated (`FutureWarning`) | `dtype=IMG_DTYPE` |
| 셀 33 | `StableDiffusionImg2ImgPipeline`·`InpaintPipeline` 을 **import 만 하고 안 쓴다** | 실습 셀을 추가하거나 import 를 줄인다 |
| 셀 33 | `pipe` 를 **지우지 않는다** → VRAM 을 계속 차지한다 | 끝에 `del pipe; free_memory()` |
| 셀 33 | 이미지를 **저장하지 않는다.** `WORK/'images'` 폴더가 비어 있다 | `img.save(WORK/'images'/f'{name}.png')` |
| 셀 34 | 이미지 크기를 바꿀 수 있게 열어 뒀지만 **512 를 벗어나면 품질이 떨어진다** | 주석으로 명시 |
| 셀 33 | `safety_checker=None` | 실습에서는 맞다. **배포 시 켠다**는 주석 |
| 공통 | `assets/assets/puppymask.png` 가 있는데 **inpaint 셀이 없다** | 마스크 실습을 추가 |
| 공통 | 마스코트 트리거 단어 (`05_자주묻는질문.txt`) 를 쓰는 셀이 없다 | LoRA·DreamBooth 실습이 다음 단계로 보인다 |

---

## 이 장 정리

### 한 줄 요약

**확산 모델은 노이즈에서 그림을 깎아 내고, 그 노브는 스텝 수와 guidance 둘이다.**
스케줄러를 바꿔 50스텝을 20스텝으로 줄였고, **3장의 LLM 을 한국어→영어 프롬프트 번역기로 재사용했다.**

### 전체 흐름

```
한국어 요청
   |  gemma3:4b (3장의 chat)        temperature=0.4, max 35 words, 키워드 나열
   v
영어 프롬프트
   |  + NEGATIVE (low quality, blurry, deformed, ugly, text, watermark)
   |  + torch.Generator('cpu').manual_seed(seed)    <- 장치 간 재현 가능
   v
StableDiffusionPipeline  (SD1.5, fp16, DPMSolverMultistep)
   |  num_inference_steps=20     품질/시간
   |  guidance_scale=7.5         충실도/과포화
   |  512x512                    학습 해상도
   v
PIL 이미지
```

### 숫자 한눈에

| 항목 | 값 |
| --- | --- |
| 모델 | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| 정밀도 | **`fp16`** (`variant` + `dtype` 둘 다) |
| 스케줄러 | `PNDM`(기본 50스텝) → **`DPMSolverMultistep`(20스텝)** |
| 기본 스텝 | **20** |
| 기본 guidance | **7.5** |
| 해상도 | **512 × 512** (SD1.5 학습 해상도) |
| 부정 프롬프트 | `low quality, blurry, deformed, ugly, text, watermark` |
| 프롬프트 길이 제한 | **35단어** (CLIP 77토큰) |
| 비교 실험 | 스텝 **2·5·10·25** / guidance **1·4·7.5·15** |
| 시드 | `Generator('cpu')`, 3장 비교는 3·4·5 |
| 생성 시간 | **측정되지 않았다** |

### 외워 둘 코드

```python
# fp16 은 두 곳에 줘야 완전히 절약된다
StableDiffusionPipeline.from_pretrained(ID, dtype=torch.float16, variant='fp16',
                                        safety_checker=None,
                                        requires_safety_checker=False).to(DEVICE)

# 스케줄러만 교체 — 설정은 물려받는다
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 장치가 달라도 같은 그림 — CPU 제너레이터
gen = torch.Generator('cpu').manual_seed(seed)
pipe(prompt, negative_prompt=NEG, num_inference_steps=20,
     guidance_scale=7.5, width=512, height=512, generator=gen).images[0]

# 부정 프롬프트는 "반대로 밀기"다
NEGATIVE = 'low quality, blurry, deformed, ugly, text, watermark'

# LLM 을 프롬프트 번역기로 — 따옴표 방어까지
chat([{'role':'system','content':'...comma-separated keywords, max 35 words. '
                                 'Output only the prompt.'},
      {'role':'user','content': korean}],
     temperature=0.4, max_tokens=100).strip().strip('"')
```

### 자주 틀리는 것

- `variant='fp16'` 만 주고 `dtype` 을 안 준다 → 다운로드만 절약, 메모리는 그대로
- CPU 에서 `float16` 을 쓴다 → 더 느리거나 에러 (1장과 같은 교훈)
- 스케줄러를 `from_config` 없이 새로 만든다 → 학습 때 노이즈 일정과 안 맞아 망가진다
- **`torch.Generator('cuda')` 를 쓴다** → 장치·드라이버마다 그림이 달라 재현이 안 된다
- 512 를 벗어난 해상도를 쓴다 → SD1.5 는 512 학습. 품질이 떨어진다
- 부정 프롬프트를 비워 둔다 → 워터마크·글자가 섞여 나온다
- 프롬프트를 35단어 넘게 쓴다 → CLIP 77토큰에서 **조용히 잘린다**
- LLM 출력을 그대로 넣는다 → 따옴표·머리말이 섞인다. `.strip('"')`
- guidance 를 15 이상 준다 → 색이 타고 대비가 세진다
- guidance 를 1로 두면 빠르다고 쓴다 → **프롬프트가 안 먹는다**
- 스텝을 50 이상 준다 → 10~20에서 포화한다. 시간만 쓴다
- 비교 그림에 제목을 안 붙인다 → 어느 설정인지 알 수 없다
- 모델을 올려 두고 안 지운다 → VRAM 이 안 돌아온다. `del` + `free_memory()`
- 생성 이미지를 저장하지 않는다 → 커널을 끄면 사라진다
