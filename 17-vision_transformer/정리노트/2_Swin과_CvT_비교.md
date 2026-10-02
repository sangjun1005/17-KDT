# 2. Swin 과 CvT — 같은 조건으로 비교한다

**실습 파일**: `17-vision_transformer/2.ipynb`(Swin), `3.ipynb`(CvT)

1장에서 ViT 를 미세조정했다. 2장은 **같은 데이터·같은 설정으로 다른 비전 트랜스포머 둘을 돌려 비교한다.**

> **2026-10-02 — 둘 다 끝까지 돌았다**
> Swin 3에폭 **macro-F1 0.9190** (328.1초), CvT 3에폭 **macro-F1 0.9221** (516.9초).
> 1장의 ViT 는 3에폭 지점에 0.9218, 5에폭에 0.9269 였다 (1,326.9초).
> **데이터·전처리·학습률·배치·시드가 전부 같다.** 모델 클래스와 체크포인트 이름만 바뀌었다.

| 절 | 내용 |
| --- | --- |
| 2-1 | 세 노트북의 차이는 세 줄뿐이다 |
| 2-2 | Swin — 창을 밀어 가며 본다 |
| 2-3 | CvT — 합성곱을 다시 들여온다 |
| 2-4 | 실측 비교 |
| 2-5 | 무엇을 골라야 하나 |
| 2-6 | 고칠 것 |

---

## 2-1. 세 노트북의 차이는 세 줄뿐이다

```python
# 1.ipynb
from transformers import ViTForImageClassification
ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", ...)
output_dir = './model/Vit-FashionMNIST'

# 2.ipynb
from transformers import SwinForImageClassification
SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", ...)
output_dir = './model/Swin-FashionMNIST'

# 3.ipynb
from transformers import CvtForImageClassification
CvtForImageClassification.from_pretrained("microsoft/cvt-21", ...)
output_dir = './model/Cvt-FashionMNIST'
```

**나머지는 전부 같다.**

| 항목 | 값 |
| --- | --- |
| 데이터 | FashionMNIST 클래스별 1,000장 = 10,000장 (테스트 1,000장) |
| 전처리 | `ToTensor` → `Resize` → `cat([x,x,x])` → `Normalize` |
| 학습률 | `1e-5` |
| 배치 | 16 |
| `weight_decay` | 0.001 |
| 지표 | macro-F1, `load_best_model_at_end=True` |
| 시드 | 42 |
| 에폭 | **2·3장은 3, 1장의 저장된 출력은 5** |

**이것이 좋은 비교 설계다.** 한 번에 하나만 바꿨으므로 **차이를 모델 탓으로 돌릴 수 있다.**
딥러닝 P장의 "반복 홀드아웃 5회"처럼 엄격하지는 않지만(시드 1개, 1회 실행), **같은 조건**이라는 조건은 지켰다.

### 입력 크기를 읽는 방법이 하나 다르다

```python
# 1·2장 — ViT·Swin
transforms.Resize(size=(image_processor.size['height'], image_processor.size['width']))

# 3장 — CvT
transforms.Resize(size=(image_processor.size['shortest_edge'],
                        image_processor.size['shortest_edge']))
```

**`image_processor.size` 의 키가 모델마다 다르다.**
ViT·Swin 은 `{'height': 224, 'width': 224}`, CvT 는 `{'shortest_edge': 224}` 를 쓴다.
`size['height']` 로 고정해 두면 CvT 에서 `KeyError` 가 난다 — **설정은 모델에서 읽으라**는 1장의 교훈이 여기서 한 번 더 걸린다.

### 1장의 버그가 고쳐졌다

```python
def model_init(classes, class_to_idx):            # ← 이제 정의돼 있다
    return SwinForImageClassification.from_pretrained(...)

for idx, label in enumerate(dataset.targets):     # ← train_labels 가 아니라 targets
```

1장 정리본에서 지적한 **`model_init` 미정의**와 **`dataset.train_labels` deprecated** 가 둘 다 고쳐졌다.
(단 `Trainer(model_init=lambda x: model_init(classes, class_to_idx))` 형태는 그대로다 — 람다로 인자를 넘기는 우회다.)

---

## 2-2. Swin — 창을 밀어 가며 본다

**`microsoft/swin-tiny-patch4-window7-224`**

| 이름 조각 | 뜻 |
| --- | --- |
| `swin` | **S**hifted **Win**dow Transformer |
| `tiny` | 가장 작은 변형 (약 2,800만) |
| `patch4` | 패치 크기 **4×4** (ViT 는 16×16) |
| `window7` | 어텐션 창 **7×7 패치** |
| `224` | 입력 224×224 |

### ViT 와 무엇이 다른가

**ViT 는 196개 패치가 전부 서로를 본다.** 전역 어텐션이라 비용이 `O(N²)` 이다.
**Swin 은 7×7 창 안에서만 본다.** 창 안의 49개 패치끼리만 어텐션하므로 비용이 **입력 크기에 선형**이다.

```
ViT (전역)                    Swin (창 안에서만)
┌─────────────────┐          ┌───┬───┬───┬───┐
│ 모든 패치가     │          │창1│창2│창3│창4│  ← 창 안에서만 어텐션
│ 서로를 본다     │          ├───┼───┼───┼───┤
│  O(N²)          │          │창5│창6│창7│창8│     O(N)
└─────────────────┘          └───┴───┴───┴───┘
```

**그러면 창 밖은 어떻게 보나 — 창을 민다(shift).**

```
층 1:  [AAAA][BBBB][CCCC]        창 경계가 고정
층 2:    [AAB][BCC][CDD]         창을 절반 밀어서 경계를 넘게 한다
```

**짝수 층은 창을 절반 밀어** 이전 층의 경계를 가로지르게 만든다. 층을 쌓으면 정보가 전체로 퍼진다.
이 "밀기"가 Swin 의 이름이고 핵심이다.

### 계층 구조 — CNN 처럼 줄여 간다

| | ViT | Swin |
| --- | --- | --- |
| 패치 수 | **처음부터 끝까지 196** | 단계마다 **1/4 로 줄인다** |
| 해상도 | 고정 | 56×56 → 28×28 → 14×14 → 7×7 |
| 채널 | 768 고정 | 96 → 192 → 384 → 768 |
| 비유 | 한 덩어리 | **CNN 의 피라미드** |

`patch4` 로 잘게 시작해(224/4 = 56 → 56² = 3,136 패치) 단계마다 2×2 를 하나로 합친다(patch merging).
**해상도를 줄이면서 채널을 늘리는 것**은 ResNet·VGG 가 하던 방식 그대로다.

**그래서 Swin 은 분할·검출에 쓰기 쉽다.** 여러 해상도의 특징 맵이 자연히 나오므로
딥러닝 6장의 UNet 이나 9장의 FPN 같은 구조에 바로 끼울 수 있다. ViT 는 해상도가 하나라 그게 어렵다.

---

## 2-3. CvT — 합성곱을 다시 들여온다

**`microsoft/cvt-21`**

| 이름 조각 | 뜻 |
| --- | --- |
| `cvt` | **C**onvolutional **v**ision **T**ransformer |
| `21` | 층 수 21개 (약 3,200만) |

**CvT 의 발상은 "트랜스포머에 CNN 의 좋은 점을 되돌려 넣자"다.** 두 군데에 합성곱을 넣는다.

**① 패치를 겹쳐 자른다 (Convolutional Token Embedding)**

```
ViT  : Conv2d(3, 768, kernel=16, stride=16)   겹치지 않는다
CvT  : Conv2d(..., kernel=7, stride=4, padding=2)   ← 겹친다
```

**ViT 의 패치 나누기는 `kernel = stride` 라 조각이 서로 안 겹친다**(1장 1-4).
그래서 패치 경계에서 정보가 끊긴다. CvT 는 `stride < kernel` 로 **겹쳐 자른다** — 경계가 부드러워진다.
Swin 처럼 **단계마다 해상도를 줄이고 채널을 늘린다**(3단계).

**② Q·K·V 를 Linear 가 아니라 Conv 로 만든다 (Convolutional Projection)**

```python
# ViT·Swin
Q = nn.Linear(E, H*D)(x)

# CvT (개념)
Q = depthwise_conv(x)      # 주변 픽셀을 함께 본다
```

**깊이별 분리 합성곱(depthwise separable conv)으로 Q·K·V 를 만든다.**
한 토큰의 Q 를 만들 때 **그 토큰만 보지 않고 주변도 본다** — 지역성이 어텐션 안으로 들어간다.
K·V 에는 stride 를 줘서 개수를 줄이기도 한다(비용 절감).

**③ 위치 임베딩이 없다**

**합성곱 자체가 위치 정보를 담으므로 위치 임베딩을 아예 안 쓴다.**
1장에서 "위치 임베딩이 학습식이라 입력 크기를 바꾸면 보간해야 한다"고 적었는데, CvT 는 그 제약이 없다.

### 세 모델의 위치 정보 처리

| 모델 | 위치 정보 | 입력 크기를 바꾸면 |
| --- | --- | --- |
| ViT | 학습되는 `nn.Embedding` (197자리) | **보간 필요** |
| Swin | 창 안 상대 위치 편향 | 창 크기만 맞으면 된다 |
| **CvT** | **합성곱에 내재** | **그냥 된다** |
| 코드 생성 LM 4장 | RoPE (회전) | 이론상 외삽 가능 |

**"위치를 어떻게 넣느냐"가 트랜스포머 변형의 큰 축이다.** 4장의 RoPE 도 같은 문제에 대한 다른 답이다.

---

## 2-4. 실측 비교

**모두 FashionMNIST 10,000장 학습 / 1,000장 검증, lr `1e-5`, 배치 16, 시드 42.**

### 에폭별 macro-F1

| 에폭 | ViT-base | Swin-tiny | CvT-21 |
| --- | --- | --- | --- |
| 1 | 0.8950 | 0.8893 | 0.8928 |
| 2 | 0.9191 | 0.9103 | 0.9203 |
| **3** | **0.9218** | **0.9190** | **0.9221** |
| 4 | 0.9246 | — | — |
| 5 | **0.9269** | — | — |

### 에폭별 검증 손실

| 에폭 | ViT-base | Swin-tiny | CvT-21 |
| --- | --- | --- | --- |
| 1 | 0.6240 | 0.3030 | 0.3097 |
| 2 | 0.4349 | 0.2537 | 0.2638 |
| 3 | 0.3678 | **0.2379** | **0.2465** |
| 5 | 0.3321 | — | — |

**손실은 Swin·CvT 가 ViT 보다 훨씬 낮은데 F1 은 비슷하다.**
ViT 쪽 손실이 큰 이유는 1에폭 시작점이 높아(0.6240) 아직 내려오는 중이기 때문이다 —
5에폭 0.3321 도 Swin 의 3에폭 0.2379 보다 높다. **같은 F1 에서 확신도(confidence)가 다르다**는 뜻이다.

### 시간과 크기

| 항목 | ViT-base | Swin-tiny | CvT-21 |
| --- | --- | --- | --- |
| 에폭 | 5 | 3 | 3 |
| 전체 학습 시간 | **1,326.9초** | **328.1초** | **516.9초** |
| 에폭당 | 265.4초 | 109.4초 | 172.3초 |
| 학습 샘플/초 | 37.7 | **91.4** | 58.0 |
| 검증 1회 | 14.8초 | **6.3초** | 8.0초 |
| 검증 샘플/초 | 67.7 | **157.9** | 124.6 |
| 평균 학습 손실 | 0.5069 | **0.3743** | 0.7829 |
| 파라미터 (알려진 값) | 약 8,600만 | 약 2,800만 | 약 3,200만 |

**에폭당 시간으로 보면 Swin 이 ViT 의 2.4배 빠르고 CvT 는 1.5배 빠르다.**
파라미터가 1/3 인데 속도는 2.4배다 — 전역 어텐션의 `O(N²)` 이 빠지면서 얻은 차이다.

### 3에폭 기준으로 정렬하면

| 순위 | 모델 | macro-F1 | 시간 | F1 / 시간 |
| --- | --- | --- | --- | --- |
| 1 | **CvT-21** | **0.9221** | 516.9초 | 중간 |
| 2 | ViT-base | 0.9218 | 약 796초 (265.4 × 3) | 가장 느리다 |
| 3 | Swin-tiny | 0.9190 | **328.1초** | **가장 빠르다** |

**세 모델의 F1 차이는 0.0031 이다.** 1,000장 검증에서 이 차이는 **3장 정도**다.
**이 실험으로 "어느 모델이 더 좋다"를 결론 내리면 안 된다.**

> **주의 — 비교의 한계**
> **시드 1개, 1회 실행이다.** 1장에서 본 것처럼 같은 설정도 돌릴 때마다 흔들린다
> (코드 생성 LM 3장은 재실행에서 손실이 0.6441 → 0.6080 으로 바뀌었다).
> 0.003 차이를 말하려면 **시드를 여러 개 두고 평균 ± 표준편차**를 봐야 한다.
>
> **에폭이 다르다.** ViT 만 5에폭이다. 3에폭 지점으로 맞춰 비교하는 것이 공정하다.
>
> **과제가 너무 쉽다.** 28×28 흑백 옷 사진 10종이다. 세 모델 모두 2에폭에 0.89~0.92 에 도달했으니
> **모델의 차이가 드러날 여지가 없다.** 해상도가 높고 클래스가 많은 과제여야 갈린다.
>
> **크기가 공정하지 않다.** ViT 는 `base`(8,600만), Swin 은 `tiny`(2,800만)다.
> `swin-base` 와 비교해야 구조의 차이를 본다. 지금은 **"크기 + 구조"가 섞인 비교**다.

---

## 2-5. 무엇을 골라야 하나

세 모델의 성격을 정리하면 고르는 기준이 나온다.

| | ViT | Swin | CvT |
| --- | --- | --- | --- |
| 어텐션 범위 | **전역** | 창 안 (밀기로 확장) | 전역 + 합성곱 지역성 |
| 비용 | `O(N²)` | **`O(N)`** | `O(N²)`이지만 K·V 를 줄인다 |
| 해상도 | 고정 1단 | **다단계 피라미드** | 다단계 |
| 위치 정보 | 학습 임베딩 | 상대 위치 편향 | **합성곱에 내재** |
| 귀납적 편향 | **가장 약하다** | 중간 (지역성·계층) | **가장 강하다** |
| 작은 데이터 | 불리 | 중간 | **유리** |
| 분할·검출 | 어렵다 | **쉽다** | 쉽다 |
| 고해상도 입력 | 비용 폭발 | **감당된다** | 감당된다 |

**선택 기준**

- **분류만 하고 사전학습 가중치가 크다** → ViT. 가장 단순하고 생태계가 가장 넓다.
- **분할·검출을 하거나 입력이 크다** → Swin. 다단계 특징 맵이 그대로 쓰인다.
- **데이터가 적다** → CvT. 합성곱의 귀납적 편향이 적은 데이터를 메워 준다.
- **속도가 중요하다** → Swin-tiny. 이 실험에서 2.4배 빨랐다.

**세 모델 모두 "트랜스포머에 CNN 의 성질을 얼마나 되돌려 넣을까"의 서로 다른 답이다.**
ViT 는 0%, Swin 은 계층 구조만, CvT 는 합성곱 연산까지. 그리고 **1장에서 본 대로,
귀납적 편향이 적을수록 데이터가 많이 필요하고 많을 때 더 멀리 간다.**

---

## 2-6. 고칠 것

| 위치 | 내용 | 수정안 |
| --- | --- | --- |
| 1·2·3.ipynb 공통 | `Trainer(model_init=lambda x: model_init(...))` — `model_init` 은 `trial` 하나를 받는다 | `functools.partial` 이나 클로저로 감싼다 |
| 1·2·3.ipynb 공통 | `evaluation_strategy` 는 최신 transformers 에서 `eval_strategy` | 버전 확인 후 교체 |
| 1·2·3.ipynb 공통 | `Trainer(tokenizer=...)` 는 deprecated | `processing_class=image_processor` |
| 1·2·3.ipynb 공통 | `compute_metrics` 가 `evaluate.load('f1')` 를 매 평가마다 다시 불러온다 | 바깥에서 한 번 |
| 1·2·3.ipynb 공통 | 데이터 증강이 없다 | `RandomHorizontalFlip` 등 |
| 2.ipynb | **혼동 행렬 셀에 출력이 없다** (`ec=None`) | 돌려서 저장 |
| 1.ipynb | 코드의 `num_train_epochs=3` 과 저장된 5에폭 출력이 어긋난다 | 값을 맞춰 다시 저장 |
| 비교 전체 | 시드 1개, 1회 실행 | 시드 3~5개로 평균 ± 표준편차 |
| 비교 전체 | 에폭이 다르다 (ViT 5 vs 나머지 3) | 같은 에폭으로 맞춘다 |
| 비교 전체 | 모델 크기가 다르다 (base vs tiny) | `swin-base`·`vit-small` 등으로 급을 맞춘다 |
| `model/` | 체크포인트가 **6.8GB** 로 늘었다 (3모델 × 3~5에폭) | `save_total_limit=1`, `.gitignore` 에 이미 넣었다 |

---

## 이 장 정리

### 한 줄 요약

**ViT·Swin·CvT 는 "트랜스포머에 CNN 의 지역성과 계층 구조를 얼마나 되돌려 넣을까"의 세 가지 답이다.**
FashionMNIST 에서는 셋의 F1 차이가 0.003 이고, **속도는 Swin 이 2.4배 빨랐다.**

### 숫자 한눈에

| | ViT-base | Swin-tiny | CvT-21 |
| --- | --- | --- | --- |
| 체크포인트 | `google/vit-base-patch16-224-in21k` | `microsoft/swin-tiny-patch4-window7-224` | `microsoft/cvt-21` |
| 3에폭 macro-F1 | 0.9218 | **0.9190** | **0.9221** |
| 3에폭 검증 손실 | 0.3678 | **0.2379** | 0.2465 |
| 전체 학습 시간 | 1,326.9초 (5에폭) | **328.1초** (3에폭) | 516.9초 (3에폭) |
| 학습 샘플/초 | 37.7 | **91.4** | 58.0 |
| 패치 크기 | 16×16 (겹치지 않음) | 4×4 | 7×7 (**겹침**) |
| 어텐션 | 전역 | **7×7 창 + 밀기** | 전역 + **깊이별 합성곱 Q·K·V** |
| 위치 정보 | 학습 임베딩 | 상대 위치 편향 | **없음 (합성곱에 내재)** |

### 외워 둘 코드

```python
# 모델만 갈아 끼우면 나머지는 그대로 돈다
from transformers import SwinForImageClassification   # 또는 CvtForImageClassification
Model.from_pretrained(체크포인트, num_labels=10, ignore_mismatched_sizes=True)

# size 의 키가 모델마다 다르다
image_processor.size['height']          # ViT · Swin
image_processor.size['shortest_edge']   # CvT

# 1장에서 지적한 것이 고쳐졌다
for idx, label in enumerate(dataset.targets):      # train_labels 아님
def model_init(classes, class_to_idx): ...         # 정의돼 있다
```

### 자주 틀리는 것

- `size['height']` 로 고정한다 → CvT 는 `shortest_edge` 다. `KeyError`
- 에폭이 다른 결과를 나란히 놓고 비교한다 → 같은 지점으로 맞춘다
- `base` 와 `tiny` 를 비교해 "구조가 더 좋다"고 말한다 → 크기 차이가 섞인다
- 1회 실행의 0.003 차이로 순위를 매긴다 → 재실행하면 뒤집힐 수 있다
- 손실이 낮으니 더 좋은 모델이라고 본다 → **F1 과 손실은 다른 것을 잰다**
- 검증 손실이 아직 내려가는데 멈춘다 → ViT 는 5에폭에서도 내려가는 중이었다
- 체크포인트를 모델마다 전부 남긴다 → 3모델 × 5에폭이면 수 GB 다
