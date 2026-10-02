# ViT 미세조정 — 트랜스포머로 이미지를 분류한다

**실습 파일**: `17-KDT/17-vision_transformer/1.ipynb`

> **2026-10-02 — 노트북이 저장됐고, 비교 대상이 둘 늘었다**
> `model_init` 이 제대로 정의되고 **학습 출력이 노트북에 남았다**(코드 셀 12개 중 9개).
> 1-8의 숫자는 이제 체크포인트가 아니라 **노트북 출력에서 확인한 값**이다. 최종 macro-F1 **0.9269**.
> 전체 학습 시간은 **1,326.9초(22분 7초)**, 평균 학습 손실 0.5069 였다.
> 같은 파이프라인으로 **Swin(2.ipynb)·CvT(3.ipynb)** 를 돌린 결과는 **2장**에 모았다.
>
> **다만 코드와 출력이 어긋나 있다.** 셀의 `num_train_epochs` 는 **3**인데 저장된 출력은 **5에폭**(3,125스텝)짜리다.
> 돌린 뒤 값을 3으로 줄여 놓았다 — 다시 돌리면 3에폭에서 끝나고 **F1 은 0.9218** 이 된다.

| 절 | 내용 |
| --- | --- |
| 1-1 | 왜 ViT 인가 |
| 1-2 | 데이터 — FashionMNIST 부분 집합 |
| 1-3 | 전처리 — 흑백 1채널을 3채널 224×224 로 |
| 1-4 | 패치 임베딩 — 196 + 1 = 197 |
| 1-5 | 분류 헤드 교체 |
| 1-6 | `Trainer` 설정 |
| 1-7 | 실행 상태와 고칠 것 |
| 1-8 | **실측 결과 (노트북 출력)** |

---

## 1-1. 왜 ViT 인가

**ViT(Vision Transformer)는 이미지를 "패치의 나열"로 보고 NLP 트랜스포머를 그대로 쓴다.**

| | CNN (영상처리·딥러닝 3~6장) | **ViT** |
| --- | --- | --- |
| 기본 연산 | 합성곱 (국소 수용영역) | **셀프 어텐션 (전역)** |
| 위치 정보 | 구조에 내재 | **위치 임베딩으로 넣는다** |
| 귀납적 편향 | 강함 (지역성·평행이동 불변) | **약함** |
| 작은 데이터 | 잘 된다 | **잘 안 된다** → 사전학습 필수 |
| 큰 데이터 | 포화 | **계속 좋아진다** |

**귀납적 편향이 약한 것이 ViT 의 장단점을 전부 설명한다.**
CNN 은 "근처 픽셀이 관련 있다"를 구조로 알고 시작한다. ViT 는 그것조차 데이터로 배워야 한다.
그래서 작은 데이터셋에서 처음부터 학습하면 CNN 에 진다. **대신 사전학습된 가중치를 가져오면 강하다.**

이 노트북이 `google/vit-base-patch16-224-in21k` 를 쓰는 이유가 그것이다.

| 이름 조각 | 뜻 |
| --- | --- |
| `vit-base` | 12층 × 12헤드 × 768차원 (약 8,600만) |
| `patch16` | 패치 크기 16×16 |
| `224` | 입력 이미지 224×224 |
| `in21k` | **ImageNet-21k**(2만 1천 클래스, 1,400만 장)로 사전학습 |

**코드 생성 LM 과 정확히 같은 구조다.** 임베딩 → (어텐션 + FFN) × 12 → 헤드.
바뀐 것은 입력을 토큰으로 만드는 방법뿐이다.

---

## 1-2. 데이터 — FashionMNIST 부분 집합

```python
train_dataset = datasets.FashionMNIST(root='./datasets', download=True, train=True)
test_dataset  = datasets.FashionMNIST(root='./datasets', download=True, train=False)
```

```python
def subset_sampler(dataset, classes, max_len):
    target_idx = defaultdict(list)
    for idx, label in enumerate(dataset.train_labels):
        target_idx[int(label)].append(idx)
    indices = list(chain.from_iterable([target_idx[idx][:max_len] for idx in range(len(classes))]))
    return Subset(dataset, indices)
```

**클래스별로 앞에서 `max_len` 개씩 뽑는다.** 10종이므로 `max_len × 10` 장이 된다.

| | 원본 | 뽑은 수 | 실측 |
| --- | --- | --- | --- |
| 학습 | 60,000 | 1,000 / 클래스 | **10,000** |
| 테스트 | 10,000 | 100 / 클래스 | **1,000** |

```
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

**왜 줄이나.** 224×224×3 으로 키운 이미지를 8,600만 파라미터 모델에 넣는다.
6만 장을 5에폭 돌리면 GPU 시간이 크게 든다. **구조와 절차를 확인하는 것이 목적이므로 1/6으로 줄였다.**
딥러닝 5장의 `RUN_LEVEL = 'test'` 와 같은 발상이다.

**클래스별로 균등하게 뽑은 것이 중요하다.** 앞에서 10,000장을 그냥 자르면 클래스가 치우친다.
`defaultdict`로 라벨별 인덱스를 모아 각각 앞 1,000개를 쓰므로 **완전 균형 데이터**가 된다.
그래서 macro-F1 과 정확도가 비슷하게 나올 조건이다.

> **참고** — `dataset.train_labels` 는 deprecated 다.
> ```
> UserWarning: train_labels has been renamed targets
> ```
> `dataset.targets` 로 쓰면 경고가 사라진다. 동작은 같다.

---

## 1-3. 전처리 — 흑백 1채널을 3채널 224×224 로

```python
image_processor = AutoImageProcessor.from_pretrained(
    pretrained_model_name_or_path="google/vit-base-patch16-224-in21k")

print(image_processor.size)         # {'height': 224, 'width': 224}
print(image_processor.image_mean)   # [0.5, 0.5, 0.5]
print(image_processor.image_std)    # [0.5, 0.5, 0.5]
```

**정규화 값을 직접 적지 않고 모델에서 읽어 온다.** 사전학습 때 쓴 값과 달라지면 성능이 떨어진다.
`in21k` 모델은 ImageNet 평균(0.485·0.456·0.406)이 아니라 **0.5 를 쓴다** — 외워 쓰면 틀린다.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(image_processor.size['height'], image_processor.size['width'])),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),      # ← 1채널 → 3채널
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])
```

```
FashionMNIST 원본  (1, 28, 28)   흑백
   │ ToTensor        0~1 범위 텐서
   │ Resize          (1, 224, 224)
   │ cat([x,x,x], 0) (3, 224, 224)   ← 같은 채널을 3번 복제
   │ Normalize       평균 0.5, 표준편차 0.5 → -1 ~ 1
   ↓
ViT 입력            (3, 224, 224)
```

**`torch.cat([x, x, x], 0)` 이 이 절의 핵심이다.**
ViT 는 컬러 이미지로 사전학습됐으니 **입력 채널이 3개여야 한다.** 흑백을 그대로 넣으면 모양 오류다.
회색 이미지를 R=G=B 로 복제하면 색이 없는 컬러 이미지가 된다.

**순서가 중요하다.** `Normalize` 가 **채널 복제 뒤**에 와야 한다.
`mean`·`std` 가 3개씩이므로 1채널 상태에서 부르면 모양이 안 맞는다.

```python
def collator(data, transform):
    images, labels = zip(*data)
    pixel_values = torch.stack([transform(image) for image in images])
    labels = torch.tensor([label for label in labels])
    return {'pixel_values': pixel_values, 'labels': labels}
```

**실측**

```
pixel_values torch.Size([32, 3, 224, 224])
labels       torch.Size([32])
```

**딕셔너리로 돌려주는 것이 HuggingFace 방식이다.** `Trainer`가 `model(**batch)` 로 풀어 넣기 때문에
키 이름이 모델의 `forward` 인자와 같아야 한다 — `pixel_values`, `labels`.

**`transform` 을 `Dataset` 에 걸지 않고 `collate_fn` 에서 적용한 것도 의도적이다.**
`datasets.FashionMNIST(...)` 를 `transform` 없이 만들었으므로 PIL 이미지가 나온다.
배치 단위로 변환하면 같은 원본 데이터셋을 다른 변환으로 재사용할 수 있다.

---

## 1-4. 패치 임베딩 — 196 + 1 = 197

```python
print(model.vit.embeddings)
```

```
ViTEmbeddings(
  (patch_embeddings): ViTPatchEmbeddings(
    (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  )
  (dropout): Dropout(p=0.0, inplace=False)
)
```

**패치 나누기가 `Conv2d` 한 줄이다.** `kernel_size = stride = 16` 이면 **겹치지 않게 16×16 씩 잘라**
각 조각을 768차원으로 선형 변환한다. 자르기와 임베딩이 한 연산에 들어 있다.

```python
print(batch['pixel_values'].shape)                                      # (32, 3, 224, 224)
print(model.vit.embeddings.patch_embeddings(batch['pixel_values']).shape) # (32, 196, 768)
print(model.vit.embeddings(batch['pixel_values']).shape)                 # (32, 197, 768)
```

**실측값 세 줄이 ViT 의 전부를 설명한다.**

```
224 / 16 = 14          가로 14개, 세로 14개
14 × 14  = 196         패치 196개  ← "토큰 196개"
196 + 1  = 197         [CLS] 토큰 1개를 앞에 붙인다
```

| 단계 | 모양 | 의미 |
| --- | --- | --- |
| 입력 | `(32, 3, 224, 224)` | 이미지 |
| `patch_embeddings` | `(32, 196, 768)` | **패치 196개 × 768차원** |
| `embeddings` | `(32, 197, 768)` | **+ [CLS] + 위치 임베딩** |

**`(32, 197, 768)` 은 NLP 의 `(배치, 토큰 수, 임베딩 차원)` 과 완전히 같은 모양이다.**
여기서부터 뒤는 코드 생성 LM 3·4장의 트랜스포머 블록과 똑같이 돌아간다.

**[CLS] 토큰**은 딥러닝 8장 BERT 에서 본 그것이다. 학습되는 벡터 1개를 맨 앞에 붙여 두고,
12층을 통과한 **그 자리의 출력만** 분류에 쓴다. 196개 패치 정보가 어텐션으로 거기 모인다.

**위치 임베딩은 학습식이다.** 197개 자리에 각각 768차원 벡터가 있다 (3장 `nn.Embedding` 방식).
그래서 **입력 크기를 224 에서 바꾸면 패치 수가 달라져 위치 임베딩을 보간해야 한다.**
`Resize(224, 224)` 가 선택이 아니라 필수인 이유다.

---

## 1-5. 분류 헤드 교체

```python
model = ViTForImageClassification.from_pretrained(
    pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
    num_labels=len(classes),
    id2label={idx: label for label, idx in class_to_idx.items()},
    label2id=class_to_idx,
    ignore_mismatched_sizes=True)

print(model.classifier)     # Linear(in_features=768, out_features=10, bias=True)
```

```
Some weights of ViTForImageClassification were not initialized from the model checkpoint
at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task ...
```

**이 경고는 정상이다.** 사전학습 모델의 분류 헤드는 2만 1천 클래스용이다.
FashionMNIST 는 10종이므로 **헤드만 버리고 새로 만든다.** 몸통(12층)은 그대로 가져온다.

| 인자 | 역할 |
| --- | --- |
| `num_labels=10` | 새 헤드의 출력 수 |
| `ignore_mismatched_sizes=True` | **모양이 안 맞는 가중치를 건너뛴다.** 없으면 에러 |
| `id2label` / `label2id` | 숫자 ↔ 이름 매핑을 모델에 저장 → 추론 결과를 바로 읽을 수 있다 |

**딥러닝 4장의 전이학습과 같은 구조다.** `resnet18`의 `fc` 를 바꿔 끼운 것과 같다.
다만 여기서는 **몸통을 얼리지 않는다** — 전체를 작은 학습률로 미세조정한다 (full fine-tuning).

---

## 1-6. `Trainer` 설정

```python
def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    macro_f1 = metric.compute(predictions=predictions, references=labels, average='macro')
    return macro_f1
```

**`average='macro'`** — 클래스별 F1 을 구해 단순 평균한다. 데이터가 균형이라(1-2) 정확도와 비슷하게 나올 것이다.
딥러닝 5장·P장에서 쓴 그 지표다.

```python
args = TrainingArguments(
    output_dir='./model/Vit-FashionMNIST',
    save_strategy='epoch',
    evaluation_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='logs',
    logging_steps=125,
    remove_unused_columns=False,
    seed=42)
```

| 설정 | 왜 |
| --- | --- |
| `learning_rate=1e-5` | **미세조정용.** 처음부터 학습하면 `1e-4`~`1e-3`, 사전학습 가중치를 지키려면 이 정도 |
| `save_strategy` + `evaluation_strategy='epoch'` | 에폭마다 저장·평가 → 둘이 같아야 `load_best_model_at_end` 가 동작한다 |
| `load_best_model_at_end=True` | **마지막이 아니라 가장 좋았던 가중치**를 돌려준다 |
| `metric_for_best_model='f1'` | `compute_metrics` 가 돌려주는 키와 같아야 한다 ✓ |
| `remove_unused_columns=False` | **반드시 필요하다.** `Trainer`가 모델 인자에 없는 열을 지우는데, 여기선 `collator`가 직접 만든다 |
| `weight_decay=0.001` | 1만 장에 8,600만 파라미터 → 과적합 억제 |
| `seed=42` | 재현성 |

**`load_best_model_at_end` 가 딥러닝 3장의 `restore_best_weights=True` 와 같은 역할이다.**
17-image `3_1.ipynb`에서 이 옵션이 없어 마지막 가중치로 평가한 것을 지적했었다. 여기서는 켜져 있다.

**`logging_steps=125`** — 1만 장 / 배치 16 = 625 스텝/에폭. 125 스텝마다 찍으면 **에폭당 5회**다.
딱 떨어지게 고른 값이다.

---

## 1-7. 실행 상태와 고칠 것

### 멈춘 지점

```python
trainer = Trainer(model_init=lambda x: model_init(classes, class_to_idx), ...)
```

```
NameError: name 'model_init' is not defined
```

**`model_init` 이라는 함수를 정의하지 않고 람다 안에서 자기 이름을 부르고 있다.**
노트북에 그 함수가 없다. 두 가지 방법 중 하나로 고친다.

```python
# 방법 1 — 이미 만든 모델을 그대로 넘긴다 (가장 간단)
trainer = Trainer(model=model, args=args, ...)

# 방법 2 — model_init 을 제대로 정의한다 (하이퍼파라미터 탐색용)
def model_init(trial=None):
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(classes),
        id2label={idx: label for label, idx in class_to_idx.items()},
        label2id=class_to_idx,
        ignore_mismatched_sizes=True)

trainer = Trainer(model_init=model_init, args=args, ...)
```

**`model_init` 은 `Trainer`가 매 시도마다 모델을 새로 만들어야 할 때 쓰는 인자다.**
`hyperparameter_search` 를 쓸 계획이 아니면 `model=model` 이면 된다.
인자는 `trial` 하나(또는 없음)를 받는다 — `lambda x:` 형태로 `classes` 를 넘길 수 없다.

### 고칠 것

| 위치 | 내용 | 수정안 |
| --- | --- | --- |
| `Trainer(...)` | **저장된 노트북은 `model_init` 미정의로 `NameError`** 다. 커널에서는 고쳐 돌렸으나 노트북에 반영되지 않았다 | 고친 코드로 한 번 더 저장한다 |
| `subset_sampler` | `dataset.train_labels` 는 deprecated | `dataset.targets` |
| `TrainingArguments` | `evaluation_strategy` 는 최신 transformers 에서 `eval_strategy` 로 바뀌었다 | 버전 확인 후 교체 |
| `Trainer(tokenizer=...)` | 이미지 모델에 `tokenizer` 인자는 deprecated | `processing_class=image_processor` |
| `compute_metrics` | `evaluate.load('f1')` 를 **매 평가마다** 새로 불러온다 | 바깥에서 한 번 만들어 재사용 |
| 셀 10 | 혼동 행렬이 아직 출력되지 않았다 → **클래스별 성능을 모른다** | 학습된 모델로 실행 (셔츠·티셔츠·풀오버·코트 혼동 확인) |
| 데이터 증강 | `transform` 에 증강이 없다. 5에폭에서 학습 0.2596 vs 검증 0.3321 로 간격이 벌어지기 시작했다 | `RandomHorizontalFlip` 등 (신발·가방은 좌우 반전이 자연스럽다) |
| `datasets/` | FashionMNIST 원본 **82MB**가 폴더에 있다 | `.gitignore` 에 추가했다 (2026-10-01) |
| `model/` | 체크포인트 5개 = **4.8GB**. `optimizer.pt` 가 모델의 2배다 | `.gitignore` 에 추가했다. `save_total_limit=2` 권장 |

### 환경

| 항목 | 값 |
| --- | --- |
| 장치 | `cuda` |
| 가상환경 | `.venv-cuda` (CUDA용), `.venv` 둘 다 있다 |
| `requirements.txt` | `torch==2.0.1`, `transformers==4.33.2`, `timm==0.9.7` 등 고정 버전 |
| 경고 | `IProgress not found` — `ipywidgets` 를 올리면 진행 표시줄이 제대로 나온다 |

---

## 1-8. 실측 결과 — 노트북 출력

**2026-10-02 에 노트북이 저장되어 아래 값은 셀 출력에서 그대로 읽은 것이다.**
(2026-10-01 에는 출력이 없어 `checkpoint-3125/trainer_state.json` 에서 읽었고, **두 값이 완전히 일치했다.**)

### 에폭별 지표

| 에폭 | 스텝 | 학습률 | 학습 손실 | 검증 손실 | **검증 macro-F1** |
| --- | --- | --- | --- | --- | --- |
| 1 | 625 | `8e-6` | 0.6815 | 0.6240 | 0.8950 |
| 2 | 1,250 | `6e-6` | 0.4492 | 0.4349 | 0.9191 |
| 3 | 1,875 | `4e-6` | 0.3402 | 0.3678 | 0.9218 |
| 4 | 2,500 | `2e-6` | 0.2638 | 0.3435 | 0.9246 |
| **5** | **3,125** | `0.0` | **0.2596** | **0.3321** | **0.9269** |

```
best_metric            0.926895693711025
best_model_checkpoint  ./model/Vit-FashionMNIST/checkpoint-3125
epoch 5.0   global_step 3125
```

**읽어 낼 것 네 가지.**

**① 1에폭 만에 0.8950 이 나온다.**
무작위는 0.1 이다. 클래스별 1,000장만 보고 한 바퀴 돌려 **89.5%** 를 찍은 것은
몸통 12층이 ImageNet-21k 로 이미 배워 둔 덕이다. **전이학습의 효과가 그대로 보인다.**

**② 학습 손실과 검증 손실이 같이 내려갔다.** 과적합이 아직 시작되지 않았다.

```
학습 손실  0.6815 → 0.4492 → 0.3402 → 0.2638 → 0.2596
검증 손실  0.6240 → 0.4349 → 0.3678 → 0.3435 → 0.3321
```

5에폭 지점에서 **검증 손실이 아직 떨어지는 중**이고 `best_model_checkpoint` 가 **마지막 에폭**이다.
`load_best_model_at_end=True` 가 마지막을 고른 것은 "더 돌려도 좋아질 여지가 있다"는 뜻이다.
다만 학습 손실(0.2596)과 검증 손실(0.3321)의 간격이 벌어지기 시작했으니 **에폭을 늘리면 증강이 필요하다.**

**③ F1 상승폭이 빠르게 줄었다.**

| 구간 | 상승폭 |
| --- | --- |
| 1 → 2 | **+0.0241** |
| 2 → 3 | +0.0027 |
| 3 → 4 | +0.0028 |
| 4 → 5 | +0.0023 |

**2에폭에서 거의 끝났다.** 그 뒤 3에폭은 0.008 을 더 얻었다.
미세조정은 "조금만 돌려도 대부분 얻는다"는 특징이 수치로 나온 것이다.

**④ 학습률이 선형으로 0까지 내려갔다.**

```
1e-5 → 8e-6 → 6e-6 → 4e-6 → 2e-6 → 0.0
```

`TrainingArguments` 에 스케줄러를 지정하지 않았는데도 감쇠했다.
**HuggingFace `Trainer` 의 기본값이 `lr_scheduler_type="linear"`**(워밍업 없는 선형 감쇠)다.
코드 생성 LM 4장에서 손으로 구현한 `get_lr` 과 같은 모양이다 — 거기는 워밍업이 붙어 있었다.

### 산출물

| 항목 | 값 |
| --- | --- |
| 체크포인트 | 5개 (`checkpoint-625` ~ `checkpoint-3125`) |
| `pytorch_model.bin` | **343,294,147 바이트** |
| `optimizer.pt` | 686,618,507 바이트 (**모델의 약 2배**) |
| `model/` 폴더 전체 | **4.8 GB** |
| 검증 1회 소요 | 1,000장에 **약 14.8초** |
| 전체 학습 시간 | **1,326.9초** (22분 7초, 37.68 샘플/초) |
| 평균 학습 손실 | 0.5069 |

**파라미터 수 역산**: 343,294,147 / 4바이트 ≈ **8,580만**. ViT-base 의 알려진 규모와 맞는다.

**`optimizer.pt` 가 모델의 2배인 이유**는 AdamW 가 파라미터마다 **1차·2차 모멘트 두 개**를 들고 있기 때문이다.
그래서 체크포인트 하나가 1GB 다. 5개면 4.8GB — **`.gitignore` 에 넣었다 (2026-10-01)**.
이어서 학습할 계획이 없으면 `save_total_limit=2` 로 개수를 제한하는 편이 좋다.

### 비교 — 같은 과제를 CNN 으로 하면

| | 이 ViT 미세조정 | 참고 |
| --- | --- | --- |
| 데이터 | 10,000장 (전체의 1/6) | FashionMNIST 전체는 60,000장 |
| 파라미터 | 8,580만 | `resnet18` 은 1,100만 |
| 입력 | 224×224×3 (28×28 흑백을 키움) | CNN 은 28×28 그대로도 된다 |
| 5에폭 macro-F1 | **0.9269** | FashionMNIST 는 간단한 CNN 으로도 0.90 대가 나온다 |

**"ViT 가 CNN 보다 좋다"를 보여 주는 실험이 아니다.**
28×28 흑백 이미지를 224×224 컬러로 8배 키워 8,580만 파라미터 모델에 넣는 것은
이 과제에 과한 구성이다. **목적은 ViT 의 구조와 미세조정 절차를 익히는 것**이다.
ViT 의 진짜 강점은 데이터가 충분히 크고 해상도가 높은 과제에서 나온다.

> **주의 — 이 숫자를 인용할 때**
> 노트북 출력과 `checkpoint-3125/trainer_state.json` 이 일치하는 값이다.
> 다만 **셀의 `num_train_epochs` 는 3으로 바뀌어 있다.** 그대로 다시 돌리면 0.9218 에서 끝난다.
> 혼동 행렬 셀은 출력이 생겼지만 그림뿐이라 **클래스별 수치는 읽을 수 없다.**
> FashionMNIST 는 셔츠(Shirt)·티셔츠(T-shirt/top)·풀오버(Pullover)·코트(Coat)가 서로 헷갈리기로 유명하다.
> 그 셀을 돌려 보면 0.9269 안에서 **어디가 약한지** 드러날 것이다.

---

## 이 장 정리

### 한 줄 요약

**ViT 는 이미지를 패치 196개로 쪼개 [CLS] 를 붙인 뒤, NLP 트랜스포머에 그대로 넣는 것이다.**
`(32, 3, 224, 224)` → `(32, 197, 768)` 이 되는 그 한 번의 변환이 전부다.

### 파이프라인

```
FashionMNIST (1, 28, 28) 흑백
   │  클래스별 1,000장 → 10,000장 (테스트 100장씩 → 1,000장)
   │  ToTensor → Resize 224 → cat([x,x,x]) → Normalize(0.5, 0.5)
   ↓
(3, 224, 224)
   │  Conv2d(3, 768, kernel=16, stride=16)      14 × 14 = 196
   ↓
(196, 768)
   │  + [CLS] 토큰 + 위치 임베딩
   ↓
(197, 768)
   │  ViT-base: 12층 × 12헤드 × 768  (약 8,600만, in21k 사전학습)
   ↓
[CLS] 자리의 출력 (768)
   │  Linear(768, 10)   ← 새로 초기화
   ↓
10종 로짓  → 검증 macro-F1 **0.9269** (5에폭)
```

### 외워 둘 숫자

| 항목 | 값 | 출처 |
| --- | --- | --- |
| 패치 수 | 224 / 16 = 14 → **14² = 196** | 노트북 출력 |
| 토큰 수 | 196 + 1([CLS]) = **197** | 노트북 출력 |
| 임베딩 차원 | **768** | 노트북 출력 |
| 입력 크기 | 224 × 224 | `image_processor.size` |
| 정규화 | 평균 0.5, 표준편차 0.5 (`in21k`) | `image_processor` |
| 학습 데이터 | **10,000** (1,000 × 10종) | 노트북 출력 |
| 테스트 데이터 | **1,000** (100 × 10종) | 노트북 출력 |
| 미세조정 학습률 | `1e-5` | 설정 |
| 에폭당 스텝 | 10,000 / 16 = 625 | 계산 |
| **최종 macro-F1** | **0.9269** | 체크포인트 실측 |
| 1에폭 macro-F1 | 0.8950 | 체크포인트 실측 |
| 모델 파일 | 343,294,147 B ≈ 8,580만 파라미터 | 파일 실측 |

### 외워 둘 코드

```python
# 정규화 값은 모델에서 읽는다
image_processor = AutoImageProcessor.from_pretrained(모델명)
transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# 흑백 → 3채널 (Normalize 앞에 와야 한다)
transforms.Lambda(lambda x: torch.cat([x, x, x], 0))

# 패치 나누기 = 겹치지 않는 Conv2d
Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))

# 헤드만 교체
ViTForImageClassification.from_pretrained(모델명, num_labels=10,
                                          ignore_mismatched_sizes=True)

# collator 는 모델 인자 이름과 같은 키로
return {'pixel_values': ..., 'labels': ...}
# 그리고 TrainingArguments(remove_unused_columns=False)
```

### 자주 틀리는 것

- 흑백 1채널을 그대로 넣는다 → 모양 오류. 3채널로 복제해야 한다
- `Normalize` 를 채널 복제 **앞**에 둔다 → `mean` 3개와 채널 1개가 안 맞는다
- ImageNet 평균(0.485·0.456·0.406)을 외워 쓴다 → `in21k` 는 **0.5** 다
- `ignore_mismatched_sizes` 를 빼먹는다 → 헤드 모양 불일치로 로드 실패
- `remove_unused_columns=True`(기본) 로 둔다 → `Trainer`가 열을 지워 `collator`가 받을 게 없어진다
- `metric_for_best_model` 이름을 `compute_metrics` 반환 키와 다르게 쓴다 → best 선택이 안 된다
- `save_strategy` 와 `evaluation_strategy` 가 다르다 → `load_best_model_at_end` 가 에러
- 입력 크기를 224 에서 바꾼다 → 패치 수가 달라져 위치 임베딩 보간이 필요하다
- 학습 없이 `trainer.predict` 를 돌린다 → 헤드가 난수다. 결과가 무의미하다
- 체크포인트를 그대로 쌓아 둔다 → `optimizer.pt` 때문에 하나가 1GB 다. `save_total_limit` 을 준다
- 노트북을 저장하지 않고 커널만 돌린다 → **출력이 남지 않아 결과를 되짚을 수 없다**
