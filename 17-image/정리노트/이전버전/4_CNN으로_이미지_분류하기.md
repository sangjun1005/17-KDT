# 4. CNN으로 이미지 분류하기

**실습 파일**: `3_1.ipynb`

같은 데이터(CIFAR-10)를 **MLP와 CNN으로 각각 학습시켜 성능을 비교**한다.
"이미지에는 왜 CNN을 쓰는가"에 대한 답이 숫자로 나온다.

---

## 4-1. CIFAR-10 데이터 준비

### 개념

CIFAR-10은 **32×32 컬러 이미지 6만 장**을 10개 클래스로 나눈 데이터셋이다.
MNIST 다음 단계로 자주 쓰는 표준 벤치마크다.

| | 장수 | shape |
|---|---|---|
| 학습 | 50,000 | `(50000, 32, 32, 3)` |
| 테스트 | 10,000 | `(10000, 32, 32, 3)` |

### 핵심 코드

```python
from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape, train_labels.shape)   # (50000, 32, 32, 3) (50000, 1)
print(test_images.shape,  test_labels.shape)    # (10000, 32, 32, 3) (10000, 1)
```

레이블 shape가 `(50000, 1)`이다. `(50000,)`이 아니라 **2차원**이라는 점을 기억해 둔다.
나중에 개별 레이블을 꺼낼 때 `test_labels[i][0]`처럼 인덱스가 하나 더 필요하다.

### 정규화

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

0~255 정수를 0~1 실수로 바꾼다. **정규화 없이 학습하면 손실이 잘 안 떨어진다.**
입력 값이 크면 가중치 업데이트가 널뛰기 때문이다.

정규화 전후를 직접 찍어 보면 차이가 보인다.

```
전: [ 59,  62,  63]
후: [0.23137255, 0.24313725, 0.24705882]
```

### 검증 데이터 분리

```python
val_images = train_images[45000:]
val_labels = train_labels[45000:]

train_images = train_images[:45000]
train_labels = train_labels[:45000]
```

학습 5만 장 중 뒤 5천 장을 **검증용으로 떼어낸다.**

| 데이터 | 용도 | 쓰이는 곳 |
|---|---|---|
| 학습(45,000) | 가중치 업데이트 | `fit`의 x, y |
| 검증(5,000) | 학습 중 성능 확인, 조기 종료 판단 | `validation_data` |
| 테스트(10,000) | **마지막에 딱 한 번** 최종 평가 | `evaluate` |

테스트 데이터를 보고 모델을 고치면 그건 이미 테스트가 아니다.
**검증 데이터가 따로 필요한 이유**가 여기에 있다.

주의: **잘라내는 순서를 바꾸면 안 된다.**
`train_images`를 먼저 45000개로 자르고 나서 `val_images`를 만들면 검증 데이터가 사라진다.
위 코드처럼 **뒤쪽을 먼저 떼고 나서** 앞쪽을 자른다.

---

## 4-2. MLP로 먼저 해 보기 (비교 기준)

### 개념

3장에서 쓴 전결합층만으로 이미지를 분류한다.
이미지를 **한 줄로 펴서(Flatten)** 넣는 방식이다.

```
32×32×3 = 3072개의 숫자 → Dense 층들 → 10개 클래스
```

### 핵심 코드

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

mlp_model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

mlp_model.fit(train_images, train_labels,
              epochs=5,
              validation_data=(val_images, val_labels))
```

### `sparse_categorical_crossentropy`를 쓰는 이유

| 레이블 형태 | 예시 | 손실 함수 |
|---|---|---|
| 정수 | `3` | **`sparse_categorical_crossentropy`** |
| 원-핫 | `[0,0,0,1,0,0,0,0,0,0]` | `categorical_crossentropy` |

CIFAR-10의 레이블은 `3` 같은 정수다. 그래서 `sparse_`가 붙은 쪽을 쓴다.
원-핫으로 바꾸는 수고를 덜어 준다.
**이 둘을 바꿔 쓰면 shape 에러가 나거나 학습이 전혀 안 된다.**

### 파라미터 수

```
flatten (Flatten)      (None, 3072)         0
dense   (Dense)        (None, 512)      1,573,376
dense_1 (Dense)        (None, 256)        131,328
dense_2 (Dense)        (None, 128)         32,896
dense_3 (Dense)        (None, 10)           1,290
=================================================
Total params: 1,738,890  (6.63 MB)
```

첫 층 하나가 157만 개다. `3072 × 512 + 512 = 1,573,376`.
**입력을 다 펴서 전부 연결하니 파라미터가 폭발한다.**

### 결과

```
Epoch 5/5  loss: 1.5185  accuracy: 0.4542  val_accuracy: 0.4432

evaluate → [1.5693, 0.4393]
```

**테스트 정확도 43.9%.** 10개 클래스에서 찍으면 10%이니 배우기는 했다.
하지만 절반도 못 맞힌다.

### 왜 이것밖에 안 되나

`Flatten`이 **위치 정보를 버리기 때문**이다.

이미지에서 바로 옆 픽셀은 서로 관계가 깊다. 눈과 눈 사이, 바퀴의 둥근 테두리 같은 것들이다.
그런데 3072개를 한 줄로 펴는 순간 "이 픽셀과 저 픽셀이 붙어 있었다"는 정보가 사라진다.
모델은 그 관계를 **처음부터 다시 배워야** 한다.

---

## 4-3. CNN

### 개념

합성곱(Convolution)은 **작은 필터를 이미지 위에서 미끄러뜨리며** 특징을 뽑는다.
2장에서 손으로 만든 Sobel 커널과 계산이 똑같다.
**차이는 커널 값을 사람이 정하지 않고 모델이 학습한다는 것**이다.

```
2장:  사람이 [[-1,0,1],[-2,0,2],[-1,0,1]] 를 직접 씀
4장:  이 9개 숫자를 모델이 스스로 찾음
```

### 핵심 코드

```python
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout

cnn_model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu',
           input_shape=(32, 32, 3)),
    MaxPool2D((2, 2)),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPool2D((2, 2)),

    Conv2D(64, (3, 3), padding='same', activation='relu'),

    Flatten(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### Conv2D 파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| 첫 번째 인자 | `32` | **필터 개수.** 서로 다른 특징 32가지를 뽑는다 |
| `(3, 3)` | 커널 크기 | 한 번에 보는 영역 |
| `padding='same'` | 테두리 채움 | **출력 크기를 입력과 같게** 유지 |
| `activation='relu'` | 활성화 | 음수 응답을 0으로 |

`padding='valid'`(기본값)로 두면 3×3 커널이 지날 때마다 가장자리가 깎여
32×32가 30×30이 된다. `'same'`은 0으로 테두리를 채워 크기를 지킨다.

### MaxPool2D

```python
MaxPool2D((2, 2))
```

2×2 영역에서 **가장 큰 값 하나만** 남긴다. 가로세로가 절반이 된다.

- 계산량이 4분의 1로 준다
- 특징이 몇 픽셀 움직여도 결과가 같다 (**위치 변화에 강해진다**)

### 층별 크기 변화

```
conv2d          (None, 32, 32, 32)      896
max_pooling2d   (None, 16, 16, 32)        0
conv2d_1        (None, 16, 16, 64)   18,496
max_pooling2d_1 (None,  8,  8, 64)        0
conv2d_2        (None,  8,  8, 64)   36,928
flatten_1       (None, 4096)             0
dropout         (None, 4096)             0
dense_4         (None, 64)         262,208
dropout_1       (None, 64)               0
dense_5         (None, 10)             650
================================================
Total params: 319,178  (1.22 MB)
```

**가로세로는 줄고, 채널은 늘어난다.** 32×32×3 → 8×8×64.
"넓고 얕게"에서 "좁고 깊게"로 바뀐다. CNN의 전형적인 모양이다.

파라미터를 보면 첫 Conv2D가 **896개**뿐이다.
`3 × 3 × 3 × 32 + 32 = 896`. 필터 하나가 이미지 전체를 훑기 때문에
**위치마다 가중치를 따로 두지 않는다.** 이게 파라미터가 적은 이유다.

전체 파라미터는 319,178개로 **MLP(1,738,890개)의 약 18%**다.

### Dropout

```python
Dropout(0.3)   # 30%를 끈다
Dropout(0.5)   # 50%를 끈다
```

학습할 때마다 뉴런 일부를 **무작위로 0으로 만든다.**
특정 뉴런에만 의존하지 못하게 해서 과적합을 막는다.

**예측할 때는 자동으로 꺼진다.** 따로 신경 쓸 필요 없다.
`Flatten` 직후 4096개 지점이 과적합에 가장 취약해서 여기에 넣는다.

### 콜백 — 조기 종료와 체크포인트

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
save_best_only = ModelCheckpoint('best_cifar10_cnn_model.h5',
                                 save_best_only=True)

history = cnn_model.fit(train_images, train_labels,
                        batch_size=512,
                        epochs=100,
                        validation_data=(val_images, val_labels),
                        callbacks=[early_stopping, save_best_only])
```

| 콜백 | 하는 일 |
|---|---|
| `EarlyStopping(patience=5)` | 검증 손실이 5에폭 연속 안 좋아지면 **학습 중단** |
| `ModelCheckpoint(save_best_only=True)` | 검증 손실이 최고 기록일 때만 **파일로 저장** |

`epochs=100`으로 크게 잡아도 조기 종료가 알아서 멈춘다.
실제로 **45에폭에서 멈췄다.**

### 학습 곡선

```python
plt.plot(history.history['loss'], 'b--')
plt.plot(history.history['val_loss'], 'r--')
plt.show()
```

파란 선(학습 손실)은 계속 내려가는데 빨간 선(검증 손실)이 어느 순간부터 안 내려가면
**그 지점이 과적합 시작점**이다.

---

## 4-4. 결과 비교

```python
cnn_model.evaluate(test_images, test_labels)
# [0.7250, 0.7490]
```

| | MLP | CNN |
|---|---|---|
| 파라미터 | 1,738,890 | **319,178** |
| 에폭 | 5 | 45 (조기 종료) |
| 테스트 정확도 | 43.9% | **74.9%** |

**파라미터는 5분의 1인데 정확도는 31%p 높다.**

이게 이미지에 CNN을 쓰는 이유다.
"파라미터를 많이 넣어서" 잘하는 게 아니라, **이미지의 구조에 맞는 연산을 쓰기 때문에** 잘한다.

### 주의: 마지막 가중치와 최고 가중치는 다르다

`EarlyStopping`에 **`restore_best_weights=True`가 빠져 있다.**

그래서 `evaluate`가 쓴 건 **45에폭(마지막) 가중치**이지,
`best_cifar10_cnn_model.h5`에 저장된 **최고 성능 가중치가 아니다.**

저장된 최고 모델로 평가하려면 다시 불러와야 한다.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                               restore_best_weights=True)
```

또는

```python
from tensorflow.keras.models import load_model
best = load_model('best_cifar10_cnn_model.h5')
best.evaluate(test_images, test_labels)
```

`.h5`로 저장하면 최신 케라스에서 경고가 뜬다. `'best_model.keras'`가 권장 형식이다.

---

## 4-5. 예측 결과 확인

### 확률을 클래스 번호로

```python
predicted_labels = cnn_model.predict(test_images)
predicted_labels.shape        # (10000, 10)

import tensorflow as tf
predicted_labels = tf.argmax(predicted_labels, axis=1)
# <tf.Tensor: shape=(10000,), numpy=array([3, 8, 8, ..., 5, 4, 7])>
```

`predict`는 **각 클래스의 확률 10개**를 돌려준다. `(10000, 10)`이다.
`argmax(axis=1)`이 그중 가장 큰 값의 **위치(=클래스 번호)**를 뽑는다.

`axis=1`이 핵심이다. `axis=0`으로 하면 샘플 방향으로 최댓값을 찾아 전혀 다른 결과가 나온다.

### 번호를 이름으로

```python
label_to_name = {
    0: 'airplane', 1: 'automobile', 2: 'bird',  3: 'cat',  4: 'deer',
    5: 'dog',      6: 'frog',       7: 'horse', 8: 'ship', 9: 'truck'
}
```

### 맞은 것과 틀린 것 색으로 구분

```python
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i])

    xlabel = f"{label_to_name[int(test_labels[i][0])]} ({label_to_name[int(predicted_labels[i])]})"
    plt.xlabel(xlabel,
               color='red' if test_labels[i][0] != predicted_labels[i] else 'black')
plt.show()
```

정답과 예측을 함께 적고, **틀린 것만 빨간 글씨**로 표시한다.
정확도 숫자 하나보다 **어떤 클래스를 어떤 클래스로 헷갈리는지** 보는 게 훨씬 유용하다.
(고양이와 개, 자동차와 트럭이 대표적으로 잘 헷갈린다.)

참고: 원본 코드에는 `cmap=plt.cm.binary`가 붙어 있는데,
**컬러 이미지에는 `cmap`이 무시된다.** 흑백 1채널일 때만 적용된다.

---

## 이 장 정리

| 층 | 하는 일 | 기억할 것 |
|---|---|---|
| `Conv2D(n, (3,3))` | 필터 n개로 특징 추출 | `padding='same'`이면 크기 유지 |
| `MaxPool2D((2,2))` | 가로세로 절반 | 계산량 감소 + 위치 변화에 강해짐 |
| `Flatten()` | 1차원으로 펴기 | 여기서 파라미터가 폭발 |
| `Dropout(0.5)` | 일부 뉴런 끄기 | 학습 때만 작동 |
| `Dense(10, 'softmax')` | 10개 클래스 확률 | 합이 1 |

### 전체 흐름

```
이미지 → [Conv → Pool] 반복 → Flatten → Dense → softmax
         특징 추출부                    분류부
```

### 자주 틀리는 것

- 레이블이 정수면 **`sparse_categorical_crossentropy`**, 원-핫이면 `categorical_crossentropy`
- 검증 데이터를 자를 때 **뒤쪽을 먼저 떼고** 앞쪽을 자른다
- `EarlyStopping`에 **`restore_best_weights=True`**를 넣지 않으면 마지막 가중치가 남는다
- `argmax(axis=1)` — 축을 틀리면 결과가 완전히 달라진다
- 레이블 shape가 `(N, 1)`이라 개별 값은 `labels[i][0]`으로 꺼낸다
