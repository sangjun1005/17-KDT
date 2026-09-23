# 5. 얼굴 검출과 VAE

**실습 파일**: `3_2.ipynb`

앞부분은 **학습이 필요 없는 고전 얼굴 검출**, 뒷부분은 **확률로 이미지를 만드는 VAE**다.
두 주제가 한 파일에 있지만 서로 독립적이다.

---

## 5-1. Haar Cascade 얼굴 검출

### 개념

딥러닝 이전부터 쓰이던 방법이다. 2001년 Viola-Jones 논문에서 나왔다.

원리는 **명암 대비 패턴**이다.
얼굴에는 일정한 규칙이 있다. 눈 부위는 볼보다 어둡고, 콧대는 양옆보다 밝다.
이런 흑백 대비 패턴(Haar 특징) 수천 개를 순서대로 검사한다.

**cascade(폭포)라는 이름의 이유**: 검사를 단계로 나눠 놓고,
초반 단계에서 탈락한 영역은 **나머지 검사를 건너뛴다.**
이미지 대부분은 얼굴이 아니므로 몇 단계 만에 걸러진다. 그래서 빠르다.

**학습이 필요 없다.** 이미 훈련된 xml 파일이 OpenCV에 들어 있다.

### 핵심 코드

```python
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('three_young_man.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30, 30))
print(faces)
```

실행 결과

```
[[449 112 172 172]
 [718 103 175 175]
 [160 150 158 158]]
```

**얼굴 세 개를 찾았다.** 각 줄이 `(x, y, 너비, 높이)`다.
`(449, 112)`에서 시작하는 172×172 영역이 첫 번째 얼굴이다.

### detectMultiScale 파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| `scaleFactor` | 1.1 | 이미지를 10%씩 줄여 가며 반복 탐색 |
| `minNeighbors` | 5 | 겹쳐 검출된 사각형이 5개 이상이어야 인정 |
| `minSize` | (30, 30) | 이보다 작으면 무시 |

**`scaleFactor`가 필요한 이유**

검사 창의 크기는 고정이다. 그래서 큰 얼굴을 찾으려면 **이미지 쪽을 줄인다.**
1.1이면 촘촘히 훑어 잘 찾지만 느리다. 1.3이면 빠른 대신 놓치는 얼굴이 생긴다.

**`minNeighbors` 조절 기준**

| 값 | 결과 |
|---|---|
| 작게 (1~3) | 얼굴을 많이 찾지만 **오검출**이 섞인다 |
| 크게 (6~10) | 확실한 것만 찾지만 **놓치는 얼굴**이 생긴다 |

얼굴이 안 잡히면 이 값을 낮추고, 엉뚱한 데가 잡히면 올린다.

### 찾은 얼굴에 사각형 그리기

```python
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

`cv2.rectangle(이미지, 좌상단, 우하단, 색, 두께)`.
`faces`가 주는 건 `(x, y, w, h)`이므로 우하단은 **`(x + w, y + h)`로 직접 계산**해야 한다.

색 `(255, 0, 0)`은 OpenCV 기준 **BGR**이라 파란색이다.
표시할 때 RGB로 변환하므로 화면에도 파란 테두리로 나온다.

### 다른 검출기

같은 방식으로 다른 것도 찾을 수 있다.

```python
cv2.data.haarcascades + 'haarcascade_eye.xml'
cv2.data.haarcascades + 'haarcascade_smile.xml'
cv2.data.haarcascades + 'haarcascade_fullbody.xml'
```

### 언제 쓰나

- 정면 얼굴, 조명이 괜찮은 환경
- 실시간 처리가 필요하고 GPU가 없을 때
- 얼굴 위치만 대충 잡아서 다음 단계(인식 모델)에 넘길 때

### 주의점

- **정면 얼굴에만 잘 맞는다.** 옆얼굴, 고개 숙인 얼굴, 가려진 얼굴은 놓친다
- 조명이 어둡거나 역광이면 성능이 크게 떨어진다
- 정확도가 중요하면 딥러닝 기반(MTCNN, RetinaFace, YOLO-face)을 쓴다

참고: 실습 코드의 변수명 `gray`

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # 이름은 gray인데 RGB다
```

이름은 `gray`인데 실제로는 **RGB 3채널**이다. `COLOR_BGR2GRAY`가 아니다.
결과는 정상이다. `detectMultiScale`이 3채널을 받으면 **내부에서 알아서 흑백으로 바꾸기** 때문이다.
다만 읽는 사람이 헷갈리므로 이름을 `rgb`로 바꾸거나, 실제로 흑백 변환을 하는 게 낫다.

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # 이쪽이 원래 의도로 보인다
```

---

## 5-2. VAE — 확률로 이미지 만들기

### 오토인코더부터

**오토인코더**는 입력을 압축했다가 다시 펼치는 모델이다.

```
이미지(784) → 인코더 → 잠재벡터 z(2) → 디코더 → 이미지(784)
```

가운데를 좁게 만들어 놓고 "원본을 복원하라"고 시키면,
모델은 **꼭 필요한 정보만 골라 담는 법**을 배운다.

문제는 이걸로 **새 이미지를 만들 수 없다는 것**이다.
잠재 공간에 아무 값이나 넣으면 의미 없는 얼룩이 나온다.
학습된 지점들 사이가 텅 비어 있기 때문이다.

### VAE의 해결책

VAE는 인코더가 **점 하나가 아니라 확률분포를 내놓게** 한다.

```
일반 오토인코더:  이미지 → z = [0.3, -1.2]           (점)
VAE:            이미지 → 평균 [0.3, -1.2], 분산 [...]  (분포)
```

그리고 그 분포가 **표준정규분포 N(0,1)에 가깝도록** 손실로 강제한다.
결과적으로 잠재 공간이 빈틈없이 채워지고, **아무 데서나 뽑아도 그럴듯한 이미지가 나온다.**

### 3장 GAN과의 차이

| | GAN (3-6) | VAE |
|---|---|---|
| 구조 | 생성자 + 판별자 경쟁 | 인코더 + 디코더 |
| 학습 | 불안정 | **안정적** |
| 결과 | 선명하다 | **흐릿하다** |
| 잠재 공간 | 해석 어려움 | **의미가 이어진다** |

### 데이터 준비

```python
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

def preprocess_image(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_image(train_images)

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(60000)
                 .batch(128))
```

`np.where(images > .5, 1.0, 0.0)`가 **0과 1만 남긴다.**
회색 없이 흑백으로만 만드는 것이다.

**이진화하는 이유**는 손실 함수 때문이다.
아래에서 `sigmoid_cross_entropy_with_logits`를 쓰는데,
이건 정답이 0 또는 1인 걸 전제로 한 함수다.
**픽셀 하나하나를 동전 던지기로 보는 셈**이다.

3장 GAN에서 셔플 버퍼를 600으로 뒀던 것과 달리 여기는 60000이다. 이쪽이 올바른 사용이다.

### 모델 구조

```python
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(28 * 28 * 1),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        ])
```

**인코더 마지막이 `latent_dim + latent_dim`인 이유**

잠재 차원이 2인데 출력은 4다. 앞 2개가 **평균**, 뒤 2개가 **로그분산**이다.
분포를 표현하려면 두 값이 다 필요하다.

**디코더 마지막에 활성화 함수가 없다**
`sigmoid`를 붙이지 않고 로짓(logit)을 그대로 내보낸다.
손실 함수 `sigmoid_cross_entropy_with_logits`가 내부에서 시그모이드를 적용하기 때문이다.
여기서 또 붙이면 두 번 적용되어 학습이 망가진다.

### 네 개의 메서드

```python
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
```

| 메서드 | 하는 일 |
|---|---|
| `encode` | 이미지 → 평균, 로그분산 |
| `reparameterize` | 분포에서 z 하나 뽑기 |
| `decode` | z → 이미지 |
| `sample` | 노이즈에서 바로 이미지 생성 (확률로 보려고 sigmoid 적용) |

### reparameterization trick — VAE의 핵심 아이디어

```python
z = eps * tf.exp(logvar * .5) + mean
```

이 한 줄이 VAE를 가능하게 만든다.

문제는 이렇다. **분포에서 무작위로 뽑는 연산은 미분이 안 된다.**
미분이 안 되면 역전파가 끊기고 학습이 멈춘다.

해결책은 무작위성을 **밖으로 빼내는 것**이다.

```
안 되는 방식:  z ~ N(mean, var)              ← 뽑는 행위 자체가 랜덤
되는 방식:    eps ~ N(0, 1)   (랜덤은 여기만)
             z = mean + eps × 표준편차       ← 이 식은 미분 가능
```

`eps`는 모델과 무관한 상수처럼 취급되고,
`mean`과 `logvar`로 가는 길은 사칙연산뿐이라 기울기가 그대로 흐른다.

**`logvar * .5`를 지수에 넣는 이유**
`exp(logvar × 0.5) = exp(logvar)^0.5 = √분산 = 표준편차`다.

**분산이 아니라 로그분산을 다루는 이유**
분산은 항상 양수여야 하는데 신경망 출력은 음수가 나올 수 있다.
로그로 두면 어떤 실수든 받아서 `exp`로 양수로 되돌릴 수 있다.

### 손실 함수

```python
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
```

수식이 복잡해 보이지만 **두 덩어리**다.

| 항 | 이름 | 뜻 |
|---|---|---|
| `logpx_z` | 복원 손실 | 원본과 얼마나 같게 복원했나 |
| `logpz - logqz_x` | KL 발산 | 잠재분포가 N(0,1)에서 얼마나 벗어났나 |

이 둘의 합을 **ELBO**라고 부르고, **ELBO를 최대화**하는 게 목표다.
손실은 최소화해야 하므로 앞에 마이너스를 붙였다.

**두 항이 서로 반대로 당긴다.**

- 복원 손실만 줄이려 하면 → 각 이미지를 서로 멀리 떼어놓는 게 유리 → 잠재공간에 구멍이 생김
- KL만 줄이려 하면 → 전부 원점으로 뭉침 → 어떤 이미지든 똑같이 복원
- **균형점에서** 잘 복원되면서도 빈틈없는 잠재공간이 만들어진다

`axis=[1, 2, 3]`은 **한 이미지 안의 픽셀 전부**를 더한다는 뜻이다.
0번 축(배치)은 남겨 두었다가 마지막에 `reduce_mean`으로 평균 낸다.

### 학습

```python
optimizer = tf.keras.optimizers.Adam(4e-4)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

model = VAE(latent_dim=2)

for epoch in range(1, epochs + 1):
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)

    loss = tf.keras.metrics.Mean()
    for test_x in train_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()

    print('Epoch {0}, ELBO {1}, time: {2}'.format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)
```

3장 GAN과 마찬가지로 `GradientTape`를 직접 쓴다.
다만 **손실이 하나, 모델이 하나**라 테이프도 하나다. GAN보다 단순하다.

`tf.keras.metrics.Mean()`은 값을 계속 넣으면 평균을 누적해 준다.
배치별 손실을 모아 에폭 평균을 내는 용도다.

### 결과

```
Epoch 50, ELBO -130.77, time elapse for current epoch: 4.20
```

**ELBO는 음수이고, 0에 가까울수록 좋다.**
-130.77은 잠재 차원이 2뿐인 것치고 무난한 값이다.
차원을 늘리면(예: 8, 16) 더 올라가지만 잠재 공간을 그림으로 보기는 어려워진다.

### 이미지 생성

```python
def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
```

에폭마다 같은 샘플 25장을 복원해 저장한다.
**같은 입력을 계속 보기 때문에 개선 과정이 눈에 보인다.**
초반에는 뿌연 얼룩이다가 점점 숫자 모양이 잡힌다.

`'{:04d}'.format(epoch)`는 `0001`처럼 0을 채운다.
파일 이름 정렬이 맞아야 나중에 GIF로 이어 붙이기 좋다.

### 주의점

- **결과가 흐릿하다.** 확률분포의 평균을 그리는 방식이라 어쩔 수 없다. 선명한 걸 원하면 GAN이나 디퓨전 모델을 쓴다
- 잠재 차원이 2면 시각화에는 좋지만 표현력이 부족하다
- 전결합층만 써서 위치 정보를 못 살린다. `Conv2D`와 `Conv2DTranspose`로 바꾸면 결과가 나아진다

참고: 실습 코드의 잔가지

| 위치 | 내용 |
|---|---|
| `import tensorflow_probability as tfp` | 임포트만 하고 쓰지 않는다. `glob`, `imageio`도 마찬가지 |
| `random_vector_for_generation` | 만들어 두고 쓰지 않는다. `generate_and_save_images`가 `test_sample`을 인코딩해서 z를 얻기 때문 |
| ELBO 계산 | `for test_x in train_dataset`으로 **학습 데이터에서** 계산한다. 원래 튜토리얼은 별도 테스트셋을 쓴다. 지금 값은 검증 점수가 아니라 학습 점수다 |

---

## 이 장 정리

| 주제 | 학습 필요 | 용도 |
|---|---|---|
| Haar Cascade | 없음 (사전 학습됨) | 빠른 얼굴 검출 |
| VAE | 있음 | 이미지 생성, 차원 축소 |

### VAE 전체 흐름

```
이미지 → 인코더 → [평균, 로그분산]
                      ↓ reparameterize (여기서 랜덤)
                      z
                      ↓ 디코더
                   복원 이미지

손실 = 복원 손실 + KL 발산
```

### 생성 모델 비교

| | GAN | VAE |
|---|---|---|
| 핵심 아이디어 | 경쟁 | 확률분포 학습 |
| 학습 안정성 | 낮다 | **높다** |
| 이미지 품질 | **선명** | 흐릿 |
| 손실 해석 | 어렵다 (균형을 봐야 함) | **쉽다 (ELBO 하나)** |

### 자주 틀리는 것

- `detectMultiScale`은 **`(x, y, w, h)`**를 준다. 우하단은 `(x+w, y+h)`로 계산한다
- 얼굴이 안 잡히면 `minNeighbors`를 낮추고, 오검출이 많으면 올린다
- VAE 디코더 마지막에 **`sigmoid`를 붙이지 않는다.** 손실 함수가 내부에서 처리한다
- `reparameterize`가 없으면 **역전파가 끊긴다.** 랜덤을 밖으로 빼는 게 핵심
- 분산이 아니라 **로그분산**을 다룬다. 음수 출력을 허용하기 위해서
