# 3. 역전파와 CNN

**실습 파일**: `3_1.ipynb`, `3_0.ipynb`

2장에서 학습 루프 다섯 줄을 외웠다. 이 장은 그 안에서 **무슨 일이 일어나는지**를 열어 본다.
그리고 이미지를 제대로 다루는 구조인 **CNN**으로 넘어간다.

| 파일 | 절 | 내용 |
|---|---|---|
| `3_1.ipynb` | 3-1 ~ 3-7 | 순전파 → 손실 → 역전파 → autograd → 활성화 함수 → MNIST |
| `3_0.ipynb` | 3-8 ~ 3-13 | 합성곱 → CNN → CIFAR-10 → ResNet → 스케줄러 |

파일 이름은 `3_0`이 앞이지만, **개념 순서는 `3_1`이 먼저**다.
역전파를 이해하고 나서 CNN으로 가는 게 자연스럽다.

---

## 3-1. 가장 단순한 모델 — 피자 가격

### 개념

신경망은 결국 **입력에 가중치를 곱하고 편향을 더하는 것**이다.
피자 크기로 가격을 맞히는 함수 하나로 시작한다.

```python
def pizza_price_model(size):
    weight = 2
    bias = 3
    price = weight * size + bias
    return price

size = 5
predicted_price = pizza_price_model(size)
print(predicted_price)   # 13
```

`가격 = 2 × 크기 + 3`. 이게 뉴런 하나가 하는 일 전부다.

지금은 `weight = 2`, `bias = 3`을 사람이 적었다.
**학습이란 이 두 숫자를 데이터를 보고 스스로 정하는 것**이다.

---

## 3-2. 순전파 — 행렬로 한 번에

### 개념

뉴런이 여러 개면 곱셈을 하나씩 쓸 수 없다. **행렬 곱**으로 한 번에 한다.

```python
import torch

def forward_pass(x, w, b):
    z = torch.mm(x, w) + b
    output = torch.relu(z)
    return output

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
w = torch.tensor([[0.5, -0.5], [1.0, 1.0]])
b = torch.tensor([[1.0, 1.0]])

output = forward_pass(x, w, b)
print(output)
```

```
tensor([[3.5000, 2.5000],
        [6.5000, 3.5000]])
```

### 손으로 따라가 보기

첫 번째 샘플 `[1, 2]`의 첫 번째 출력을 계산해 보면 이렇다.

```
1 × 0.5 + 2 × 1.0 + 1.0 = 3.5
```

두 번째 출력은 `1 × (-0.5) + 2 × 1.0 + 1.0 = 2.5`다. 결과와 맞는다.

| 항목 | 모양 | 의미 |
|---|---|---|
| `x` | (2, 2) | 샘플 2개, 피처 2개 |
| `w` | (2, 2) | 입력 2 → 출력 2 |
| `b` | (1, 2) | 출력마다 편향 하나 |

**`b`의 모양이 `(1, 2)`인데 결과는 `(2, 2)`다.**
넘파이·파이토치가 알아서 행 방향으로 복제해 더한다(브로드캐스팅).
편향은 샘플마다 같은 값을 쓰므로 이게 맞는 동작이다.

`torch.relu`가 음수를 0으로 만든다. 여기서는 전부 양수라 변화가 없다.

---

## 3-3. 손실 — 얼마나 틀렸나

```python
import torch.nn as nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = 2 * x + 1                              # 정답: 3, 5, 7, 9

model_output = torch.tensor([[2.5], [3.5], [4.5], [5.5]])
criterion = nn.MSELoss()

loss = criterion(model_output, y_true)
print(loss)      # tensor(5.2500)
```

**인자 순서가 (예측, 정답)이다.** 반대로 넣으면 MSE는 결과가 같지만,
`CrossEntropyLoss` 같은 건 에러가 나거나 엉뚱한 값이 나온다. 습관을 (예측, 정답)으로 굳힌다.

계산을 확인해 보면

```
오차:   -0.5, -1.5, -2.5, -3.5
제곱:    0.25, 2.25, 6.25, 12.25
평균:   21 / 4 = 5.25
```

---

## 3-4. 역전파 — 손으로 한 번

### 개념

손실을 알았으니 이제 **가중치를 어느 방향으로 얼마나 옮길지** 정해야 한다.

```python
actual_price = 11
predicted_price = 13
error = predicted_price - actual_price       # 2

weight = 2
bias = 3
learning_rate = 0.01
size = 5

weight = weight - learning_rate * error * size    # 2 - 0.01*2*5 = 1.9
bias = bias - learning_rate * error               # 3 - 0.01*2   = 2.98

price = weight * size + bias
print(price)    # 12.48
```

13에서 12.48로 **정답 11 쪽으로 움직였다.** 한 번에 닿지는 않는다.
이걸 수천 번 반복하는 게 학습이다.

### 왜 `error * size`인가

가중치는 `size`가 곱해져서 출력에 영향을 준다.
그래서 가중치를 1 바꿀 때 출력은 `size`만큼 바뀐다. 편향은 그냥 더해지므로 계수가 1이다.

**이게 미분이 하는 말이다.** "이 값을 조금 바꾸면 결과가 얼마나 바뀌는가."

| 파라미터 | 기울기 | 왜 |
|---|---|---|
| `weight` | `error × size` | 출력에 `size`배로 반영됨 |
| `bias` | `error` | 출력에 그대로 더해짐 |

학습률 0.01을 곱하는 이유는 **한 번에 너무 많이 움직이지 않기 위해서**다.
크게 잡으면 정답을 지나쳐 발산하고, 작게 잡으면 느리다.

---

## 3-5. autograd — 파이토치가 대신 미분한다

### 개념

3-4처럼 손으로 미분식을 쓰는 건 층이 두세 개만 돼도 불가능하다.
파이토치는 연산을 기록해 두었다가 **자동으로 미분한다.**

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2 + 3 * x + 4      # 순전파
y.backward()                # 역전파

print(x.grad)               # tensor(7.)
```

손으로 확인해 보면 `dy/dx = 2x + 3`이고, `x = 2`이므로 `2×2 + 3 = 7`이다. 맞는다.

**`requires_grad=True`가 핵심이다.** 이게 붙은 텐서만 기울기를 추적한다.
`nn.Linear` 같은 층의 파라미터는 자동으로 `True`다.

### 텐서에도 똑같이

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = (x ** 2).sum()

y.backward()
print(x.grad)
```

```
tensor([[2., 4.],
        [6., 8.]])
```

`d(x²)/dx = 2x`이므로 원소마다 2배가 나온다.

**`backward()`를 부르려면 결과가 스칼라여야 한다.**
그래서 `.sum()`으로 하나의 숫자로 줄였다.
손실 함수가 항상 숫자 하나를 돌려주는 것도 같은 이유다.

### 3-4와 이어 보기

```
3-4  error * size 를 사람이 계산    →  손으로 미분
3-5  loss.backward()               →  파이토치가 미분
```

2장에서 외운 다섯 줄 중 `loss.backward()`가 하는 일이 바로 이것이다.

---

## 3-6. 선형 회귀 — 다섯 줄로 학습시키기

```python
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = 2 * x + 1

model = LinearRegressionModel(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y_true)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(epoch + 1, loss.item())
```

```
 20  0.16900856792926788
 40  0.08326765149831772
 60  0.07381182163953781
 80  0.06546936929225922
100  0.05806984007358551
```

3-1의 피자 모델과 **같은 일을 하는데 가중치를 스스로 찾는다.**
정답이 `y = 2x + 1`이므로 손실이 0에 가까워지는 게 맞다.

100 에폭에 0.058이면 아직 완전히 수렴하지 않았다.
40 에폭 이후로는 내려가는 속도가 눈에 띄게 느려진다.
`lr`을 키우거나 `Adam`으로 바꾸면 빨라진다.

---

## 3-7. 활성화 함수 세 가지

### 개념

활성화 함수가 없으면 층을 아무리 쌓아도 **하나의 선형 변환으로 합쳐진다.**
비선형을 넣어야 층을 쌓는 의미가 생긴다.

```python
import numpy as np

x = np.linspace(-10, 10, 400)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)
```

| 함수 | 출력 범위 | 특징 | 주로 |
|---|---|---|---|
| **Sigmoid** | 0 ~ 1 | 확률로 읽힌다. 양 끝에서 기울기가 0에 가까워짐 | 이진 분류 출력층 |
| **Tanh** | -1 ~ 1 | 0을 중심으로 대칭. 시그모이드보다 학습이 잘 됨 | RNN 계열 |
| **ReLU** | 0 ~ ∞ | 계산이 싸고 기울기 소실이 적다 | **은닉층 기본값** |

**기울기 소실**이 핵심 개념이다.
시그모이드는 입력이 5만 넘어가도 기울기가 거의 0이다.
층이 깊어지면 이 작은 값들이 곱해지면서 앞쪽 층까지 신호가 도달하지 못한다.

ReLU는 양수 구간에서 기울기가 항상 1이라 이 문제가 훨씬 덜하다.
대신 음수 입력에서는 기울기가 0이라 뉴런이 죽을 수 있다(dying ReLU).
영상처리 5장에서 본 LeakyReLU가 그 보완책이다.

---

## 3-8. MNIST로 확인하기

### 데이터 준비

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)
```

`0.1307`과 `0.3081`은 MNIST 전체의 평균과 표준편차다.
**데이터셋마다 이 값이 다르다.** CIFAR-10은 3-11에서 다른 값을 쓴다.

`transforms.RandomRotation(10)`처럼 증강을 넣을 수도 있는데,
**테스트 데이터에는 증강을 넣지 않는다.** 평가가 흔들리기 때문이다.

### 모델과 학습

```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(epoch + 1, total_loss / len(train_loader))
```

```
1 0.2632080554795354
2 0.11573933447693298
3 0.07989479099531402
4 0.06032238197453352
5 0.0474413006736744
```

**2장과 달리 에폭 평균 손실을 낸다.** `total_loss`를 모아서 배치 수로 나눈다.
2장에서는 마지막 배치의 손실만 찍어 값이 튀었다. 이쪽이 맞는 방식이다.

`x.view(-1, 28 * 28)`이 이미지를 한 줄로 편다. 케라스의 `Flatten`과 같다.

### 평가

```python
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

print(evaluate(model, test_loader))    # 97.48
```

**테스트 정확도 97.48%.**

평가 함수에 반드시 들어가야 할 두 가지가 있다.

| 코드 | 없으면 |
|---|---|
| `model.eval()` | 드롭아웃과 배치정규화가 학습 모드로 돌아 결과가 흔들린다 |
| `torch.no_grad()` | 기울기를 계산하느라 메모리를 낭비하고 느려진다 |

`torch.max(outputs, 1)`은 (최댓값, 위치)를 돌려준다. 위치가 곧 예측 클래스다.
`_`로 최댓값을 버리는 게 관용적인 표현이다.

---

## 3-9. 학습률 비교

```python
learning_rates = [0.1, 0.01, 0.001, 0.0001]
final_losses = []

for lr in learning_rates:
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    final_losses.append(evaluate(model, test_loader))

plt.plot(learning_rates, final_losses, marker='o')
plt.xscale('log')
plt.show()
```

**학습률마다 모델을 새로 만드는 게 중요하다.** `SimpleMLP()`를 루프 안에서 다시 생성한다.
안 그러면 앞 학습률의 결과 위에 이어서 학습하게 되어 비교가 안 된다.

`plt.xscale('log')`를 쓰는 이유는 학습률이 10배씩 차이 나기 때문이다.
선형 축으로 그리면 0.0001, 0.001, 0.01이 한 점에 뭉친다.

참고: 변수 이름이 `final_losses`인데 담기는 값은 `evaluate`가 돌려주는 **정확도**다.
y축 라벨도 `final loss`로 되어 있어 그래프를 잘못 읽기 쉽다.
`final_accuracies` / `accuracy`로 바꾸는 게 맞다.

---

## 3-10. 오분류 들여다보기

### 모델 정의

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten * nn.Flatten()          # ← 여기에 버그가 있다
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),      nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 10))

    def forward(self, x):
        return self.network(self.flatten(x))
```

주의: `=`가 아니라 `*`가 들어갔다

`self.flatten * nn.Flatten()`은 **대입이 아니라 곱셈**이다.
`self.flatten`이 아직 없는 상태에서 읽으려 하므로 `MNISTNet()`을 만드는 순간
`AttributeError: 'MNISTNet' object has no attribute 'flatten'`이 난다.

실제로 노트북에서 **이 모델을 만드는 셀은 실행되지 않았다**(실행 번호가 비어 있다).
그래서 아래 오분류 분석과 혼동행렬은 `MNISTNet`이 아니라
**3-9 루프의 마지막 모델(`SimpleMLP`, 학습률 0.0001)**로 계산된 것이다.

```python
self.flatten = nn.Flatten()      # 이렇게 고친다
```

### 틀린 것만 골라 보기

```python
wrong_images, wrong_preds, wrong_targets = [], [], []
model.eval()

with torch.no_grad():
    for data, target in test_loader:
        out = model(data)
        preds = out.argmax(dim=1)
        mask = preds != target                 # 틀린 것만 True
        idxs = mask.nonzero().squeeze()
        for i in idxs:
            if len(wrong_images) >= 6:
                break
            wrong_images.append(data[i])
            wrong_preds.append(preds[i].item())
            wrong_targets.append(target[i].item())

plt.figure(figsize=(10, 4))
for i in range(len(wrong_images)):
    ax = plt.subplot(1, 6, i + 1)
    ax.imshow(wrong_images[i].squeeze(), cmap='gray')
    ax.axis('off')
    ax.set_title(f"{wrong_targets[i]}, {wrong_preds[i]}")
plt.tight_layout()
plt.show()
```

**정확도 숫자 하나보다 틀린 사례를 직접 보는 게 훨씬 유용하다.**
사람이 봐도 헷갈리는 글씨인지, 모델이 특정 숫자를 계속 놓치는지가 보인다.

`preds != target`이 불리언 마스크를 만들고, `nonzero()`가 True인 위치를 뽑는다.
`.squeeze()`는 `(N, 1)`을 `(N,)`으로 눌러 준다.

참고: 안쪽 `break`는 for 루프 하나만 빠져나오므로
바깥 배치 루프는 계속 돈다. 6개를 채운 뒤에도 테스트셋 전체를 훑는다.
빨리 끝내려면 플래그를 두거나 함수로 감싸 `return`한다.

### 혼동행렬

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        out = model(data)
        all_preds.extend(out.argmax(dim=1).tolist())
        all_labels.extend(target.tolist())

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

대각선이 맞힌 것, 나머지가 틀린 것이다.
**어떤 숫자를 어떤 숫자로 헷갈리는지**가 한눈에 보인다. 4와 9, 3과 5가 대표적이다.

`fmt='d'`를 빼면 정수가 `1.2e+03` 같은 지수 표기로 나온다.

---

## 3-11. 합성곱 — 이미지를 이미지답게

### 왜 CNN인가

3-8의 `SimpleMLP`는 28×28을 784개 숫자로 펴서 넣었다.
**그 순간 "옆 픽셀끼리 붙어 있다"는 정보가 사라진다.**

```python
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
```

합성곱은 **작은 필터를 이미지 위에서 미끄러뜨리며** 지역 패턴을 찾는다. 위치 정보가 유지된다.

### 소벨 필터로 확인

```python
sobel_filter = torch.tensor([[[
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [1.0, 0.0, -1.0]
]]])

sample = torch.randn(1, 1, 28, 28)
sobel_output = nn.functional.conv2d(sample, sobel_filter)
print(sobel_output.shape)     # torch.Size([1, 1, 26, 26])
```

영상처리 2장에서 손으로 만들었던 그 소벨 커널이다.
**CNN은 이 9개 숫자를 사람이 정하지 않고 학습으로 찾는다.**

28×28이 26×26이 된 이유는 3×3 커널이 지나갈 때 가장자리가 깎이기 때문이다.

### 세로선 검출 실험

```python
filter_2d = torch.tensor([[
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0]
]]).unsqueeze(0)

sample_image_2 = torch.zeros(1, 1, 8, 8)
sample_image_2[0, 0, :, 3:5] = 1.0      # 세로 흰 띠

output_2 = nn.functional.conv2d(sample_image_2, filter_2d)
```

왼쪽이 +1, 오른쪽이 -1인 필터다. **밝기가 왼→오로 떨어지는 곳**에서 큰 값이 나온다.
띠의 왼쪽 경계와 오른쪽 경계가 각각 밝게, 어둡게 잡힌다.

### 텐서 모양 규칙

파이토치의 합성곱 입력은 **`(배치, 채널, 높이, 너비)`**다.
케라스는 `(배치, 높이, 너비, 채널)`로 **채널이 뒤에 온다.** 자주 헷갈리는 지점이다.

| | 순서 |
|---|---|
| 파이토치 | `(N, C, H, W)` — 채널 먼저 |
| 케라스 | `(N, H, W, C)` — 채널 나중 |

`.unsqueeze(0)`은 앞에 차원 하나를 끼워 넣는다. 필터도 4차원이어야 하기 때문이다.

### 출력 크기 계산

```python
input_image = torch.randn(1, 1, 6, 6)
filter1 = torch.randn(1, 1, 3, 3)

# 출력 = (입력 - 커널 + 2×패딩) / 스트라이드 + 1
print(F.conv2d(input_image, filter1, stride=1, padding=0).shape)   # [1, 1, 4, 4]
print(F.conv2d(input_image, filter1, stride=1, padding=1).shape)   # [1, 1, 6, 6]
print(F.conv2d(input_image, filter1, stride=2, padding=0).shape)   # [1, 1, 2, 2]
```

**이 식은 외워 두면 계속 쓴다.**

| 설정 | 계산 | 결과 |
|---|---|---|
| stride 1, padding 0 | (6-3+0)/1 + 1 | 4 |
| stride 1, padding 1 | (6-3+2)/1 + 1 | **6 (크기 유지)** |
| stride 2, padding 0 | (6-3+0)/2 + 1 | 2 |

`padding=1`이 3×3 커널에서 크기를 유지하는 표준 조합이다. 케라스의 `padding='same'`에 해당한다.

### 여러 필터, 여러 채널

```python
input_rgb = torch.randn(1, 3, 32, 32)
filters = torch.randn(16, 3, 3, 3)

output4 = F.conv2d(input_rgb, filters, stride=1, padding=1)
print(output4.shape)    # torch.Size([1, 16, 32, 32])
```

필터 모양 `(16, 3, 3, 3)`을 읽는 법이 중요하다.

```
(16,   3,    3, 3)
  │    │     └──┴── 커널 크기 3×3
  │    └────────── 입력 채널 3개(RGB)를 모두 받는다
  └─────────────── 출력 채널 16개 = 필터 16장
```

**필터 하나가 입력 채널 전부를 한꺼번에 본다.** 그래서 출력 채널이 16개가 된다.

---

## 3-12. CNN 쌓기 — BatchNorm과 Dropout

### 기본 CNN

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

**`nn.Linear(2048, 10)`의 2048이 어디서 나왔는지**를 따라갈 수 있어야 한다.

```
입력      (3, 32, 32)
conv1  →  (16, 32, 32)    padding=1이라 크기 유지
pool1  →  (16, 16, 16)    절반
conv2  →  (32, 16, 16)
pool2  →  (32, 8, 8)      절반
편다   →  32 × 8 × 8 = 2048
```

이 숫자를 틀리면 `shape mismatch` 에러가 난다. 파이토치에서 가장 흔한 에러다.
`nn.LazyLinear`를 쓰면 자동으로 맞춰 주지만, 직접 계산해 보는 편이 구조 이해에 낫다.

`x.view(x.size(0), -1)`에서 `x.size(0)`은 배치 크기다. **배치는 건드리지 않고** 나머지만 편다.

### BatchNorm 추가

```python
self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
self.bn1 = nn.BatchNorm2d(16)
self.relu1 = nn.ReLU()
```

**순서가 Conv → BN → ReLU다.** 이게 표준이다.

`BatchNorm2d(16)`의 16은 **채널 수**다. 앞 Conv의 출력 채널과 반드시 같아야 한다.

배치 단위로 출력을 평균 0, 분산 1로 맞춰서

- 층이 깊어져도 신호가 죽거나 폭발하지 않는다
- 학습률을 크게 잡을 수 있다
- 그 자체로 약한 규제 효과가 있다

### Dropout 추가

```python
self.dropout1 = nn.Dropout(0.25)     # 합성곱 블록 뒤
self.dropout2 = nn.Dropout(0.25)
self.fc1 = nn.Linear(32 * 8 * 8, 128)
self.dropout3 = nn.Dropout(0.5)      # 전결합층 앞
self.fc2 = nn.Linear(128, 10)
```

**전결합층 앞의 비율이 더 높다(0.5).** 파라미터가 거기 몰려 있어 과적합에 취약하기 때문이다.

`model.eval()`을 부르면 드롭아웃이 자동으로 꺼진다.
**케라스는 이게 자동인데 파이토치는 직접 불러야 한다.** 파이토치에서 가장 흔한 실수다.

---

## 3-13. CIFAR-10 학습

### 데이터 증강

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 테스트 데이터는 증강하지 않는다
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
```

**증강은 학습 데이터에만 건다.** 테스트에 걸면 평가가 매번 달라진다.

`ToTensor()`가 `Normalize()`보다 먼저 와야 한다.
`Normalize`는 텐서에만 동작하기 때문이다. 순서를 바꾸면 에러가 난다.

세 개의 평균·표준편차는 CIFAR-10 전용 값이다. MNIST의 `(0.1307,)`과 다르다.
채널이 3개라 값도 3개다.

### 모델

```python
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10))

    def forward(self, x):
        return self.classifier(self.features(x))
```

**전체 파라미터 1,305,066개.**

`features`와 `classifier`로 나누는 건 관례다. 4장의 전이학습에서 이 구분이 그대로 쓰인다.

`Dropout2d`는 일반 `Dropout`과 다르다. **채널 하나를 통째로 끈다.**
합성곱 출력은 옆 픽셀끼리 값이 비슷해서, 픽셀 단위로 끄면 효과가 약하다.

`inplace=True`는 새 텐서를 만들지 않고 제자리에서 바꿔 메모리를 아낀다.

### 학습 — 그리고 여기서 문제가 생긴다

```python
cifar_model = CIFAR10_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)     # ← model 은 3-12의 CNNDropout
```

버그: 옵티마이저가 다른 모델을 붙들고 있다

`cifar_model`을 만들어 놓고 옵티마이저에는 `model.parameters()`를 넘겼다.
`model`은 3-12에서 마지막으로 만든 `CNNDropout`이다.

그래서 학습 루프가 `cifar_model`로 순전파·역전파를 해도
**`optimizer.step()`은 엉뚱한 모델의 가중치를 갱신한다.**
`cifar_model`의 파라미터는 처음 그대로 남는다.

실행 결과가 이를 그대로 보여 준다.

| 에폭 | 배치 | 손실 | 정확도 |
|---|---|---|---|
| 1 | 10 | 2.3888 | 9.53% |
| 10 | 10 | 2.3976 | 8.98% |

```
Test Loss: 1.9299, Test Acc: 10.6
```

10개 클래스에서 **찍는 것과 같은 10%**다. 손실도 `ln(10) ≈ 2.303` 근처에서 움직이지 않는다.
**학습이 전혀 일어나지 않았다는 신호다.**

```python
optimizer = optim.Adam(cifar_model.parameters(), lr=0.001)   # 이렇게 고친다
```

참고: 이 실습은 `if i >= 10: break`로 배치 10개만 돌린다.
버그를 고쳐도 배치 10개(=1280장)로는 제대로 학습되지 않는다. 구조 확인용 코드다.

참고: 시각화 두 가지

```python
print(classes[labels[j]] for j in range(8))
```

괄호 안이 제너레이터라 `<generator object ...>`가 찍힌다.
`print([classes[labels[j]] for j in range(8)])`처럼 리스트로 감싸야 한다.

`imshow`의 `img / 2 + 0.5`도 맞지 않는다.
이 역정규화는 평균 0.5, 표준편차 0.5로 정규화했을 때의 식인데,
실제로는 CIFAR-10 통계값을 썼다. 출력에 범위 초과 경고가 그대로 남아 있다.

```
Clipping input data to the valid range for imshow ... Got range [-0.4947..1.5632]
```

---

## 3-14. ResNet — 잔차 연결

### 개념

층을 깊게 쌓으면 기울기가 앞까지 도달하지 못한다.
ResNet은 **입력을 출력에 그대로 더해서** 이 문제를 푼다.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)     # 잔차 연결
        out = self.relu(out)
        return out
```

**`out += self.shortcut(residual)` 한 줄이 ResNet의 전부다.**

층이 "정답을 만드는 일" 대신 **"입력에서 얼마나 바꿀지(잔차)만 정하는 일"**을 하게 된다.
바꿀 게 없으면 0을 출력하면 되므로, 층을 늘려도 성능이 나빠지지 않는다.

### shortcut이 조건부인 이유

더하려면 두 텐서의 모양이 같아야 한다.
채널 수가 바뀌거나 stride로 크기가 줄면 **1×1 합성곱으로 맞춰 준다.**

모양이 같으면 `nn.Sequential()` — 빈 시퀀셜이라 입력을 그대로 통과시킨다.
`if`문 없이 처리하는 깔끔한 방법이다.

`bias=False`인 이유는 바로 뒤 BatchNorm이 평균을 빼기 때문이다. 편향이 상쇄되어 의미가 없다.

### ResNet18 조립

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

resnet18_model = ResNet18(num_classes=10)
print(sum(p.numel() for p in resnet18_model.parameters()))    # 11173962
```

**파라미터 11,173,962개.** 3-13의 CNN(1,305,066개)의 약 8.6배다.

`[2, 2, 2, 2]`는 단계별 블록 수다. 블록 하나에 합성곱이 2개이므로
`2×(2+2+2+2) + 1(첫 conv) + 1(fc) = 18`, 그래서 ResNet**18**이다.
`[3, 4, 6, 3]`으로 바꾸면 ResNet34가 된다.

`_make_layer`의 `strides = [stride] + [1] * (num_blocks - 1)`은
**첫 블록만 크기를 줄이고 나머지는 유지**한다는 뜻이다.

`AdaptiveAvgPool2d((1, 1))`은 입력 크기와 무관하게 1×1로 만든다.
덕분에 32×32든 224×224든 같은 모델이 동작한다.
영상처리 4장의 GoogLeNet에서 `AveragePooling2D`를 쓴 것과 같은 이유다.

`expansion`은 병목 블록(ResNet50 이상)에서 채널이 4배로 늘어나는 걸 처리하려는 장치다.
`BasicBlock`은 1이라 실질적으로 아무 일도 하지 않는다.

---

## 3-15. 학습률 스케줄러와 가중치 초기화

### 스케줄러 세 가지

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

demo_model = SimpleCNN()
demo_optimizer = optim.SGD(demo_model.parameters(), lr=0.1,
                           momentum=0.9, weight_decay=5e-4)

step_scheduler    = StepLR(demo_optimizer, step_size=30, gamma=0.1)
plateau_scheduler = ReduceLROnPlateau(demo_optimizer, mode='min', factor=0.1, patience=10)
cosine_scheduler  = CosineAnnealingLR(demo_optimizer, T_max=200)
```

| 스케줄러 | 동작 | 언제 |
|---|---|---|
| `StepLR` | 30에폭마다 0.1배 | 계획대로 줄일 때 |
| `ReduceLROnPlateau` | 검증 손실이 10에폭 정체되면 0.1배 | **손실을 보고 반응** |
| `CosineAnnealingLR` | 코사인 곡선으로 부드럽게 0까지 | 트랜스포머·최신 모델 |

영상처리 4장에서 케라스로 본 것과 같은 개념이다. 이름까지 거의 같다.

**호출 위치가 하나만 다르다.**

```python
scheduler.step()              # StepLR, CosineAnnealingLR — 인자 없음
scheduler.step(val_loss)      # ReduceLROnPlateau — 검증 손실을 넘겨야 한다
```

`ReduceLROnPlateau`에 인자를 안 넘기면 에러가 난다. 자주 걸리는 함정이다.

`weight_decay=5e-4`는 L2 규제다. 옵티마이저에 직접 넣는다.

### 가중치 초기화

```python
def initializer_weight(model, init_type='kaiming'):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

initializer_weight(demo_model, 'kaiming')
```

| 방법 | 맞는 활성화 |
|---|---|
| **Kaiming (He)** | ReLU 계열 — 음수를 버리는 만큼 분산을 키워 준다 |
| **Xavier (Glorot)** | Sigmoid, Tanh |

**ReLU를 쓴다면 Kaiming이 맞다.** 요즘 파이토치는 기본값이 이미 Kaiming 계열이라
직접 부를 일은 많지 않지만, 왜 초기화가 중요한지는 알아 둘 만하다.

`model.modules()`가 중첩된 것까지 **모든 하위 층을 재귀적으로** 돌려준다.
`isinstance`로 층 종류를 구분해 다르게 초기화한다.

BatchNorm은 가중치 1, 편향 0으로 둔다. 처음에는 **아무것도 바꾸지 않는 상태**로 시작하는 것이다.

참고: 함수 안에 오타가 있다

```python
nn.init.normal_(m.wiehgt, mean=0, std=0.01)     # wiehgt → weight
```

`init_type='normal'`로 부르고 `nn.Linear`를 만났을 때만 터진다.
기본값이 `'kaiming'`이라 지금까지는 드러나지 않았다.

---

## 이 장 정리

### 흐름

```
3-1 ~ 3-4   순전파 → 손실 → 역전파를 손으로 계산
   ↓
3-5 ~ 3-6   autograd 가 대신 미분한다
   ↓
3-7 ~ 3-10  활성화 함수, MNIST, 오분류 분석
   ↓
3-11 ~ 3-13 합성곱과 CNN
   ↓
3-14 ~ 3-15 ResNet, 스케줄러, 초기화
```

### 텐서 모양 규칙 — 파이토치 vs 케라스

| | 파이토치 | 케라스 |
|---|---|---|
| 이미지 배치 | `(N, C, H, W)` | `(N, H, W, C)` |
| 채널 위치 | **앞** | 뒤 |
| 평탄화 | `x.view(x.size(0), -1)` | `Flatten()` |

### 합성곱 출력 크기

```
출력 = (입력 - 커널 + 2 × 패딩) / 스트라이드 + 1
```

3×3 커널에 `padding=1`, `stride=1`이면 크기가 유지된다. 가장 많이 쓰는 조합이다.

### 모델 크기 비교

| 모델 | 파라미터 | 절 |
|---|---|---|
| CIFAR10_CNN | 1,305,066 | 3-13 |
| ResNet18 | 11,173,962 | 3-14 |

### 이 장에서 나온 층

| 층 | 인자의 의미 | 주의 |
|---|---|---|
| `nn.Conv2d(in, out, k, s, p)` | 입력 채널, 출력 채널 | 채널이 먼저 |
| `nn.BatchNorm2d(n)` | **채널 수** | 앞 Conv의 out과 같아야 함 |
| `nn.MaxPool2d(2, 2)` | 커널, 스트라이드 | 크기 절반 |
| `nn.Dropout2d(p)` | 끄는 비율 | **채널 통째로** 끈다 |
| `nn.Linear(in, out)` | 입력 크기 계산 필요 | shape 에러의 주범 |
| `nn.AdaptiveAvgPool2d((1,1))` | 목표 크기 | 입력 크기와 무관 |

### 자주 틀리는 것

- `__init__`에서 `self.layer = nn.Xxx()` — **`=`를 확인한다.** `*`나 `+`면 AttributeError
- 옵티마이저에 **학습시킬 그 모델**의 `parameters()`를 넘긴다
- `nn.Linear`의 입력 크기를 직접 계산한다. 틀리면 shape 에러
- `BatchNorm2d`의 인자는 채널 수다. 배치 크기가 아니다
- 평가 전에 `model.eval()`, 그리고 `torch.no_grad()`
- 증강은 **학습 데이터에만** 건다
- `ToTensor()`가 `Normalize()`보다 먼저 와야 한다
- `ReduceLROnPlateau`는 `scheduler.step(val_loss)`로 값을 넘겨야 한다
- 정규화에 쓴 평균·표준편차로 **역정규화**해야 이미지가 제대로 보인다
