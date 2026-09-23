# 6. 시맨틱 분할 — U-Net으로 도로 장면 나누기

**실습 파일**: `6.ipynb`

5장까지는 **사진 한 장에 답 하나**였다. 폐렴이냐 아니냐.
6장부터는 **픽셀 하나하나에 답이 있다.** 이 픽셀은 도로, 저 픽셀은 사람.

| 절 | 내용 |
|---|---|
| 6-1 | 분할이란 무엇인가 — 분류와 무엇이 다른가 |
| 6-2 | Cityscapes 데이터 — 한 장에 사진과 정답이 붙어 있다 |
| 6-3 | 색 마스크를 클래스 번호로 바꾸기 |
| 6-4 | U-Net 구조 — 줄였다가 다시 키운다 |
| 6-5 | 학습과 픽셀 정확도 |
| 6-6 | mIoU — 분할의 진짜 점수 |

---

## 6-1. 분할이란 무엇인가

### 출력 모양이 달라진다

```
분류 (1~5장)     입력 (3, 128, 128)  →  출력 (클래스 수,)     ← 벡터 하나
분할 (6장)       입력 (3, 128, 128)  →  출력 (클래스 수, 128, 128)  ← 픽셀마다 벡터
```

**손실 함수는 똑같이 `CrossEntropyLoss`를 쓴다.**
PyTorch의 `CrossEntropyLoss`는 `(N, C, H, W)` 로짓과 `(N, H, W)` 정답을 받으면
알아서 픽셀마다 교차 엔트로피를 계산하고 평균낸다. 따로 손댈 게 없다.

<table fit-page-width="true" header-row="true">
<tr><td></td><td>분류</td><td>분할</td></tr>
<tr><td>로짓 모양</td><td>(N, C)</td><td>(N, C, H, W)</td></tr>
<tr><td>정답 모양</td><td>(N,)</td><td>(N, H, W)</td></tr>
<tr><td>정답 dtype</td><td>long</td><td>long</td></tr>
<tr><td>예측</td><td>argmax(dim=1)</td><td>argmax(dim=1) → (N, H, W)</td></tr>
</table>

**정답 마스크는 정수 라벨 맵이다.** 색 이미지가 아니다.
그래서 6-3에서 색 → 번호 변환이 필요하다.

### 왜 일반 CNN으로는 안 되나

분류 CNN은 풀링으로 계속 크기를 줄인다. 128 → 64 → 32 → 16 → 8.
마지막엔 공간 정보가 거의 없다. **"어디에" 있는지를 버리고 "무엇이" 있는지만 남긴다.**

분할은 "어디에"가 답이다. 그래서 **줄인 다음 다시 키우는** 구조가 필요하다.

---

## 6-2. Cityscapes 데이터

### 파일 구조

```python
train_dir = './cityscapes/train/'
val_dir   = './cityscapes/val/'

train_lists = sorted(glob(f'{train_dir}*.jpg'))
val_lists   = sorted(glob(f'{val_dir}*.jpg'))
print(f'train {len(train_lists)}장, val {len(val_lists)}장')
```

```
train 2975장, val 500장
```

**이 버전은 사진과 정답이 한 파일에 가로로 붙어 있다.**
512×256 이미지 한 장의 왼쪽 절반이 사진, 오른쪽 절반이 색칠된 정답 마스크다.

```python
def image_mask_split(filename, image_size):
    image_mask = Image.open(filename)
    image = image_mask.crop([0, 0, 256, 256])      # 왼쪽 = 사진
    mask  = image_mask.crop([256, 0, 512, 256])    # 오른쪽 = 마스크
    image = image.resize((image_size, image_size))
    mask  = mask.resize((image_size, image_size), Image.NEAREST)
    return np.array(image), np.array(mask)
```

**`Image.NEAREST`가 이 함수에서 가장 중요한 한 글자다.**

```
사진  : 기본 보간(BILINEAR) — 이웃 픽셀을 섞어서 부드럽게
마스크: NEAREST            — 절대 섞으면 안 된다
```

마스크를 보간하면 `(128,64,128)`(도로)과 `(70,70,70)`(건물) 사이에
**세상에 없는 중간 색**이 생긴다. 그 픽셀은 어느 클래스도 아니게 된다.

**라벨은 숫자가 아니라 기호다. 평균을 내면 안 된다.** 이 원칙은 분할 전체에서 통한다.

### 라벨 정의

Cityscapes 공식 라벨표를 `namedtuple`로 옮겨 적는다.

```python
from collections import namedtuple

Label = namedtuple('Label', ['name', 'id', 'trainId', 'category',
                             'categoryId', 'hasInstances',
                             'ignoreInEval', 'color'])

labels = [
    Label('unlabeled',  0, 255, 'void',   0, False, True,  (0,0,0)),
    # ...
    Label('road',       7,   0, 'ground', 1, False, False, (128,64,128)),
    Label('building',  11,   2, 'construction', 2, False, False, (70,70,70)),
    # ...
]

N_CLASSES = len(labels)      # 35
IMAGE_SIZE = 128
BATCH_SIZE = 16

id2color_arr = np.array([l.color for l in labels], dtype=np.uint8)
```

```
클래스 35개, 색상표 (35, 3)
```

**`namedtuple`은 필드 이름이 붙은 튜플이다.** `l[7]` 대신 `l.color`로 쓸 수 있다.
클래스를 만들 만큼은 아니지만 인덱스로 접근하기엔 헷갈릴 때 쓴다.

각 필드의 뜻은 이렇다.

<table fit-page-width="true" header-row="true">
<tr><td>필드</td><td>뜻</td></tr>
<tr><td>id</td><td>원본 라벨 번호 (0~34)</td></tr>
<tr><td>trainId</td><td>학습용으로 추린 번호 (0~18, 무시할 건 255)</td></tr>
<tr><td>ignoreInEval</td><td>평가에서 빼야 하는 클래스인가</td></tr>
<tr><td>color</td><td>마스크에 칠해진 RGB</td></tr>
</table>

### 주의 — `trainId`를 만들어 놓고 안 쓴다

Cityscapes를 제대로 쓰려면 **`trainId` 19개**로 학습한다.
`ignoreInEval=True`인 클래스(주차장, 철길, 터널, 가드레일 등)는
정답이 불안정해서 손실 계산에서 빼는 게 표준이다.

이 노트북은 `N_CLASSES = 35`, 즉 **원본 `id` 전부**로 학습한다.
그래서 6-6의 mIoU가 실제보다 낮게 나온다. 쓰지도 못할 클래스가 평균에 섞인다.

고치려면 이렇게 한다. (원본은 그대로 두었다.)

```python
id2train = np.array([l.trainId for l in labels], dtype=np.uint8)
mask_train = id2train[encode_mask(mask)]        # 0~18, 무시는 255
criterion = nn.CrossEntropyLoss(ignore_index=255)
```

**`ignore_index=255`가 핵심이다.** 그 값이 든 픽셀은 손실에서 통째로 빠진다.

---

## 6-3. 색 마스크를 클래스 번호로

### 가장 가까운 색을 찾는다

```python
def encode_mask(mask, colors=id2color_arr):
    diff = (mask[:, :, None, :].astype(np.int16)
            - colors[None, None, :, :].astype(np.int16))
    dist = np.linalg.norm(diff, axis=-1)
    return np.argmin(dist, axis=-1).astype(np.uint8)


def decode_mask(enc, colors=id2color_arr):
    return colors[enc]
```

브로드캐스팅으로 한 번에 계산한다. 모양을 따라가 보면 이렇다.

```
mask[:, :, None, :]        (H, W, 1, 3)
colors[None, None, :, :]   (1, 1, 35, 3)
        ↓  빼기 (브로드캐스팅)
diff                       (H, W, 35, 3)
        ↓  마지막 축으로 norm
dist                       (H, W, 35)     ← 픽셀마다 35개 색과의 거리
        ↓  argmin(axis=-1)
enc                        (H, W)         ← 가장 가까운 색의 번호
```

**`.astype(np.int16)`이 꼭 필요하다.** uint8끼리 빼면 음수가 255 쪽으로 되감긴다.
`50 - 70`이 `-20`이 아니라 `236`이 된다. 거리가 완전히 틀어진다.

**`decode_mask`는 그냥 팬시 인덱싱이다.** `colors[enc]`가 `(H, W)` → `(H, W, 3)`.
번호 배열의 각 원소를 색으로 치환한다. 되돌리기가 이렇게 한 줄이다.

### 결과

```python
train_images, train_masks_enc = build_dataset(train_lists, 'Train Dataset')
val_images,   val_masks_enc   = build_dataset(val_lists,   'Validation Dataset')

print(train_images.shape, train_images.dtype)
print(train_masks_enc.shape, train_masks_enc.dtype)
print(train_masks_enc.min(), train_masks_enc.max())
```

```
(2975, 128, 128, 3) uint8
(2975, 128, 128) uint8
0 33
```

**2,975장을 통째로 메모리에 올린다.** 128×128이라 가능한 선택이다.
`2975 × 128 × 128 × 3 ≈ 146MB` + 마스크 49MB. 원본 해상도였다면 불가능하다.

`tqdm`으로 진행률을 보여 주는데, 전처리가 한 번에 끝나므로
학습 중에는 디스크를 안 읽는다. 5장의 `ImageFolder`와 반대되는 전략이다.

<table fit-page-width="true" header-row="true">
<tr><td></td><td>미리 전부 로드 (6장)</td><td>매번 읽기 (5장 ImageFolder)</td></tr>
<tr><td>메모리</td><td>많이 씀</td><td>적게 씀</td></tr>
<tr><td>학습 속도</td><td>빠름</td><td>디스크 I/O가 병목</td></tr>
<tr><td>증강</td><td>매번 다르게 주기 어려움</td><td>자연스럽게 가능</td></tr>
<tr><td>쓸 수 있는 조건</td><td>데이터가 작을 때</td><td>항상</td></tr>
</table>

### 주의 — 최댓값이 34가 아니라 33이다

색상표에 **같은 색이 세 쌍 있다.**

<table fit-page-width="true" header-row="true">
<tr><td>색</td><td>해당 id</td><td>결과</td></tr>
<tr><td>(0, 0, 0)</td><td>0 unlabeled, 1 egovehicle, 2 rectificationborder, 3 outofroi, 4 static</td><td>전부 0으로</td></tr>
<tr><td>(153, 153, 153)</td><td>17 pole, 18 polegroup</td><td>전부 17로</td></tr>
<tr><td>(0, 0, 142)</td><td>26 car, 34 licenseplate</td><td>전부 26으로</td></tr>
</table>

`np.argmin`은 동점일 때 **가장 앞 인덱스**를 준다.
그래서 1, 2, 3, 4, 18, 34번은 **한 번도 나오지 않는다.**
고유한 색은 35개 중 **29개**뿐이고, 도달 가능한 최대 번호가 **33**이다.

출력의 `max = 33`이 이걸 그대로 보여 준다.
6-6에서 "평가된 클래스 수 29"가 나오는 것도 같은 이유다.

**이건 버그라기보다 이 방식의 한계다.** 색으로 라벨을 복원하는 건 원래 손실이 있다.
Cityscapes 공식 배포본은 `*_labelIds.png`라는 **번호가 직접 들어 있는 파일**을 준다.
그쪽을 쓰면 이 문제가 없다.

### 주의 — 시각화 셀의 인덱스

```python
for i in range(2):
    for j, (arr, title) in enumerate([...]):
        plt.subplot(2, 3, 1 * 3 + j + 1)      # ← i가 아니라 1
```

`i * 3`이어야 하는데 `1 * 3`으로 되어 있다.
두 번 도는 루프가 **같은 자리(4~6번 칸)에 두 번 그린다.**
첫째 줄은 비고 둘째 줄에 마지막 이미지만 남는다.

---

## 6-4. U-Net 구조

### 전체 모양

```
입력 (3, 128, 128)
  │
  ├─ enc1 ──f1(64,128,128)──────────────────────────┐
  │   ↓ p1 (64, 64, 64)                             │
  ├─ enc2 ──f2(128,64,64)────────────────────┐      │
  │   ↓ p2 (128, 32, 32)                     │      │
  ├─ enc3 ──f3(256,32,32)─────────────┐      │      │
  │   ↓ p3 (256, 16, 16)              │      │      │
  ├─ enc4 ──f4(512,16,16)──────┐      │      │      │
  │   ↓ p4 (512, 8, 8)         │      │      │      │
  └─ bottleneck (1024, 8, 8)   │      │      │      │
       ↓ dec1 ←────────────────┘      │      │      │
      (512, 16, 16)                   │      │      │
       ↓ dec2 ←───────────────────────┘      │      │
      (256, 32, 32)                          │      │
       ↓ dec3 ←──────────────────────────────┘      │
      (128, 64, 64)                                 │
       ↓ dec4 ←─────────────────────────────────────┘
      (64, 128, 128)
       ↓ 1×1 conv
    출력 (35, 128, 128)
```

**U자로 생겨서 U-Net이다.** 왼쪽 내리막이 인코더, 오른쪽 오르막이 디코더,
가로로 건너가는 화살표가 **스킵 연결(skip connection)**이다.

### 기본 블록

```python
def conv2d_block(in_ch, out_ch, kernel_size=3):
    layers = []
    for i in range(2):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch,
                      kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)
```

**합성곱 두 번이 한 세트다.** U-Net의 원 논문부터 그렇다.
`padding = kernel_size // 2`라서 크기가 안 변한다. 3//2 = 1.

`nn.Sequential(*layers)`의 `*`는 리스트를 풀어서 인자로 넘기는 것이다.
`nn.Sequential(layers)`로 쓰면 에러가 난다.

### 인코더 — 두 개를 돌려준다

```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.conv = conv2d_block(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        f = self.conv(x)                 # 스킵용 (크기 그대로)
        p = self.drop(self.pool(f))      # 다음 단계용 (반으로)
        return f, p
```

**`forward`가 튜플을 돌려주는 게 이 장에서 처음 보는 패턴이다.**
`f`는 옆으로(스킵), `p`는 아래로(다음 인코더). 두 갈래로 나간다.

**풀링 전의 `f`를 보내는 게 핵심이다.** 풀링 후를 보내면 스킵의 의미가 없다.

### 디코더 — 키우고 이어 붙인다

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, dropout=0.3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3,
                                     stride=2, padding=1, output_padding=1)
        self.drop = nn.Dropout(dropout)
        self.conv = conv2d_block(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        u = self.up(x)
        c = torch.cat([u, skip], dim=1)      # 채널 방향으로 붙임
        return self.conv(self.drop(c))
```

**`nn.ConvTranspose2d`가 이 장의 새 층이다.**
합성곱이 크기를 줄인다면, 전치 합성곱은 **학습 가능한 방식으로 키운다.**

```
stride=2, padding=1, output_padding=1, kernel=3
  →  출력 크기 = (입력 - 1) × 2 - 2×1 + 3 + 1 = 입력 × 2
```

`output_padding=1`이 없으면 `입력 × 2 - 1`이 나와서 홀수가 된다. 스킵과 안 맞는다.

<table fit-page-width="true" header-row="true">
<tr><td>키우는 방법</td><td>학습 파라미터</td><td>특징</td></tr>
<tr><td>nn.ConvTranspose2d</td><td>있음</td><td>표현력 좋음. 체커보드 무늬가 생길 수 있음</td></tr>
<tr><td>nn.Upsample + Conv2d</td><td>Conv에만</td><td>무늬 없음. 최근에 더 많이 씀</td></tr>
</table>

**`torch.cat([u, skip], dim=1)`의 `dim=1`은 채널 축이다.**
`(N, C, H, W)`에서 1번이 C다. `u`가 512채널, `skip`이 512채널이면 1024채널이 된다.
그래서 뒤의 `conv2d_block`이 `out_ch + skip_ch`를 입력으로 받는다.

**여기서 H, W가 안 맞으면 바로 에러가 난다.** 스킵 연결에서 제일 흔한 실수다.
`output_padding`을 넣은 이유가 이것이다.

### 스킵 연결이 왜 필요한가

```
풀링 4번 : 128 → 64 → 32 → 16 → 8
```

8×8까지 줄어들면 "이 사진에 도로가 있다"는 알아도
**"도로의 경계가 정확히 어느 픽셀인지"는 이미 버려졌다.**

스킵 연결은 인코더의 **고해상도 특징맵을 디코더에 직접 전달한다.**

```
디코더가 받는 것 = 깊은 층의 "무엇인가"(의미) + 얕은 층의 "어디인가"(위치)
```

**이게 U-Net이 경계를 잘 그리는 이유다.** ResNet의 잔차 연결과는 목적이 다르다.

<table fit-page-width="true" header-row="true">
<tr><td></td><td>ResNet 잔차 연결</td><td>U-Net 스킵 연결</td></tr>
<tr><td>합치는 방법</td><td>더하기 (+)</td><td>이어 붙이기 (cat)</td></tr>
<tr><td>거리</td><td>바로 옆 블록</td><td>인코더 ↔ 디코더 (멀다)</td></tr>
<tr><td>목적</td><td>기울기 소실 방지</td><td>위치 정보 복원</td></tr>
</table>

### 가중치 초기화

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

**`self.modules()`는 중첩된 층까지 전부 훑는다.** (`children()`은 바로 아래만)

**He 초기화(`kaiming_normal_`)는 ReLU에 맞춘 초기화다.**
ReLU가 음수를 0으로 만들어 분산이 반으로 줄어드는 걸 보정한다.
층이 깊을수록 차이가 커진다. 3,400만 파라미터 34층 모델이라 효과가 있다.

### 파라미터

```python
model = UNet()
out = model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
print(tuple(out.shape))
print(sum(p.numel() for p in model.parameters()))
```

```
(1, 35, 128, 128)
34515555
```

**3,451만 개.** 5장의 ResNetV2-50(2,376만)보다 크다.
사전학습 가중치 없이 **밑바닥부터** 학습한다는 점이 5장과 결정적으로 다르다.

출력이 `(1, 35, 128, 128)` — 입력과 같은 해상도에 클래스 축이 붙었다.
**모양이 맞는지 먼저 확인하는 습관**은 분할에서 특히 중요하다.

---

## 6-5. 학습과 픽셀 정확도

### 평가 함수

```python
@torch.no_grad()
def evaluate_seg(model, loader, criterion):
    model.eval()
    tot_loss, tot_correct, tot_pixels = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        tot_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        tot_correct += (preds == yb).sum().item()
        tot_pixels += yb.numel()

    n = len(loader.dataset)
    return tot_loss / n, tot_correct / tot_pixels
```

**정확도의 분모가 `yb.numel()`, 즉 픽셀 수다.** 이미지 수가 아니다.
배치 16장 × 128 × 128 = 262,144픽셀이 한 번에 더해진다.

`logits.argmax(dim=1)`은 `(N, 35, H, W)` → `(N, H, W)`.
클래스 축을 따라 가장 큰 걸 고르면 모양이 정답과 같아진다.

### 학습

```python
EPOCHS = 2
model = UNet()
history = train_unet(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3)
```

```
1/2 loss 1.6619 acc 0.5514 val_loss 1.1371 val_acc 0.6866 저장
2/2 loss 1.0086 acc 0.7256 val_loss 0.9726 val_acc 0.7405 저장
최저 0.9726: cityscapes-unet.pt
```

**시작 손실이 얼마여야 정상인지 계산해 보자.**
35개 클래스를 균등하게 찍으면 `ln(35) = 3.56`이다.
1에폭 **평균**이 1.66이니 이미 학습이 되고 있다는 뜻이다.

(5장까지 여러 번 나온 확인법이다. 손실이 `ln(클래스 수)` 근처에서
안 내려가면 학습이 전혀 안 되고 있는 것이다.)

**2에폭은 너무 적다.** 손실이 계속 내려가는 중이고 과적합 기미도 없다.
val_acc 0.74는 아직 한참 개선될 수 있는 값이다.

### 픽셀 정확도의 함정

`val_acc 0.7405`는 좋아 보이는데, **도로와 하늘과 건물만 맞혀도 나오는 수치다.**

도시 사진에서 픽셀이 차지하는 면적은 대략 이렇다.

```
도로 + 건물 + 하늘 + 식생  ≈ 전체 픽셀의 70~80%
사람, 자전거, 신호등        ≈ 전체 픽셀의 1% 미만
```

**면적이 큰 클래스가 정확도를 독점한다.**
자율주행에서 제일 중요한 사람과 자전거를 통째로 놓쳐도 0.74가 나온다.

이건 5장의 "폐렴 74.3%라 전부 폐렴이라 찍어도 정확도 0.743" 과 같은 문제다.
**클래스 불균형이 픽셀 단위로 벌어진 것이다.**

그래서 분할은 정확도를 쓰지 않는다.

---

## 6-6. mIoU — 분할의 진짜 점수

### IoU란

```
        예측 ∩ 정답        겹친 픽셀
IoU  =  ───────────  =  ─────────────
        예측 ∪ 정답     둘 중 하나라도 해당되는 픽셀
```

**클래스마다 따로 계산하고, 그걸 평균낸 게 mIoU(mean IoU)다.**

면적이 작아도 자기 클래스 안에서만 비교하므로
**"사람"의 IoU가 낮으면 평균이 바로 내려간다.** 정확도와 결정적으로 다른 점이다.

<table fit-page-width="true" header-row="true">
<tr><td>지표</td><td>분모</td><td>작은 클래스</td></tr>
<tr><td>픽셀 정확도</td><td>전체 픽셀</td><td>묻힌다</td></tr>
<tr><td>mIoU</td><td>클래스별로 따로</td><td>같은 무게로 반영된다</td></tr>
</table>

### 구현

```python
@torch.no_grad()
def compute_miou(model, loader, n_classes=N_CLASSES):
    model.eval()
    inter = torch.zeros(n_classes)
    union = torch.zeros(n_classes)

    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(dim=1).cpu()

        for c in range(n_classes):
            p = pred == c
            t = yb == c
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()

    present = (union > 0).nonzero().squeeze(dim=1)
    iou = inter[present] / union[present]
    return iou.mean().item(), present.numpy(), iou.numpy()
```

**배치마다 IoU를 구해서 평균내면 안 된다.** 교집합·합집합을 **누적**한 뒤 마지막에 나눈다.
배치마다 계산하면 그 배치에 없는 클래스가 0/0이 되어 평균이 망가진다.

**`p & t`, `p | t`는 불리언 텐서의 원소별 AND / OR다.**
`&`가 교집합, `|`이 합집합. 파이썬의 `and`/`or`가 아니다.

**`present`가 중요하다.** `union == 0`인 클래스는 **예측에도 정답에도 없다.**
나누면 `0/0 = nan`이 되어 평균이 전부 nan이 된다. 그래서 먼저 걸러낸다.

`(union > 0).nonzero().squeeze(dim=1)`는 조건을 만족하는 인덱스를 1차원으로 뽑는 관용구다.

### 결과

```
Pixel Accuracy: 0.7405
mIoU: 0.1549
평가된 클래스 수: 29
```

**픽셀 정확도 0.74 vs mIoU 0.15.** 다섯 배 가까이 차이가 난다.
6-5에서 말한 함정이 숫자로 드러난 것이다.

"평가된 클래스 수 29"는 6-3에서 본 그대로다.
색이 겹쳐서 실제로 나타날 수 있는 번호가 29개뿐이다.

### 클래스별로 뜯어보기

```python
names = [label.name for label in labels]
order = np.argsort(-iou_vals)         # 내림차순
```

```
잘 맞힌 클래스
road        0.7875
sky         0.7767
vegetation  0.6928
unlabeled   0.6530
building    0.5763

잘못 맞힌 클래스
railtrack    0.0
parking      0.0
dynamic      0.0
trafficsign  0.0
bicycle      0.0
```

**`np.argsort(-x)`가 내림차순 정렬 관용구다.** numpy에는 `reverse=True`가 없다.

읽어 보면 이야기가 분명하다.

<table fit-page-width="true" header-row="true">
<tr><td>잘 맞힌 것</td><td>공통점</td></tr>
<tr><td>road, sky, vegetation, building</td><td>면적이 크고 덩어리로 존재한다</td></tr>
</table>

<table fit-page-width="true" header-row="true">
<tr><td>0.0이 나온 것</td><td>이유</td></tr>
<tr><td>trafficsign, bicycle</td><td>너무 작아서 2에폭으로는 학습이 안 됨</td></tr>
<tr><td>railtrack, parking, dynamic</td><td>애초에 ignoreInEval=True — 빼야 할 클래스</td></tr>
</table>

**IoU 0.0은 그 클래스를 단 한 픽셀도 못 맞혔다는 뜻이다.**
`inter = 0`인데 `union > 0`이니 정답에는 있었다.

`railtrack`, `parking`, `dynamic`은 **`ignoreInEval=True`인 클래스**다.
6-2에서 말한 대로 `trainId`를 썼다면 이들은 평균에서 빠졌을 것이고,
mIoU 0.1549는 상당히 올라갔을 것이다.

### 그러니 0.1549를 어떻게 읽어야 하나

```
낮게 나온 이유 세 가지
  1. 2에폭밖에 안 돌렸다               ← 가장 큼
  2. ignoreInEval 클래스가 평균에 섞였다  ← trainId를 안 씀
  3. 색 충돌로 일부 클래스가 소실됐다     ← encode_mask의 한계
```

**모델 구조가 나쁜 게 아니다.** 참고로 Cityscapes에서 잘 만든 모델의 mIoU는 0.7~0.8대다.

### 주의 — 없는 Early Stopping

```python
# 에폭을 많이 돌리고 Early Stopping으로 저장한 Best Model을 불러온다.
model.load_state_dict(torch.load('cityscapes-unet.pt', map_location=device))
```

주석은 Early Stopping을 말하는데 `train_unet`에는 **그런 게 없다.**
검증 손실이 최저일 때 저장(`best_val`)은 하지만, 조기 종료는 안 한다.
2에폭이라 어차피 마지막이 최저였다.

### 주의 — 정규화도 증강도 없다

```python
img = self.images[i].astype(np.float32) / 255.0      # 0~1로만
```

5장에서는 ImageNet 평균·표준편차로 정규화했다. 여기는 안 한다.
사전학습 백본을 안 쓰니 **틀린 건 아니지만**, 수렴은 느려진다.

증강도 전혀 없다. 분할에서 증강을 넣으려면
**사진과 마스크에 똑같은 기하 변환을 적용**해야 한다 (좌우 반전, 크롭 등).
마스크에는 색상 변환(밝기, 대비)을 주면 안 된다. 이게 분할 증강의 까다로운 점이다.

---

## 이 장 정리

### 흐름

```
데이터           6-2   한 장에 사진+마스크가 붙어 있음, NEAREST로 리사이즈
   ↓
색 → 번호        6-3   가장 가까운 색 찾기, int16 캐스팅 필수
   ↓
U-Net           6-4   인코더/디코더 + 스킵 연결, ConvTranspose2d
   ↓
학습            6-5   CrossEntropyLoss 그대로, 픽셀 정확도
   ↓
mIoU            6-6   클래스별 IoU의 평균 — 이게 진짜 점수
```

### 분류와 분할 대조표

<table fit-page-width="true" header-row="true">
<tr><td></td><td>분류 (1~5장)</td><td>분할 (6장)</td></tr>
<tr><td>출력</td><td>(N, C)</td><td>(N, C, H, W)</td></tr>
<tr><td>정답</td><td>(N,)</td><td>(N, H, W)</td></tr>
<tr><td>손실</td><td>CrossEntropyLoss</td><td>CrossEntropyLoss (그대로)</td></tr>
<tr><td>구조</td><td>줄이기만</td><td>줄였다가 키우기</td></tr>
<tr><td>지표</td><td>정확도 / AUC</td><td>mIoU</td></tr>
<tr><td>리사이즈</td><td>BILINEAR</td><td>마스크는 NEAREST</td></tr>
</table>

### U-Net 조립 체크리스트

```python
# 1. 인코더는 두 개를 돌려준다 (스킵용 f, 다음 단계용 p)
def forward(self, x):
    f = self.conv(x)
    p = self.drop(self.pool(f))
    return f, p

# 2. 디코더는 키운 뒤 스킵과 채널 방향으로 합친다
u = self.up(x)
c = torch.cat([u, skip], dim=1)

# 3. 그래서 다음 conv의 입력 채널은 out_ch + skip_ch
self.conv = conv2d_block(out_ch + skip_ch, out_ch)
```

### 전치 합성곱 크기 공식

```
출력 = (입력 - 1) × stride - 2 × padding + kernel + output_padding
```

정확히 두 배로 키우려면 `stride=2, padding=1, kernel=3, output_padding=1`.

### 자주 틀리는 것

- 마스크 리사이즈에 **`Image.NEAREST`**를 안 쓴다 (없는 색이 생긴다)
- uint8끼리 빼서 음수가 되감긴다 → **`.astype(np.int16)`**
- `torch.cat`의 `dim`을 1(채널)이 아닌 다른 축으로 준다
- `output_padding`을 빼먹어 스킵과 크기가 안 맞는다
- IoU를 배치마다 평균낸다 → **교집합·합집합을 누적**한 뒤 마지막에 나눈다
- `union == 0`인 클래스를 안 걸러서 mIoU가 nan이 된다
- **픽셀 정확도만 보고 잘 됐다고 판단한다** → mIoU를 같이 본다
- 분할 증강에서 마스크에 색상 변환을 준다 (기하 변환만 같이 준다)
