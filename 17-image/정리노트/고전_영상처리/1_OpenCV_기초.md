# 1. OpenCV 기초 — 이미지 읽기와 기하 변환

**실습 파일**: `1.ipynb`

컴퓨터에게 이미지는 **숫자가 든 행렬**일 뿐이다.
이 장은 그 행렬을 읽고, 크기를 바꾸고, 돌리고, 잘라내는 기본기를 다룬다.

---

## 1-1. 이미지 읽기

### 개념

`cv2.imread()`는 이미지 파일을 넘파이 배열로 읽어온다.
**파일이 없거나 경로가 틀려도 에러를 내지 않고 `None`을 돌려준다.**
그래서 읽은 직후에 확인하는 습관이 중요하다.

### 핵심 코드

```python
import cv2

image = cv2.imread('like_lenna.png', cv2.IMREAD_GRAYSCALE)

if image is not None:
    print('read!')
else:
    print('not!')

print(type(image))   # <class 'numpy.ndarray'>
```

### 읽기 모드

| 플래그 | 값 | 결과 |
|---|---|---|
| `cv2.IMREAD_COLOR` | 1 | 컬러 3채널 (기본값). **BGR 순서** |
| `cv2.IMREAD_GRAYSCALE` | 0 | 흑백 1채널 |
| `cv2.IMREAD_UNCHANGED` | -1 | 알파 채널까지 그대로 |

### shape로 확인하기

```python
image.shape   # (224, 224)  ← 흑백은 (높이, 너비) 2차원
```

컬러로 읽으면 `(224, 224, 3)`이 된다. 순서는 **(높이, 너비, 채널)**이다.
가로세로가 아니라 **세로가 먼저**라는 점을 자주 헷갈린다.

### 주의: OpenCV는 BGR이다

`cv2.imread()`가 돌려주는 컬러 이미지는 RGB가 아니라 **BGR 순서**다.
matplotlib은 RGB로 그리기 때문에, 그냥 넘기면 **빨강과 파랑이 뒤바뀐 그림**이 나온다.

```python
image = cv2.imread('like_lenna.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # 반드시 변환

plt.imshow(image)
plt.axis('off')
plt.show()
```

흑백으로 읽었다면 채널이 하나뿐이라 이 문제가 없다.

---

## 1-2. 이미지 표시

```python
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')
plt.title('Like Lenna')
plt.axis('off')      # 축 눈금 제거
plt.show()
```

**`cmap='gray'`를 빼먹으면** 흑백 이미지가 노랑·보라 계열(matplotlib 기본 컬러맵 viridis)로 나온다.
1채널 이미지를 그릴 때는 항상 붙인다.

`plt.axis('off')`는 이미지 주변의 좌표축을 숨긴다. 이미지를 볼 때는 축이 필요 없다.

### 참고: `cv2_imshow`

```python
from google.colab.patches import cv2_imshow
cv2_imshow(image[50:150, 50:150])
```

이건 **구글 코랩 전용 함수**다. 로컬 주피터에서는 `google.colab` 모듈이 없어 임포트 에러가 난다.
(이 저장소에는 `tools/colab_shim`이라는 대체 패키지가 들어 있어서 로컬에서도 돌아간다.)

로컬에서 직접 쓸 거면 `plt.imshow()`로 바꾸는 게 안전하다.

---

## 1-3. 크기 조정 (resize)

```python
image_small = cv2.resize(image, (100, 100))
```

**인자 순서가 `(너비, 높이)`다.** `shape`의 `(높이, 너비)`와 **반대**라서 실수하기 쉽다.

| | 순서 |
|---|---|
| `image.shape` | (높이, 너비) |
| `cv2.resize(img, dsize)` | (너비, 높이) |

### 보간 방법 (interpolation)

크기를 바꾸면 원래 없던 픽셀을 만들어내야 한다. 그 방식을 정하는 게 `interpolation`이다.

| 값 | 특징 | 언제 |
|---|---|---|
| `cv2.INTER_LINEAR` | 기본값. 빠르고 무난 | 대부분 |
| `cv2.INTER_AREA` | 축소할 때 깨끗함 | **줄일 때** |
| `cv2.INTER_CUBIC` | 느리지만 부드러움 | **키울 때** |
| `cv2.INTER_NEAREST` | 계단 형태로 뭉개짐 | 세그멘테이션 마스크 등 값이 바뀌면 안 될 때 |

```python
small = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
```

---

## 1-4. 뒤집기 (flip)

```python
image_flip = cv2.flip(image, 0)    # 상하 반전
image_flip = cv2.flip(image, 1)    # 좌우 반전
image_flip = cv2.flip(image, -1)   # 상하 + 좌우 (180도 회전과 동일)
```

| flipCode | 축 | 결과 |
|---|---|---|
| `0` | x축 기준 | 위아래 뒤집기 |
| `1` (양수) | y축 기준 | 좌우 뒤집기 |
| `-1` (음수) | 양쪽 | 180도 돌린 것과 같음 |

### 언제 쓰나

**데이터 증강(augmentation)**에 가장 많이 쓴다.
사진 한 장을 좌우 반전하면 학습 데이터가 두 배가 된다.

단, **좌우 반전이 의미를 바꾸는 경우**에는 쓰면 안 된다.
글자 인식, 숫자 인식(2와 5), 좌/우 방향 표지판 같은 데이터가 그렇다.

---

## 1-5. 회전 (rotation)

### 개념

OpenCV의 회전은 두 단계다.

1. `getRotationMatrix2D()`로 **변환 행렬**을 만든다
2. `warpAffine()`으로 이미지에 그 행렬을 **적용**한다

### 핵심 코드

```python
height, width = image.shape

matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
res = cv2.warpAffine(image, matrix, (width, height))

plt.imshow(res, cmap='gray')
plt.axis('off')
plt.show()
```

### getRotationMatrix2D 파라미터

| 파라미터 | 의미 |
|---|---|
| `center` | 회전의 중심점. 보통 `(width/2, height/2)` |
| `angle` | 각도. **양수가 반시계 방향** |
| `scale` | 확대/축소 배율. 1이면 크기 유지 |

돌려주는 건 2×3 크기의 어파인 변환 행렬이다.

### warpAffine 파라미터

| 파라미터 | 의미 |
|---|---|
| `src` | 원본 이미지 |
| `M` | 변환 행렬 |
| `dsize` | 출력 크기 `(너비, 높이)` |
| `borderValue` | 빈 공간을 채울 값 (기본 0 = 검정) |

```python
matrix = cv2.getRotationMatrix2D((width/2, height/2), 30, 1)
res = cv2.warpAffine(image, matrix, matrix_size, borderValue=200)
```

### 주의: 45도 회전하면 모서리가 잘린다

출력 크기를 원본과 같게(`(width, height)`) 주면, 비스듬히 돌아간 이미지의 **네 귀퉁이가 화면 밖으로 나가 잘린다.**
빈 자리는 `borderValue`로 채워진다.

90도, 180도처럼 직각으로만 돌릴 거라면 전용 함수가 더 정확하고 빠르다.

```python
res = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
```

---

## 1-6. 자르기 (crop)

### 개념

OpenCV에 크롭 함수는 없다. **넘파이 슬라이싱을 그대로 쓴다.**

```python
image[y_start:y_end, x_start:x_end]
```

**세로(y)가 먼저**다. `shape`와 같은 순서다.

```python
plt.imshow(image[:100, :100], cmap='gray')      # 왼쪽 위 100×100
plt.imshow(image[50:100, 50:150], cmap='gray')  # y 50~100, x 50~150
```

두 번째 예시는 **세로 50, 가로 100** 크기다. 숫자만 보면 반대로 읽기 쉽다.

### 반드시 알아야 할 함정: 슬라이싱은 복사가 아니다

```python
croped_image = image[50:150, 50:150]
croped_image[:] = 255          # 잘라낸 부분만 흰색으로?

plt.imshow(image, cmap='gray')  # ← 원본에 흰 사각형이 생겨 있다
```

넘파이 슬라이싱은 **뷰(view)**를 돌려준다. 새 배열을 만드는 게 아니라
원본의 그 영역을 가리키는 창을 하나 여는 것이다.
그래서 잘라낸 쪽을 고치면 **원본이 같이 바뀐다.**

원본을 지키려면 `.copy()`를 붙인다.

```python
croped_image = image[50:150, 50:150].copy()
croped_image[:] = 255           # 이제 원본은 그대로
```

> 이 실습에서는 원본이 바뀌는 걸 눈으로 보여주려고 일부러 `.copy()`를 뺐다.
> 실제 코드에서는 잘라낸 이미지를 수정할 계획이라면 항상 `.copy()`를 붙인다.

### 언제 쓰나

- 관심 영역(ROI)만 떼어내 처리할 때
- 랜덤 크롭으로 데이터 증강할 때
- 객체 검출 결과의 박스 영역만 잘라낼 때

---

## 이 장 정리

| 작업 | 함수 | 주의점 |
|---|---|---|
| 읽기 | `cv2.imread(path, flag)` | 실패해도 에러 대신 `None` |
| 색 순서 | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | **OpenCV는 BGR** |
| 표시 | `plt.imshow(img, cmap='gray')` | 1채널이면 `cmap` 필수 |
| 크기 | `cv2.resize(img, (w, h))` | **(너비, 높이) 순서** |
| 뒤집기 | `cv2.flip(img, 0/1/-1)` | 의미가 바뀌는 데이터엔 금지 |
| 회전 | `getRotationMatrix2D` + `warpAffine` | 모서리 잘림, `borderValue` |
| 자르기 | `img[y1:y2, x1:x2]` | **뷰다. 수정하려면 `.copy()`** |

가장 많이 실수하는 세 가지

1. `shape`는 (높이, 너비), `resize`는 (너비, 높이) — **순서가 반대**
2. OpenCV는 BGR, matplotlib은 RGB — **변환 없이 그리면 색이 뒤집힌다**
3. 슬라이싱은 뷰 — **원본이 같이 바뀐다**
