# 17-KDT

KDT 17기 과정에서 실습한 노트북과 프로젝트를 한 저장소로 모았습니다.
파이썬 기초 → 데이터 분석·머신러닝 → 딥러닝(이미지·자연어) → 웹 백엔드·배포 → 팀 프로젝트 순서로 진행했습니다.

## 폴더 안내

### 파이썬 · 데이터 분석

| 폴더 | 내용 | 주요 기술 |
|---|---|---|
| [17-data](./17-data/) | 파이썬 기초 문법 (변수, 자료형, 조건문, 반복문, FizzBuzz 등) | Python |
| [17-data-analyze](./17-data-analyze/) | NumPy·Pandas 데이터 처리, 시각화, 사이킷런 머신러닝(분류·회귀·차원축소·군집화), 통계 실습 | NumPy, Pandas, Matplotlib, Seaborn, Folium, scikit-learn, XGBoost, LightGBM, Hyperopt |
| [17-laser](./17-laser/) | 정밀가공 레이저 센서 신호로 이상을 탐지하는 미니 프로젝트 | Pandas, scikit-learn, TensorFlow |

### 딥러닝

| 폴더 | 내용 | 주요 기술 |
|---|---|---|
| [17-image](./17-image/) | 컴퓨터 비전 — OpenCV 영상처리, CNN·비전 트랜스포머, 생성 모델, 객체 검출 | TensorFlow, Keras, OpenCV, TF Hub, TF Object Detection API |
| [17-deep-learning](./17-deep-learning/) | 파이토치 딥러닝 — CNN, 전이학습, 의료 영상 분류, U-Net 분할, RNN·LSTM, 트랜스포머, 강화학습 | PyTorch, torchvision, OpenCV, Hugging Face Transformers, Gym |
| [17-pytorch](./17-pytorch/) | 파이토치 자연어 처리 — 토큰화, 임베딩, RNN, 트랜스포머, 사전학습 모델 미세조정 | PyTorch, Hugging Face Transformers·Tokenizers, KoNLPy, Korpora, Gensim, SentencePiece, spaCy |
| [17-codebot](./17-codebot/) | 코드 생성 언어모델을 처음부터 만들기 — 문자·바이트·BPE 토크나이저 | PyTorch |

### 웹 백엔드 · 배포

| 폴더 | 내용 | 주요 기술 |
|---|---|---|
| [17-job-scrapper](./17-job-scrapper/) | 인크루트 채용공고를 스크래핑해 보여 주는 웹앱 | Flask, Requests, BeautifulSoup |
| [17-assignment](./17-assignment/) | TJ·금영 노래방 인기차트 크롤러와 웹 페이지 | Flask, BeautifulSoup, APScheduler |
| [17-apartment](./17-apartment/) | 국토교통부 아파트 실거래가 공공 API 조회 웹앱 | Flask, 공공데이터 API |
| [17-fast-api](./17-fast-api/) | FastAPI 기초 — 라우팅, HTTP 메서드, Pydantic, 응답 모델, 예외 처리, 템플릿 | FastAPI, Pydantic, Jinja2, uv |
| [17-todos](./17-todos/) | 할 일 관리 앱을 단계별로 확장 — DB 연동, 로그인, 프런트엔드, 외래키, MVC 구조 | FastAPI, SQLAlchemy, bcrypt, Jinja2 |
| [17-aws-docker](./17-aws-docker/) | 천안시 의료취약지역 지도를 FastAPI로 옮겨 Docker·Nginx로 배포 | FastAPI, Docker, Docker Compose, Nginx |

### 팀 프로젝트

| 폴더 | 내용 | 주요 기술 |
|---|---|---|
| [17-team4](./17-team4/) | 천안시 읍·면·동 의료취약지역 분석과 지도 대시보드, 대기오염과 질환 관계 분석 | Pandas, statsmodels, pyGAM, scikit-learn, FastAPI, Next.js·TypeScript |
| [17-2조](./17-2조/) | 백혈구(WBC) 현미경 이미지 4종 분류 — 모델·증강·학습률 비교 실험과 가설검정 | PyTorch, torchvision, timm, scikit-learn, SciPy |
| [pro4_team3_최종](./pro4_team3_최종/pro4_team3_최종/) | 한국어 맞춤법 교정 모델 — 말뭉치 정제, T5·KoBART 비교, 디코딩·오류 분석 | PyTorch, Hugging Face Transformers, SentencePiece |

> `17-machine_learning` 폴더에는 강의 슬라이드만 있어 저장소에 포함하지 않았습니다.
> 머신러닝 실습 노트북은 [17-data-analyze/ML](./17-data-analyze/ML/)에 있습니다.

## 저장소에 포함하지 않은 파일

용량이나 라이선스 문제로 데이터셋, 모델 가중치, 가상환경(`.venv`)은 올리지 않았습니다.
노트북을 다시 실행하려면 아래 데이터를 받아 **표에 적힌 경로**에 넣어 주세요.
모델 가중치(`.pt`, `.h5` 등)는 노트북을 실행하면 다시 만들어집니다.

### 데이터셋

| 사용하는 곳 | 데이터 | 넣을 경로 | 받는 곳 |
|---|---|---|---|
| 17-2조 | Blood Cell Images (BCCD 포함) | `17-2조/dataset-master/`, `17-2조/dataset2-master/` | [Kaggle: paultimothymooney/blood-cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) |
| 17-deep-learning | MNIST, CIFAR-10 | `17-deep-learning/data/` | 노트북 실행 시 torchvision이 자동으로 내려받음 |
| 17-deep-learning | Chest X-Ray (폐렴) | `17-deep-learning/chest_xray/` | [Kaggle: paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| 17-deep-learning | Cityscapes Image Pairs | `17-deep-learning/cityscapes/` | [Kaggle: dansbecker/cityscapes-image-pairs](https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs) |
| 17-deep-learning | Helmet Detection | `17-deep-learning/helmet/` | [Kaggle: andrewmvd/helmet-detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) |
| 17-image | PASCAL VOC 2007 | `17-image/VOCtrainval_2007/`, `17-image/VOCtest_2007/` | [VOCtrainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar), [VOCtest](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) |
| 17-data-analyze | Credit Card Fraud | `17-data-analyze/ML/data/creditcard.csv` | [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| 17-data-analyze | Santander Customer Satisfaction | `17-data-analyze/ML/data/santander_*.csv` | [Kaggle 대회](https://www.kaggle.com/competitions/santander-customer-satisfaction) |
| 17-data-analyze | Human Activity Recognition | `17-data-analyze/ML/data/human_activity_*.csv` | [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| 17-data-analyze | Online Retail | `17-data-analyze/ML/data/Online_Retail.xlsx` | [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) |
| 17-data-analyze | Default of Credit Card Clients | `17-data-analyze/ML/data/pca_credit_card.xls` | [UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| 17-pytorch | 국민청원 말뭉치 (`corpus.txt`) | `17-pytorch/dataset/corpus.txt` | `02.ipynb`에서 Korpora `korean_petitions`로 받아 자동 생성 |
| 17-codebot | `tiny_codes.txt` (파이썬 코드 예제 말뭉치) | `17-codebot/tiny_codes.txt` | [Hugging Face: flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k) (MIT) — `output` 열에서 코드블록 표시(`` ```python ``)를 지우고 샘플 사이를 `<\|endoftext\|>`로 이어 붙인 파일 |
| 17-laser | 정밀가공 자원최적화 AI 데이터셋 | `17-laser/data/normal.xlsx`, `anomaly.xlsx` | [KAMP 인공지능 중소벤처 제조 플랫폼](https://www.kamp-ai.kr/) |
| 17-team4 | 전국 버스정류장 위치정보 | `17-team4/Project/data/raw/` | [공공데이터포털](https://www.data.go.kr/) — "국토교통부_전국 버스정류장 위치정보" 검색 |
| pro4_team3_최종 | 맞춤법 교정 말뭉치 2022 | 노트북의 `DATA_DIR` | [국립국어원 모두의 말뭉치](https://kli.korean.go.kr/corpus/main/requestMain.do) (이용 신청 필요) |

### 그 밖에

- **한글 폰트:** `17-data-analyze/04_시각화도구`의 노트북은 `data/malgun.ttf`를 사용합니다. `C:\Windows\Fonts\malgun.ttf`를 해당 폴더에 복사해 주세요.
- **`.env` 설정:** 비밀값이 들어 있어 올리지 않았습니다. 직접 만들어 주세요.
  - `17-apartment/.env`: `SERVICEKEY` (공공데이터포털에서 발급받은 API 키)
  - `17-aws-docker/.env`, `17-todos/*/.env`: `DATABASE_URL` (연결할 데이터베이스 주소)
- **외부 저장소:** `17-image`의 객체 검출 실습은 [tensorflow/models](https://github.com/tensorflow/models)와 [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)을 `17-image/` 아래에 clone해서 사용합니다.
