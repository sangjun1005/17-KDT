# 17-pytorch — 파이토치로 배우는 자연어 처리

파이토치(PyTorch)와 허깅페이스로 **한국어·영어 자연어 처리**를 단계별로 실습한 노트북 모음입니다.
텐서 기초에서 시작해 토큰화 → 임베딩 → RNN → 트랜스포머 → 사전학습 모델 미세조정까지 이어집니다.

## 노트북 구성

| 파일 | 내용 | 주요 도구 |
|---|---|---|
| `01.ipynb` | 파이토치 기초 — 텐서, 자동미분, 학습 루프 | `torch` |
| `02.ipynb` | 한글 토큰화 — 자모 분해와 문자 단위 처리 | `jamo` |
| `03.ipynb` | 텍스트 벡터화 — TF-IDF와 임베딩 층 | `nltk`, `scikit-learn`, `torch.nn` |
| `03_1.ipynb` | 순환 신경망으로 문장 분류 | `torch.nn`, `pandas` |
| `03_2.ipynb` | 한국어 코퍼스 실습 — 형태소 분석기 연결 | `Korpora`, `konlpy(Okt)` |
| `04.ipynb` | 트랜스포머 — 위치 인코딩과 어텐션 | `torch.nn`, `torchtext` |
| `04_1.ipynb` | Multi30k 번역 데이터 어휘 사전 만들기 | `torchtext` |
| `04_2.ipynb` | 사전학습 생성 모델 다뤄 보기 | `transformers` (CausalLM) |
| `04_3.ipynb` | KoELECTRA 문장 분류 미세조정 | `transformers` (Electra) |
| `04_4.ipynb` | T5 시퀀스-투-시퀀스 미세조정 | `transformers` (T5), `datasets` |

`torchtext` / `torchtext_compat` 폴더는 torchtext가 개발 종료되면서
최신 파이토치에서도 예제가 돌아가도록 직접 맞춰 둔 호환 모듈입니다.

## 실행 환경

```bash
# Windows + RTX 50 시리즈 (CUDA 12.8)
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-windows.txt
```

| 파일 | 대상 |
|---|---|
| `requirements-windows.txt` | Windows + NVIDIA RTX 50 시리즈 |
| `requirements-macos.txt` | macOS (Apple Silicon) |
| `requirements-notebooks.txt` | 노트북 실행에만 필요한 추가 패키지 |

첫 셀에서 GPU가 잡히는지 먼저 확인합니다.

```python
import torch
print(torch.cuda.is_available(), torch.__version__)
```

## 저장소에 포함하지 않은 것

용량이 크거나 직접 내려받는 편이 나은 것들은 제외했습니다.

| 경로 | 크기 | 받는 방법 |
|---|---|---|
| `dataset/corpus.txt` | 약 517MB | `Korpora`로 내려받아 같은 경로에 저장 |
| `dataset/multi30k/` | 약 2MB | `torchtext.datasets.Multi30k`가 자동으로 내려받음 |
| `models/` | 약 1.9GB | 각 노트북을 실행하면 다시 생성됨 (미세조정 체크포인트) |
| `.venv/`, `.venv-notebooks/` | — | 위 설치 명령으로 다시 만듦 |

`dataset/binary.csv`, `dataset/non_linear.csv`는 크기가 작아 그대로 포함했습니다.
