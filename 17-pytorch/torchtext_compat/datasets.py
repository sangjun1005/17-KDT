"""
torchtext.datasets.Multi30k 의 최소 대체 구현.

원본 torchtext 가 참조하던 다운로드 URL(구 S3 버킷)이 죽어서 실제 torchtext 로는
Multi30k 를 받을 수 없습니다. 같은 데이터를 미러링하고 있는
https://github.com/multi30k/dataset 저장소에서 원문 텍스트 파일을 받아
(source_sentence, target_sentence) 튜플의 리스트로 돌려줍니다.

사용법은 원본과 동일합니다:

    from torchtext_compat.datasets import Multi30k
    train_iter = Multi30k(split="train", language_pair=("de", "en"))
    first = next(iter(train_iter))
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

_BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"

# split 이름 -> {언어: 원본 파일명}
_FILES = {
    "train": {"de": "train.de.gz", "en": "train.en.gz"},
    "valid": {"de": "val.de.gz", "en": "val.en.gz"},
    "test": {"de": "test_2016_flickr.de.gz", "en": "test_2016_flickr.en.gz"},
}

_CACHE_DIR = Path(__file__).resolve().parent.parent / "dataset" / "multi30k"


def _cached_gz_path(split: str, lang: str) -> Path:
    filename = _FILES[split][lang]
    return _CACHE_DIR / filename


def _download(split: str, lang: str) -> Path:
    dest = _cached_gz_path(split, lang)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _BASE_URL + _FILES[split][lang]
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.rename(dest)
    return dest


def _load_lines(split: str, lang: str) -> List[str]:
    gz_path = _download(split, lang)
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _build_split(split: str, language_pair: Sequence[str]) -> List[Tuple[str, str]]:
    if split not in _FILES:
        raise ValueError(f"알 수 없는 split: {split!r} (지원: {sorted(_FILES)})")
    src_lang, trg_lang = language_pair
    src_lines = _load_lines(split, src_lang)
    trg_lines = _load_lines(split, trg_lang)
    if len(src_lines) != len(trg_lines):
        raise RuntimeError(
            f"{split} split 의 {src_lang}/{trg_lang} 문장 수가 다릅니다: "
            f"{len(src_lines)} vs {len(trg_lines)}"
        )
    return list(zip(src_lines, trg_lines))


def Multi30k(
    root: Union[str, Path, None] = None,
    split: Union[str, Sequence[str]] = ("train", "valid", "test"),
    language_pair: Tuple[str, str] = ("de", "en"),
) -> Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]]:
    """torchtext.datasets.Multi30k 호환 함수.

    반환값은 (source_sentence, target_sentence) 튜플들의 리스트(iterable)입니다.
    split 이 문자열이면 해당 split 하나만, 리스트/튜플이면 split 별 리스트를 순서대로 담은
    리스트를 반환합니다(원본 torchtext 와 동일한 언패킹 방식 지원).
    """
    if isinstance(split, str):
        return _build_split(split, language_pair)
    return [_build_split(s, language_pair) for s in split]
