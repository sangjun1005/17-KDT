"""
torch 2.7 에서는 진짜 torchtext(개발 종료, torch<=2.3 까지만 동작)를 import 할 수 없어서,
노트북에서 필요한 최소 기능(Multi30k 데이터셋)만 흉내내는 대체 모듈입니다.
"""

from . import datasets

__version__ = "0.0.0-compat"

__all__ = ["datasets"]
