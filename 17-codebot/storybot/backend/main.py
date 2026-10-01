import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictStr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 기존 스토리봇 코드를 그대로 사용한다.
from model import GPT
from tokenizer import BPETokenizer
from utils import generate, get_device
from backend import auth, stories
from backend.auth import current_user
from backend.storage import database, initialize


DATA_DIR = Path(os.environ.get("STORYBOT_DATA_DIR", ROOT / "backend" / "data"))
MODEL_PATH = ROOT / "model_pretrain.pt"
TOKENIZER_PATH = ROOT / "merge_rules.pkl"
MAX_INPUT_TOKENS = 56
MAX_NEW_TOKENS = 200  # 56 + 200 = 256 = max_context_len
TEMPERATURE = 1.0
END_TOKEN = "<|endoftext|>"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = BPETokenizer.load_from(TOKENIZER_PATH)
    model = GPT.load_from(MODEL_PATH, device=get_device())
    if tokenizer.vocab_size != model.vocab_size:
        raise RuntimeError("모델과 토크나이저의 어휘 크기가 일치하지 않습니다.")
    if MAX_INPUT_TOKENS + MAX_NEW_TOKENS > model.max_context_len:
        raise RuntimeError("입력 + 생성 토큰 수가 모델의 최대 문맥 길이를 넘습니다.")
    model.eval()
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.model_error = (
        "스토리봇 모델 가중치에 NaN 또는 무한대 값이 있어 생성할 수 없습니다."
        if any(not p.isfinite().all().item() for p in model.parameters()) else None
    )
    app.state.generation_lock = Lock()
    app.state.db_path, app.state.jwt_key = initialize(DATA_DIR)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(auth.router)
app.include_router(stories.router)


@app.middleware("http")
async def check_origin(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        allowed = {str(request.base_url).rstrip("/"), "http://127.0.0.1:5173", "http://localhost:5173"}
        if (origin and origin not in allowed) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "허용되지 않은 요청 출처입니다."}, status_code=403)
    return await call_next(request)


class StoryRequest(BaseModel):
    prompt: StrictStr


@app.post("/api/stories/generate", status_code=201)
def generate_story(payload: StoryRequest, request: Request, user: dict = Depends(current_user)):
    prompt = payload.prompt
    if not prompt.strip():
        raise HTTPException(400, "시작 문장을 입력해주세요.")
    if END_TOKEN in prompt:
        raise HTTPException(400, f"입력에 사용할 수 없는 문자열({END_TOKEN})이 포함되어 있습니다.")
    tokenizer = request.app.state.tokenizer
    input_tokens = len(tokenizer.encode(prompt))
    if input_tokens > MAX_INPUT_TOKENS:
        raise HTTPException(400, f"입력이 너무 깁니다. (현재 {input_tokens} 토큰 / 최대 {MAX_INPUT_TOKENS} 토큰)")
    if request.app.state.model_error:
        raise HTTPException(503, request.app.state.model_error)

    lock = request.app.state.generation_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(503, "다른 스토리를 생성하고 있습니다. 잠시 후 다시 시도해 주세요.")
    try:
        content = generate(request.app.state.model, tokenizer, prompt,
                           max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    except Exception:
        logger.exception("스토리 생성에 실패했습니다.")
        raise HTTPException(500, "스토리 생성에 실패했습니다. 다시 시도해 주세요.") from None
    finally:
        lock.release()

    with database(request.app.state.db_path) as db:
        story = stories.save_story(db, user["id"], prompt, content)
    return {**story, "completion": content[len(prompt):], "input_tokens": input_tokens}


DIST = ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
