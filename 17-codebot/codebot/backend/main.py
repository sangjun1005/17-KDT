import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictStr

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device
from codebot.backend import auth, stories
from codebot.backend.auth import current_user
from codebot.backend.storage import database, initialize
from codebot.backend.story_model import load_storybot


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "backend" / "data"
STORY_DIR = ROOT.parent / "storybot"
MAX_NEW_TOKENS = 200
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = BPETokenizer.load_from(ROOT / "merge_rules.pkl")
    model = GPT.load_from(ROOT / "model_pretrain.pt", device=get_device())
    if tokenizer.vocab_size != model.vocab_size:
        raise RuntimeError("모델과 토크나이저의 어휘 크기가 일치하지 않습니다.")
    model.eval()
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.generation_lock = Lock()
    app.state.story_model, app.state.story_tokenizer = load_storybot(STORY_DIR, get_device())
    app.state.story_model_error = (
        "스토리봇 모델 가중치에 NaN 또는 무한대 값이 있어 생성할 수 없습니다. 정상 모델 파일이 필요합니다."
        if any(not parameter.isfinite().all().item() for parameter in app.state.story_model.parameters())
        else None
    )
    app.state.db_path, app.state.jwt_key = initialize(DATA_DIR)
    try:
        yield
    finally:
        del app.state.model
        del app.state.tokenizer
        del app.state.story_model
        del app.state.story_tokenizer


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


class GenerateRequest(BaseModel):
    code: StrictStr


class GenerateResponse(BaseModel):
    completion: str


@app.post("/api/generate", response_model=GenerateResponse)
def generate_code(payload: GenerateRequest, request: Request, user: dict = Depends(current_user)):
    code = payload.code
    if not code.strip():
        raise HTTPException(status_code=422, detail="코드를 입력해 주세요.")

    tokenizer = request.app.state.tokenizer
    model = request.app.state.model
    input_tokens = len(tokenizer.encode(code))
    if input_tokens > model.max_context_len:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"입력은 최대 {model.max_context_len}토큰입니다. "
                    f"현재 입력은 {input_tokens}토큰입니다. 코드를 줄여 주세요."
                ),
                "input_tokens": input_tokens,
                "max_input_tokens": model.max_context_len,
            },
        )

    lock = request.app.state.generation_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="다른 코드를 생성하고 있습니다. 잠시 후 다시 시도해 주세요.",
        )
    try:
        full_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=code,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
        )
        return GenerateResponse(completion=full_text[len(code):])
    except Exception:
        logger.exception("코드 생성에 실패했습니다.")
        raise HTTPException(
            status_code=500,
            detail="코드 생성에 실패했습니다. 다시 시도해 주세요.",
        ) from None
    finally:
        lock.release()


class StoryRequest(BaseModel):
    prompt: StrictStr


@app.post("/api/stories/generate", status_code=201)
def generate_story(payload: StoryRequest, request: Request, user: dict = Depends(current_user)):
    prompt = payload.prompt
    if not prompt.strip():
        raise HTTPException(422, "이야기의 시작을 입력해 주세요.")
    tokenizer = request.app.state.story_tokenizer
    model = request.app.state.story_model
    input_tokens = len(tokenizer.encode(prompt))
    if input_tokens > model.max_context_len:
        raise HTTPException(422, detail={
            "message": f"입력은 최대 {model.max_context_len}토큰입니다. 현재 입력은 {input_tokens}토큰입니다.",
            "input_tokens": input_tokens, "max_input_tokens": model.max_context_len,
        })
    lock = request.app.state.generation_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(503, "다른 내용을 생성하고 있습니다. 잠시 후 다시 시도해 주세요.")
    try:
        # Reuse codebot's verified generator with the original storybot model.
        # Non-cached inference keeps its RoPE context within the 256-token window.
        if request.app.state.story_model_error:
            raise HTTPException(503, request.app.state.story_model_error)
        content = generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0)
        current_user(request)
        title = prompt.strip().splitlines()[0][:80]
        with database(request.app.state.db_path) as db:
            cursor = db.execute(
                "INSERT INTO stories (user_id, title, prompt, content) VALUES (?, ?, ?, ?)",
                (user["id"], title, prompt, content),
            )
            return stories.get_story(db, cursor.lastrowid, user["id"])
    except HTTPException:
        raise
    except Exception:
        logger.exception("스토리 생성에 실패했습니다.")
        raise HTTPException(500, "스토리 생성에 실패했습니다. 다시 시도해 주세요.") from None
    finally:
        lock.release()


DIST = ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
