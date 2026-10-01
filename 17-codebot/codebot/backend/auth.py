import secrets
import sqlite3
import time
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, StringConstraints, field_validator
from pwdlib import PasswordHash

from codebot.backend.storage import database


router = APIRouter(prefix="/api/auth")
password_hasher = PasswordHash.recommended()
COOKIE = "storybot_session"
SESSION_SECONDS = 7 * 24 * 60 * 60
Username = Annotated[str, StringConstraints(strict=True, pattern=r"^[a-zA-Z0-9_]{3,30}$")]
Password = Annotated[str, StringConstraints(strict=True, min_length=8, max_length=128)]
Nickname = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=30)]


class SignupRequest(BaseModel):
    username: Username
    password: Password
    nickname: Nickname

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value):
        return value.lower()


class LoginRequest(BaseModel):
    username: Username
    password: Password


class ProfileRequest(BaseModel):
    nickname: Nickname
    current_password: Password | None = None
    new_password: Password | None = None


def public_user(row):
    return {key: row[key] for key in ("id", "username", "nickname")}


def start_session(request, response, user_id):
    now = int(time.time())
    session_id = secrets.token_urlsafe(32)
    expires = now + SESSION_SECONDS
    with database(request.app.state.db_path) as db:
        db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        db.execute(
            "INSERT INTO sessions (id, user_id, expires_at) VALUES (?, ?, ?)",
            (session_id, user_id, expires),
        )
    token = jwt.encode(
        {"sub": str(user_id), "jti": session_id, "iat": now, "exp": expires,
         "iss": "storybot", "aud": "storybot-web"},
        request.app.state.jwt_key,
        algorithm="HS256",
    )
    response.set_cookie(
        COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
        samesite="strict", secure=request.url.scheme == "https", path="/",
    )


def current_user(request: Request):
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "로그인이 필요합니다.")
    try:
        claims = jwt.decode(
            token, request.app.state.jwt_key, algorithms=["HS256"],
            issuer="storybot", audience="storybot-web",
            options={"require": ["sub", "jti", "iat", "exp"]},
        )
        user_id = int(claims["sub"])
        with database(request.app.state.db_path) as db:
            user = db.execute(
                """SELECT u.id, u.username, u.nickname FROM users u
                   JOIN sessions s ON s.user_id = u.id
                   WHERE s.id = ? AND u.id = ? AND s.expires_at > ?""",
                (claims["jti"], user_id, int(time.time())),
            ).fetchone()
        if user is None:
            raise HTTPException(401, "로그인이 만료되었습니다. 다시 로그인해 주세요.")
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise HTTPException(401, "로그인이 만료되었습니다. 다시 로그인해 주세요.") from None
    request.state.session_id = claims["jti"]
    return public_user(user)


@router.post("/signup", status_code=201)
def signup(payload: SignupRequest, request: Request, response: Response):
    hashed = password_hasher.hash(payload.password)
    try:
        with database(request.app.state.db_path) as db:
            cursor = db.execute(
                "INSERT INTO users (username, nickname, password_hash) VALUES (?, ?, ?)",
                (payload.username, payload.nickname, hashed),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "이미 사용 중인 아이디입니다.") from None
    start_session(request, response, user_id)
    return {"id": user_id, "username": payload.username, "nickname": payload.nickname}


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response):
    with database(request.app.state.db_path) as db:
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (payload.username,),
        ).fetchone()
    if user is None or not password_hasher.verify(payload.password, user["password_hash"]):
        raise HTTPException(401, "아이디 또는 비밀번호가 올바르지 않습니다.")
    start_session(request, response, user["id"])
    return public_user(user)


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return user


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, user: dict = Depends(current_user)):
    with database(request.app.state.db_path) as db:
        db.execute("DELETE FROM sessions WHERE id = ?", (request.state.session_id,))
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict")


@router.patch("/me")
def update_profile(
    payload: ProfileRequest, request: Request, response: Response,
    user: dict = Depends(current_user),
):
    with database(request.app.state.db_path) as db:
        if payload.new_password is not None:
            row = db.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
            if not payload.current_password or not password_hasher.verify(payload.current_password, row[0]):
                raise HTTPException(403, "현재 비밀번호가 올바르지 않습니다.")
            hashed = password_hasher.hash(payload.new_password)
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hashed, user["id"]))
            db.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))
        db.execute("UPDATE users SET nickname = ? WHERE id = ?", (payload.nickname, user["id"]))
    if payload.new_password is not None:
        start_session(request, response, user["id"])
    return {**user, "nickname": payload.nickname}
