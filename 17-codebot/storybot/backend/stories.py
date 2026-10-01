from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, StringConstraints

from backend.auth import current_user
from backend.storage import database


router = APIRouter(prefix="/api/stories")
PAGE_SIZE = 10


class StoryUpdate(BaseModel):
    title: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=100)]
    content: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=50000)]


def story_query():
    return """SELECT s.*, u.nickname AS author,
              (SELECT COUNT(*) FROM likes l WHERE l.story_id = s.id) AS like_count,
              EXISTS(SELECT 1 FROM likes l WHERE l.story_id = s.id AND l.user_id = ?) AS liked
              FROM stories s JOIN users u ON u.id = s.user_id"""


def serialize_story(row, viewer_id):
    result = dict(row)
    result["liked"] = bool(result["liked"])
    result["is_owner"] = result["user_id"] == viewer_id
    return result


def get_story(db, story_id, user_id):
    row = db.execute(story_query() + " WHERE s.id = ?", (user_id, story_id)).fetchone()
    if row is None:
        raise HTTPException(404, "스토리를 찾을 수 없습니다.")
    return serialize_story(row, user_id)


def check_owner(db, story_id, user_id):
    row = db.execute("SELECT user_id FROM stories WHERE id = ?", (story_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "스토리를 찾을 수 없습니다.")
    if row[0] != user_id:
        raise HTTPException(403, "내 스토리만 수정·삭제할 수 있습니다.")


def save_story(db, user_id, prompt, content):
    title = prompt.strip().splitlines()[0][:100]
    cursor = db.execute(
        "INSERT INTO stories (user_id, title, prompt, content) VALUES (?, ?, ?, ?)",
        (user_id, title, prompt, content),
    )
    return get_story(db, cursor.lastrowid, user_id)


@router.get("")
def list_stories(
    request: Request, page: Annotated[int, Query(ge=1)] = 1,
    user: dict = Depends(current_user),
):
    with database(request.app.state.db_path) as db:
        total = db.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
        rows = db.execute(
            story_query() + " ORDER BY s.id DESC LIMIT ? OFFSET ?",
            (user["id"], PAGE_SIZE, (page - 1) * PAGE_SIZE),
        ).fetchall()
    return {"items": [serialize_story(row, user["id"]) for row in rows],
            "page": page, "page_size": PAGE_SIZE, "total": total,
            "total_pages": (total + PAGE_SIZE - 1) // PAGE_SIZE}


@router.patch("/{story_id}")
def update_story(
    story_id: int, payload: StoryUpdate, request: Request,
    user: dict = Depends(current_user),
):
    if not payload.content.strip():
        raise HTTPException(422, "스토리 내용을 입력해 주세요.")
    with database(request.app.state.db_path) as db:
        check_owner(db, story_id, user["id"])
        db.execute(
            """UPDATE stories SET title = ?, content = ?,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?""",
            (payload.title, payload.content, story_id),
        )
        return get_story(db, story_id, user["id"])


@router.delete("/{story_id}", status_code=204)
def delete_story(story_id: int, request: Request, user: dict = Depends(current_user)):
    with database(request.app.state.db_path) as db:
        check_owner(db, story_id, user["id"])
        db.execute("DELETE FROM stories WHERE id = ?", (story_id,))


@router.put("/{story_id}/like")
def like_story(story_id: int, request: Request, user: dict = Depends(current_user)):
    with database(request.app.state.db_path) as db:
        get_story(db, story_id, user["id"])
        db.execute("INSERT OR IGNORE INTO likes (user_id, story_id) VALUES (?, ?)", (user["id"], story_id))
        return get_story(db, story_id, user["id"])


@router.delete("/{story_id}/like")
def unlike_story(story_id: int, request: Request, user: dict = Depends(current_user)):
    with database(request.app.state.db_path) as db:
        get_story(db, story_id, user["id"])
        db.execute("DELETE FROM likes WHERE user_id = ? AND story_id = ?", (user["id"], story_id))
        return get_story(db, story_id, user["id"])
