"""스토리봇 API 검증. 실행: storybot 폴더에서 python -m unittest backend.tests.test_api -v"""
import os
import shutil
import tempfile
import time
import unittest
from contextlib import contextmanager
from unittest import mock

import jwt

TEST_DIR = tempfile.mkdtemp(prefix="storybot-test-")
os.environ["STORYBOT_DATA_DIR"] = TEST_DIR

from fastapi.testclient import TestClient  # noqa: E402

from backend import main  # noqa: E402

PROMPT = "Once upon a time, there was a little girl named Lily."
counter = 0


def fake_generate(model, tokenizer, prompt, max_new_tokens, temperature):
    return prompt + " She found a red ball in the park."


@contextmanager
def fake_model(**kwargs):
    """모델 호출만 대체 (모델 상태와 무관하게 API 동작 검증)"""
    saved = main.app.state.model_error
    main.app.state.model_error = None
    try:
        with mock.patch.object(main, "generate", **(kwargs or {"new": fake_generate})):
            yield
    finally:
        main.app.state.model_error = saved


def signup(client, name=None, password="password123"):
    global counter
    counter += 1
    name = name or f"user{counter}"
    r = client.post("/api/auth/signup", json={"username": name, "password": password, "nickname": f"닉{counter}"})
    assert r.status_code == 201, r.text
    return name


def tokens_text(tokenizer, n):
    """정확히 n토큰이 되는 영어 문자열"""
    words = []
    while len(tokenizer.encode(" ".join(words))) < n:
        words.append("cat")
    text = " ".join(words)
    assert len(tokenizer.encode(text)) == n
    return text


class StorybotApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = TestClient(main.app)
        cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        shutil.rmtree(TEST_DIR, ignore_errors=True)

    def client(self):
        return TestClient(main.app)

    def make_story(self, client):
        with fake_model():
            r = client.post("/api/stories/generate", json={"prompt": PROMPT})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    # 회원
    def test_signup_login_logout_me(self):
        c = self.client()
        name = signup(c)
        self.assertEqual(c.get("/api/auth/me").json()["username"], name)
        dup = self.client().post("/api/auth/signup", json={"username": name.upper(), "password": "password123", "nickname": "x"})
        self.assertEqual(dup.status_code, 409)
        token = c.cookies.get(main.auth.COOKIE)
        self.assertEqual(c.post("/api/auth/logout").status_code, 204)
        self.assertEqual(c.get("/api/auth/me").status_code, 401)
        # 로그아웃한 토큰을 다시 넣어도 거절
        c2 = self.client()
        c2.cookies.set(main.auth.COOKIE, token)
        self.assertEqual(c2.get("/api/auth/me").status_code, 401)
        bad = self.client().post("/api/auth/login", json={"username": name, "password": "wrongpass1"})
        self.assertEqual(bad.status_code, 401)
        ok = self.client().post("/api/auth/login", json={"username": name, "password": "password123"})
        self.assertEqual(ok.status_code, 200)

    def test_tampered_and_expired_token_rejected(self):
        c = self.client()
        signup(c)
        token = c.cookies.get(main.auth.COOKIE)
        claims = jwt.decode(token, options={"verify_signature": False})
        forged = jwt.encode({**claims}, "wrong-key-" * 4, algorithm="HS256")
        expired = jwt.encode({**claims, "exp": int(time.time()) - 10}, main.app.state.jwt_key, algorithm="HS256")
        for value in (forged, expired, token + "x"):
            c2 = self.client()
            c2.cookies.set(main.auth.COOKIE, value)
            self.assertEqual(c2.get("/api/auth/me").status_code, 401)

    def test_profile_update(self):
        c = self.client()
        name = signup(c)
        other = self.client()
        other.post("/api/auth/login", json={"username": name, "password": "password123"})
        r = c.patch("/api/auth/me", json={"nickname": "새닉네임"})
        self.assertEqual(r.json()["nickname"], "새닉네임")
        r = c.patch("/api/auth/me", json={"nickname": "새닉네임", "current_password": "wrongpass1", "new_password": "newpass123"})
        self.assertEqual(r.status_code, 403)
        r = c.patch("/api/auth/me", json={"nickname": "새닉네임", "current_password": "password123", "new_password": "newpass123"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(c.get("/api/auth/me").status_code, 200)
        self.assertEqual(other.get("/api/auth/me").status_code, 401)
        self.assertEqual(self.client().post("/api/auth/login", json={"username": name, "password": "newpass123"}).status_code, 200)

    # 비로그인 차단
    def test_anonymous_blocked(self):
        c = self.client()
        calls = [("post", "/api/stories/generate", {"prompt": PROMPT}), ("get", "/api/stories", None),
                 ("patch", "/api/stories/1", {"title": "t", "content": "c"}), ("delete", "/api/stories/1", None),
                 ("put", "/api/stories/1/like", None), ("delete", "/api/stories/1/like", None),
                 ("get", "/api/auth/me", None), ("patch", "/api/auth/me", {"nickname": "x"})]
        for method, path, body in calls:
            r = getattr(c, method)(path, **({"json": body} if body else {}))
            self.assertEqual(r.status_code, 401, f"{method} {path}")

    # 입력 제한
    def test_input_validation(self):
        c = self.client()
        signup(c)
        tok = main.app.state.tokenizer
        with fake_model():
            self.assertEqual(c.post("/api/stories/generate", json={"prompt": "   \n"}).status_code, 400)
            self.assertEqual(c.post("/api/stories/generate", json={"prompt": "Hi <|endoftext|>"}).status_code, 400)
            self.assertEqual(c.post("/api/stories/generate", json={"prompt": tokens_text(tok, 56)}).status_code, 201)
            r = c.post("/api/stories/generate", json={"prompt": tokens_text(tok, 57)})
            self.assertEqual(r.status_code, 400)
            self.assertIn("57 토큰", r.json()["detail"])
        with fake_model(side_effect=AssertionError("must not call")):
            self.assertEqual(c.post("/api/stories/generate", json={"prompt": tokens_text(tok, 57)}).status_code, 400)

    def test_failed_generation_not_saved(self):
        c = self.client()
        signup(c)
        before = c.get("/api/stories").json()["total"]
        with fake_model(side_effect=RuntimeError("boom")):
            self.assertEqual(c.post("/api/stories/generate", json={"prompt": PROMPT}).status_code, 500)
        self.assertEqual(c.get("/api/stories").json()["total"], before)

    # 기록·페이지네이션
    def test_generate_saves_and_paginates(self):
        c = self.client()
        signup(c)
        story = self.make_story(c)
        self.assertEqual(story["title"], PROMPT)
        self.assertEqual(story["completion"], " She found a red ball in the park.")
        self.assertTrue(story["content"].startswith(PROMPT))
        for _ in range(11):
            self.make_story(c)
        p1 = c.get("/api/stories?page=1").json()
        self.assertEqual(len(p1["items"]), 10)
        self.assertGreaterEqual(p1["total_pages"], 2)
        ids = [s["id"] for s in p1["items"]]
        self.assertEqual(ids, sorted(ids, reverse=True))
        p2 = c.get("/api/stories?page=2").json()
        self.assertLess(p2["items"][0]["id"], ids[-1])
        # 다른 회원도 전체 기록을 본다
        other = self.client()
        signup(other)
        seen = other.get("/api/stories").json()["items"]
        self.assertIn(ids[0], [s["id"] for s in seen])
        self.assertFalse(next(s for s in seen if s["id"] == ids[0])["is_owner"])

    # 수정·삭제 권한
    def test_owner_only_edit_delete(self):
        owner, other = self.client(), self.client()
        signup(owner)
        signup(other)
        sid = self.make_story(owner)["id"]
        body = {"title": "새 제목", "content": "새 내용\n  줄바꿈"}
        self.assertEqual(other.patch(f"/api/stories/{sid}", json=body).status_code, 403)
        self.assertEqual(other.delete(f"/api/stories/{sid}").status_code, 403)
        r = owner.patch(f"/api/stories/{sid}", json=body)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["content"], "새 내용\n  줄바꿈")
        # 작성자·좋아요 수는 수정 요청으로 바뀌지 않음
        r = owner.patch(f"/api/stories/{sid}", json={**body, "user_id": 999, "like_count": 50})
        self.assertEqual(r.json()["like_count"], 0)
        self.assertTrue(r.json()["is_owner"])
        self.assertEqual(owner.patch(f"/api/stories/{sid}", json={"title": " ", "content": "x"}).status_code, 422)
        other.put(f"/api/stories/{sid}/like")
        self.assertEqual(owner.delete(f"/api/stories/{sid}").status_code, 204)
        self.assertEqual(owner.delete(f"/api/stories/{sid}").status_code, 404)
        with main.database(main.app.state.db_path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM likes WHERE story_id = ?", (sid,)).fetchone()[0], 0)

    # 좋아요
    def test_like_toggle_no_duplicates(self):
        owner, a, b = self.client(), self.client(), self.client()
        for c in (owner, a, b):
            signup(c)
        sid = self.make_story(owner)["id"]
        self.assertEqual(a.put(f"/api/stories/{sid}/like").json()["like_count"], 1)
        r = a.put(f"/api/stories/{sid}/like").json()
        self.assertEqual((r["like_count"], r["liked"]), (1, True))
        self.assertEqual(b.put(f"/api/stories/{sid}/like").json()["like_count"], 2)
        r = a.delete(f"/api/stories/{sid}/like").json()
        self.assertEqual((r["like_count"], r["liked"]), (1, False))

    def test_cross_origin_rejected(self):
        r = self.client().post("/api/auth/login", json={"username": "abc", "password": "password123"},
                               headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)

    # 실제 모델 생성
    def test_real_model_generation(self):
        self.assertIsNone(main.app.state.model_error)
        c = self.client()
        signup(c)
        r = c.post("/api/stories/generate", json={"prompt": "Once upon a time"})
        self.assertEqual(r.status_code, 201, r.text)
        story = r.json()
        self.assertTrue(story["content"].startswith("Once upon a time"))
        self.assertGreater(len(story["completion"].strip()), 0)
        self.assertEqual(c.get("/api/stories").json()["items"][0]["id"], story["id"])
        print("\n[실제 생성 결과]\n" + story["content"][:400])


if __name__ == "__main__":
    unittest.main()
