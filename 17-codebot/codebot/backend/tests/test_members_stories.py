import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import jwt
import torch
from fastapi.testclient import TestClient

from codebot.backend.auth import COOKIE, password_hasher
from codebot.backend.main import app
from codebot.backend.storage import database
from codebot.utils import generate


PASSWORD = "TestPassword123!"


class MembersAndStoriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = TemporaryDirectory()
        cls.data_patch = patch("codebot.backend.main.DATA_DIR", Path(cls.temp.name))
        cls.data_patch.start()
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.data_patch.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.client.cookies.clear()
        with database(app.state.db_path) as db:
            for table in ("likes", "stories", "sessions", "users"):
                db.execute(f"DELETE FROM {table}")

    def signup(self, name="alice"):
        response = self.client.post("/api/auth/signup", json={
            "username": name, "password": PASSWORD, "nickname": name,
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def create_story(self, prompt="Once upon a time, "):
        with patch.object(app.state, "story_model_error", None), patch("codebot.backend.main.generate", return_value=prompt + "a fox found a friend."):
            response = self.client.post("/api/stories/generate", json={"prompt": prompt})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_signup_hashes_password_and_sets_persistent_httponly_cookie(self):
        response = self.client.post("/api/auth/signup", json={
            "username": "Alice", "password": PASSWORD, "nickname": "앨리스",
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["username"], "alice")
        self.assertNotIn("password_hash", response.json())
        header = response.headers["set-cookie"].lower()
        self.assertIn("httponly", header)
        self.assertIn("max-age=604800", header)
        self.assertIn("samesite=strict", header)
        with database(app.state.db_path) as db:
            hashed = db.execute("SELECT password_hash FROM users").fetchone()[0]
        self.assertNotEqual(hashed, PASSWORD)
        self.assertTrue(password_hasher.verify(PASSWORD, hashed))
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)

    def test_duplicate_signup_is_case_insensitive(self):
        self.signup()
        response = self.client.post("/api/auth/signup", json={
            "username": "ALICE", "password": PASSWORD, "nickname": "다른 앨리스",
        })
        self.assertEqual(response.status_code, 409)

    def test_invalid_signup_is_rejected(self):
        for username, password, nickname in [("ab", PASSWORD, "a"), ("alice", "short", "a"), ("alice", PASSWORD, " ")]:
            with self.subTest(username=username, password=password):
                response = self.client.post("/api/auth/signup", json={"username": username, "password": password, "nickname": nickname})
                self.assertEqual(response.status_code, 422)

    def test_login_and_cookie_restore(self):
        user = self.signup()
        self.client.cookies.clear()
        bad = self.client.post("/api/auth/login", json={"username": "alice", "password": "WrongPassword!"})
        self.assertEqual(bad.status_code, 401)
        good = self.client.post("/api/auth/login", json={"username": "ALICE", "password": PASSWORD})
        self.assertEqual(good.status_code, 200)
        token = self.client.cookies.get(COOKIE)
        restored = TestClient(app)
        restored.cookies.set(COOKIE, token)
        self.assertEqual(restored.get("/api/auth/me").json(), user)
        restored.close()

    def test_logout_revokes_token_even_if_replayed(self):
        self.signup()
        token = self.client.cookies.get(COOKIE)
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 204)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.client.cookies.set(COOKIE, token)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_session_and_database_survive_application_restart(self):
        user = self.signup()
        story = self.create_story()
        key = app.state.jwt_key
        self.client.__exit__(None, None, None)
        self.client.__enter__()
        self.assertEqual(app.state.jwt_key, key)
        self.assertEqual(self.client.get("/api/auth/me").json(), user)
        self.assertEqual(self.client.get("/api/stories?mine=true").json()["items"][0]["id"], story["id"])

    def test_tampered_expired_or_revoked_token_is_rejected(self):
        self.signup()
        token = self.client.cookies.get(COOKIE)
        claims = jwt.decode(token, app.state.jwt_key, algorithms=["HS256"], audience="storybot-web", issuer="storybot")
        for bad in [token[:-8] + "xxxxxxxx", jwt.encode({**claims, "exp": int(time.time()) - 1}, app.state.jwt_key, algorithm="HS256")]:
            self.client.cookies.clear()
            self.client.cookies.set(COOKIE, bad)
            self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_unauthenticated_feature_access_is_rejected(self):
        with patch("codebot.backend.main.generate") as mock:
            self.assertEqual(self.client.post("/api/generate", json={"code": "def"}).status_code, 401)
            self.assertEqual(self.client.post("/api/stories/generate", json={"prompt": "Once"}).status_code, 401)
            self.assertEqual(self.client.get("/api/stories").status_code, 401)
            self.assertEqual(self.client.put("/api/stories/1/like").status_code, 401)
        mock.assert_not_called()

    def test_nickname_update_reflects_in_story_author(self):
        self.signup()
        self.create_story()
        updated = self.client.patch("/api/auth/me", json={"nickname": "새 닉네임"})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(self.client.get("/api/stories").json()["items"][0]["author"], "새 닉네임")

    def test_password_change_requires_old_password_and_revokes_other_sessions(self):
        self.signup()
        old_token = self.client.cookies.get(COOKIE)
        bad = self.client.patch("/api/auth/me", json={"nickname": "alice", "current_password": "WrongPassword!", "new_password": "NewPassword123!"})
        self.assertEqual(bad.status_code, 403)
        good = self.client.patch("/api/auth/me", json={"nickname": "alice", "current_password": PASSWORD, "new_password": "NewPassword123!"})
        self.assertEqual(good.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        self.client.cookies.clear()
        self.client.cookies.set(COOKIE, old_token)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD}).status_code, 401)
        self.assertEqual(self.client.post("/api/auth/login", json={"username": "alice", "password": "NewPassword123!"}).status_code, 200)

    def test_real_original_story_model_generation_and_auto_save(self):
        if app.state.story_model_error:
            self.skipTest("Existing storybot checkpoints contain non-finite weights; a valid checkpoint is required.")
        self.signup()
        prompt = "Once upon a time, a little fox"
        torch.manual_seed(0)
        expected = generate(app.state.story_model, app.state.story_tokenizer, prompt, max_new_tokens=200, temperature=1.0)
        torch.manual_seed(0)
        response = self.client.post("/api/stories/generate", json={"prompt": prompt})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["content"], expected)
        self.assertGreater(len(expected), len(prompt))
        self.assertEqual(self.client.get("/api/stories?mine=true").json()["total"], 1)

    def test_story_token_boundary_and_whitespace_preservation(self):
        self.signup()
        tokenizer = app.state.story_tokenizer
        prompt = "a\n" * 128
        self.assertEqual(len(tokenizer.encode(prompt)), 256)
        with patch.object(app.state, "story_model_error", None), patch("codebot.backend.main.generate", return_value=prompt + " end\n") as mock:
            accepted = self.client.post("/api/stories/generate", json={"prompt": prompt})
        self.assertEqual(accepted.status_code, 201)
        self.assertEqual(mock.call_args.args[2], prompt)
        self.assertEqual(accepted.json()["content"], prompt + " end\n")
        with patch("codebot.backend.main.generate") as mock:
            rejected = self.client.post("/api/stories/generate", json={"prompt": prompt + "a"})
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(rejected.json()["detail"]["input_tokens"], 257)
        mock.assert_not_called()

    def test_failed_generation_does_not_save_record_and_releases_lock(self):
        self.signup()
        with patch.object(app.state, "story_model_error", None), patch("codebot.backend.main.generate", side_effect=RuntimeError("test")):
            with self.assertLogs("codebot.backend.main", level="ERROR"):
                response = self.client.post("/api/stories/generate", json={"prompt": "Once"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.client.get("/api/stories").json()["total"], 0)
        self.create_story()

    def test_invalid_model_is_reported_without_saving_a_story(self):
        self.signup()
        with patch.object(app.state, "story_model_error", "스토리봇 모델 가중치에 NaN 값이 있습니다."), patch("codebot.backend.main.generate") as mock:
            response = self.client.post("/api/stories/generate", json={"prompt": "Once"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("NaN", response.json()["detail"])
        self.assertEqual(self.client.get("/api/stories").json()["total"], 0)
        mock.assert_not_called()

    def test_pagination_and_my_stories_are_filtered_before_pagination(self):
        alice = self.signup()
        with database(app.state.db_path) as db:
            for i in range(12):
                db.execute("INSERT INTO stories (user_id, title, prompt, content) VALUES (?, ?, 'p', 'c')", (alice["id"], f"story {i}"))
        self.client.cookies.clear()
        bob = self.signup("bob")
        self.create_story()
        first = self.client.get("/api/stories?page=1").json()
        second = self.client.get("/api/stories?page=2").json()
        self.assertEqual(first["total"], 13)
        self.assertEqual(first["total_pages"], 2)
        self.assertEqual(len(first["items"]), 10)
        self.assertEqual(len(second["items"]), 3)
        self.assertGreater(first["items"][0]["id"], second["items"][0]["id"])
        mine = self.client.get("/api/stories?mine=true").json()
        self.assertEqual(mine["total"], 1)
        self.assertEqual(mine["items"][0]["user_id"], bob["id"])
        self.assertEqual(self.client.get("/api/stories?page=0").status_code, 422)
        self.assertEqual(self.client.get("/api/stories?page=999").json()["items"], [])

    def test_other_user_cannot_edit_or_delete(self):
        self.signup()
        story = self.create_story()
        self.client.cookies.clear()
        self.signup("bob")
        edit = self.client.patch(f'/api/stories/{story["id"]}', json={"title": "hacked", "content": "hacked"})
        self.assertEqual(edit.status_code, 403)
        self.assertEqual(self.client.delete(f'/api/stories/{story["id"]}').status_code, 403)
        record = self.client.get("/api/stories").json()["items"][0]
        self.assertEqual(record["content"], story["content"])
        self.assertFalse(record["is_owner"])

    def test_owner_edit_preserves_content_and_delete_cascades_likes(self):
        self.signup()
        story = self.create_story()
        path = f'/api/stories/{story["id"]}'
        content = "  edited\n\n<script>alert(1)</script>"
        edited = self.client.patch(path, json={"title": "new title", "content": content})
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(edited.json()["content"], content)
        self.client.put(path + "/like")
        self.assertEqual(self.client.delete(path).status_code, 204)
        self.assertEqual(self.client.get("/api/stories").json()["total"], 0)
        with database(app.state.db_path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM likes").fetchone()[0], 0)
        self.assertEqual(self.client.patch(path, json={"title": "x", "content": "y"}).status_code, 404)

    def test_like_is_idempotent_and_can_be_cancelled_by_other_reader(self):
        self.signup()
        story = self.create_story()
        self.client.cookies.clear()
        self.signup("bob")
        path = f'/api/stories/{story["id"]}/like'
        for _ in range(2):
            liked = self.client.put(path).json()
            self.assertEqual(liked["like_count"], 1)
            self.assertTrue(liked["liked"])
        for _ in range(2):
            unliked = self.client.delete(path).json()
            self.assertEqual(unliked["like_count"], 0)
            self.assertFalse(unliked["liked"])

    def test_foreign_origin_cannot_change_state(self):
        response = self.client.post("/api/auth/signup", json={"username": "alice", "password": PASSWORD, "nickname": "alice"}, headers={"Origin": "https://other.example"})
        self.assertEqual(response.status_code, 403)
        self.signup()
        response = self.client.post("/api/auth/logout", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)


if __name__ == "__main__":
    unittest.main()
