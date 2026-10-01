import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from codebot.backend.main import app
from codebot.utils import generate


class GenerateApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = TemporaryDirectory()
        cls.data_patch = patch("codebot.backend.main.DATA_DIR", Path(cls.temp.name))
        cls.data_patch.start()
        cls.client = TestClient(app)
        cls.client.__enter__()
        response = cls.client.post("/api/auth/signup", json={
            "username": "code_tester", "password": "TestPassword123!", "nickname": "코드 검증",
        })
        assert response.status_code == 201, response.text

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.data_patch.stop()
        cls.temp.cleanup()

    def test_real_model_matches_existing_generation_function(self):
        code = "def add(a, b):\n    "
        torch.manual_seed(0)
        expected = generate(
            app.state.model, app.state.tokenizer, code,
            max_new_tokens=200, temperature=1.0,
        )[len(code):]
        torch.manual_seed(0)
        response = self.client.post("/api/generate", json={"code": code})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"completion": expected})
        self.assertTrue(expected)

    def test_preserves_input_and_completion_whitespace(self):
        code = "def add(a, b):\n    "
        completion = "    return a + b\n\n"
        with patch("codebot.backend.main.generate", return_value=code + completion) as mock:
            response = self.client.post("/api/generate", json={"code": code})
        self.assertEqual(response.json(), {"completion": completion})
        self.assertEqual(mock.call_args.kwargs["prompt"], code)
        self.assertEqual(mock.call_args.kwargs["max_new_tokens"], 200)
        self.assertEqual(mock.call_args.kwargs["temperature"], 1.0)

    def test_removes_only_input_prefix(self):
        code = "def"
        with patch("codebot.backend.main.generate", return_value="def\n# def again\n"):
            response = self.client.post("/api/generate", json={"code": code})
        self.assertEqual(response.json()["completion"], "\n# def again\n")

    def test_accepts_exactly_256_tokens(self):
        code = "a\n" * 128
        self.assertEqual(len(app.state.tokenizer.encode(code)), 256)
        with patch("codebot.backend.main.generate", return_value=code) as mock:
            response = self.client.post("/api/generate", json={"code": code})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"completion": ""})
        mock.assert_called_once()

    def test_rejects_257_tokens_before_generation(self):
        code = "a\n" * 128 + "a"
        self.assertEqual(len(app.state.tokenizer.encode(code)), 257)
        with patch("codebot.backend.main.generate") as mock:
            response = self.client.post("/api/generate", json={"code": code})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["input_tokens"], 257)
        self.assertEqual(response.json()["detail"]["max_input_tokens"], 256)
        mock.assert_not_called()

    def test_uses_bpe_tokens_for_korean_input(self):
        code = "가" * 128
        self.assertLess(len(code), 256)
        token_count = len(app.state.tokenizer.encode(code))
        self.assertGreater(token_count, 256)
        with patch("codebot.backend.main.generate") as mock:
            response = self.client.post("/api/generate", json={"code": code})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["input_tokens"], token_count)
        mock.assert_not_called()

    def test_rejects_blank_input(self):
        with patch("codebot.backend.main.generate") as mock:
            for code in ["", " \t\n", "　"]:
                with self.subTest(code=code):
                    response = self.client.post("/api/generate", json={"code": code})
                    self.assertEqual(response.status_code, 422)
        mock.assert_not_called()

    def test_rejects_invalid_request_shape(self):
        with patch("codebot.backend.main.generate") as mock:
            for payload in [{}, {"code": None}, {"code": 123}, {"code": []}]:
                with self.subTest(payload=payload):
                    response = self.client.post("/api/generate", json=payload)
                    self.assertEqual(response.status_code, 422)
        mock.assert_not_called()

    def test_rejects_malformed_json(self):
        with patch("codebot.backend.main.generate") as mock:
            response = self.client.post(
                "/api/generate", content="{", headers={"Content-Type": "application/json"},
            )
        self.assertEqual(response.status_code, 422)
        mock.assert_not_called()

    def test_failure_releases_lock_and_allows_retry(self):
        with patch("codebot.backend.main.generate", side_effect=RuntimeError("test failure")):
            with self.assertLogs("codebot.backend.main", level="ERROR"):
                response = self.client.post("/api/generate", json={"code": "def"})
        self.assertEqual(response.status_code, 500)
        with patch("codebot.backend.main.generate", return_value="def done"):
            retry = self.client.post("/api/generate", json={"code": "def"})
        self.assertEqual(retry.status_code, 200)

    def test_concurrent_request_returns_503_without_waiting(self):
        started = Event()
        release = Event()

        def slow_generate(**kwargs):
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test request was not released")
            return kwargs["prompt"] + " done"

        with patch("codebot.backend.main.generate", side_effect=slow_generate) as mock:
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(self.client.post, "/api/generate", json={"code": "def"})
                try:
                    self.assertTrue(started.wait(timeout=5))
                    second = self.client.post("/api/generate", json={"code": "def"})
                    self.assertEqual(second.status_code, 503)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5).status_code, 200)
            mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
