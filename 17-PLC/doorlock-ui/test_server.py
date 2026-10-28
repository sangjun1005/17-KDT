import ast
import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import Handler, ThreadingHTTPServer, camera


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_static_and_state(self):
        for path in ("/", "/styles.css", "/app.js", "/logic.mjs"):
            with urlopen(self.base + path) as response:
                self.assertEqual(response.status, 200)
        with urlopen(self.base + "/api/state") as response:
            state = json.load(response)
            self.assertEqual(state["status"], "off")
            self.assertEqual(state["events"], [])

    def test_camera_not_started_by_page(self):
        self.assertIsNone(camera.worker)

    def test_cross_origin_camera_request_rejected(self):
        request = Request(self.base + "/api/camera/start", method="POST",
                          headers={"Origin": "https://example.com"})
        with self.assertRaises(HTTPError) as result:
            urlopen(request)
        self.assertEqual(result.exception.code, 403)
        self.assertIsNone(camera.worker)

    def test_private_source_not_served(self):
        with self.assertRaises(HTTPError) as result:
            urlopen(self.base + "/server.py")
        self.assertEqual(result.exception.code, 404)

    def test_original_finger_rules_preserved(self):
        root = Path(__file__).parent
        def source_ast(path):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                 and node.name == "count_fingers"), include_attributes=False)
        # Ignore the original function's descriptive docstring, not its executable rules.
        original = ast.parse((root.parent / "gesture_doorlock.py").read_text(encoding="utf-8-sig"))
        fn = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "count_fingers")
        fn.body = fn.body[1:]
        self.assertEqual(ast.dump(fn, include_attributes=False), source_ast(root / "server.py"))


if __name__ == "__main__":
    unittest.main()
