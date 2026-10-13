import json
import threading
import time
import unittest
import urllib.error
import urllib.request

import uvicorn

from toolgate.server import app

PORT = 8765


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}") as r:
        return r.status, r.read()


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(5)

    def test_pages_and_config(self):
        self.assertEqual(get("/")[0], 200)
        self.assertEqual(get("/static/app.js")[0], 200)
        cfg = json.loads(get("/api/config")[1])
        self.assertGreater(cfg["tools"], 0)
        self.assertIn("hybrid", cfg["methods"])

    def test_search_and_errors(self):
        st, d = post("/api/search", {"query": "close issue 12", "k": 3, "methods": ["hybrid"]})
        self.assertEqual(st, 200)
        self.assertEqual(len(d["results"][0]["hits"]), 3)
        self.assertEqual(post("/api/search", {"query": ""})[0], 400)
        self.assertEqual(post("/api/search", {"query": "x", "methods": ["nope"]})[0], 400)

    def test_eval_endpoint(self):
        st, d = post("/api/eval/run", {"split": "test", "ablations": False})
        self.assertEqual(st, 200)
        self.assertIn("hybrid", d["methods"])
        self.assertEqual(post("/api/eval/run", {"split": "bogus"})[0], 400)


if __name__ == "__main__":
    unittest.main()
