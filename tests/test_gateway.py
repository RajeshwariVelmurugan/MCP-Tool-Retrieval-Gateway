"""End-to-end: collector -> corpus -> gateway (stdio) -> search_tools -> call_tool -> real downstream server."""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
DEMO = str(ROOT / "tests" / "demo_server.py")


class GatewayE2E(unittest.TestCase):
    def test_collect_then_gateway(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            servers = tmp / "servers.json"
            servers.write_text(json.dumps({"servers": {"demo": {"command": sys.executable, "args": [DEMO]}}}))
            corpus = tmp / "tools.json"

            from toolgate import collect

            self.assertEqual(collect.main(["--servers", str(servers), "--out", str(corpus), "--timeout", "30"]), 0)
            data = json.loads(corpus.read_text())
            self.assertEqual({t["id"] for t in data["tools"]}, {"demo__add_numbers", "demo__reverse_text", "demo__convert_temperature"})

            env = {**os.environ, "TOOLGATE_CORPUS": str(corpus), "TOOLGATE_SERVERS": str(servers),
                   "TOOLGATE_RESULTS_DIR": str(tmp / "res"), "TOOLGATE_CACHE_DIR": str(tmp / "cache"), "PYTHONPATH": str(ROOT)}
            params = StdioServerParameters(command=sys.executable, args=["-m", "toolgate.gateway"], env=env, cwd=str(ROOT))

            async def run():
                async with Client(params, read_timeout_seconds=60) as c:
                    names = {t.name for t in (await c.list_tools()).tools}
                    self.assertEqual(names, {"search_tools", "call_tool"})
                    res = await c.call_tool("search_tools", {"query": "reverse this text", "k": 2})
                    payload = json.loads(res.content[0].text)
                    ids = [t["id"] for t in payload["tools"]]
                    self.assertEqual(ids[0], "demo__reverse_text")
                    res = await c.call_tool("call_tool", {"tool_id": "demo__add_numbers", "arguments": {"a": 2, "b": 40}})
                    self.assertEqual(res.content[0].text.strip(), "42.0")
                    res = await c.call_tool("call_tool", {"tool_id": "nope__x", "arguments": {}})
                    self.assertIn("unknown tool_id", res.content[0].text)

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
