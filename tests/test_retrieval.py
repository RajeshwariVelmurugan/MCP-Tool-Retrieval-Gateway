import unittest

from toolgate.config import get_settings
from toolgate.corpus import load_queries, load_tools
from toolgate.engine import Engine
from toolgate.evaluation import metrics as M
from toolgate.evaluation.runner import run_eval
from toolgate.retrievers.hybrid import rrf_fuse


class Metrics(unittest.TestCase):
    def test_recall_modes(self):
        ranked = ["a", "b", "c", "d"]
        self.assertEqual(M.recall_at_k(ranked, ["a", "d"], 3, "all"), 0.5)
        self.assertEqual(M.recall_at_k(ranked, ["a", "d"], 3, "any"), 1.0)
        self.assertEqual(M.full_hit_at_k(ranked, ["a", "d"], 3, "all"), 0.0)
        self.assertEqual(M.full_hit_at_k(ranked, ["a", "d"], 4, "all"), 1.0)

    def test_mrr_ndcg(self):
        self.assertEqual(M.mrr(["x", "a"], ["a"]), 0.5)
        self.assertEqual(M.mrr(["x", "y"], ["a"]), 0.0)
        self.assertAlmostEqual(M.ndcg_at_k(["a", "b"], ["a"], 2), 1.0)
        self.assertLess(M.ndcg_at_k(["b", "a"], ["a"], 2), 1.0)

    def test_percentile(self):
        self.assertEqual(M.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(M.percentile([], 95), 0.0)

    def test_rrf(self):
        fused = rrf_fuse([[(1, 0), (2, 0)], [(2, 0), (3, 0)]])
        self.assertEqual(fused[0][0], 2)


class Engine_(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = get_settings()
        cls.engine = Engine(load_tools(s.corpus_path), s, use_cache=False)
        cls.queries = load_queries(s.queries_path)

    def test_corpus_and_gold_consistent(self):
        ids = {t.id for t in self.engine.tools}
        self.assertEqual(len(ids), len(self.engine.tools))
        for q in self.queries:
            for g in q.gold:
                self.assertIn(g, ids, q.id)

    def test_obvious_queries(self):
        for query, want in [("close issue 12", "github__update_issue"), ("get the current time in Tokyo", "time__get_current_time")]:
            for m in ("bm25", "hybrid", "hybrid_rerank"):
                ids = [h["id"] for h in self.engine.search(query, m, 5).hits]
                self.assertIn(want, ids, (m, query))

    def test_all_tools_and_token_savings(self):
        r = self.engine.search("anything", "all_tools", 5)
        self.assertEqual(len(r.hits), len(self.engine.tools))
        r2 = self.engine.search("read a file", "hybrid", 5)
        self.assertLess(r2.tokens_returned, r2.tokens_all * 0.2)

    def test_eval_runs_and_all_tools_is_perfect(self):
        res = run_eval(self.engine, self.queries, split="test")
        self.assertEqual(res["methods"]["all_tools"]["metrics"]["recall@5"], 1.0)
        self.assertIsNone(res["methods"]["all_tools"]["metrics"]["mrr"])
        for m in ("bm25", "hybrid", "hybrid_rerank"):
            self.assertGreater(res["methods"][m]["metrics"]["full_hit@5"], 0.4)
            self.assertGreater(res["methods"][m]["tokens_saved_pct"], 80)
        self.assertIn("buckets", res["failures"]["hybrid"])

    def test_dev_test_split_disjoint(self):
        dev = {q.id for q in self.queries if q.split == "dev"}
        test = {q.id for q in self.queries if q.split == "test"}
        self.assertFalse(dev & test)
        self.assertTrue(dev and test)


if __name__ == "__main__":
    unittest.main()
