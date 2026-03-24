"""
AIL v0.3 — Benchmark Suite
Compares optimized vs unoptimized AIL graphs across multiple scenarios.

Measures:
  - Node count reduction
  - Token cost savings
  - Binary size compression
  - Confidence accuracy (does optimization preserve semantics?)
  - Hash-based dedup effectiveness
  - Timing for compile/optimize/execute phases

Run: python3 benchmark.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ail_types import AILProgram, AILScope, AILNode, InputBinding, InlineExpr, GraphRef
from transpiler import transpile
from serializer import encode
from executor import execute_program
from cost_model import analyze_program, ModelTier
from optimizer import optimize_program
from hash_ids import hash_program, dedup_by_hash


# ─── Benchmark Scenarios ─────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "Simple filter pipeline",
        "description": "Filter → sort → limit (pure compute, no LLM)",
        "source": """
async def get_top_users(db, min_score=0.8):
    users = await db.query("SELECT * FROM users")
    scored = [u for u in users if u['score'] >= min_score]
    ranked = sorted(scored, key=lambda u: u['score'], reverse=True)
    return ranked[:10]
""",
    },
    {
        "name": "Aggregation pipeline",
        "description": "Multiple aggregations from same source",
        "source": """
def user_stats(users):
    total = len(users)
    avg_score = sum(u['score'] for u in users) / total
    return avg_score
""",
    },
    {
        "name": "Conditional branching",
        "description": "If/elif/else chain with multiple branches",
        "source": """
def classify_user(user):
    if user['score'] > 0.9:
        return 'premium'
    elif user['score'] > 0.5:
        return 'standard'
    else:
        return 'basic'
""",
    },
    {
        "name": "Multi-source merge",
        "description": "Read from multiple sources and combine",
        "source": """
async def aggregate_data(db, cache, api):
    users = await db.query("users")
    cached = await cache.get("recent_users")
    external = await api.fetch("partners")
    combined = users + cached + external
    return combined
""",
    },
    {
        "name": "Complex recommendation",
        "description": "Search → filter → sort → limit (async pipeline)",
        "source": """
async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""",
    },
]

# ─── Manual AI-native scenarios (can't transpile from Python) ────────────────

def _build_ai_pipeline() -> AILProgram:
    """AI-native pipeline: embed → infer → rank → threshold."""
    prog = AILProgram()
    scope = AILScope("ai_recommendation")
    refs = [scope.next_ref() for _ in range(8)]

    scope.add_node(AILNode("io:in",
        inputs=[InputBinding("param", '"user_query"')],
        outputs=[refs[0]]))
    scope.add_node(AILNode("~:embed",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))
    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"vector_db"')],
        outputs=[refs[2]]))
    scope.add_node(AILNode("~:sim",
        inputs=[InputBinding("a", refs[1]), InputBinding("b", refs[2])],
        outputs=[refs[3]]))
    scope.add_node(AILNode("~:threshold",
        inputs=[InputBinding("val", refs[3]), InputBinding("min", "~+")],
        outputs=[refs[4]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[4])],
        outputs=[refs[5]]))
    scope.add_node(AILNode("~:rank",
        inputs=[InputBinding("col", refs[5])],
        outputs=[refs[6]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[6])]))

    prog.add_scope(scope)
    return prog


def _build_redundant_pipeline() -> AILProgram:
    """Pipeline with deliberate redundancy for optimizer to eliminate."""
    prog = AILProgram()
    scope = AILScope("redundant_ops")
    refs = [scope.next_ref() for _ in range(10)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    # Duplicate filters
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", InlineExpr("§.active = true"))],
        outputs=[refs[1]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[1]), InputBinding("cond", InlineExpr("§.score > 0.5"))],
        outputs=[refs[2]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[2]), InputBinding("cond", InlineExpr("§.verified = true"))],
        outputs=[refs[3]]))
    # Duplicate maps on same input
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[3]), InputBinding("fn", InlineExpr("§.name"))],
        outputs=[refs[4]]))
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[3]), InputBinding("fn", InlineExpr("§.name"))],
        outputs=[refs[5]]))
    # Dead node
    scope.add_node(AILNode("∇:asc",
        inputs=[InputBinding("col", refs[5]), InputBinding("key", '"name"')],
        outputs=[refs[6]]))
    # Only use first map
    scope.add_node(AILNode("∇:desc",
        inputs=[InputBinding("col", refs[4]), InputBinding("key", '"name"')],
        outputs=[refs[7]]))
    scope.add_node(AILNode("⊗:limit",
        inputs=[InputBinding("col", refs[7]), InputBinding("n", "#5")],
        outputs=[refs[8]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[8])]))

    prog.add_scope(scope)
    return prog


MANUAL_SCENARIOS = [
    {
        "name": "AI-native pipeline",
        "description": "embed → sim → infer → rank (full LLM cost)",
        "program": _build_ai_pipeline(),
    },
    {
        "name": "Redundant operations",
        "description": "3 sequential filters + duplicate maps + dead node",
        "program": _build_redundant_pipeline(),
    },
]


# ─── Benchmark Runner ────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    description: str
    nodes_before: int
    nodes_after: int
    tokens_before: int
    tokens_after: int
    binary_before: int
    binary_after: int
    confidence_before: float
    confidence_after: float
    max_tier: str
    cacheable_ratio_before: float
    cacheable_ratio_after: float
    unique_hashes: int
    total_hashes: int
    hash_duplicates: int
    compile_ms: float
    optimize_ms: float
    execute_ms: float

    @property
    def node_reduction(self) -> float:
        return (1 - self.nodes_after / max(self.nodes_before, 1)) * 100

    @property
    def token_savings(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def binary_reduction(self) -> float:
        return (1 - self.binary_after / max(self.binary_before, 1)) * 100


def run_benchmark(name: str, description: str, prog: AILProgram) -> BenchmarkResult:
    """Run full benchmark on a program."""
    # Cost before
    costs_before = analyze_program(prog)
    binary_before = encode(prog)

    # Execute before
    t0 = time.perf_counter()
    results_before = execute_program(prog)
    exec_time = (time.perf_counter() - t0) * 1000

    # Optimize
    t0 = time.perf_counter()
    opt_prog, reports = optimize_program(prog, confidence_threshold=0.1)
    opt_time = (time.perf_counter() - t0) * 1000

    # Cost after
    costs_after = analyze_program(opt_prog)
    binary_after = encode(opt_prog)
    results_after = execute_program(opt_prog)

    # Hash analysis
    hashed = hash_program(opt_prog)

    # Aggregate across scopes
    total_before = sum(c.total_nodes for c in costs_before)
    total_after = sum(c.total_nodes for c in costs_after)
    tok_before = sum(c.total_token_cost for c in costs_before)
    tok_after = sum(c.total_token_cost for c in costs_after)

    cache_before = sum(c.cacheable_nodes for c in costs_before) / max(total_before, 1)
    cache_after = sum(c.cacheable_nodes for c in costs_after) / max(total_after, 1)

    conf_before = results_before[-1].output_confidence if results_before else 1.0
    conf_after = results_after[-1].output_confidence if results_after else 1.0

    tier = max((c.max_model_tier for c in costs_before),
               key=lambda t: {"none": 0, "cheap": 1, "mid": 2, "powerful": 3}[t.value])

    unique = sum(len(hs.reverse_map) for hs in hashed)
    total_h = sum(len(hs.nodes) for hs in hashed)
    dups = sum(len(hs.duplicates) for hs in hashed)

    return BenchmarkResult(
        name=name,
        description=description,
        nodes_before=total_before,
        nodes_after=total_after,
        tokens_before=tok_before,
        tokens_after=tok_after,
        binary_before=len(binary_before),
        binary_after=len(binary_after),
        confidence_before=conf_before,
        confidence_after=conf_after,
        max_tier=tier.value,
        cacheable_ratio_before=cache_before,
        cacheable_ratio_after=cache_after,
        unique_hashes=unique,
        total_hashes=total_h,
        hash_duplicates=dups,
        compile_ms=0.0,  # set separately for transpiled
        optimize_ms=opt_time,
        execute_ms=exec_time,
    )


# ─── Pretty Print ────────────────────────────────────────────────────────────

def print_results(results: list[BenchmarkResult]):
    """Print benchmark results as a formatted table."""
    W = 80

    print("=" * W)
    print("AIL BENCHMARK SUITE — v0.3".center(W))
    print("=" * W)

    # Table header
    print(f"\n{'Scenario':<30s} {'Nodes':>8s} {'Tokens':>8s} {'Binary':>8s} "
          f"{'Cache%':>7s} {'~Conf':>7s}")
    print("-" * W)

    for r in results:
        node_str = f"{r.nodes_before}→{r.nodes_after}"
        tok_str = f"{r.tokens_before}→{r.tokens_after}"
        bin_str = f"{r.binary_before}→{r.binary_after}"
        cache_str = f"{r.cacheable_ratio_before:.0%}→{r.cacheable_ratio_after:.0%}"
        conf_str = f"~{r.confidence_after:.3f}"

        print(f"{r.name:<30s} {node_str:>8s} {tok_str:>8s} {bin_str:>8s} "
              f"{cache_str:>7s} {conf_str:>7s}")

    print("-" * W)

    # Summary stats
    total_nodes_before = sum(r.nodes_before for r in results)
    total_nodes_after = sum(r.nodes_after for r in results)
    total_tok_before = sum(r.tokens_before for r in results)
    total_tok_after = sum(r.tokens_after for r in results)
    total_dups = sum(r.hash_duplicates for r in results)
    avg_opt_ms = sum(r.optimize_ms for r in results) / len(results)

    print(f"\n{'TOTALS':<30s}")
    print(f"  Nodes eliminated: {total_nodes_before - total_nodes_after} "
          f"({(1-total_nodes_after/max(total_nodes_before,1))*100:.1f}% reduction)")
    print(f"  Tokens saved:     {total_tok_before - total_tok_after}")
    print(f"  Hash duplicates:  {total_dups}")
    print(f"  Avg optimize:     {avg_opt_ms:.2f}ms")

    # Detailed reports
    print(f"\n{'=' * W}")
    print("DETAILED REPORTS".center(W))
    print("=" * W)

    for r in results:
        print(f"\n┌─ {r.name} {'─' * (W - len(r.name) - 4)}")
        print(f"│  {r.description}")
        print(f"│")
        print(f"│  Nodes:      {r.nodes_before:>4d} → {r.nodes_after:>4d}  "
              f"({r.node_reduction:+.1f}%)")
        print(f"│  Tokens:     {r.tokens_before:>4d} → {r.tokens_after:>4d}  "
              f"(saved {r.token_savings})")
        print(f"│  Binary:     {r.binary_before:>4d}B → {r.binary_after:>4d}B  "
              f"({r.binary_reduction:+.1f}%)")
        print(f"│  Max tier:   {r.max_tier}")
        print(f"│  Cache:      {r.cacheable_ratio_before:.0%} → {r.cacheable_ratio_after:.0%}")
        print(f"│  Confidence: ~{r.confidence_before:.4f} → ~{r.confidence_after:.4f}")
        print(f"│  Hashes:     {r.unique_hashes} unique / {r.total_hashes} total "
              f"({r.hash_duplicates} duplicates)")
        print(f"│  Timing:     optimize={r.optimize_ms:.2f}ms  exec={r.execute_ms:.2f}ms")
        print(f"└{'─' * (W - 1)}")

    print(f"\n{'=' * W}")
    print("ALL BENCHMARKS COMPLETE".center(W))
    print("=" * W)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results: list[BenchmarkResult] = []

    # Transpiled scenarios
    for scenario in SCENARIOS:
        t0 = time.perf_counter()
        prog = transpile(scenario["source"])
        compile_ms = (time.perf_counter() - t0) * 1000

        r = run_benchmark(scenario["name"], scenario["description"], prog)
        r.compile_ms = compile_ms
        results.append(r)

    # Manual AI-native scenarios
    for scenario in MANUAL_SCENARIOS:
        r = run_benchmark(scenario["name"], scenario["description"], scenario["program"])
        results.append(r)

    print_results(results)
