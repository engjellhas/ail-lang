"""
AIL v0.2 — Cost Model
Token cost estimation, model tier selection, cacheability analysis.

Every node in an AIL graph gets three annotations:
  1. token_cost  — estimated LLM tokens consumed (0 for compute-only nodes)
  2. model_tier  — suggested model tier (none/cheap/mid/powerful)
  3. cacheable   — can the output be memoized? (deterministic + no side effects)

These annotations feed the optimizer (P4): you can't minimize cost
without knowing what things cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ail_types import AILNode, AILScope, AILProgram, GraphRef, InputBinding


# ─── Model Tiers ─────────────────────────────────────────────────────────────

class ModelTier(Enum):
    """Suggested LLM tier for executing a node."""
    NONE     = "none"       # no LLM needed — pure compute
    CHEAP    = "cheap"      # small/fast model (haiku-class)
    MID      = "mid"        # balanced model (sonnet-class)
    POWERFUL = "powerful"   # frontier model (opus-class)


# ─── Cost Profile ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostProfile:
    """Static cost annotation for a node type."""
    token_cost: int          # estimated tokens consumed per execution
    model_tier: ModelTier    # suggested model tier
    cacheable: bool          # deterministic + no side effects → memoizable

    def __str__(self):
        tier = self.model_tier.value
        cache = "$$" if self.cacheable else "  "
        if self.token_cost == 0:
            return f"[{tier}] {cache} free"
        return f"[{tier}] {cache} ~{self.token_cost}tok"


# ─── Per-Node Cost Table ─────────────────────────────────────────────────────
# Token costs are order-of-magnitude estimates for a single invocation.
# Real costs depend on input size — these are base costs for the optimizer
# to reason about relative expense.

NODE_COSTS: dict[str, CostProfile] = {
    # IO — no LLM, but not cacheable (side effects)
    "io:in":        CostProfile(0,    ModelTier.NONE, False),
    "io:out":       CostProfile(0,    ModelTier.NONE, False),
    "io:read":      CostProfile(0,    ModelTier.NONE, False),   # network/DB call
    "io:write":     CostProfile(0,    ModelTier.NONE, False),   # side effect
    "io:emit":      CostProfile(0,    ModelTier.NONE, False),   # fire-and-forget

    # Transform — pure compute, all cacheable
    "∿:map":        CostProfile(0,    ModelTier.NONE, True),
    "∿:flat":       CostProfile(0,    ModelTier.NONE, True),
    "∿:cast":       CostProfile(0,    ModelTier.NONE, True),
    "∿:norm":       CostProfile(0,    ModelTier.NONE, True),
    "∿:fmt":        CostProfile(0,    ModelTier.NONE, True),

    # Filter — pure compute, all cacheable
    "⊗:where":      CostProfile(0,    ModelTier.NONE, True),
    "⊗:dedup":      CostProfile(0,    ModelTier.NONE, True),
    "⊗:limit":      CostProfile(0,    ModelTier.NONE, True),
    "⊗:skip":       CostProfile(0,    ModelTier.NONE, True),
    "⊗:null":       CostProfile(0,    ModelTier.NONE, True),

    # Aggregate — pure compute, all cacheable
    "Σ:sum":        CostProfile(0,    ModelTier.NONE, True),
    "Σ:count":      CostProfile(0,    ModelTier.NONE, True),
    "Σ:avg":        CostProfile(0,    ModelTier.NONE, True),
    "Σ:max":        CostProfile(0,    ModelTier.NONE, True),
    "Σ:min":        CostProfile(0,    ModelTier.NONE, True),
    "Σ:group":      CostProfile(0,    ModelTier.NONE, True),

    # Control — pure compute, cacheable (same inputs → same branch)
    "⊢:if":         CostProfile(0,    ModelTier.NONE, True),
    "⊢:match":      CostProfile(0,    ModelTier.NONE, True),
    "⊢:guard":      CostProfile(0,    ModelTier.NONE, True),
    "⊢:prob":       CostProfile(0,    ModelTier.NONE, True),

    # Loop — pure compute, cacheable with same inputs
    "⟳:loop":       CostProfile(0,    ModelTier.NONE, True),
    "⟳:rec":        CostProfile(0,    ModelTier.NONE, True),

    # Merge — pure compute, all cacheable
    "⊕:zip":        CostProfile(0,    ModelTier.NONE, True),
    "⊕:concat":     CostProfile(0,    ModelTier.NONE, True),
    "⊕:merge":      CostProfile(0,    ModelTier.NONE, True),
    "⊕:fan-in":     CostProfile(0,    ModelTier.NONE, True),

    # Sort — pure compute, cacheable
    "∇:asc":        CostProfile(0,    ModelTier.NONE, True),
    "∇:desc":       CostProfile(0,    ModelTier.NONE, True),

    # AI-native — THE EXPENSIVE ONES
    # These are the only nodes that consume LLM tokens.
    "~:infer":      CostProfile(500,  ModelTier.POWERFUL, False),  # general inference
    "~:intent":     CostProfile(200,  ModelTier.MID,      False),  # classification
    "~:embed":      CostProfile(100,  ModelTier.CHEAP,    True),   # embeddings are cacheable!
    "~:sim":        CostProfile(0,    ModelTier.NONE,     True),   # cosine sim = pure math
    "~:rank":       CostProfile(300,  ModelTier.MID,      False),  # LLM-assisted ranking
    "~:threshold":  CostProfile(0,    ModelTier.NONE,     True),   # pure comparison
    "~:sample":     CostProfile(0,    ModelTier.NONE,     False),  # random → not cacheable

    # Call — depends on callee, conservative defaults
    "call":         CostProfile(0,    ModelTier.NONE, False),
}


# ─── Graph Cost Analysis ─────────────────────────────────────────────────────

@dataclass
class NodeCost:
    """Cost annotation for a single node instance."""
    node_index: int
    node_type: str
    profile: CostProfile
    cacheable_subgraph: bool = False   # all ancestors also cacheable?

    def __str__(self):
        sub = " [subgraph-cacheable]" if self.cacheable_subgraph else ""
        return f"§{self.node_index} [{self.node_type}] {self.profile}{sub}"


@dataclass
class ScopeCost:
    """Cost analysis for an entire scope."""
    scope_name: str
    node_costs: list[NodeCost]
    total_token_cost: int
    max_model_tier: ModelTier
    cacheable_nodes: int
    total_nodes: int

    @property
    def cacheable_ratio(self) -> float:
        return self.cacheable_nodes / self.total_nodes if self.total_nodes else 0.0

    @property
    def llm_nodes(self) -> int:
        return sum(1 for nc in self.node_costs if nc.profile.model_tier != ModelTier.NONE)

    def summary(self) -> str:
        lines = [
            f"⟦ {self.scope_name}  — cost analysis",
            f"  Total tokens:   ~{self.total_token_cost}",
            f"  Max model tier: {self.max_model_tier.value}",
            f"  LLM nodes:      {self.llm_nodes}/{self.total_nodes}",
            f"  Cacheable:      {self.cacheable_nodes}/{self.total_nodes} "
            f"({self.cacheable_ratio:.0%})",
        ]
        lines.append("  ─────────────────────────────────────")
        for nc in self.node_costs:
            lines.append(f"  {nc}")
        lines.append("⟧")
        return "\n".join(lines)


def _build_ancestor_map(scope: AILScope) -> dict[int, set[int]]:
    """Map each node index to the set of all its transitive ancestors."""
    ref_to_producer: dict[int, int] = {}
    for i, node in enumerate(scope.nodes):
        for out in node.outputs:
            if isinstance(out, GraphRef):
                ref_to_producer[out.index] = i

    # Direct parents
    parents: dict[int, set[int]] = {i: set() for i in range(len(scope.nodes))}
    for i, node in enumerate(scope.nodes):
        for inp in node.inputs:
            if isinstance(inp.ref, GraphRef) and inp.ref.index in ref_to_producer:
                parents[i].add(ref_to_producer[inp.ref.index])

    # Transitive closure (BFS per node)
    ancestors: dict[int, set[int]] = {}
    for i in range(len(scope.nodes)):
        visited = set()
        queue = list(parents[i])
        while queue:
            p = queue.pop()
            if p not in visited:
                visited.add(p)
                queue.extend(parents.get(p, set()))
        ancestors[i] = visited

    return ancestors


TIER_RANK = {
    ModelTier.NONE: 0,
    ModelTier.CHEAP: 1,
    ModelTier.MID: 2,
    ModelTier.POWERFUL: 3,
}


def analyze_scope(scope: AILScope) -> ScopeCost:
    """Compute cost annotations for every node in a scope."""
    ancestors = _build_ancestor_map(scope)

    node_costs: list[NodeCost] = []
    total_tokens = 0
    max_tier = ModelTier.NONE
    cacheable_count = 0

    for i, node in enumerate(scope.nodes):
        profile = NODE_COSTS.get(node.node_type)
        if profile is None:
            raise ValueError(f"No cost profile for node type: {node.node_type}")

        # A node's subgraph is cacheable if the node itself AND all ancestors are cacheable
        subgraph_cacheable = profile.cacheable and all(
            NODE_COSTS.get(scope.nodes[a].node_type, CostProfile(0, ModelTier.NONE, False)).cacheable
            for a in ancestors[i]
        )

        nc = NodeCost(
            node_index=i,
            node_type=node.node_type,
            profile=profile,
            cacheable_subgraph=subgraph_cacheable,
        )
        node_costs.append(nc)

        total_tokens += profile.token_cost
        if TIER_RANK[profile.model_tier] > TIER_RANK[max_tier]:
            max_tier = profile.model_tier
        if profile.cacheable:
            cacheable_count += 1

    return ScopeCost(
        scope_name=scope.name,
        node_costs=node_costs,
        total_token_cost=total_tokens,
        max_model_tier=max_tier,
        cacheable_nodes=cacheable_count,
        total_nodes=len(scope.nodes),
    )


def analyze_program(program: AILProgram) -> list[ScopeCost]:
    """Compute cost analysis for all scopes in a program."""
    return [analyze_scope(scope) for scope in program.scopes]


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ail_types import AILScope, AILNode, InputBinding, GraphRef, ScopeRef

    SEP = "─" * 60

    # ── Test 1: Pure compute pipeline — zero LLM cost ──
    print(f"\n{SEP}")
    print("TEST 1: Pure compute pipeline — zero cost")
    print(SEP)

    scope = AILScope("pure_compute")
    refs = [scope.next_ref() for _ in range(5)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", "⟨§.x > 0⟩")],
        outputs=[refs[1]]))
    scope.add_node(AILNode("∇:desc",
        inputs=[InputBinding("col", refs[1]), InputBinding("key", '"x"')],
        outputs=[refs[2]]))
    scope.add_node(AILNode("⊗:limit",
        inputs=[InputBinding("col", refs[2]), InputBinding("n", "#10")],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    cost = analyze_scope(scope)
    print(cost.summary())
    assert cost.total_token_cost == 0, "Pure compute should cost 0 tokens"
    assert cost.max_model_tier == ModelTier.NONE
    assert cost.cacheable_nodes == 3  # where, desc, limit (not io:in, io:out)
    print("✓ Zero LLM cost, 3/5 nodes cacheable")

    # ── Test 2: AI-native pipeline — expensive ──
    print(f"\n{SEP}")
    print("TEST 2: AI-native pipeline — LLM cost")
    print(SEP)

    scope = AILScope("ai_pipeline")
    refs = [scope.next_ref() for _ in range(6)]

    scope.add_node(AILNode("io:in",
        inputs=[InputBinding("param", '"user_message"')],
        outputs=[refs[0]]))
    scope.add_node(AILNode("~:intent",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))                          # 200 tok, mid
    scope.add_node(AILNode("~:threshold",
        inputs=[InputBinding("val", refs[1]), InputBinding("min", "~+")],
        outputs=[refs[2]]))                          # 0 tok, none
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[2])],
        outputs=[refs[3]]))                          # 500 tok, powerful
    scope.add_node(AILNode("~:rank",
        inputs=[InputBinding("col", refs[3])],
        outputs=[refs[4]]))                          # 300 tok, mid
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[4])]))

    cost = analyze_scope(scope)
    print(cost.summary())
    assert cost.total_token_cost == 1000, f"Expected ~1000 tokens, got {cost.total_token_cost}"
    assert cost.max_model_tier == ModelTier.POWERFUL
    assert cost.llm_nodes == 3  # intent, infer, rank
    print("✓ ~1000 tokens, max tier: powerful, 3 LLM nodes")

    # ── Test 3: Subgraph cacheability ──
    print(f"\n{SEP}")
    print("TEST 3: Subgraph cacheability analysis")
    print(SEP)

    scope = AILScope("cache_analysis")
    refs = [scope.next_ref() for _ in range(5)]

    # io:in (not cacheable) → ⊗:where (cacheable node, but ancestor isn't)
    # vs pure chain: ⊗:where → ∇:desc (both cacheable → subgraph cacheable)
    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", "⟨§.x > 0⟩")],
        outputs=[refs[1]]))
    scope.add_node(AILNode("∇:desc",
        inputs=[InputBinding("col", refs[1]), InputBinding("key", '"x"')],
        outputs=[refs[2]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[2])],
        outputs=[refs[3]]))                          # resets cacheability
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    cost = analyze_scope(scope)
    print(cost.summary())

    # io:in is not cacheable → no downstream node has a fully cacheable subgraph
    nc = cost.node_costs
    assert not nc[0].cacheable_subgraph, "io:in subgraph should not be cacheable"
    assert not nc[1].cacheable_subgraph, "where's ancestor (io:in) breaks cacheability"
    assert not nc[2].cacheable_subgraph, "desc's ancestor chain includes io:in"
    assert not nc[3].cacheable_subgraph, "~:infer is not cacheable"
    assert not nc[4].cacheable_subgraph, "io:out is not cacheable"
    print("✓ io:in breaks subgraph cacheability for all descendants")

    # ── Test 4: Embeddings are cacheable ──
    print(f"\n{SEP}")
    print("TEST 4: ~:embed is cacheable (same input → same vector)")
    print(SEP)

    scope = AILScope("embed_cache")
    refs = [scope.next_ref() for _ in range(4)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("~:embed",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))
    scope.add_node(AILNode("~:sim",
        inputs=[InputBinding("a", refs[1]), InputBinding("b", refs[1])],
        outputs=[refs[2]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[2])]))

    cost = analyze_scope(scope)
    print(cost.summary())
    assert cost.node_costs[1].profile.cacheable, "~:embed should be cacheable"
    assert cost.node_costs[2].profile.cacheable, "~:sim should be cacheable"
    assert cost.total_token_cost == 100, f"Only ~:embed costs tokens, got {cost.total_token_cost}"
    print("✓ ~:embed cacheable (100 tok), ~:sim cacheable (0 tok, pure math)")

    # ── Test 5: Full transpiled pipeline ──
    print(f"\n{SEP}")
    print("TEST 5: Transpiled pipeline — cost analysis")
    print(SEP)

    from transpiler import transpile

    prog = transpile("""
async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""")

    costs = analyze_program(prog)
    for c in costs:
        print(c.summary())

    print(f"\n{SEP}")
    print("ALL TESTS PASSED ✓")
    print(SEP)
