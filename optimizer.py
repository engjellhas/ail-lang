"""
AIL v0.2 — Graph Optimizer
Takes an AIL graph, returns a cheaper equivalent graph.

Optimization passes (applied in order):
  1. Filter fusion     — collapse sequential ⊗:where nodes into one
  2. Dead node elim    — remove nodes whose outputs are never consumed
  3. Confidence pruning — cut branches where ~ drops below a threshold
  4. Duplicate elim    — merge identical nodes (same type + same inputs)
  5. Model annotation  — stamp each AI-native node with suggested model tier
  6. Cache marking     — flag cacheable subgraphs in metadata

Each pass is independent and produces a valid AIL graph.
The optimizer is the monetizable core — everything else is infrastructure for this.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from ail_types import (
    AILNode, AILScope, AILProgram,
    GraphRef, ScopeRef, InputBinding, InlineExpr,
)
from executor import (
    execute_scope, build_dependency_graph, schedule,
    NODE_PROPAGATION, PropagationRule,
)
from cost_model import (
    analyze_scope, NODE_COSTS, ModelTier, CostProfile, TIER_RANK,
)


# ─── Optimization Result ─────────────────────────────────────────────────────

@dataclass
class OptimizationReport:
    """What the optimizer did."""
    original_nodes: int
    optimized_nodes: int
    filters_fused: int = 0
    dead_nodes_removed: int = 0
    branches_pruned: int = 0
    duplicates_merged: int = 0
    nodes_annotated: int = 0
    cache_subgraphs: int = 0
    original_token_cost: int = 0
    optimized_token_cost: int = 0

    @property
    def nodes_eliminated(self) -> int:
        return self.original_nodes - self.optimized_nodes

    @property
    def token_savings(self) -> int:
        return self.original_token_cost - self.optimized_token_cost

    def summary(self) -> str:
        lines = [
            "╔═ Optimization Report ═══════════════════╗",
            f"  Nodes:    {self.original_nodes} → {self.optimized_nodes} "
            f"(−{self.nodes_eliminated})",
            f"  Tokens:   ~{self.original_token_cost} → ~{self.optimized_token_cost} "
            f"(saved ~{self.token_savings})",
        ]
        if self.filters_fused:
            lines.append(f"  Filters fused:      {self.filters_fused}")
        if self.dead_nodes_removed:
            lines.append(f"  Dead nodes removed: {self.dead_nodes_removed}")
        if self.branches_pruned:
            lines.append(f"  Branches pruned:    {self.branches_pruned}")
        if self.duplicates_merged:
            lines.append(f"  Duplicates merged:  {self.duplicates_merged}")
        if self.nodes_annotated:
            lines.append(f"  Model annotations:  {self.nodes_annotated}")
        if self.cache_subgraphs:
            lines.append(f"  Cache subgraphs:    {self.cache_subgraphs}")
        lines.append("╚═════════════════════════════════════════╝")
        return "\n".join(lines)


# ─── Pass 1: Filter Fusion ───────────────────────────────────────────────────

def _fuse_filters(scope: AILScope) -> tuple[AILScope, int]:
    """
    Collapse sequential ⊗:where → ⊗:where into a single ⊗:where
    with a combined condition (AND of both conditions).

    Before: [⊗:where] ← cond:⟨A⟩ → §1  ...  [⊗:where] ← col:§1, cond:⟨B⟩ → §2
    After:  [⊗:where] ← cond:⟨A ∧ B⟩ → §2
    """
    if len(scope.nodes) < 2:
        return scope, 0

    # Map output ref → node index
    ref_to_idx: dict[int, int] = {}
    for i, node in enumerate(scope.nodes):
        for out in node.outputs:
            if isinstance(out, GraphRef):
                ref_to_idx[out.index] = i

    fused_count = 0
    skip = set()

    new_nodes = []
    for i, node in enumerate(scope.nodes):
        if i in skip:
            continue

        if node.node_type != "⊗:where":
            new_nodes.append(copy.deepcopy(node))
            continue

        # Look for a chain: this ⊗:where → next ⊗:where consuming our output
        current = node
        conditions = []
        col_input = None
        final_outputs = current.outputs

        # Gather condition from current node
        for inp in current.inputs:
            if inp.role == "cond":
                conditions.append(str(inp.ref))
            elif inp.role == "col":
                col_input = inp.ref

        # Walk forward through consecutive filters
        next_idx = i
        while True:
            # Find consumer of current node's output
            if not final_outputs:
                break
            out_ref = final_outputs[0] if final_outputs else None
            if not isinstance(out_ref, GraphRef):
                break

            # Find who consumes this ref
            consumer_idx = None
            for j, n in enumerate(scope.nodes):
                if j <= next_idx or j in skip:
                    continue
                for inp in n.inputs:
                    if isinstance(inp.ref, GraphRef) and inp.ref.index == out_ref.index:
                        consumer_idx = j
                        break
                if consumer_idx is not None:
                    break

            if consumer_idx is None:
                break
            consumer = scope.nodes[consumer_idx]
            if consumer.node_type != "⊗:where":
                break

            # Check that consumer's col input is our output (direct chain)
            is_chain = False
            for inp in consumer.inputs:
                if inp.role == "col" and isinstance(inp.ref, GraphRef):
                    if inp.ref.index == out_ref.index:
                        is_chain = True
                        break
            if not is_chain:
                break

            # Fuse: grab consumer's condition
            for inp in consumer.inputs:
                if inp.role == "cond":
                    conditions.append(str(inp.ref))

            skip.add(consumer_idx)
            final_outputs = consumer.outputs
            next_idx = consumer_idx
            fused_count += 1

        # Build fused node
        fused_cond = " ∧ ".join(conditions) if len(conditions) > 1 else (conditions[0] if conditions else "⟨true⟩")
        inputs = []
        if col_input is not None:
            inputs.append(InputBinding("col", col_input))
        inputs.append(InputBinding("cond", InlineExpr(fused_cond) if len(conditions) > 1 else (InlineExpr(fused_cond.strip("⟨⟩")) if fused_cond.startswith("⟨") else fused_cond)))

        # Preserve original condition format if no fusion happened
        if fused_count == 0 or len(conditions) <= 1:
            new_nodes.append(copy.deepcopy(node))
        else:
            fused_node = AILNode(
                "⊗:where",
                inputs=inputs,
                outputs=list(final_outputs),
                metadata={"fused_from": len(conditions)},
            )
            new_nodes.append(fused_node)

    result = AILScope(scope.name)
    result.nodes = new_nodes
    result.ref_counter = scope.ref_counter
    return result, fused_count


# ─── Pass 2: Dead Node Elimination ───────────────────────────────────────────

def _eliminate_dead_nodes(scope: AILScope) -> tuple[AILScope, int]:
    """
    Remove nodes whose outputs are never consumed by any other node,
    EXCEPT io:out and io:write and io:emit (side effects).
    """
    # Collect all consumed refs
    consumed_refs: set[int] = set()
    for node in scope.nodes:
        for inp in node.inputs:
            if isinstance(inp.ref, GraphRef):
                consumed_refs.add(inp.ref.index)

    # Which nodes are side-effect sinks (always keep)
    SINK_TYPES = {"io:out", "io:write", "io:emit"}

    # Mark nodes to keep: sinks + nodes whose outputs are consumed
    keep = set()
    for i, node in enumerate(scope.nodes):
        if node.node_type in SINK_TYPES:
            keep.add(i)
            continue
        for out in node.outputs:
            if isinstance(out, GraphRef) and out.index in consumed_refs:
                keep.add(i)
                break

    # Also keep all ancestors of kept nodes
    deps, _ = build_dependency_graph(scope)
    to_visit = list(keep)
    while to_visit:
        idx = to_visit.pop()
        for dep in deps.get(idx, set()):
            if dep not in keep:
                keep.add(dep)
                to_visit.append(dep)

    removed = len(scope.nodes) - len(keep)
    result = AILScope(scope.name)
    result.nodes = [copy.deepcopy(scope.nodes[i]) for i in sorted(keep)]
    result.ref_counter = scope.ref_counter
    return result, removed


# ─── Pass 3: Confidence Pruning ──────────────────────────────────────────────

def _prune_low_confidence(
    scope: AILScope,
    threshold: float = 0.1,
    node_overrides: Optional[dict[int, float]] = None,
) -> tuple[AILScope, int]:
    """
    Run confidence propagation, then remove nodes (and their descendants)
    where confidence drops below threshold.

    Nodes below threshold are replaced with nothing — their downstream
    consumers will be caught by dead node elimination.
    """
    exec_result = execute_scope(scope, node_overrides)

    # Find nodes below threshold
    below = set()
    for ns in exec_result.node_states:
        if ns.confidence < threshold and ns.node.node_type not in {"io:in", "io:out"}:
            below.add(ns.index)

    if not below:
        return scope, 0

    # Also remove all descendants of pruned nodes
    _, dependents = build_dependency_graph(scope)
    to_remove = set(below)
    queue = list(below)
    while queue:
        idx = queue.pop()
        for dep in dependents.get(idx, set()):
            if dep not in to_remove:
                to_remove.add(dep)
                queue.append(dep)

    # Keep io:out even if pruned (it's the program output)
    to_remove = {i for i in to_remove if scope.nodes[i].node_type != "io:out"}

    result = AILScope(scope.name)
    result.nodes = [copy.deepcopy(scope.nodes[i]) for i in range(len(scope.nodes)) if i not in to_remove]
    result.ref_counter = scope.ref_counter
    return result, len(to_remove)


# ─── Pass 4: Duplicate Elimination ───────────────────────────────────────────

def _node_signature(node: AILNode) -> str:
    """Content-addressable signature: type + sorted inputs."""
    inputs_sig = ",".join(
        f"{inp.role}:{inp.ref}" for inp in sorted(node.inputs, key=lambda i: i.role)
    )
    return f"{node.node_type}|{inputs_sig}"


def _eliminate_duplicates(scope: AILScope) -> tuple[AILScope, int]:
    """
    Merge nodes with identical signatures (same type + same inputs).
    The first occurrence wins; later duplicates are removed and their
    output refs are rewritten to point to the first occurrence.
    """
    seen: dict[str, int] = {}  # signature → first node index
    ref_remap: dict[int, int] = {}  # old output ref → replacement ref
    keep = []
    merged = 0

    for i, node in enumerate(scope.nodes):
        sig = _node_signature(node)

        if sig in seen and node.node_type not in {"io:in", "io:out", "io:write", "io:emit"}:
            # Duplicate — remap outputs
            original_idx = seen[sig]
            original_node = scope.nodes[original_idx]
            for old_out, new_out in zip(node.outputs, original_node.outputs):
                if isinstance(old_out, GraphRef) and isinstance(new_out, GraphRef):
                    ref_remap[old_out.index] = new_out.index
            merged += 1
        else:
            seen[sig] = i
            keep.append(i)

    if not ref_remap:
        return scope, 0

    # Rewrite refs in kept nodes
    result = AILScope(scope.name)
    for i in keep:
        node = copy.deepcopy(scope.nodes[i])
        for inp in node.inputs:
            if isinstance(inp.ref, GraphRef) and inp.ref.index in ref_remap:
                inp.ref = GraphRef(ref_remap[inp.ref.index], inp.ref.streaming)
        result.nodes.append(node)
    result.ref_counter = scope.ref_counter
    return result, merged


# ─── Pass 5: Model Annotation ────────────────────────────────────────────────

def _annotate_models(scope: AILScope) -> tuple[AILScope, int]:
    """
    Stamp each AI-native node's metadata with the suggested model tier.
    """
    result = copy.deepcopy(scope)
    annotated = 0

    for node in result.nodes:
        profile = NODE_COSTS.get(node.node_type)
        if profile and profile.model_tier != ModelTier.NONE:
            node.metadata["model_tier"] = profile.model_tier.value
            node.metadata["token_cost"] = profile.token_cost
            annotated += 1

    return result, annotated


# ─── Pass 6: Cache Marking ───────────────────────────────────────────────────

def _mark_cacheable(scope: AILScope) -> tuple[AILScope, int]:
    """
    Flag cacheable subgraphs in node metadata.
    A subgraph is cacheable if the node AND all its ancestors are cacheable.
    """
    cost = analyze_scope(scope)
    result = copy.deepcopy(scope)
    marked = 0

    for nc in cost.node_costs:
        if nc.cacheable_subgraph:
            result.nodes[nc.node_index].metadata["cacheable_subgraph"] = True
            marked += 1
        if nc.profile.cacheable:
            result.nodes[nc.node_index].metadata["cacheable"] = True

    return result, marked


# ─── Main Optimizer ──────────────────────────────────────────────────────────

def optimize_scope(
    scope: AILScope,
    confidence_threshold: float = 0.1,
    node_overrides: Optional[dict[int, float]] = None,
) -> tuple[AILScope, OptimizationReport]:
    """
    Run all optimization passes on a scope.
    Returns (optimized_scope, report).
    """
    original_cost = analyze_scope(scope)
    report = OptimizationReport(
        original_nodes=len(scope.nodes),
        optimized_nodes=0,
        original_token_cost=original_cost.total_token_cost,
    )

    current = scope

    # Pass 1: Filter fusion
    current, fused = _fuse_filters(current)
    report.filters_fused = fused

    # Pass 2: Dead node elimination
    current, dead = _eliminate_dead_nodes(current)
    report.dead_nodes_removed = dead

    # Pass 3: Confidence pruning
    current, pruned = _prune_low_confidence(current, confidence_threshold, node_overrides)
    report.branches_pruned = pruned

    # Pass 4: Duplicate elimination
    current, dupes = _eliminate_duplicates(current)
    report.duplicates_merged = dupes

    # Pass 5: Model annotation
    current, annotated = _annotate_models(current)
    report.nodes_annotated = annotated

    # Pass 6: Cache marking
    current, cached = _mark_cacheable(current)
    report.cache_subgraphs = cached

    optimized_cost = analyze_scope(current)
    report.optimized_nodes = len(current.nodes)
    report.optimized_token_cost = optimized_cost.total_token_cost

    return current, report


def optimize_program(
    program: AILProgram,
    confidence_threshold: float = 0.1,
    scope_overrides: Optional[dict[str, dict[int, float]]] = None,
) -> tuple[AILProgram, list[OptimizationReport]]:
    """Optimize all scopes in a program."""
    overrides = scope_overrides or {}
    result = AILProgram(version=program.version)
    reports = []

    for scope in program.scopes:
        opt_scope, report = optimize_scope(
            scope, confidence_threshold, overrides.get(scope.name)
        )
        result.add_scope(opt_scope)
        reports.append(report)

    return result, reports


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "─" * 60

    # ── Test 1: Filter fusion ──
    print(f"\n{SEP}")
    print("TEST 1: Filter fusion — two sequential ⊗:where → one")
    print(SEP)

    scope = AILScope("filter_fusion")
    refs = [scope.next_ref() for _ in range(5)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", InlineExpr("§.age ≥ 18"))],
        outputs=[refs[1]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[1]), InputBinding("cond", InlineExpr("§.score > 0.5"))],
        outputs=[refs[2]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[2]), InputBinding("cond", InlineExpr("§.active = true"))],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    print(f"BEFORE ({len(scope.nodes)} nodes):")
    print(scope.to_ail())

    opt, report = optimize_scope(scope)
    print(f"\nAFTER ({len(opt.nodes)} nodes):")
    print(opt.to_ail())
    print(f"\n{report.summary()}")
    assert report.filters_fused > 0, "Should fuse sequential filters"
    print("✓ Sequential filters fused")

    # ── Test 2: Dead node elimination ──
    print(f"\n{SEP}")
    print("TEST 2: Dead node elimination — unused branch removed")
    print(SEP)

    scope = AILScope("dead_elim")
    refs = [scope.next_ref() for _ in range(5)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[0]), InputBinding("fn", InlineExpr("§.x * 2"))],
        outputs=[refs[1]]))
    # §2 is computed but never consumed — dead node
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[0]), InputBinding("fn", InlineExpr("§.x * 3"))],
        outputs=[refs[2]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[1])]))

    print(f"BEFORE ({len(scope.nodes)} nodes):")
    print(scope.to_ail())

    opt, report = optimize_scope(scope)
    print(f"\nAFTER ({len(opt.nodes)} nodes):")
    print(opt.to_ail())
    print(f"\n{report.summary()}")
    assert report.dead_nodes_removed >= 1, "Should remove dead §2 node"
    print("✓ Dead node eliminated")

    # ── Test 3: Confidence pruning ──
    print(f"\n{SEP}")
    print("TEST 3: Confidence pruning — low-confidence branch cut")
    print(SEP)

    scope = AILScope("conf_prune")
    refs = [scope.next_ref() for _ in range(6)]

    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"noisy_a"')],
        outputs=[refs[0]]))  # ~0.5
    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"noisy_b"')],
        outputs=[refs[1]]))  # ~0.5
    scope.add_node(AILNode("⊕:merge",
        inputs=[InputBinding("a", refs[0]), InputBinding("b", refs[1])],
        outputs=[refs[2]]))  # ~0.25
    scope.add_node(AILNode("⊕:merge",
        inputs=[InputBinding("a", refs[2]), InputBinding("b", refs[2])],
        outputs=[refs[3]]))  # ~0.0625 — below 0.1 threshold!
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[3]), InputBinding("fn", InlineExpr("§.x"))],
        outputs=[refs[4]]))  # would be ~0.0625, descendant of pruned
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[4])]))

    print(f"BEFORE ({len(scope.nodes)} nodes):")
    print(scope.to_ail())

    # Run confidence trace first
    exec_result = execute_scope(scope)
    print("\nConfidence trace:")
    print(exec_result.confidence_trace())

    opt, report = optimize_scope(scope, confidence_threshold=0.1)
    print(f"\nAFTER ({len(opt.nodes)} nodes):")
    print(opt.to_ail())
    print(f"\n{report.summary()}")
    assert report.branches_pruned >= 1, "Should prune nodes below 0.1 confidence"
    print("✓ Low-confidence branch pruned")

    # ── Test 4: Duplicate elimination ──
    print(f"\n{SEP}")
    print("TEST 4: Duplicate elimination — identical nodes merged")
    print(SEP)

    scope = AILScope("dedup")
    refs = [scope.next_ref() for _ in range(6)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    # Two identical maps on the same input
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[0]), InputBinding("fn", InlineExpr("§.upper"))],
        outputs=[refs[1]]))
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[0]), InputBinding("fn", InlineExpr("§.upper"))],
        outputs=[refs[2]]))
    # Both consumed downstream
    scope.add_node(AILNode("⊕:concat",
        inputs=[InputBinding("a", refs[1]), InputBinding("b", refs[2])],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    print(f"BEFORE ({len(scope.nodes)} nodes):")
    print(scope.to_ail())

    opt, report = optimize_scope(scope)
    print(f"\nAFTER ({len(opt.nodes)} nodes):")
    print(opt.to_ail())
    print(f"\n{report.summary()}")
    assert report.duplicates_merged >= 1, "Should merge identical maps"
    print("✓ Duplicate nodes merged")

    # ── Test 5: Model annotation + cache marking ──
    print(f"\n{SEP}")
    print("TEST 5: Model annotation + cache marking")
    print(SEP)

    scope = AILScope("annotate")
    refs = [scope.next_ref() for _ in range(5)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("~:embed",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[1])],
        outputs=[refs[2]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[2])]))

    opt, report = optimize_scope(scope)
    print(f"Optimized:\n{opt.to_ail()}")
    print(f"\nNode metadata:")
    for i, node in enumerate(opt.nodes):
        if node.metadata:
            print(f"  §{i} [{node.node_type}] → {node.metadata}")
    print(f"\n{report.summary()}")
    assert report.nodes_annotated >= 2, "Should annotate ~:embed and ~:infer"
    print("✓ AI-native nodes annotated with model tier + cost")

    # ── Test 6: Full pipeline — all passes ──
    print(f"\n{SEP}")
    print("TEST 6: Full pipeline — transpiled + optimized")
    print(SEP)

    from transpiler import transpile

    prog = transpile("""
async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""")

    print(f"BEFORE:")
    print(prog.to_ail())

    opt_prog, reports = optimize_program(prog)
    print(f"\nAFTER:")
    print(opt_prog.to_ail())
    for r in reports:
        print(f"\n{r.summary()}")

    print(f"\n{SEP}")
    print("ALL TESTS PASSED ✓")
    print(SEP)
