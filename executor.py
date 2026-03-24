"""
AIL v0.2 — Execution Semantics
Confidence propagation, DAG scheduling, error propagation.

Core invariant: confidence compounds multiplicatively through merge nodes,
passes through transforms, and resets at AI-native inference nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Optional, Callable

from ail_types import (
    AILNode, AILScope, AILProgram, AILType, AILBaseType,
    GraphRef, ScopeRef, InputBinding, CONFIDENCE_ANCHORS,
)


# ─── Confidence Propagation ──────────────────────────────────────────────────

class PropagationRule(Enum):
    """How a node category handles incoming confidence."""
    PASSTHROUGH     = "passthrough"      # output = single input confidence
    MULTIPLY        = "multiply"         # output = product of input confidences
    MINIMUM         = "minimum"          # output = min of input confidences
    RESET           = "reset"            # node produces its own confidence
    BRANCH          = "branch"           # output = confidence of taken branch
    IDENTITY        = "identity"         # output = 1.0 (ground truth)
    UNCERTAIN       = "uncertain"        # output = 0.5 (unknown source)


# Which propagation rule each node type uses
NODE_PROPAGATION: dict[str, PropagationRule] = {
    # IO — ground truth in, uncertain reads, passthrough out
    "io:in":        PropagationRule.IDENTITY,
    "io:out":       PropagationRule.PASSTHROUGH,
    "io:read":      PropagationRule.UNCERTAIN,
    "io:write":     PropagationRule.PASSTHROUGH,
    "io:emit":      PropagationRule.PASSTHROUGH,

    # Transform — confidence preserved
    "∿:map":        PropagationRule.PASSTHROUGH,
    "∿:flat":       PropagationRule.PASSTHROUGH,
    "∿:cast":       PropagationRule.PASSTHROUGH,
    "∿:norm":       PropagationRule.PASSTHROUGH,
    "∿:fmt":        PropagationRule.PASSTHROUGH,

    # Filter — confidence preserved (survivors keep their confidence)
    "⊗:where":      PropagationRule.PASSTHROUGH,
    "⊗:dedup":      PropagationRule.PASSTHROUGH,
    "⊗:limit":      PropagationRule.PASSTHROUGH,
    "⊗:skip":       PropagationRule.PASSTHROUGH,
    "⊗:null":       PropagationRule.PASSTHROUGH,

    # Aggregate — weakest link
    "Σ:sum":        PropagationRule.MINIMUM,
    "Σ:count":      PropagationRule.MINIMUM,
    "Σ:avg":        PropagationRule.MINIMUM,
    "Σ:max":        PropagationRule.MINIMUM,
    "Σ:min":        PropagationRule.MINIMUM,
    "Σ:group":      PropagationRule.MINIMUM,

    # Control — branch confidence
    "⊢:if":         PropagationRule.BRANCH,
    "⊢:match":      PropagationRule.BRANCH,
    "⊢:guard":      PropagationRule.PASSTHROUGH,
    "⊢:prob":       PropagationRule.BRANCH,

    # Loop — weakest link (each iteration can degrade)
    "⟳:loop":       PropagationRule.MINIMUM,
    "⟳:rec":        PropagationRule.MINIMUM,

    # Merge — multiplicative compounding
    "⊕:zip":        PropagationRule.MULTIPLY,
    "⊕:concat":     PropagationRule.MINIMUM,    # concat preserves weakest
    "⊕:merge":      PropagationRule.MULTIPLY,
    "⊕:fan-in":     PropagationRule.MINIMUM,    # fan-in collects, weakest link

    # Sort — confidence preserved
    "∇:asc":        PropagationRule.PASSTHROUGH,
    "∇:desc":       PropagationRule.PASSTHROUGH,

    # AI-native — reset (inference produces new confidence)
    "~:infer":      PropagationRule.RESET,
    "~:intent":     PropagationRule.RESET,
    "~:embed":      PropagationRule.RESET,

    # AI-native — operate on confidence (passthrough or domain-specific)
    "~:rank":       PropagationRule.PASSTHROUGH,
    "~:threshold":  PropagationRule.PASSTHROUGH,
    "~:sample":     PropagationRule.PASSTHROUGH,
    "~:sim":        PropagationRule.RESET,

    # Call — passthrough (callee scope handles its own propagation)
    "call":         PropagationRule.PASSTHROUGH,
}


def clamp(v: float) -> float:
    """Clamp confidence to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))


def propagate_confidence(
    node: AILNode,
    input_confidences: list[float],
    node_output_confidence: Optional[float] = None,
) -> float:
    """
    Compute the output confidence for a node given its input confidences.

    Args:
        node: The AIL node being executed.
        input_confidences: Confidence values from each input edge.
        node_output_confidence: For RESET nodes (e.g. ~:infer), the confidence
                                the node itself produces. If None, defaults to 0.5.

    Returns:
        Output confidence in [0.0, 1.0].

    The math:
        PASSTHROUGH:  c_out = c_in[0]                    (single input forwarded)
        MULTIPLY:     c_out = ∏(c_in)                    (uncertainty compounds)
        MINIMUM:      c_out = min(c_in)                  (weakest link)
        RESET:        c_out = node_output_confidence      (fresh from inference)
        BRANCH:       c_out = c_in[branch_taken]          (follows the branch)
        IDENTITY:     c_out = 1.0                         (ground truth)
        UNCERTAIN:    c_out = 0.5                         (unknown source)
    """
    rule = NODE_PROPAGATION.get(node.node_type)
    if rule is None:
        raise ValueError(f"No propagation rule for node type: {node.node_type}")

    if rule == PropagationRule.IDENTITY:
        return 1.0

    if rule == PropagationRule.UNCERTAIN:
        return 0.5

    if rule == PropagationRule.RESET:
        return clamp(node_output_confidence if node_output_confidence is not None else 0.5)

    if not input_confidences:
        return 1.0  # no inputs = ground truth (e.g. literal node)

    if rule == PropagationRule.PASSTHROUGH:
        return input_confidences[0]

    if rule == PropagationRule.MULTIPLY:
        result = 1.0
        for c in input_confidences:
            result *= c
        return clamp(result)

    if rule == PropagationRule.MINIMUM:
        return min(input_confidences)

    if rule == PropagationRule.BRANCH:
        # For branch nodes, the last input is typically the taken branch.
        # At static analysis time without runtime data, use minimum as conservative estimate.
        return min(input_confidences)

    raise ValueError(f"Unhandled propagation rule: {rule}")


# ─── DAG Scheduler ───────────────────────────────────────────────────────────

@dataclass
class NodeState:
    """Runtime state of a single node during execution."""
    node: AILNode
    index: int                              # position in scope's node list
    confidence: float = 1.0                 # output confidence after execution
    error: Optional[str] = None             # ! type error, if any
    executed: bool = False
    value: Any = None                       # runtime output (opaque)


@dataclass
class ExecutionPlan:
    """Topological ordering + parallelism groups for a scope."""
    levels: list[list[int]]                 # each level = nodes that can run in parallel
    total_nodes: int = 0

    def __str__(self):
        lines = []
        for i, level in enumerate(self.levels):
            par = " | " if len(level) > 1 else "   "
            nodes_str = par.join(f"§{n}" for n in level)
            lines.append(f"  L{i}: [{nodes_str}]")
        return "\n".join(lines)


def build_dependency_graph(scope: AILScope) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """
    Build adjacency lists from a scope's nodes.

    Returns (deps, dependents):
        deps[i] = set of node indices that node i depends on
        dependents[i] = set of node indices that depend on node i
    """
    # Map §N output ref → node index that produces it
    ref_to_producer: dict[int, int] = {}
    for i, node in enumerate(scope.nodes):
        for out_ref in node.outputs:
            if isinstance(out_ref, GraphRef):
                ref_to_producer[out_ref.index] = i

    deps: dict[int, set[int]] = defaultdict(set)
    dependents: dict[int, set[int]] = defaultdict(set)

    for i, node in enumerate(scope.nodes):
        for inp in node.inputs:
            if isinstance(inp.ref, GraphRef) and inp.ref.index in ref_to_producer:
                producer = ref_to_producer[inp.ref.index]
                if producer != i:
                    deps[i].add(producer)
                    dependents[producer].add(i)

    return deps, dependents


def schedule(scope: AILScope) -> ExecutionPlan:
    """
    Topological sort with parallelism extraction.

    Returns an ExecutionPlan where each level contains nodes that
    can execute concurrently (all dependencies satisfied by prior levels).

    Algorithm: Kahn's algorithm, grouping nodes by wavefront.
    """
    n = len(scope.nodes)
    deps, dependents = build_dependency_graph(scope)

    in_degree = [0] * n
    for i in range(n):
        in_degree[i] = len(deps.get(i, set()))

    # Seed: all nodes with no dependencies
    current_level = [i for i in range(n) if in_degree[i] == 0]
    levels: list[list[int]] = []
    visited = 0

    while current_level:
        levels.append(sorted(current_level))
        visited += len(current_level)
        next_level = []
        for node_idx in current_level:
            for dep_idx in dependents.get(node_idx, set()):
                in_degree[dep_idx] -= 1
                if in_degree[dep_idx] == 0:
                    next_level.append(dep_idx)
        current_level = next_level

    if visited != n:
        raise ValueError(
            f"Cycle detected in scope '{scope.name}': "
            f"scheduled {visited}/{n} nodes"
        )

    return ExecutionPlan(levels=levels, total_nodes=n)


# ─── Error Propagation ───────────────────────────────────────────────────────

class ErrorBehavior(Enum):
    """How a node handles an incoming ! error."""
    PROPAGATE   = "propagate"    # error passes through (most nodes)
    CATCH       = "catch"        # node handles error (guard)
    HALT        = "halt"         # stop execution entirely

# Most nodes propagate errors. Guards catch them.
NODE_ERROR_BEHAVIOR: dict[str, ErrorBehavior] = {
    "⊢:guard":  ErrorBehavior.CATCH,
}
# Everything else defaults to PROPAGATE.


def propagate_error(
    node: AILNode,
    input_errors: list[Optional[str]],
) -> Optional[str]:
    """
    Determine if a node's output carries an error.

    If ANY input has an error and the node doesn't catch it,
    the error propagates to the output.
    """
    behavior = NODE_ERROR_BEHAVIOR.get(node.node_type, ErrorBehavior.PROPAGATE)

    errors = [e for e in input_errors if e is not None]
    if not errors:
        return None

    if behavior == ErrorBehavior.CATCH:
        return None  # guard absorbs the error

    # Propagate first error encountered
    return errors[0]


# ─── Scope Executor ──────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Result of executing a scope."""
    scope_name: str
    node_states: list[NodeState]
    plan: ExecutionPlan
    output_confidence: float
    output_error: Optional[str]

    def confidence_trace(self) -> str:
        """Human-readable confidence flow through the graph."""
        lines = [f"⟦ {self.scope_name}  — confidence trace"]
        for ns in self.node_states:
            err = f" !{ns.error}" if ns.error else ""
            conf_label = _confidence_label(ns.confidence)
            lines.append(
                f"  §{ns.index} [{ns.node.node_type}] → "
                f"~{ns.confidence:.4f} ({conf_label}){err}"
            )
        lines.append(f"⟧ → ~{self.output_confidence:.4f}")
        return "\n".join(lines)


def _confidence_label(c: float) -> str:
    """Map confidence to nearest named anchor."""
    if c >= 1.0:
        return "~!"
    if c > 0.9:
        return "~+"
    if c > 0.3:
        return "~?"
    if c > 0.0:
        return "~-"
    return "~∅"


def execute_scope(
    scope: AILScope,
    node_overrides: Optional[dict[int, float]] = None,
) -> ExecutionResult:
    """
    Execute a scope: schedule nodes, propagate confidence and errors.

    Args:
        scope: The AIL scope to execute.
        node_overrides: Optional dict mapping node index → confidence override.
                        Used for AI-native RESET nodes (e.g. ~:infer produces
                        a specific confidence at runtime).

    Returns:
        ExecutionResult with full confidence trace.
    """
    overrides = node_overrides or {}
    plan = schedule(scope)
    deps, _ = build_dependency_graph(scope)

    # Map §N → node index that produces it (for confidence lookup)
    ref_to_node: dict[int, int] = {}
    for i, node in enumerate(scope.nodes):
        for out_ref in node.outputs:
            if isinstance(out_ref, GraphRef):
                ref_to_node[out_ref.index] = i

    states = [NodeState(node=node, index=i) for i, node in enumerate(scope.nodes)]

    for level in plan.levels:
        for node_idx in level:
            node = scope.nodes[node_idx]
            state = states[node_idx]

            # Gather input confidences
            input_confidences = []
            input_errors = []
            for inp in node.inputs:
                if isinstance(inp.ref, GraphRef) and inp.ref.index in ref_to_node:
                    producer_idx = ref_to_node[inp.ref.index]
                    input_confidences.append(states[producer_idx].confidence)
                    input_errors.append(states[producer_idx].error)

            # Error propagation
            state.error = propagate_error(node, input_errors)

            # Confidence propagation
            override = overrides.get(node_idx)
            state.confidence = propagate_confidence(
                node, input_confidences, node_output_confidence=override
            )
            state.executed = True

    # Output confidence = confidence of last node
    output_conf = states[-1].confidence if states else 1.0
    output_err = states[-1].error if states else None

    return ExecutionResult(
        scope_name=scope.name,
        node_states=states,
        plan=plan,
        output_confidence=output_conf,
        output_error=output_err,
    )


# ─── Convenience: Execute full program ───────────────────────────────────────

def execute_program(
    program: AILProgram,
    scope_overrides: Optional[dict[str, dict[int, float]]] = None,
) -> list[ExecutionResult]:
    """Execute all scopes in a program."""
    overrides = scope_overrides or {}
    return [
        execute_scope(scope, overrides.get(scope.name))
        for scope in program.scopes
    ]


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "─" * 60

    # ── Test 1: Passthrough chain ──
    print(f"\n{SEP}")
    print("TEST 1: Transform chain — confidence preserved")
    print(SEP)

    scope = AILScope("passthrough_chain")
    refs = [scope.next_ref() for _ in range(5)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", "⟨§.x > 0⟩")],
        outputs=[refs[1]]))
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[1]), InputBinding("fn", "⟨§.x * 2⟩")],
        outputs=[refs[2]]))
    scope.add_node(AILNode("∇:desc",
        inputs=[InputBinding("col", refs[2]), InputBinding("key", '"x"')],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    result = execute_scope(scope)
    print(result.confidence_trace())
    print(f"\nExecution plan:\n{result.plan}")
    assert result.output_confidence == 1.0, "Passthrough should preserve 1.0"
    print("✓ Confidence preserved through transform chain")

    # ── Test 2: Multiplicative merge ──
    print(f"\n{SEP}")
    print("TEST 2: Merge — multiplicative compounding")
    print(SEP)

    scope = AILScope("merge_compound")
    refs = [scope.next_ref() for _ in range(4)]

    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"api_a"')],
        outputs=[refs[0]]))  # → ~0.5 (uncertain)
    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"api_b"')],
        outputs=[refs[1]]))  # → ~0.5 (uncertain)
    scope.add_node(AILNode("⊕:merge",
        inputs=[InputBinding("a", refs[0]), InputBinding("b", refs[1])],
        outputs=[refs[2]]))  # → ~0.25 (0.5 × 0.5)
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[2])]))

    result = execute_scope(scope)
    print(result.confidence_trace())
    assert abs(result.output_confidence - 0.25) < 1e-9, \
        f"Merge of two ~0.5 should be ~0.25, got {result.output_confidence}"
    print("✓ Multiplicative compounding: ~0.5 × ~0.5 = ~0.25")

    # ── Test 3: AI-native reset ──
    print(f"\n{SEP}")
    print("TEST 3: ~:infer resets confidence")
    print(SEP)

    scope = AILScope("infer_reset")
    refs = [scope.next_ref() for _ in range(4)]

    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"noisy_data"')],
        outputs=[refs[0]]))  # → ~0.5
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))  # → ~0.95 (override)
    scope.add_node(AILNode("~:threshold",
        inputs=[InputBinding("val", refs[1]), InputBinding("min", "~+")],
        outputs=[refs[2]]))  # → ~0.95 (passthrough)
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[2])]))

    result = execute_scope(scope, node_overrides={1: 0.95})
    print(result.confidence_trace())
    assert abs(result.output_confidence - 0.95) < 1e-9, \
        f"Infer should reset to 0.95, got {result.output_confidence}"
    print("✓ ~:infer resets confidence: ~0.5 → [~:infer] → ~0.95")

    # ── Test 4: Error propagation ──
    print(f"\n{SEP}")
    print("TEST 4: Error propagation through ! type")
    print(SEP)

    scope = AILScope("error_prop")
    refs = [scope.next_ref() for _ in range(4)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[0]), InputBinding("fn", "⟨fail⟩")],
        outputs=[refs[1]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[1]), InputBinding("cond", "⟨true⟩")],
        outputs=[refs[2]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[2])]))

    # Simulate: node 1 errors
    result = execute_scope(scope)
    # Manually inject error to test propagation
    states = result.node_states
    states[1].error = "!\"division by zero\""
    # Re-propagate from node 2 onward
    for i in [2, 3]:
        input_errors = []
        for inp in scope.nodes[i].inputs:
            if isinstance(inp.ref, GraphRef):
                ref_to_node_map = {}
                for j, n in enumerate(scope.nodes):
                    for o in n.outputs:
                        if isinstance(o, GraphRef):
                            ref_to_node_map[o.index] = j
                if inp.ref.index in ref_to_node_map:
                    input_errors.append(states[ref_to_node_map[inp.ref.index]].error)
        states[i].error = propagate_error(scope.nodes[i], input_errors)

    print(f"  §0 [io:in]     → error: {states[0].error}")
    print(f"  §1 [∿:map]     → error: {states[1].error}")
    print(f"  §2 [⊗:where]   → error: {states[2].error}")
    print(f"  §3 [io:out]    → error: {states[3].error}")
    assert states[2].error is not None, "Error should propagate through filter"
    assert states[3].error is not None, "Error should propagate to output"
    print("✓ Error propagates: ∿:map ! → ⊗:where ! → io:out !")

    # ── Test 5: Parallel scheduling ──
    print(f"\n{SEP}")
    print("TEST 5: Parallel scheduling — independent reads")
    print(SEP)

    scope = AILScope("parallel_reads")
    refs = [scope.next_ref() for _ in range(5)]

    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"db"')],
        outputs=[refs[0]]))
    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"cache"')],
        outputs=[refs[1]]))
    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"api"')],
        outputs=[refs[2]]))
    scope.add_node(AILNode("⊕:fan-in",
        inputs=[
            InputBinding("a", refs[0]),
            InputBinding("b", refs[1]),
            InputBinding("c", refs[2]),
        ],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    result = execute_scope(scope)
    print(f"Execution plan:\n{result.plan}")
    print(result.confidence_trace())
    assert len(result.plan.levels) == 3, \
        f"Expected 3 levels (reads | fan-in | out), got {len(result.plan.levels)}"
    assert len(result.plan.levels[0]) == 3, \
        f"Expected 3 parallel reads in L0, got {len(result.plan.levels[0])}"
    print("✓ 3 reads execute in parallel at L0, fan-in at L1, out at L2")

    # ── Test 6: Full pipeline from transpiler ──
    print(f"\n{SEP}")
    print("TEST 6: Transpiled pipeline — end-to-end confidence")
    print(SEP)

    from transpiler import transpile

    prog = transpile("""
async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""")

    results = execute_program(prog)
    for r in results:
        print(r.confidence_trace())
        print(f"\nExecution plan:\n{r.plan}")

    print(f"\n{SEP}")
    print("ALL TESTS PASSED ✓")
    print(SEP)
