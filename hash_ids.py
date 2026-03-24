"""
AIL v0.2 — Hash-Based Node IDs (P3)
Content-addressable node identifiers that replace positional §N refs.

Why:
  §1827 is meaningless. A hash of (node_type + inputs) is:
  1. Stable across graph restructuring
  2. Enables subgraph deduplication automatically
  3. Makes caching trivial — same hash = same result
  4. Scales to arbitrarily large graphs

Hash format: first 8 chars of SHA-256 hex → e.g. "§a3f7c012"
Short enough for human readability, long enough to avoid collisions
at any practical graph size (4 billion unique addresses).

This module provides:
  - hash_node()       — compute content-addressable ID for a single node
  - hash_scope()      — rewrite a scope from §N refs to §<hash> refs
  - hash_program()    — rewrite an entire program
  - dedup_by_hash()   — eliminate duplicate nodes (same hash = same computation)
"""

from __future__ import annotations

import hashlib
import copy
from dataclasses import dataclass, field
from typing import Optional

from ail_types import (
    AILNode, AILScope, AILProgram,
    GraphRef, ScopeRef, InputBinding, InlineExpr,
)


# ─── Hash Computation ───────────────────────────────────────────────────────

def _canonical_ref(ref) -> str:
    """Produce a canonical string representation of a ref for hashing."""
    if isinstance(ref, GraphRef):
        tag = "S" if ref.streaming else "G"
        return f"{tag}:{ref.index}"
    elif isinstance(ref, ScopeRef):
        inner = f".{ref.inner_ref.index}" if ref.inner_ref else ""
        return f"@{ref.scope_name}{inner}"
    elif isinstance(ref, InlineExpr):
        return f"E:{ref.expression}"
    else:
        return f"L:{ref}"


def _canonical_inputs(inputs: list[InputBinding]) -> str:
    """Sorted, deterministic string of all inputs."""
    parts = sorted(f"{inp.role}={_canonical_ref(inp.ref)}" for inp in inputs)
    return "|".join(parts)


def hash_node(node: AILNode) -> str:
    """
    Compute content-addressable hash for a node.

    hash = SHA-256(node_type + canonical_inputs)[:8]

    Two nodes with the same type and same inputs will always
    produce the same hash, enabling automatic deduplication.
    """
    payload = f"{node.node_type}:{_canonical_inputs(node.inputs)}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return h


def hash_node_with_deps(node: AILNode, dep_hashes: dict[int, str]) -> str:
    """
    Compute content-addressable hash using dependency hashes instead of §N.

    This is the "deep" hash: it recursively depends on the content of
    all ancestor nodes, not just their positional index. This means:
    - Identical subgraphs always produce identical hashes
    - Moving a subgraph doesn't change its hash
    - Renumbering §N refs doesn't change the hash

    Args:
        node: The node to hash.
        dep_hashes: Map from §N index → hash of the node that produces §N.
    """
    def resolve_ref(ref) -> str:
        if isinstance(ref, GraphRef) and ref.index in dep_hashes:
            tag = "S" if ref.streaming else "G"
            return f"{tag}:{dep_hashes[ref.index]}"
        return _canonical_ref(ref)

    parts = sorted(
        f"{inp.role}={resolve_ref(inp.ref)}" for inp in node.inputs
    )
    payload = f"{node.node_type}:{'|'.join(parts)}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return h


# ─── Hashed Node ────────────────────────────────────────────────────────────

@dataclass
class HashedNode:
    """A node with both its positional §N index and content-addressable hash."""
    index: int
    hash_id: str
    node: AILNode
    dep_hashes: list[str] = field(default_factory=list)

    def __str__(self):
        return f"§{self.hash_id} [{self.node.node_type}] (was §{self.index})"


@dataclass
class HashedScope:
    """A scope with all nodes hashed."""
    name: str
    nodes: list[HashedNode]
    hash_map: dict[int, str]          # §N → hash
    reverse_map: dict[str, int]       # hash → §N (first occurrence)
    duplicates: list[tuple[int, int]] # (duplicate_idx, original_idx)

    def to_ail(self, indent: int = 2) -> str:
        """Render with hash-based refs instead of positional."""
        pad = " " * indent
        lines = [f"⟦ {self.name}"]
        for hn in self.nodes:
            node = hn.node
            parts = [f"[{node.node_type}]"]
            if node.inputs:
                parts.append("←")
                inp_strs = []
                for inp in node.inputs:
                    if isinstance(inp.ref, GraphRef) and inp.ref.index in self.hash_map:
                        h = self.hash_map[inp.ref.index]
                        prefix = "§∞" if inp.ref.streaming else "§"
                        inp_strs.append(f"{inp.role}:{prefix}{h}")
                    else:
                        inp_strs.append(str(inp))
                parts.append(", ".join(inp_strs))
            if node.outputs:
                parts.append("→")
                out_strs = []
                for out in node.outputs:
                    if isinstance(out, GraphRef) and out.index in self.hash_map:
                        h = self.hash_map[out.index]
                        prefix = "§∞" if out.streaming else "§"
                        out_strs.append(f"{prefix}{h}")
                    else:
                        out_strs.append(str(out))
                parts.append(", ".join(out_strs))
            lines.append(f"{pad}{' '.join(parts)}")
        lines.append("⟧")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"⟦ {self.name} — hash analysis",
            f"  Total nodes:    {len(self.nodes)}",
            f"  Unique hashes:  {len(self.reverse_map)}",
            f"  Duplicates:     {len(self.duplicates)}",
        ]
        if self.duplicates:
            lines.append("  Duplicate pairs:")
            for dup, orig in self.duplicates:
                h = self.hash_map[dup]
                lines.append(f"    §{dup} = §{orig} (hash: {h})")
        lines.append("⟧")
        return "\n".join(lines)


# ─── Scope Hashing ──────────────────────────────────────────────────────────

def hash_scope(scope: AILScope) -> HashedScope:
    """
    Compute content-addressable hashes for every node in a scope.

    Algorithm:
    1. Topological order (nodes are already in dependency order by construction)
    2. For each node, resolve input refs to their producer's hash
    3. Compute deep hash: SHA-256(node_type + resolved_inputs)
    4. Detect duplicates: same hash = same computation

    Returns HashedScope with full hash map and duplicate detection.
    """
    # Build output ref → node index map
    ref_to_producer: dict[int, int] = {}
    for i, node in enumerate(scope.nodes):
        for out in node.outputs:
            if isinstance(out, GraphRef):
                ref_to_producer[out.index] = i

    # Compute hashes in topological order (nodes are already ordered)
    node_hashes: dict[int, str] = {}     # node_index → hash
    output_hashes: dict[int, str] = {}   # output_ref_index → hash

    hashed_nodes: list[HashedNode] = []
    hash_to_first: dict[str, int] = {}   # hash → first node index
    duplicates: list[tuple[int, int]] = []

    for i, node in enumerate(scope.nodes):
        # Gather dependency hashes (resolved through output refs)
        dep_hashes: dict[int, str] = {}
        dep_hash_list: list[str] = []
        for inp in node.inputs:
            if isinstance(inp.ref, GraphRef) and inp.ref.index in output_hashes:
                dep_hashes[inp.ref.index] = output_hashes[inp.ref.index]
                dep_hash_list.append(output_hashes[inp.ref.index])

        # Compute deep hash
        h = hash_node_with_deps(node, dep_hashes)
        node_hashes[i] = h

        # Map all outputs of this node to its hash
        for out in node.outputs:
            if isinstance(out, GraphRef):
                output_hashes[out.index] = h

        # Detect duplicates
        if h in hash_to_first and node.node_type not in {"io:in", "io:out", "io:write", "io:emit"}:
            duplicates.append((i, hash_to_first[h]))
        else:
            hash_to_first[h] = i

        hashed_nodes.append(HashedNode(
            index=i,
            hash_id=h,
            node=node,
            dep_hashes=dep_hash_list,
        ))

    # Build the output ref → hash map (for rendering)
    ref_hash_map: dict[int, str] = {}
    for i, node in enumerate(scope.nodes):
        for out in node.outputs:
            if isinstance(out, GraphRef):
                ref_hash_map[out.index] = node_hashes[i]

    return HashedScope(
        name=scope.name,
        nodes=hashed_nodes,
        hash_map=ref_hash_map,
        reverse_map=hash_to_first,
        duplicates=duplicates,
    )


def hash_program(program: AILProgram) -> list[HashedScope]:
    """Hash all scopes in a program."""
    return [hash_scope(scope) for scope in program.scopes]


# ─── Deduplication via Hash ──────────────────────────────────────────────────

def dedup_by_hash(scope: AILScope) -> tuple[AILScope, int]:
    """
    Eliminate duplicate nodes using content-addressable hashes.

    Unlike the optimizer's _eliminate_duplicates (which uses string signatures),
    this uses deep hashes that account for the entire dependency chain.
    Two subgraphs that compute the same thing will always be detected,
    even if they appear at different positions in the graph.

    Returns (deduplicated_scope, num_eliminated).
    """
    hs = hash_scope(scope)

    if not hs.duplicates:
        return scope, 0

    # Build remap: duplicate output refs → original output refs
    dup_set = {dup for dup, _ in hs.duplicates}
    dup_to_orig: dict[int, int] = {dup: orig for dup, orig in hs.duplicates}

    # Build output ref remap
    ref_remap: dict[int, int] = {}
    for dup_idx, orig_idx in hs.duplicates:
        dup_node = scope.nodes[dup_idx]
        orig_node = scope.nodes[orig_idx]
        for d_out, o_out in zip(dup_node.outputs, orig_node.outputs):
            if isinstance(d_out, GraphRef) and isinstance(o_out, GraphRef):
                ref_remap[d_out.index] = o_out.index

    # Rebuild scope without duplicates, rewriting refs
    result = AILScope(scope.name)
    for i, node in enumerate(scope.nodes):
        if i in dup_set:
            continue
        new_node = copy.deepcopy(node)
        for inp in new_node.inputs:
            if isinstance(inp.ref, GraphRef) and inp.ref.index in ref_remap:
                inp.ref = GraphRef(ref_remap[inp.ref.index], inp.ref.streaming)
        result.nodes.append(new_node)
    result.ref_counter = scope.ref_counter

    return result, len(hs.duplicates)


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "─" * 60

    # ── Test 1: Basic hashing — unique nodes get unique hashes ──
    print(f"\n{SEP}")
    print("TEST 1: Unique nodes → unique hashes")
    print(SEP)

    scope = AILScope("basic_hash")
    refs = [scope.next_ref() for _ in range(5)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", InlineExpr("§.x > 0"))],
        outputs=[refs[1]]))
    scope.add_node(AILNode("∇:desc",
        inputs=[InputBinding("col", refs[1]), InputBinding("key", '"score"')],
        outputs=[refs[2]]))
    scope.add_node(AILNode("⊗:limit",
        inputs=[InputBinding("col", refs[2]), InputBinding("n", "#10")],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    hs = hash_scope(scope)
    print(hs.to_ail())
    print(f"\n{hs.summary()}")

    # All hashes should be unique
    hashes = [hn.hash_id for hn in hs.nodes]
    assert len(set(hashes)) == len(hashes), "All hashes should be unique"
    print("✓ All nodes have unique content-addressable hashes")

    # ── Test 2: Deterministic — same graph always produces same hashes ──
    print(f"\n{SEP}")
    print("TEST 2: Deterministic hashing")
    print(SEP)

    hs2 = hash_scope(scope)
    for a, b in zip(hs.nodes, hs2.nodes):
        assert a.hash_id == b.hash_id, f"Hash mismatch: {a.hash_id} vs {b.hash_id}"
    print("✓ Hashing is deterministic — same graph → same hashes")

    # ── Test 3: Duplicate detection ──
    print(f"\n{SEP}")
    print("TEST 3: Duplicate detection via hash")
    print(SEP)

    scope = AILScope("dup_detect")
    refs = [scope.next_ref() for _ in range(6)]

    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    # Two identical maps
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[0]), InputBinding("fn", InlineExpr("§.upper"))],
        outputs=[refs[1]]))
    scope.add_node(AILNode("∿:map",
        inputs=[InputBinding("col", refs[0]), InputBinding("fn", InlineExpr("§.upper"))],
        outputs=[refs[2]]))
    scope.add_node(AILNode("⊕:concat",
        inputs=[InputBinding("a", refs[1]), InputBinding("b", refs[2])],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    hs = hash_scope(scope)
    print(hs.to_ail())
    print(f"\n{hs.summary()}")
    assert len(hs.duplicates) >= 1, "Should detect duplicate maps"
    print("✓ Duplicate nodes detected by identical hash")

    # ── Test 4: Deduplication ──
    print(f"\n{SEP}")
    print("TEST 4: Automatic deduplication")
    print(SEP)

    print(f"BEFORE ({len(scope.nodes)} nodes):")
    print(scope.to_ail())

    deduped, removed = dedup_by_hash(scope)
    print(f"\nAFTER ({len(deduped.nodes)} nodes, {removed} removed):")
    print(deduped.to_ail())
    assert removed >= 1, "Should eliminate at least 1 duplicate"
    print("✓ Duplicate eliminated, refs rewritten")

    # ── Test 5: Hash stability across restructuring ──
    print(f"\n{SEP}")
    print("TEST 5: Hash stability — same computation, different position")
    print(SEP)

    # Build same computation in two different scopes with different §N numbering
    scope_a = AILScope("scope_a")
    ra = [scope_a.next_ref() for _ in range(3)]
    scope_a.add_node(AILNode("io:in", outputs=[ra[0]]))
    scope_a.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", ra[0]), InputBinding("cond", InlineExpr("§.x > 0"))],
        outputs=[ra[1]]))

    scope_b = AILScope("scope_b")
    # Burn some refs to get different numbering
    _ = scope_b.next_ref()
    _ = scope_b.next_ref()
    _ = scope_b.next_ref()
    rb = [scope_b.next_ref() for _ in range(3)]
    scope_b.add_node(AILNode("io:in", outputs=[rb[0]]))
    scope_b.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", rb[0]), InputBinding("cond", InlineExpr("§.x > 0"))],
        outputs=[rb[1]]))

    hs_a = hash_scope(scope_a)
    hs_b = hash_scope(scope_b)

    # The filter node should have the same deep hash in both scopes
    # because its dependency chain is identical
    filter_hash_a = hs_a.nodes[1].hash_id
    filter_hash_b = hs_b.nodes[1].hash_id
    print(f"  scope_a ⊗:where hash: {filter_hash_a} (§{ra[0].index} → §{ra[1].index})")
    print(f"  scope_b ⊗:where hash: {filter_hash_b} (§{rb[0].index} → §{rb[1].index})")
    assert filter_hash_a == filter_hash_b, \
        f"Same computation should have same hash regardless of §N numbering"
    print("✓ Hash is stable across different §N numbering")

    # ── Test 6: Transpiled pipeline ──
    print(f"\n{SEP}")
    print("TEST 6: Transpiled pipeline — full hash analysis")
    print(SEP)

    from transpiler import transpile

    prog = transpile("""
async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""")

    hashed_scopes = hash_program(prog)
    for hs in hashed_scopes:
        print(hs.to_ail())
        print(f"\n{hs.summary()}")

    print(f"\n{SEP}")
    print("ALL TESTS PASSED ✓")
    print(SEP)
