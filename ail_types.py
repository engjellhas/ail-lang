"""
AIL — Artificial Intent Language
Core types and node registry

Design decisions locked in v0.1:
- § refs are scoped per ⟦⟧ block
- Recursion references by scope name
- Protobuf for serialization (future)
- ~ type is float 0-1 with named anchors
- Streaming outputs use §∞N sigil
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import json


# ─── Confidence anchors for ~ type ───────────────────────────────────────────

CONFIDENCE_ANCHORS = {
    "~!":  1.0,    # certain
    "~+":  0.92,   # high confidence  
    "~?":  0.5,    # uncertain
    "~-":  0.08,   # unlikely
    "~∅":  0.0,    # impossible
}


# ─── Type System ─────────────────────────────────────────────────────────────

class AILBaseType(Enum):
    INTEGER     = "#"
    FLOAT       = "%"
    STRING      = '"'
    BOOLEAN     = "?"
    NULL        = "∅"
    REFERENCE   = "@"
    COLLECTION  = "*"
    FUNCTION    = "(→)"
    PROBABILISTIC = "~"
    ERROR       = "!"
    STREAM      = "§∞"


@dataclass
class AILType:
    base: AILBaseType
    inner: Optional["AILType"] = None        # for *, @, ~, !
    confidence: Optional[float] = None       # for ~ type only
    union_with: Optional["AILType"] = None   # for T|U

    def __str__(self):
        s = self.base.value
        if self.inner:
            s += str(self.inner)
        if self.confidence is not None:
            s = f"~{self.confidence:.2f}:{s}"
        return s

    @classmethod
    def probabilistic(cls, inner: "AILType", confidence: float) -> "AILType":
        return cls(AILBaseType.PROBABILISTIC, inner=inner, confidence=confidence)

    @classmethod
    def collection(cls, inner: "AILType") -> "AILType":
        return cls(AILBaseType.COLLECTION, inner=inner)

    @classmethod
    def nullable(cls, inner: "AILType") -> "AILType":
        t = cls(inner.base, inner.inner)
        t.union_with = cls(AILBaseType.NULL)
        return t


# ─── Graph References ─────────────────────────────────────────────────────────

@dataclass 
class GraphRef:
    """§N — positional reference, scoped to current ⟦⟧ block"""
    index: int
    streaming: bool = False     # §∞N — lazy/stream edge

    def __str__(self):
        if self.streaming:
            return f"§∞{self.index}"
        return f"§{self.index}"

    def __hash__(self):
        return hash((self.index, self.streaming))


@dataclass
class ScopeRef:
    """@scope_name or @scope_name.§N for cross-scope access"""
    scope_name: str
    inner_ref: Optional[GraphRef] = None

    def __str__(self):
        if self.inner_ref:
            return f"@{self.scope_name}.{self.inner_ref}"
        return f"@{self.scope_name}"


# ─── Node Input Binding ───────────────────────────────────────────────────────

@dataclass
class InputBinding:
    """role:§ref — named input slot on a node"""
    role: str
    ref: Union[GraphRef, ScopeRef, "InlineExpr", Any]

    def __str__(self):
        return f"{self.role}:{self.ref}"


@dataclass
class InlineExpr:
    """⟨ expr ⟩ — syntactic sugar, expands to nodes at compile time"""
    expression: str

    def __str__(self):
        return f"⟨{self.expression}⟩"


# ─── Node Type Registry ───────────────────────────────────────────────────────

class NodeCategory(Enum):
    IO          = "io"
    TRANSFORM   = "∿"
    FILTER      = "⊗"
    AGGREGATE   = "Σ"
    CONTROL     = "⊢"
    LOOP        = "⟳"
    MERGE       = "⊕"
    AI_NATIVE   = "~"
    CALL        = "call"
    SORT        = "∇"


# Full node type registry
NODE_TYPES = {
    # IO
    "io:in":        "Program input entry point",
    "io:out":       "Program output / return",
    "io:read":      "Read from external source",
    "io:write":     "Write to external source",
    "io:emit":      "Fire-and-forget event",

    # Transform
    "∿:map":        "Apply fn to each element",
    "∿:flat":       "Flatten nested collections",
    "∿:cast":       "Type conversion",
    "∿:norm":       "Normalize values",
    "∿:fmt":        "Format / serialize",

    # Filter
    "⊗:where":      "Filter by condition",
    "⊗:dedup":      "Remove duplicates",
    "⊗:limit":      "Take first N",
    "⊗:skip":       "Skip first N",
    "⊗:null":       "Remove null values",

    # Aggregate
    "Σ:sum":        "Sum numeric collection",
    "Σ:count":      "Count elements",
    "Σ:avg":        "Arithmetic mean",
    "Σ:max":        "Maximum value",
    "Σ:min":        "Minimum value",
    "Σ:group":      "Group by key",

    # Control
    "⊢:if":         "Binary conditional",
    "⊢:match":      "Pattern match",
    "⊢:guard":      "Assert — halt if false",
    "⊢:prob":       "Probabilistic branch",

    # Loop
    "⟳:loop":       "Iteration with condition",
    "⟳:rec":        "Recursive scope call",

    # Merge
    "⊕:zip":        "Pair elements from two collections",
    "⊕:concat":     "Concatenate collections",
    "⊕:merge":      "Merge two maps/objects",
    "⊕:fan-in":     "Collect parallel outputs",

    # Sort
    "∇:asc":        "Sort ascending",
    "∇:desc":       "Sort descending",

    # AI-native (no human language equivalent)
    "~:infer":      "Probabilistic inference → ~confidence:value",
    "~:rank":       "Rank collection by confidence",
    "~:threshold":  "Gate on minimum confidence",
    "~:sample":     "Sample from distribution",
    "~:embed":      "Semantic embedding",
    "~:sim":        "Cosine similarity between embeddings",
    "~:intent":     "Classify intent from unstructured input",

    # Call
    "call":         "Invoke a named scope",
}


# ─── Core Node ───────────────────────────────────────────────────────────────

@dataclass
class AILNode:
    node_type: str                              # e.g. "⊗:where", "∿:map"
    inputs: list[InputBinding] = field(default_factory=list)
    outputs: list[GraphRef] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # type hints, confidence, etc.

    def __post_init__(self):
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {self.node_type}")

    def __str__(self):
        parts = [f"[{self.node_type}]"]
        if self.inputs:
            parts.append("←")
            parts.append(", ".join(str(i) for i in self.inputs))
        if self.outputs:
            parts.append("→")
            parts.append(", ".join(str(o) for o in self.outputs))
        return " ".join(parts)


# ─── Scope (named sub-graph) ──────────────────────────────────────────────────

@dataclass
class AILScope:
    name: str
    nodes: list[AILNode] = field(default_factory=list)
    ref_counter: int = 0            # scoped § counter — resets per scope

    def next_ref(self, streaming: bool = False) -> GraphRef:
        ref = GraphRef(self.ref_counter, streaming)
        self.ref_counter += 1
        return ref

    def add_node(self, node: AILNode) -> "AILScope":
        self.nodes.append(node)
        return self

    def to_ail(self, indent: int = 2) -> str:
        pad = " " * indent
        lines = [f"⟦ {self.name}"]
        for node in self.nodes:
            lines.append(f"{pad}{node}")
        lines.append("⟧")
        return "\n".join(lines)

    def __str__(self):
        return self.to_ail()


# ─── Program ──────────────────────────────────────────────────────────────────

@dataclass
class AILProgram:
    scopes: list[AILScope] = field(default_factory=list)
    version: str = "0.1"

    def add_scope(self, scope: AILScope) -> "AILProgram":
        self.scopes.append(scope)
        return self

    def to_ail(self) -> str:
        return "\n\n".join(scope.to_ail() for scope in self.scopes)

    def to_json(self) -> str:
        """Debug serialization — not the production binary format"""
        def serialize(obj):
            if hasattr(obj, '__dict__'):
                d = {}
                for k, v in obj.__dict__.items():
                    d[k] = serialize(v)
                return {"_type": type(obj).__name__, **d}
            elif isinstance(obj, list):
                return [serialize(i) for i in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        return json.dumps(serialize(self), indent=2)

    def __str__(self):
        return self.to_ail()


if __name__ == "__main__":
    # Sanity check — build a small program manually
    prog = AILProgram()
    scope = AILScope("demo")

    r0 = scope.next_ref()
    r1 = scope.next_ref()
    r2 = scope.next_ref()
    r3 = scope.next_ref()

    scope.add_node(AILNode("io:in", outputs=[r0]))
    scope.add_node(AILNode("⊗:where", 
        inputs=[InputBinding("col", r0), InputBinding("cond", InlineExpr("§.score ≥ 0.8"))],
        outputs=[r1]))
    scope.add_node(AILNode("∇:desc",
        inputs=[InputBinding("col", r1), InputBinding("key", '"score"')],
        outputs=[r2]))
    scope.add_node(AILNode("⊗:limit",
        inputs=[InputBinding("col", r2), InputBinding("n", "#10")],
        outputs=[r3]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", r3)]))

    prog.add_scope(scope)
    print(prog.to_ail())
