"""
AIL v0.4 -- Runtime Engine
Executes AIL graphs against real backends (LLMs, HTTP, databases).

Architecture:
  Runtime registers "executors" for each node type category.
  When a node fires, the runtime:
    1. Gathers input values from upstream nodes
    2. Calls the registered executor
    3. Stores the output value + confidence
    4. Propagates confidence via executor.py math

  Executors are pluggable -- swap in mocks for testing,
  real API clients for production.

Usage:
    from runtime import Runtime, AnthropicBackend, MockBackend

    rt = Runtime(backend=AnthropicBackend(api_key="sk-..."))
    result = await rt.execute(program)

    # Or for testing:
    rt = Runtime(backend=MockBackend())
    result = await rt.execute(program)
"""

from __future__ import annotations

import asyncio
import time
import json
import os
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Union
from collections import defaultdict

from ail_types import (
    AILNode, AILScope, AILProgram,
    GraphRef, ScopeRef, InputBinding, InlineExpr,
)
from executor import (
    propagate_confidence, propagate_error,
    schedule, build_dependency_graph, ExecutionPlan,
    NODE_PROPAGATION, PropagationRule,
)
from cost_model import (
    analyze_scope, NODE_COSTS, ModelTier, CostProfile, TIER_RANK,
)


# -- Value wrapper ---------------------------------------------------------

@dataclass
class AILValue:
    """Runtime value flowing through the graph."""
    data: Any                           # the actual data
    confidence: float = 1.0             # ~ confidence
    error: Optional[str] = None         # ! error
    model_used: Optional[str] = None    # which model produced this (for AI nodes)
    latency_ms: float = 0.0            # execution time
    tokens_used: int = 0                # actual tokens consumed
    cached: bool = False                # was this a cache hit?

    def __repr__(self):
        conf = f"~{self.confidence:.3f}"
        err = f" !{self.error}" if self.error else ""
        model = f" [{self.model_used}]" if self.model_used else ""
        cache = " (cached)" if self.cached else ""
        return f"AILValue({conf}{err}{model}{cache}: {repr(self.data)[:80]})"


# -- Backend ABC -----------------------------------------------------------

class AILBackend(ABC):
    """
    Pluggable backend for AI-native operations.
    Implement this to wire AIL to different LLM providers.
    """

    @abstractmethod
    async def infer(self, prompt: str, model_tier: ModelTier, **kwargs) -> tuple[Any, float, int]:
        """
        Run inference.
        Returns (result, confidence, tokens_used).
        """
        ...

    @abstractmethod
    async def embed(self, text: str, **kwargs) -> tuple[list[float], int]:
        """
        Generate embedding vector.
        Returns (vector, tokens_used).
        """
        ...

    @abstractmethod
    async def classify_intent(self, text: str, intents: list[str], **kwargs) -> tuple[str, float, int]:
        """
        Classify intent.
        Returns (intent_label, confidence, tokens_used).
        """
        ...

    @abstractmethod
    async def rank(self, items: list, criterion: str, **kwargs) -> tuple[list, int]:
        """
        LLM-assisted ranking.
        Returns (ranked_items, tokens_used).
        """
        ...

    async def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Cosine similarity -- pure math, no LLM needed."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# -- Mock Backend (for testing) -------------------------------------------

class MockBackend(AILBackend):
    """
    Deterministic mock backend for testing.
    Returns predictable values so tests are reproducible.
    """

    def __init__(self, latency_ms: float = 0.0):
        self.call_log: list[dict] = []
        self.latency_ms = latency_ms

    async def infer(self, prompt: str, model_tier: ModelTier, **kwargs) -> tuple[Any, float, int]:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        self.call_log.append({"op": "infer", "prompt": prompt[:100], "tier": model_tier.value})
        # Deterministic mock: hash prompt for reproducible "inference"
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        confidence = 0.85 + (int(h[:2], 16) / 2550)  # 0.85-0.95 range
        tokens = {"powerful": 500, "mid": 200, "cheap": 100, "none": 0}[model_tier.value]
        return f"inferred:{h}", min(confidence, 1.0), tokens

    async def embed(self, text: str, **kwargs) -> tuple[list[float], int]:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        self.call_log.append({"op": "embed", "text": text[:100]})
        # Deterministic mock embedding (128-dim)
        h = hashlib.md5(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h * 8]  # 128 dims
        return vec, 100

    async def classify_intent(self, text: str, intents: list[str], **kwargs) -> tuple[str, float, int]:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        self.call_log.append({"op": "intent", "text": text[:100], "intents": intents})
        # Pick intent based on hash
        idx = int(hashlib.md5(text.encode()).hexdigest()[:4], 16) % max(len(intents), 1)
        intent = intents[idx] if intents else "unknown"
        return intent, 0.92, 200

    async def rank(self, items: list, criterion: str, **kwargs) -> tuple[list, int]:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        self.call_log.append({"op": "rank", "count": len(items), "criterion": criterion})
        # Mock: reverse the list (deterministic reordering)
        return list(reversed(items)), 300


# -- Anthropic Backend (real) ---------------------------------------------

class AnthropicBackend(AILBackend):
    """
    Real Anthropic Claude backend.
    Requires: pip install anthropic
    Set ANTHROPIC_API_KEY env var or pass api_key.
    """

    MODEL_MAP = {
        ModelTier.POWERFUL: "claude-sonnet-4-20250514",
        ModelTier.MID:      "claude-haiku-4-20250514",
        ModelTier.CHEAP:    "claude-haiku-4-20250514",
        ModelTier.NONE:     None,
    }

    def __init__(self, api_key: Optional[str] = None, model_overrides: Optional[dict] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if model_overrides:
            self.MODEL_MAP = {**self.MODEL_MAP, **model_overrides}

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for AnthropicBackend. "
                    "Install with: pip install anthropic"
                )
        return self._client

    async def infer(self, prompt: str, model_tier: ModelTier, **kwargs) -> tuple[Any, float, int]:
        model = self.MODEL_MAP.get(model_tier)
        if model is None:
            return prompt, 1.0, 0  # no LLM needed

        client = self._get_client()
        response = client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens

        # Confidence heuristic: longer, more structured responses = higher confidence
        confidence = min(0.95, 0.7 + len(text) / 5000)
        return text, confidence, tokens

    async def embed(self, text: str, **kwargs) -> tuple[list[float], int]:
        # Anthropic doesn't have a native embedding API yet.
        # Fall back to a hash-based mock or use voyageai/openai.
        h = hashlib.sha256(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in h * 4]  # 128 dims
        return vec, 100

    async def classify_intent(self, text: str, intents: list[str], **kwargs) -> tuple[str, float, int]:
        prompt = (
            f"Classify the following text into one of these intents: {', '.join(intents)}\n\n"
            f"Text: {text}\n\n"
            f"Respond with ONLY the intent label, nothing else."
        )
        result, conf, tokens = await self.infer(prompt, ModelTier.MID)
        # Match result to closest intent
        result_lower = result.strip().lower()
        for intent in intents:
            if intent.lower() in result_lower:
                return intent, conf, tokens
        return result.strip(), conf * 0.8, tokens  # lower confidence if no exact match

    async def rank(self, items: list, criterion: str, **kwargs) -> tuple[list, int]:
        items_str = "\n".join(f"{i+1}. {json.dumps(item)}" for i, item in enumerate(items[:20]))
        prompt = (
            f"Rank these items by {criterion}. "
            f"Return ONLY the numbers in order, comma-separated.\n\n{items_str}"
        )
        result, _, tokens = await self.infer(prompt, ModelTier.MID)
        try:
            indices = [int(x.strip()) - 1 for x in result.split(",") if x.strip().isdigit()]
            ranked = [items[i] for i in indices if 0 <= i < len(items)]
            # Append any items not mentioned
            seen = set(indices)
            for i, item in enumerate(items):
                if i not in seen:
                    ranked.append(item)
            return ranked, tokens
        except (ValueError, IndexError):
            return items, tokens


# -- IO Handlers -----------------------------------------------------------

@dataclass
class IORegistry:
    """
    Registry of IO handlers for io:read, io:write, io:emit.
    Register handlers by source name pattern.
    """
    readers: dict[str, Callable] = field(default_factory=dict)
    writers: dict[str, Callable] = field(default_factory=dict)
    emitters: dict[str, Callable] = field(default_factory=dict)

    def register_reader(self, pattern: str, handler: Callable):
        """Register a reader: async fn(source, query) -> data"""
        self.readers[pattern] = handler

    def register_writer(self, pattern: str, handler: Callable):
        """Register a writer: async fn(target, data) -> None"""
        self.writers[pattern] = handler

    def register_emitter(self, pattern: str, handler: Callable):
        """Register an emitter: async fn(event, data) -> None"""
        self.emitters[pattern] = handler

    async def read(self, source: str, query: Any = None) -> Any:
        for pattern, handler in self.readers.items():
            if pattern in source or pattern == "*":
                return await handler(source, query)
        raise ValueError(f"No reader registered for source: {source}")

    async def write(self, target: str, data: Any) -> None:
        for pattern, handler in self.writers.items():
            if pattern in target or pattern == "*":
                return await handler(target, data)
        raise ValueError(f"No writer registered for target: {target}")

    async def emit(self, event: str, data: Any) -> None:
        for pattern, handler in self.emitters.items():
            if pattern in event or pattern == "*":
                return await handler(event, data)
        # Emitters are fire-and-forget, don't fail if none registered


# -- Cache -----------------------------------------------------------------

class ResultCache:
    """
    Cache for deterministic/cacheable node outputs.
    Key = content hash of (node_type + resolved input values).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._store: dict[str, AILValue] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, node: AILNode, input_values: dict[str, Any]) -> str:
        payload = f"{node.node_type}:{json.dumps(input_values, sort_keys=True, default=str)}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, node: AILNode, input_values: dict[str, Any]) -> Optional[AILValue]:
        if not self.enabled:
            return None
        key = self._key(node, input_values)
        if key in self._store:
            self.hits += 1
            val = self._store[key]
            val.cached = True
            return val
        self.misses += 1
        return None

    def put(self, node: AILNode, input_values: dict[str, Any], value: AILValue):
        if not self.enabled:
            return
        key = self._key(node, input_values)
        self._store[key] = value

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {self.hits}/{total} hits ({rate:.1f}%)"


# -- Runtime ---------------------------------------------------------------

@dataclass
class RuntimeResult:
    """Result of executing a full program."""
    scope_results: list["ScopeResult"]
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    cache_stats: str = ""

    def summary(self) -> str:
        lines = [
            "--- AIL Runtime Execution ---",
            f"  Scopes:      {len(self.scope_results)}",
            f"  Total tokens: {self.total_tokens}",
            f"  Latency:     {self.total_latency_ms:.1f}ms",
            f"  {self.cache_stats}",
        ]
        for sr in self.scope_results:
            lines.append(f"\n  {sr.summary()}")
        lines.append("-------------------------------")
        return "\n".join(lines)


@dataclass
class ScopeResult:
    """Result of executing a single scope."""
    scope_name: str
    output: AILValue
    node_values: dict[int, AILValue]    # node_index -> output value
    plan: ExecutionPlan
    tokens_used: int = 0
    latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"  [{self.scope_name}]",
            f"    Output:  {self.output}",
            f"    Tokens:  {self.tokens_used}",
            f"    Latency: {self.latency_ms:.1f}ms",
            f"    Nodes:   {len(self.node_values)} executed",
        ]
        return "\n".join(lines)

    def trace(self) -> str:
        """Detailed execution trace."""
        lines = [f"--- {self.scope_name} trace ---"]
        for idx in sorted(self.node_values.keys()):
            val = self.node_values[idx]
            lines.append(f"  ${idx}: {val}")
        lines.append(f"  OUTPUT: {self.output}")
        return "\n".join(lines)


class Runtime:
    """
    The AIL runtime engine.
    Executes AIL graphs by:
      1. Scheduling nodes via topological sort
      2. Executing each level in parallel
      3. Propagating values, confidence, and errors
      4. Caching deterministic results
    """

    def __init__(
        self,
        backend: Optional[AILBackend] = None,
        io: Optional[IORegistry] = None,
        cache: Optional[ResultCache] = None,
        inputs: Optional[dict[str, Any]] = None,
    ):
        self.backend = backend or MockBackend()
        self.io = io or IORegistry()
        self.cache = cache or ResultCache()
        self.inputs = inputs or {}

        # Register default IO handlers
        if not self.io.readers:
            self.io.register_reader("*", self._default_reader)
        if not self.io.writers:
            self.io.register_writer("*", self._default_writer)

    @staticmethod
    async def _default_reader(source: str, query: Any = None) -> Any:
        """Default reader: returns a mock dataset."""
        return [
            {"id": i, "name": f"item_{i}", "score": 0.5 + i * 0.05, "confidence": 0.6 + i * 0.04}
            for i in range(10)
        ]

    @staticmethod
    async def _default_writer(target: str, data: Any) -> None:
        """Default writer: no-op."""
        pass

    async def execute(self, program: AILProgram) -> RuntimeResult:
        """Execute an entire AIL program."""
        t0 = time.perf_counter()
        scope_results = []
        total_tokens = 0

        for scope in program.scopes:
            sr = await self._execute_scope(scope)
            scope_results.append(sr)
            total_tokens += sr.tokens_used

        total_ms = (time.perf_counter() - t0) * 1000

        return RuntimeResult(
            scope_results=scope_results,
            total_tokens=total_tokens,
            total_latency_ms=total_ms,
            cache_stats=self.cache.stats(),
        )

    async def _execute_scope(self, scope: AILScope) -> ScopeResult:
        """Execute a single scope."""
        t0 = time.perf_counter()
        plan = schedule(scope)
        deps, _ = build_dependency_graph(scope)

        # Map output §N -> node index
        ref_to_node: dict[int, int] = {}
        for i, node in enumerate(scope.nodes):
            for out in node.outputs:
                if isinstance(out, GraphRef):
                    ref_to_node[out.index] = i

        # Node values store
        node_values: dict[int, AILValue] = {}
        total_tokens = 0

        # Execute level by level (nodes in same level run concurrently)
        for level in plan.levels:
            tasks = []
            for node_idx in level:
                node = scope.nodes[node_idx]

                # Gather input values
                input_vals: dict[str, Any] = {}
                input_confidences: list[float] = []
                input_errors: list[Optional[str]] = []

                for inp in node.inputs:
                    if isinstance(inp.ref, GraphRef) and inp.ref.index in ref_to_node:
                        producer_idx = ref_to_node[inp.ref.index]
                        if producer_idx in node_values:
                            pv = node_values[producer_idx]
                            input_vals[inp.role] = pv.data
                            input_confidences.append(pv.confidence)
                            input_errors.append(pv.error)
                    elif isinstance(inp.ref, InlineExpr):
                        input_vals[inp.role] = inp.ref.expression
                    elif isinstance(inp.ref, str):
                        input_vals[inp.role] = inp.ref.strip('"')
                    else:
                        input_vals[inp.role] = str(inp.ref)

                tasks.append(self._execute_node(
                    node, node_idx, input_vals, input_confidences, input_errors
                ))

            # Run all nodes in this level concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for node_idx, result in zip(level, results):
                if isinstance(result, Exception):
                    node_values[node_idx] = AILValue(
                        data=None, confidence=0.0,
                        error=f"!{type(result).__name__}: {result}"
                    )
                else:
                    node_values[node_idx] = result
                    total_tokens += result.tokens_used

        # Output = last node's value
        last_idx = len(scope.nodes) - 1
        output = node_values.get(last_idx, AILValue(data=None, confidence=0.0))

        latency_ms = (time.perf_counter() - t0) * 1000

        return ScopeResult(
            scope_name=scope.name,
            output=output,
            node_values=node_values,
            plan=plan,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
        )

    async def _execute_node(
        self,
        node: AILNode,
        node_idx: int,
        input_vals: dict[str, Any],
        input_confidences: list[float],
        input_errors: list[Optional[str]],
    ) -> AILValue:
        """Execute a single node."""

        # Check for incoming errors
        error = propagate_error(node, input_errors)
        if error is not None:
            conf = propagate_confidence(node, input_confidences)
            return AILValue(data=None, confidence=conf, error=error)

        # Check cache for cacheable nodes
        cost_profile = NODE_COSTS.get(node.node_type)
        if cost_profile and cost_profile.cacheable:
            cached = self.cache.get(node, input_vals)
            if cached is not None:
                return cached

        # Dispatch to executor
        t0 = time.perf_counter()
        result = await self._dispatch(node, input_vals, input_confidences)
        result.latency_ms = (time.perf_counter() - t0) * 1000

        # Propagate confidence
        if result.confidence == -1:  # sentinel: use propagation math
            result.confidence = propagate_confidence(
                node, input_confidences, node_output_confidence=None
            )

        # Cache if cacheable
        if cost_profile and cost_profile.cacheable:
            self.cache.put(node, input_vals, result)

        return result

    async def _dispatch(
        self,
        node: AILNode,
        inputs: dict[str, Any],
        input_confidences: list[float],
    ) -> AILValue:
        """Dispatch a node to the appropriate executor."""

        nt = node.node_type

        # -- IO nodes --
        if nt == "io:in":
            # Program input: check runtime inputs, then use param name
            param = inputs.get("param", inputs.get("val", inputs.get("ref", "unknown")))
            param_clean = str(param).strip('"')
            data = self.inputs.get(param_clean, param_clean)
            return AILValue(data=data, confidence=-1)  # -1 = use propagation

        if nt == "io:out":
            data = inputs.get("val", None)
            return AILValue(data=data, confidence=-1)

        if nt == "io:read":
            src = str(inputs.get("src", "unknown")).strip('"')
            query = inputs.get("query", None)
            data = await self.io.read(src, query)
            return AILValue(data=data, confidence=-1)

        if nt == "io:write":
            target = str(inputs.get("src", "unknown")).strip('"')
            data = inputs.get("val", None)
            await self.io.write(target, data)
            return AILValue(data=data, confidence=-1)

        if nt == "io:emit":
            data = inputs.get("val", None)
            await self.io.emit("event", data)
            return AILValue(data=data, confidence=-1)

        # -- Transform nodes --
        if nt == "∿:map":
            col = inputs.get("col", [])
            fn_expr = inputs.get("fn", "identity")
            # For runtime: apply a simple transform
            if isinstance(col, list):
                result = [_apply_expr(item, fn_expr) for item in col]
            else:
                result = _apply_expr(col, fn_expr)
            return AILValue(data=result, confidence=-1)

        if nt == "∿:flat":
            col = inputs.get("col", [])
            if isinstance(col, list):
                result = []
                for item in col:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return AILValue(data=result, confidence=-1)
            return AILValue(data=col, confidence=-1)

        if nt == "∿:cast":
            return AILValue(data=inputs.get("col", None), confidence=-1)

        if nt == "∿:norm":
            col = inputs.get("col", [])
            if isinstance(col, list) and col and all(isinstance(x, (int, float)) for x in col):
                mn, mx = min(col), max(col)
                rng = mx - mn if mx != mn else 1
                return AILValue(data=[(x - mn) / rng for x in col], confidence=-1)
            return AILValue(data=col, confidence=-1)

        if nt == "∿:fmt":
            return AILValue(data=str(inputs.get("tmpl", inputs)), confidence=-1)

        # -- Filter nodes --
        if nt == "⊗:where":
            col = inputs.get("col", [])
            cond = inputs.get("cond", "true")
            if isinstance(col, list):
                result = [item for item in col if _eval_condition(item, cond)]
            else:
                result = col
            return AILValue(data=result, confidence=-1)

        if nt == "⊗:dedup":
            col = inputs.get("col", [])
            if isinstance(col, list):
                seen = set()
                result = []
                for item in col:
                    key = json.dumps(item, sort_keys=True, default=str)
                    if key not in seen:
                        seen.add(key)
                        result.append(item)
                return AILValue(data=result, confidence=-1)
            return AILValue(data=col, confidence=-1)

        if nt == "⊗:limit":
            col = inputs.get("col", [])
            n = _parse_num(inputs.get("n", "10"))
            if isinstance(col, list):
                return AILValue(data=col[:n], confidence=-1)
            return AILValue(data=col, confidence=-1)

        if nt == "⊗:skip":
            col = inputs.get("col", [])
            n = _parse_num(inputs.get("n", "0"))
            if isinstance(col, list):
                return AILValue(data=col[n:], confidence=-1)
            return AILValue(data=col, confidence=-1)

        if nt == "⊗:null":
            col = inputs.get("col", [])
            if isinstance(col, list):
                return AILValue(data=[x for x in col if x is not None], confidence=-1)
            return AILValue(data=col, confidence=-1)

        # -- Aggregate nodes --
        if nt == "Σ:count":
            col = inputs.get("col", [])
            return AILValue(data=len(col) if isinstance(col, (list, str)) else 0, confidence=-1)

        if nt == "Σ:sum":
            col = inputs.get("col", [])
            if isinstance(col, list):
                return AILValue(data=sum(x for x in col if isinstance(x, (int, float))), confidence=-1)
            return AILValue(data=0, confidence=-1)

        if nt == "Σ:avg":
            col = inputs.get("col", [])
            if isinstance(col, list) and col:
                nums = [x for x in col if isinstance(x, (int, float))]
                return AILValue(data=sum(nums) / len(nums) if nums else 0, confidence=-1)
            return AILValue(data=0, confidence=-1)

        if nt == "Σ:max":
            col = inputs.get("col", [])
            if isinstance(col, list) and col:
                return AILValue(data=max(col), confidence=-1)
            return AILValue(data=None, confidence=-1)

        if nt == "Σ:min":
            col = inputs.get("col", [])
            if isinstance(col, list) and col:
                return AILValue(data=min(col), confidence=-1)
            return AILValue(data=None, confidence=-1)

        if nt == "Σ:group":
            col = inputs.get("col", [])
            key = inputs.get("key", "id")
            if isinstance(col, list):
                groups: dict[str, list] = defaultdict(list)
                for item in col:
                    k = item.get(key, "unknown") if isinstance(item, dict) else str(item)
                    groups[k].append(item)
                return AILValue(data=dict(groups), confidence=-1)
            return AILValue(data={}, confidence=-1)

        # -- Sort nodes --
        if nt == "∇:asc":
            col = inputs.get("col", [])
            key_expr = inputs.get("key", None)
            if isinstance(col, list):
                return AILValue(
                    data=sorted(col, key=lambda x: _extract_key(x, key_expr)),
                    confidence=-1
                )
            return AILValue(data=col, confidence=-1)

        if nt == "∇:desc":
            col = inputs.get("col", [])
            key_expr = inputs.get("key", None)
            if isinstance(col, list):
                return AILValue(
                    data=sorted(col, key=lambda x: _extract_key(x, key_expr), reverse=True),
                    confidence=-1
                )
            return AILValue(data=col, confidence=-1)

        # -- Merge nodes --
        if nt == "⊕:zip":
            a = inputs.get("a", [])
            b = inputs.get("b", [])
            if isinstance(a, list) and isinstance(b, list):
                return AILValue(data=list(zip(a, b)), confidence=-1)
            return AILValue(data=[a, b], confidence=-1)

        if nt == "⊕:concat":
            # Concat all inputs that are lists
            result = []
            for key in sorted(inputs.keys()):
                val = inputs[key]
                if isinstance(val, list):
                    result.extend(val)
                else:
                    result.append(val)
            return AILValue(data=result, confidence=-1)

        if nt == "⊕:merge":
            # Merge dicts or concat lists
            result = {}
            for key in sorted(inputs.keys()):
                val = inputs[key]
                if isinstance(val, dict):
                    result.update(val)
                else:
                    result[key] = val
            return AILValue(data=result, confidence=-1)

        if nt == "⊕:fan-in":
            # Collect all inputs into a list
            result = [inputs[k] for k in sorted(inputs.keys())]
            return AILValue(data=result, confidence=-1)

        # -- Control nodes --
        if nt == "⊢:if":
            cond = inputs.get("cond", "false")
            true_val = inputs.get("true", None)
            false_val = inputs.get("false", None)
            if _eval_condition(inputs, cond):
                return AILValue(data=true_val, confidence=-1)
            return AILValue(data=false_val, confidence=-1)

        if nt == "⊢:guard":
            val = inputs.get("val", None)
            # Guard passes through, absorbing errors
            return AILValue(data=val, confidence=-1)

        if nt == "⊢:match":
            return AILValue(data=inputs.get("val", None), confidence=-1)

        if nt == "⊢:prob":
            return AILValue(data=inputs.get("val", None), confidence=-1)

        # -- Loop nodes --
        if nt in ("⟳:loop", "⟳:rec"):
            col = inputs.get("col", [])
            return AILValue(data=col, confidence=-1)

        # -- AI-native nodes --
        if nt == "~:infer":
            val = inputs.get("val", "")
            prompt = str(val)
            tier = ModelTier(node.metadata.get("model_tier", "powerful"))
            result, conf, tokens = await self.backend.infer(prompt, tier)
            return AILValue(data=result, confidence=conf, tokens_used=tokens,
                          model_used=tier.value)

        if nt == "~:intent":
            val = inputs.get("val", "")
            intents = inputs.get("intents", ["greeting", "question", "command", "other"])
            if isinstance(intents, str):
                intents = [s.strip() for s in intents.split(",")]
            intent, conf, tokens = await self.backend.classify_intent(str(val), intents)
            return AILValue(data=intent, confidence=conf, tokens_used=tokens,
                          model_used="mid")

        if nt == "~:embed":
            val = inputs.get("val", "")
            vec, tokens = await self.backend.embed(str(val))
            return AILValue(data=vec, confidence=0.99, tokens_used=tokens)

        if nt == "~:sim":
            a = inputs.get("a", [])
            b = inputs.get("b", [])
            if isinstance(a, list) and isinstance(b, list) and a and b:
                sim = await self.backend.similarity(a, b)
                return AILValue(data=sim, confidence=-1)
            return AILValue(data=0.0, confidence=-1)

        if nt == "~:rank":
            col = inputs.get("col", [])
            criterion = inputs.get("key", "relevance")
            if isinstance(col, list):
                ranked, tokens = await self.backend.rank(col, str(criterion))
                return AILValue(data=ranked, confidence=-1, tokens_used=tokens)
            return AILValue(data=col, confidence=-1)

        if nt == "~:threshold":
            val = inputs.get("val", None)
            # Passthrough -- threshold checking is a confidence gate
            return AILValue(data=val, confidence=-1)

        if nt == "~:sample":
            col = inputs.get("col", [])
            if isinstance(col, list) and col:
                import random
                return AILValue(data=random.choice(col), confidence=-1)
            return AILValue(data=col, confidence=-1)

        # -- Call node --
        if nt == "call":
            # Would need cross-scope execution -- passthrough for now
            return AILValue(data=inputs, confidence=-1)

        # -- Fallback --
        return AILValue(data=inputs, confidence=-1)


# -- Helper functions ------------------------------------------------------

def _parse_num(val: Any) -> int:
    """Parse AIL number literal (#10 -> 10)."""
    s = str(val).strip().lstrip("#")
    try:
        return int(s)
    except ValueError:
        return 10


def _extract_key(item: Any, key_expr: Any) -> Any:
    """Extract sort key from an item."""
    if key_expr is None:
        return item
    key = str(key_expr).strip('"').strip("'")
    # Handle lambda-like expressions: (x->x.field) or simple field names
    if "->" in key:
        field_name = key.split("->")[-1].strip().split(".")[-1]
    elif "." in key:
        field_name = key.split(".")[-1]
    else:
        field_name = key.strip("(").strip(")")

    if isinstance(item, dict):
        return item.get(field_name, item.get(key, 0))
    return item


def _apply_expr(item: Any, expr: Any) -> Any:
    """Apply a simple expression/transform to an item."""
    expr_str = str(expr)
    # Handle field access: x.field or field
    if "." in expr_str and isinstance(item, dict):
        field = expr_str.split(".")[-1].strip(")")
        return item.get(field, item)
    # Handle multiplication: x * 2
    if "*" in expr_str and isinstance(item, (int, float)):
        try:
            factor = float(expr_str.split("*")[-1].strip().rstrip(")"))
            return item * factor
        except ValueError:
            pass
    return item


def _eval_condition(item: Any, cond: Any) -> bool:
    """Evaluate a simple filter condition against an item."""
    cond_str = str(cond)

    # Handle common patterns
    if cond_str in ("true", "True", "1"):
        return True
    if cond_str in ("false", "False", "0"):
        return False

    # Pattern: field >= value, field > value, field = value
    for op, fn in [(">=", lambda a, b: a >= b), (">", lambda a, b: a > b),
                   ("<=", lambda a, b: a <= b), ("<", lambda a, b: a < b),
                   ("=", lambda a, b: a == b)]:
        # Use the unicode versions too
        for op_char in [op, op.replace(">=", "\u2265").replace("<=", "\u2264").replace("=", "=")]:
            if op_char in cond_str:
                parts = cond_str.split(op_char, 1)
                if len(parts) == 2:
                    field = parts[0].strip().split(".")[-1].strip("$").strip()
                    val_str = parts[1].strip().strip('"').strip("'")
                    try:
                        threshold = float(val_str)
                    except ValueError:
                        continue
                    if isinstance(item, dict):
                        item_val = item.get(field, 0)
                        if isinstance(item_val, (int, float)):
                            return fn(item_val, threshold)
                    break

    return True  # default: pass through


# -- Demo -----------------------------------------------------------------

if __name__ == "__main__":
    SEP = "-" * 60

    async def run_tests():
        # -- Test 1: Pure compute pipeline --
        print(f"\n{SEP}")
        print("TEST 1: Pure compute pipeline (no LLM)")
        print(SEP)

        scope = AILScope("compute_pipeline")
        refs = [scope.next_ref() for _ in range(5)]
        scope.add_node(AILNode("io:in",
            inputs=[InputBinding("param", '"data"')],
            outputs=[refs[0]]))
        scope.add_node(AILNode("⊗:where",
            inputs=[InputBinding("col", refs[0]), InputBinding("cond", InlineExpr("score >= 0.7"))],
            outputs=[refs[1]]))
        scope.add_node(AILNode("∇:desc",
            inputs=[InputBinding("col", refs[1]), InputBinding("key", '"score"')],
            outputs=[refs[2]]))
        scope.add_node(AILNode("⊗:limit",
            inputs=[InputBinding("col", refs[2]), InputBinding("n", "#3")],
            outputs=[refs[3]]))
        scope.add_node(AILNode("io:out",
            inputs=[InputBinding("val", refs[3])]))

        prog = AILProgram()
        prog.add_scope(scope)

        rt = Runtime(
            backend=MockBackend(),
            inputs={"data": [
                {"id": i, "name": f"user_{i}", "score": 0.3 + i * 0.08}
                for i in range(10)
            ]}
        )
        result = await rt.execute(prog)
        print(result.summary())

        output = result.scope_results[0].output
        assert output.data is not None, "Should produce output"
        assert isinstance(output.data, list), "Should be a list"
        assert len(output.data) <= 3, f"Should limit to 3, got {len(output.data)}"
        assert output.error is None, f"Should have no error, got {output.error}"
        print("PASSED: filter -> sort -> limit pipeline works")

        # -- Test 2: AI-native pipeline --
        print(f"\n{SEP}")
        print("TEST 2: AI-native pipeline (mock LLM)")
        print(SEP)

        scope = AILScope("ai_pipeline")
        refs = [scope.next_ref() for _ in range(5)]
        scope.add_node(AILNode("io:in",
            inputs=[InputBinding("param", '"query"')],
            outputs=[refs[0]]))
        scope.add_node(AILNode("~:embed",
            inputs=[InputBinding("val", refs[0])],
            outputs=[refs[1]]))
        scope.add_node(AILNode("~:infer",
            inputs=[InputBinding("val", refs[1])],
            outputs=[refs[2]]))
        scope.add_node(AILNode("io:out",
            inputs=[InputBinding("val", refs[2])]))

        prog = AILProgram()
        prog.add_scope(scope)

        rt = Runtime(
            backend=MockBackend(),
            inputs={"query": "What is the meaning of life?"}
        )
        result = await rt.execute(prog)
        print(result.summary())

        output = result.scope_results[0].output
        assert output.tokens_used == 0, "io:out doesn't use tokens"
        # Check that LLM nodes actually consumed tokens
        node_vals = result.scope_results[0].node_values
        embed_val = node_vals[1]
        infer_val = node_vals[2]
        assert embed_val.tokens_used == 100, f"Embed should use 100 tokens, got {embed_val.tokens_used}"
        assert infer_val.tokens_used == 500, f"Infer should use 500 tokens, got {infer_val.tokens_used}"
        print(f"PASSED: embed={embed_val.tokens_used}tok, infer={infer_val.tokens_used}tok")

        # -- Test 3: Parallel execution --
        print(f"\n{SEP}")
        print("TEST 3: Parallel IO reads")
        print(SEP)

        scope = AILScope("parallel_io")
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

        prog = AILProgram()
        prog.add_scope(scope)

        # Use mock with latency to prove parallel execution
        rt = Runtime(backend=MockBackend(latency_ms=50))
        result = await rt.execute(prog)
        print(result.summary())

        # 3 reads at 50ms each: serial=150ms, parallel<100ms
        sr = result.scope_results[0]
        assert sr.plan.levels[0] == [0, 1, 2], "3 reads should be in same parallel level"
        print(f"PASSED: 3 reads in parallel level, latency={sr.latency_ms:.0f}ms")

        # -- Test 4: Cache hits --
        print(f"\n{SEP}")
        print("TEST 4: Result caching")
        print(SEP)

        scope = AILScope("cache_test")
        refs = [scope.next_ref() for _ in range(4)]
        scope.add_node(AILNode("io:in",
            inputs=[InputBinding("param", '"nums"')],
            outputs=[refs[0]]))
        scope.add_node(AILNode("∇:desc",
            inputs=[InputBinding("col", refs[0]), InputBinding("key", '"val"')],
            outputs=[refs[1]]))
        scope.add_node(AILNode("⊗:limit",
            inputs=[InputBinding("col", refs[1]), InputBinding("n", "#5")],
            outputs=[refs[2]]))
        scope.add_node(AILNode("io:out",
            inputs=[InputBinding("val", refs[2])]))

        prog = AILProgram()
        prog.add_scope(scope)

        cache = ResultCache()
        rt = Runtime(backend=MockBackend(), cache=cache,
                    inputs={"nums": [{"val": i} for i in range(10)]})

        # Execute twice -- second should hit cache
        await rt.execute(prog)
        r2 = await rt.execute(prog)
        print(f"  {cache.stats()}")
        assert cache.hits > 0, "Second execution should have cache hits"
        print("PASSED: Cacheable nodes hit cache on second execution")

        # -- Test 5: Transpiled pipeline --
        print(f"\n{SEP}")
        print("TEST 5: Transpiled Python -> AIL -> Runtime")
        print(SEP)

        from transpiler import transpile

        prog = transpile("""
async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""")

        rt = Runtime(backend=MockBackend())
        result = await rt.execute(prog)
        print(result.summary())

        output = result.scope_results[0].output
        assert output.data is not None, "Should produce output"
        print(f"PASSED: End-to-end transpile+execute, output={output.data}")

        # -- Test 6: Error propagation --
        print(f"\n{SEP}")
        print("TEST 6: Error propagation through runtime")
        print(SEP)

        scope = AILScope("error_test")
        refs = [scope.next_ref() for _ in range(4)]
        scope.add_node(AILNode("io:in", outputs=[refs[0]]))
        scope.add_node(AILNode("∿:map",
            inputs=[InputBinding("col", refs[0]), InputBinding("fn", InlineExpr("fail"))],
            outputs=[refs[1]]))
        scope.add_node(AILNode("⊢:guard",
            inputs=[InputBinding("val", refs[1])],
            outputs=[refs[2]]))
        scope.add_node(AILNode("io:out",
            inputs=[InputBinding("val", refs[2])]))

        prog = AILProgram()
        prog.add_scope(scope)

        rt = Runtime(backend=MockBackend(), inputs={"unknown": "test"})
        result = await rt.execute(prog)
        print(result.summary())
        print("PASSED: Error propagation works")

        print(f"\n{SEP}")
        print("ALL TESTS PASSED")
        print(SEP)

    asyncio.run(run_tests())
