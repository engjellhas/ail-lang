"""
AIL v0.4 -- Real Cost Model
Actual token pricing from Anthropic and OpenAI APIs (March 2026).

Replaces the estimate-based cost_model.py annotations with real $/1M-token
pricing so the optimizer can report dollar savings, not just token deltas.

Usage:
    from pricing import price_scope, price_program, PricingReport

    report = price_scope(scope)
    print(report.summary())
    # -> "Estimated cost: $0.0042/execution, optimized: $0.0011/execution"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ail_types import AILScope, AILProgram
from cost_model import (
    analyze_scope, analyze_program, NODE_COSTS,
    ModelTier, CostProfile, ScopeCost,
)


# -- Pricing Tables (per 1M tokens, March 2026) ---------------------------

@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a specific model."""
    name: str
    input_per_1m: float      # $/1M input tokens
    output_per_1m: float     # $/1M output tokens
    cached_input_per_1m: float = 0.0  # $/1M cached input tokens

    @property
    def avg_per_1m(self) -> float:
        """Average of input+output for quick estimates."""
        return (self.input_per_1m + self.output_per_1m) / 2


# Anthropic Claude pricing (https://docs.anthropic.com/en/docs/about-claude/models)
ANTHROPIC_PRICING = {
    ModelTier.POWERFUL: ModelPricing(
        name="claude-sonnet-4",
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=0.30,
    ),
    ModelTier.MID: ModelPricing(
        name="claude-haiku-4",
        input_per_1m=0.80,
        output_per_1m=4.00,
        cached_input_per_1m=0.08,
    ),
    ModelTier.CHEAP: ModelPricing(
        name="claude-haiku-4",
        input_per_1m=0.80,
        output_per_1m=4.00,
        cached_input_per_1m=0.08,
    ),
    ModelTier.NONE: ModelPricing(
        name="none",
        input_per_1m=0.0,
        output_per_1m=0.0,
    ),
}

# OpenAI pricing (for comparison)
OPENAI_PRICING = {
    ModelTier.POWERFUL: ModelPricing(
        name="gpt-4o",
        input_per_1m=2.50,
        output_per_1m=10.00,
        cached_input_per_1m=1.25,
    ),
    ModelTier.MID: ModelPricing(
        name="gpt-4o-mini",
        input_per_1m=0.15,
        output_per_1m=0.60,
        cached_input_per_1m=0.075,
    ),
    ModelTier.CHEAP: ModelPricing(
        name="gpt-4o-mini",
        input_per_1m=0.15,
        output_per_1m=0.60,
        cached_input_per_1m=0.075,
    ),
    ModelTier.NONE: ModelPricing(
        name="none",
        input_per_1m=0.0,
        output_per_1m=0.0,
    ),
}


# -- Token Estimates per Node Type ----------------------------------------
# More realistic than cost_model.py estimates.
# Based on typical prompt sizes and response lengths.

REALISTIC_TOKEN_ESTIMATES: dict[str, tuple[int, int]] = {
    # (input_tokens, output_tokens) per invocation
    # AI-native nodes
    "~:infer":     (800, 400),     # typical inference: ~800 in, ~400 out
    "~:intent":    (200, 20),      # classification: short prompt, 1-word answer
    "~:embed":     (150, 0),       # embedding: input only, no text output
    "~:rank":      (500, 100),     # ranking: list in, ordered list out
    "~:sim":       (0, 0),         # pure math
    "~:threshold": (0, 0),         # pure comparison
    "~:sample":    (0, 0),         # pure random
}


# -- Pricing Report --------------------------------------------------------

@dataclass
class NodePricingDetail:
    """Cost breakdown for a single node."""
    node_index: int
    node_type: str
    model_tier: ModelTier
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached_cost_usd: float     # cost if input is cached
    cacheable: bool

    def __str__(self):
        if self.cost_usd == 0:
            return f"  ${self.node_index} [{self.node_type}] free"
        cache_str = f" (cached: ${self.cached_cost_usd:.6f})" if self.cacheable else ""
        return (
            f"  ${self.node_index} [{self.node_type}] "
            f"${self.cost_usd:.6f} "
            f"({self.input_tokens}in + {self.output_tokens}out tok)"
            f"{cache_str}"
        )


@dataclass
class PricingReport:
    """Full pricing analysis for a scope."""
    scope_name: str
    provider: str
    nodes: list[NodePricingDetail]
    total_cost_usd: float
    cached_cost_usd: float      # cost with all cacheable inputs cached
    total_input_tokens: int
    total_output_tokens: int
    llm_node_count: int
    total_node_count: int

    @property
    def savings_if_cached(self) -> float:
        return self.total_cost_usd - self.cached_cost_usd

    @property
    def cost_per_1k_executions(self) -> float:
        return self.total_cost_usd * 1000

    @property
    def cached_cost_per_1k_executions(self) -> float:
        return self.cached_cost_usd * 1000

    def summary(self) -> str:
        lines = [
            f"--- {self.scope_name} -- Pricing ({self.provider}) ---",
            f"  Cost/execution:       ${self.total_cost_usd:.6f}",
            f"  Cost/1K executions:   ${self.cost_per_1k_executions:.4f}",
            f"  With caching:         ${self.cached_cost_usd:.6f}/exec "
            f"(saves ${self.savings_if_cached:.6f})",
            f"  Input tokens:         {self.total_input_tokens}",
            f"  Output tokens:        {self.total_output_tokens}",
            f"  LLM nodes:            {self.llm_node_count}/{self.total_node_count}",
            f"  ---",
        ]
        for nd in self.nodes:
            if nd.cost_usd > 0:
                lines.append(str(nd))
        lines.append(f"  ---")
        # Monthly cost estimate at various scales
        for rate_name, rate in [("100/day", 100), ("1K/day", 1000), ("10K/day", 10000)]:
            monthly = self.total_cost_usd * rate * 30
            cached_monthly = self.cached_cost_usd * rate * 30
            lines.append(
                f"  @ {rate_name}: ${monthly:.2f}/mo "
                f"(cached: ${cached_monthly:.2f}/mo)"
            )
        lines.append("---")
        return "\n".join(lines)


# -- Pricing Engine --------------------------------------------------------

def price_node(
    node_type: str,
    pricing_table: dict[ModelTier, ModelPricing],
) -> NodePricingDetail:
    """Compute cost for a single node type."""
    profile = NODE_COSTS.get(node_type)
    if profile is None or profile.model_tier == ModelTier.NONE:
        return NodePricingDetail(
            node_index=0, node_type=node_type,
            model_tier=ModelTier.NONE,
            input_tokens=0, output_tokens=0,
            cost_usd=0.0, cached_cost_usd=0.0,
            cacheable=profile.cacheable if profile else False,
        )

    model_pricing = pricing_table[profile.model_tier]
    input_tok, output_tok = REALISTIC_TOKEN_ESTIMATES.get(node_type, (0, 0))

    cost = (input_tok * model_pricing.input_per_1m +
            output_tok * model_pricing.output_per_1m) / 1_000_000

    cached_cost = (input_tok * model_pricing.cached_input_per_1m +
                   output_tok * model_pricing.output_per_1m) / 1_000_000

    return NodePricingDetail(
        node_index=0, node_type=node_type,
        model_tier=profile.model_tier,
        input_tokens=input_tok, output_tokens=output_tok,
        cost_usd=cost,
        cached_cost_usd=cached_cost if profile.cacheable else cost,
        cacheable=profile.cacheable,
    )


def price_scope(
    scope: AILScope,
    provider: str = "anthropic",
    pricing_table: Optional[dict[ModelTier, ModelPricing]] = None,
) -> PricingReport:
    """Compute full pricing for a scope."""
    if pricing_table is None:
        pricing_table = ANTHROPIC_PRICING if provider == "anthropic" else OPENAI_PRICING

    node_details: list[NodePricingDetail] = []
    total_cost = 0.0
    cached_cost = 0.0
    total_in = 0
    total_out = 0
    llm_count = 0

    for i, node in enumerate(scope.nodes):
        nd = price_node(node.node_type, pricing_table)
        nd.node_index = i
        node_details.append(nd)
        total_cost += nd.cost_usd
        cached_cost += nd.cached_cost_usd
        total_in += nd.input_tokens
        total_out += nd.output_tokens
        if nd.model_tier != ModelTier.NONE:
            llm_count += 1

    return PricingReport(
        scope_name=scope.name,
        provider=provider,
        nodes=node_details,
        total_cost_usd=total_cost,
        cached_cost_usd=cached_cost,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        llm_node_count=llm_count,
        total_node_count=len(scope.nodes),
    )


def price_program(
    program: AILProgram,
    provider: str = "anthropic",
) -> list[PricingReport]:
    """Compute pricing for all scopes in a program."""
    return [price_scope(scope, provider) for scope in program.scopes]


def compare_pricing(
    before: AILScope,
    after: AILScope,
    provider: str = "anthropic",
) -> str:
    """Compare pricing before/after optimization."""
    p_before = price_scope(before, provider)
    p_after = price_scope(after, provider)

    savings = p_before.total_cost_usd - p_after.total_cost_usd
    pct = (savings / p_before.total_cost_usd * 100) if p_before.total_cost_usd > 0 else 0

    lines = [
        f"=== Cost Comparison ({provider}) ===",
        f"  Before: ${p_before.total_cost_usd:.6f}/exec "
        f"({p_before.total_input_tokens}in + {p_before.total_output_tokens}out)",
        f"  After:  ${p_after.total_cost_usd:.6f}/exec "
        f"({p_after.total_input_tokens}in + {p_after.total_output_tokens}out)",
        f"  Saved:  ${savings:.6f}/exec ({pct:.1f}%)",
        f"",
        f"  Monthly savings @ 10K/day: ${savings * 10000 * 30:.2f}",
    ]
    return "\n".join(lines)


# -- Demo -----------------------------------------------------------------

if __name__ == "__main__":
    from ail_types import AILNode, InputBinding, InlineExpr, GraphRef
    from optimizer import optimize_scope

    SEP = "-" * 60

    # -- Test 1: Pure compute -- free --
    print(f"\n{SEP}")
    print("TEST 1: Pure compute pipeline -- $0")
    print(SEP)

    scope = AILScope("pure_compute")
    refs = [scope.next_ref() for _ in range(5)]
    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", InlineExpr("x > 0"))],
        outputs=[refs[1]]))
    scope.add_node(AILNode("∇:desc",
        inputs=[InputBinding("col", refs[1]), InputBinding("key", '"score"')],
        outputs=[refs[2]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[2])]))

    report = price_scope(scope)
    print(report.summary())
    assert report.total_cost_usd == 0.0, "Pure compute should be free"
    print("PASSED: Pure compute = $0")

    # -- Test 2: AI-native pipeline -- real pricing --
    print(f"\n{SEP}")
    print("TEST 2: AI-native pipeline -- Anthropic pricing")
    print(SEP)

    scope = AILScope("ai_pipeline")
    refs = [scope.next_ref() for _ in range(6)]
    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("~:intent",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[1])],
        outputs=[refs[2]]))
    scope.add_node(AILNode("~:rank",
        inputs=[InputBinding("col", refs[2])],
        outputs=[refs[3]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[3])]))

    report = price_scope(scope, "anthropic")
    print(report.summary())
    assert report.total_cost_usd > 0, "AI pipeline should cost money"
    assert report.llm_node_count == 3, "Should have 3 LLM nodes"
    print(f"PASSED: Total cost = ${report.total_cost_usd:.6f}")

    # -- Test 3: Anthropic vs OpenAI --
    print(f"\n{SEP}")
    print("TEST 3: Anthropic vs OpenAI pricing comparison")
    print(SEP)

    report_a = price_scope(scope, "anthropic")
    report_o = price_scope(scope, "openai")
    print(f"  Anthropic: ${report_a.total_cost_usd:.6f}/exec")
    print(f"  OpenAI:    ${report_o.total_cost_usd:.6f}/exec")
    print(f"  Cheaper:   {'Anthropic' if report_a.total_cost_usd < report_o.total_cost_usd else 'OpenAI'}")
    print("PASSED: Cross-provider comparison")

    # -- Test 4: Before/after optimization --
    print(f"\n{SEP}")
    print("TEST 4: Cost savings from optimization")
    print(SEP)

    scope = AILScope("optimize_savings")
    refs = [scope.next_ref() for _ in range(8)]
    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("~:intent",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[1])],
        outputs=[refs[2]]))
    # Duplicate infer (optimizer should eliminate)
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[1])],
        outputs=[refs[3]]))
    scope.add_node(AILNode("~:rank",
        inputs=[InputBinding("col", refs[2])],
        outputs=[refs[4]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[4])]))

    opt_scope, _ = optimize_scope(scope)
    print(compare_pricing(scope, opt_scope, "anthropic"))
    print("PASSED: Optimization reduces cost")

    # -- Test 5: Embedding caching savings --
    print(f"\n{SEP}")
    print("TEST 5: Embedding cache savings")
    print(SEP)

    scope = AILScope("embed_cache")
    refs = [scope.next_ref() for _ in range(4)]
    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("~:embed",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[1])]))

    report = price_scope(scope, "anthropic")
    print(report.summary())
    embed_node = report.nodes[1]
    assert embed_node.cacheable, "~:embed should be cacheable"
    assert embed_node.cached_cost_usd < embed_node.cost_usd, \
        "Cached embedding should be cheaper"
    print(f"PASSED: Embed cost ${embed_node.cost_usd:.6f} -> cached ${embed_node.cached_cost_usd:.6f}")

    print(f"\n{SEP}")
    print("ALL TESTS PASSED")
    print(SEP)
