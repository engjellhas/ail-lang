# AIL -- Artificial Intent Language

> An intent graph IR for AI agents with a built-in cost optimizer.
> Built for machines. Not for humans.

---

## What is AIL?

AIL is an **intent graph IR** -- a format for representing what an AI agent needs to *do*, stripped of everything that exists only for human readability.

```
Human intent / Prompt
        |
   Python / DSL
        |
   AIL graph  <-- this repo
        |
  LLMs / Tools / APIs
```

**The value proposition:** AIL's optimizer analyzes your AI pipeline and returns a cheaper equivalent -- eliminating duplicate LLM calls, caching deterministic results, routing to cheaper models, and pruning low-confidence branches. Real dollar savings, automatically.

---

## Install

```bash
pip3 install -e .
```

Then add to your PATH if needed:
```bash
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
```

No external dependencies for core usage. Optional: `pip3 install anthropic` for real Claude API execution.

---

## CLI

```bash
ail compile sample.py              # Python -> AIL text
ail compile sample.py --binary     # Python -> AIL binary
ail optimize sample.py             # Optimize + show savings report
ail run sample.py                  # Confidence trace (static analysis)
ail execute sample.py              # Execute through runtime (mock backend)
ail execute sample.py --backend anthropic  # Execute against Claude API
ail price sample.py                # Real $/execution pricing (Anthropic & OpenAI)
ail viz sample.py                  # Generate interactive HTML graph
ail viz sample.py -O               # Before/after optimization comparison
ail bench sample.py                # Full benchmark suite
ail hash sample.py                 # Content-addressable node IDs
ail info file.ail.bin              # Inspect binary AIL file
```

---

## What it does (with real numbers)

```
$ ail price sample.py

--- ai_pipeline -- Pricing (anthropic) ---
  Cost/execution:       $0.009440
  Cost/1K executions:   $9.4400
  ~:intent  $0.000240  (200in + 20out tok)
  ~:infer   $0.008400  (800in + 400out tok)
  ~:rank    $0.000800  (500in + 100out tok)
  @ 1K/day:  $283.20/mo
  @ 10K/day: $2832.00/mo

--- Optimized ---
  Before: $0.017840/exec
  After:  $0.009440/exec  (saved 47.1%)
  Monthly savings @ 10K/day: $2,520.00
```

---

## Core concepts

### Intent nodes

Every unit of computation is a typed node:

```
[io:read] <- src:@db, query:"SELECT * FROM users" -> $2
[*:where] <- col:$2, cond:<$.score >= 0.8> -> $3
[*:desc]  <- col:$3, key:"score" -> $4
[*:limit] <- col:$4, n:#10 -> $5
[io:out]  <- $5
```

### Probabilistic type (`~`)

The most important thing AIL has that no existing IR has. Confidence propagates through the graph mathematically:

```
~0.9 -> [map]   -> ~0.9    (preserved)
~0.9 + ~0.8 -> [merge] -> ~0.72  (compounded: 0.9 x 0.8)
~0.9 -> [infer] -> ~new    (reset by inference)
```

One uncertain input makes downstream outputs uncertain -- automatically. The optimizer uses this to prune branches where confidence drops below a threshold, skipping all downstream LLM calls.

### AI-native nodes

Operations that have no equivalent in any human language:

| Node | What it does | Cost |
|------|-------------|------|
| `[~:infer]` | LLM inference -> ~confidence:value | ~$0.0084 |
| `[~:intent]` | Classify intent from text | ~$0.0002 |
| `[~:embed]` | Semantic embedding (cacheable) | ~$0.0001 |
| `[~:sim]` | Cosine similarity | free |
| `[~:rank]` | LLM-assisted ranking | ~$0.0008 |
| `[~:threshold]` | Gate on minimum confidence | free |

---

## The optimizer (the monetizable core)

6-pass pipeline that takes an AIL graph and returns a cheaper equivalent:

| Pass | What it does | Example savings |
|------|-------------|-----------------|
| Filter fusion | Collapse 3 sequential filters into 1 | Fewer nodes |
| Dead node elimination | Remove nodes whose outputs are never consumed | -1 to -N nodes |
| Confidence pruning | Cut branches where ~ drops below threshold | Skip downstream LLM calls |
| Duplicate elimination | Merge identical nodes (same type + inputs) | -50% on duplicated infer |
| Model annotation | Route ~:intent to Haiku instead of Sonnet | $0.0084 -> $0.0008 |
| Cache marking | Flag cacheable subgraphs for memoization | $0.000120 -> $0.000012 |

---

## Python API

```python
from transpiler import transpile
from optimizer import optimize_program
from pricing import price_program, compare_pricing
from runtime import Runtime, MockBackend, AnthropicBackend
from visualizer import render_html
import asyncio

# Transpile
prog = transpile(open("sample.py").read())

# Optimize
opt_prog, reports = optimize_program(prog)
for r in reports:
    print(r.summary())  # nodes eliminated, tokens saved

# Price
for r in price_program(prog, "anthropic"):
    print(r.summary())  # $/execution, monthly projections

# Execute (mock)
rt = Runtime(backend=MockBackend())
result = asyncio.run(rt.execute(prog))
print(result.summary())

# Execute (real Claude API)
rt = Runtime(backend=AnthropicBackend())
result = asyncio.run(rt.execute(prog))

# Visualize
html = render_html(prog.scopes[0])
open("graph.html", "w").write(html)
```

---

## Architecture

```
ail-lang/
  ail_types.py      Core type system -- nodes, refs, scopes, ~ type
  transpiler.py      Python -> AIL transpiler
  serializer.py      Binary encoder/decoder (~49% compression)
  executor.py        ~ confidence propagation, DAG scheduler, ! error flow
  cost_model.py      Token cost estimates, model tiers, cacheability flags
  optimizer.py       6-pass graph optimizer (the monetizable core)
  hash_ids.py        Content-addressable node IDs (SHA-256)
  runtime.py         Executes graphs against real backends (Claude, mock)
  pricing.py         Real $/token pricing (Anthropic, OpenAI)
  visualizer.py      Interactive HTML DAG renderer
  benchmark.py       Full benchmark suite
  cli.py             CLI toolchain (ail compile/optimize/run/execute/price/viz)
  pyproject.toml     pip install ready
```

---

## Status

| Component | Status |
|-----------|--------|
| Type system + node registry | Done |
| Python -> AIL transpiler | Done |
| Binary serializer (round-trip verified) | Done |
| ~ confidence propagation math | Done |
| DAG scheduler (parallel execution) | Done |
| ! error propagation | Done |
| Cost model (token estimates, tiers, cacheability) | Done |
| 6-pass graph optimizer | Done |
| Content-addressable hash IDs | Done |
| Runtime engine (mock + Anthropic backends) | Done |
| Real pricing (Anthropic/OpenAI) | Done |
| Interactive graph visualizer | Done |
| CLI toolchain | Done |
| pip package | Done |
| Formal spec v0.1 | Done |
| TypeScript transpiler | Planned |
| Multi-agent communication layer | Planned |
| Hosted optimizer API | Planned |

---

## Run tests

```bash
python3 executor.py       # ~ propagation, scheduling, error flow
python3 cost_model.py     # token costs, tiers, cacheability
python3 optimizer.py      # filter fusion, dead node elim, pruning, dedup
python3 hash_ids.py       # content-addressable hashing, dedup
python3 runtime.py        # runtime execution, caching, parallel IO
python3 pricing.py        # real pricing, cross-provider comparison
python3 visualizer.py     # HTML graph generation
python3 benchmark.py      # full benchmark suite
```

All modules end with `ALL TESTS PASSED`.

---

## License

MIT

---

*AIL v0.4*
*Built by [@engjellhas](https://github.com/engjellhas)*
