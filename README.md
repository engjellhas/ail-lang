# AIL — Artificial Intent Language

> An intermediate representation for AI agent execution graphs.  
> Built for machines. Not for humans.

---

## What is AIL?

AIL is not a programming language in the traditional sense.  
It is an **intent graph IR** — a format for representing what an AI agent needs to *do*, stripped of everything that exists only for human readability.

Conceptually it sits between your high-level agent logic and your LLM execution layer:

```
Human intent / Prompt
        ↓
   Python / DSL
        ↓
🔥 AIL graph (this repo)
        ↓
  LLMs / Tools / APIs
```

---

## Why AIL exists

Every existing "AI language" or agent framework is Python with fancy syntax on top.  
They inherit all of Python's assumptions — named variables, sequential execution, binary control flow.

AI systems don't need any of that.

AIL is built around three observations:

**1. Execution is a graph, not a sequence**  
Dependencies determine order. Topology is semantics.

**2. Uncertainty is not an edge case**  
AI systems reason probabilistically. Confidence should be a first-class type, not a bolted-on heuristic.

**3. Tokens are the real cost unit**  
Every byte transmitted between agents costs money. The representation format should minimize that cost without losing semantic fidelity.

---

## Core concepts

### Intent nodes

Every unit of computation is a typed node with inputs and outputs:

```
[⊗:where] ← col:§1, cond:⟨§.score ≥ 0.8⟩ → §2
[∇:desc]  ← col:§2, key:"score" → §3
[⊗:limit] ← col:§3, n:#10 → §4
[io:out]  ← §4
```

No loops. No variable declarations. No syntax noise.  
Just a graph of what flows into what and what happens to it.

### Graph references (`§N`)

There are no named variables. Everything is referenced by graph position, scoped per `⟦⟧` block.

```
§0   — first value in scope
§1   — second derived value
§∞0  — streaming / lazy edge
```

Zero naming overhead. Zero ambiguity.

### Probabilistic type (`~`)

The most important thing AIL has that no existing IR has.

```
~0.92:"purchase"     — "purchase" with 92% confidence
~+:§0.intent         — high-confidence inference result  
~?:user.churn_risk   — uncertain boolean
```

Confidence propagates through the graph mathematically.  
One uncertain input makes downstream outputs uncertain — automatically.

### AI-native nodes

Operations that have no equivalent in any human language:

| Node | Description |
|------|-------------|
| `[~:infer]` | Probabilistic inference → returns `~confidence:value` |
| `[~:intent]` | Classify intent from unstructured input |
| `[~:embed]` | Semantic embedding |
| `[~:sim]` | Cosine similarity between embeddings |
| `[~:rank]` | Rank collection by confidence score |
| `[~:threshold]` | Gate execution on minimum confidence |

---

## A complete example

**Python source:**
```python
async def get_top_users(db, min_score=0.8):
    users = await db.query("SELECT * FROM users")
    scored = [u for u in users if u['score'] >= min_score]
    scored.sort(key=lambda u: u['score'], reverse=True)
    return scored[:10]
```

**AIL output:**
```
⟦ get_top_users
  [io:in]   ← param:"db" → §∞0
  [io:in]   ← param:"min_score" → §1
  [io:read] ← src:§∞0, query:"SELECT * FROM users" → §2
  [⊗:where] ← col:§2, cond:⟨§.score ≥ min_score⟩ → §3
  [∇:desc]  ← col:§3, key:"score" → §4
  [⊗:limit] ← col:§4, n:#10 → §5
  [io:out]  ← §5
⟧
```

**Binary encoding:** 151 bytes. Round-trip verified.  
**Text → Binary compression:** ~49%

---

## Agent-to-agent communication

Instead of transmitting natural language instructions between agents:

```
"Please fetch all user records from the database, filter those with 
engagement scores above 0.8, sort by score descending, return top 10."
```

Agents transmit AIL binary — unambiguous, compact, executable:

```
⟦ task
  [io:read] ← src:@db, tbl:"users" → §1
  [⊗:where] ← col:*§1, cond:⟨§.engagement ≥ %0.8⟩ → §2
  [∇:desc]  ← col:*§2, key:"engagement" → §3
  [⊗:limit] ← col:*§3, n:#10 → §4
  [io:out]  ← §4
⟧
```

Same semantics. No ambiguity. Dramatically lower token cost at scale.

---

## Installation

```bash
git clone https://github.com/engjellhas/ail-lang
cd ail-lang
pip install -r requirements.txt
```

No external dependencies for core usage. Python 3.10+.

---

## Usage

### Transpile Python → AIL

```python
from transpiler.transpiler import transpile

source = """
def classify_user(user):
    if user['score'] > 0.9:
        return 'premium'
    elif user['score'] > 0.5:
        return 'standard'
    else:
        return 'basic'
"""

program = transpile(source)
print(program.to_ail())
```

### Encode to binary

```python
from transpiler.serializer import encode, decode

binary = encode(program)        # bytes
recovered = decode(binary)      # AILProgram
print(recovered.to_ail())       # identical output
```

### Build a program manually

```python
from transpiler.ail_types import AILProgram, AILScope, AILNode, InputBinding, InlineExpr

scope = AILScope("my_agent")
r0 = scope.next_ref()
r1 = scope.next_ref()

scope.add_node(AILNode("io:in", outputs=[r0]))
scope.add_node(AILNode("⊗:where",
    inputs=[
        InputBinding("col", r0),
        InputBinding("cond", InlineExpr("§.score ≥ 0.8"))
    ],
    outputs=[r1]))
scope.add_node(AILNode("io:out",
    inputs=[InputBinding("val", r1)],
    outputs=[]))

program = AILProgram()
program.add_scope(scope)
print(program.to_ail())
```

---

## Repository structure

```
ail-lang/
├── transpiler/
│   ├── ail_types.py      # Core type system — nodes, refs, scopes, ~ type
│   ├── transpiler.py     # Python → AIL transpiler
│   └── serializer.py     # Binary encoder/decoder (round-trip verified)
├── spec/
│   └── AIL_SPEC_v0.1.md  # Formal language specification
├── examples/
│   └── examples.py       # Annotated usage examples
├── bench/
│   └── benchmark.py      # Token compression benchmarks
└── README.md
```

---

## Current status

| Component | Status |
|-----------|--------|
| Type system | ✅ Complete |
| Symbol vocabulary | ✅ Complete |
| Python → AIL transpiler | ✅ Working (core subset) |
| Binary serializer | ✅ Working, round-trip verified |
| Formal spec v0.1 | ✅ Published |
| Execution model / runtime | 🔄 In progress |
| Uncertainty propagation math | 🔄 In progress |
| Optimizer (node collapsing, cost model) | 📋 Planned |
| Multi-agent communication layer | 📋 Planned |

---

## What's next (v0.2)

- **Execution semantics** — node scheduling, parallelism model, uncertainty propagation
- **Cost model** — token cost per node, model selection hints, caching strategy  
- **Hash-based node IDs** — replace `§N` positional refs with content-addressable IDs for large graphs
- **AIL Optimizer** — collapse redundant nodes, reduce token cost, route by complexity

---

## The probabilistic type system (why this matters)

Every existing agent framework treats uncertainty as an afterthought — logprobs bolted on after the fact, confidence scores as metadata, not as values that flow through computation.

AIL makes uncertainty compositional:

```
[~:infer] ← input:§0 → ~0.87:§1        # inference with confidence
[~:threshold] ← val:§1, min:~+          # gate on high confidence  
[⊢:prob] ← ~§1 → §2                    # probabilistic branch
```

When a `~` value flows into a node, the output inherits and compounds that uncertainty automatically. This is the behavior AI systems actually have — AIL just makes it explicit and traceable.

---

## Contributing

This is early stage. The most valuable contributions right now:

1. **Execution model proposals** — how should nodes be scheduled? how does `~` propagate mathematically?
2. **Additional transpiler targets** — TypeScript, Rust source → AIL
3. **Real-world graph examples** — agent pipelines expressed in AIL
4. **Benchmarks** — token cost comparisons on real agent workloads

Open an issue or start a discussion.

---

## License

MIT

---

*AIL v0.1 — Initial release*  
*Built by [@engjellhas](https://github.com/engjellhas)*
