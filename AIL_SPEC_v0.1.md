# AIL — Artificial Intent Language
## Formal Specification v0.1

---

## 0. Preamble

AIL is a language designed exclusively for machine cognition. It is not intended to be written or read by humans in production use. Its goals, in order of priority:

1. **Semantic density** — maximum meaning per token
2. **Unambiguity** — one valid interpretation per construct
3. **Graph-native** — structure IS meaning, not an afterthought
4. **Uncertainty-aware** — probabilistic reasoning is a first-class citizen
5. **Execution-agnostic** — AIL describes intent; runtime decides execution

AIL is NOT a general-purpose language. It is an intent representation format for AI agents producing, consuming, and chaining programs.

---

## 1. Core Model

An AIL program is a **Directed Acyclic Graph (DAG)** of **Intent Nodes**.

- Every node has a **type**, zero or more **inputs**, and zero or more **outputs**
- Execution flows along edges (data dependencies)
- There are no statements, no lines, no sequential execution implied by order
- Order is determined entirely by the graph topology

### 1.1 Node Anatomy

```
[NODE_TYPE] ← input_role:§ref, input_role:§ref → §output_ref
```

- `[NODE_TYPE]` — the operation category and subtype
- `←` — input binding operator
- `→` — output binding operator
- `§N` — graph position reference (scoped per `⟦⟧` block)
- `role:` — semantic role of the input to this node

### 1.2 Graph References

All data is referenced by graph position `§N` scoped to the current `⟦⟧` block.  
Cross-scope references use `@scope_name.§N` syntax.

```
§0    — first value in scope
§1    — second derived value
§∞N   — streaming / lazy edge (does not block downstream)
```

---

## 2. Symbol Vocabulary

### 2.1 Flow Operators

| Symbol | Meaning |
|--------|---------|
| `←` | input binding |
| `→` | output binding |
| `⟳` | recursive / loop back |
| `⊢` | conditional branch |
| `\|` | parallel execution |
| `»` | sequential pipeline (order enforced) |
| `⊥` | halt / terminal |

### 2.2 Data Operators

| Symbol | Meaning |
|--------|---------|
| `∿` | transform / map |
| `⊕` | merge / combine |
| `⊗` | filter / exclude |
| `∂` | partial / subset |
| `Σ` | aggregate / reduce |
| `∇` | sort / order |
| `⊠` | cross-product / join |

### 2.3 Type Sigils

| Symbol | Meaning |
|--------|---------|
| `#` | integer |
| `%` | float |
| `"` | string |
| `?` | boolean |
| `@` | reference / pointer |
| `*` | collection / array |
| `~` | probabilistic value (0.0–1.0 confidence) |
| `∅` | null / empty |
| `!` | error / exception type |

### 2.4 Probabilistic Anchors

| Anchor | Value | Meaning |
|--------|-------|---------|
| `~!` | 1.0 | certain |
| `~+` | >0.9 | high confidence |
| `~?` | ~0.5 | uncertain |
| `~-` | <0.1 | unlikely |
| `~∅` | 0.0 | impossible |

### 2.5 Scope Delimiters

| Symbol | Meaning |
|--------|---------|
| `⟦` | scope open |
| `⟧` | scope close |
| `⟨` | inline expression open |
| `⟩` | inline expression close |

---

## 3. Node Type Registry

### 3.1 IO Nodes

| Node | Description |
|------|-------------|
| `[io:in]` | Program input entry point |
| `[io:out]` | Program output / return |
| `[io:read]` | Read from external source |
| `[io:write]` | Write to external source |
| `[io:emit]` | Fire-and-forget event emission |

### 3.2 Transform Nodes

| Node | Description |
|------|-------------|
| `[∿:map]` | Apply function to each element |
| `[∿:flat]` | Flatten nested collections |
| `[∿:cast]` | Type conversion |
| `[∿:norm]` | Normalize values |
| `[∿:fmt]` | Format / serialize |

### 3.3 Filter Nodes

| Node | Description |
|------|-------------|
| `[⊗:where]` | Filter by condition |
| `[⊗:dedup]` | Remove duplicates |
| `[⊗:limit]` | Take first N |
| `[⊗:skip]` | Skip first N |
| `[⊗:null]` | Remove null values |

### 3.4 Aggregate Nodes

| Node | Description |
|------|-------------|
| `[Σ:sum]` | Sum numeric collection |
| `[Σ:count]` | Count elements |
| `[Σ:avg]` | Arithmetic mean |
| `[Σ:max]` | Maximum value |
| `[Σ:min]` | Minimum value |
| `[Σ:group]` | Group by key |

### 3.5 Control Nodes

| Node | Description |
|------|-------------|
| `[⊢:if]` | Binary conditional |
| `[⊢:match]` | Pattern match (multi-branch) |
| `[⊢:guard]` | Assertion — halt if false |
| `[⊢:prob]` | Probabilistic branch |
| `[⟳:loop]` | Iteration with condition |
| `[⟳:rec]` | Recursive call |

### 3.6 Merge Nodes

| Node | Description |
|------|-------------|
| `[⊕:zip]` | Pair elements from two collections |
| `[⊕:concat]` | Concatenate collections |
| `[⊕:merge]` | Merge two maps/objects |
| `[⊕:fan-in]` | Collect parallel outputs |

### 3.7 Sort Nodes

| Node | Description |
|------|-------------|
| `[∇:asc]` | Sort ascending |
| `[∇:desc]` | Sort descending |

### 3.8 AI-Native Nodes

These have no equivalent in any human programming language.

| Node | Description |
|------|-------------|
| `[~:infer]` | Probabilistic inference → `~confidence:value` |
| `[~:rank]` | Rank collection by confidence score |
| `[~:threshold]` | Gate execution on confidence minimum |
| `[~:sample]` | Sample from probability distribution |
| `[~:embed]` | Semantic embedding of input |
| `[~:sim]` | Cosine similarity between embeddings |
| `[~:intent]` | Classify intent from unstructured input |

---

## 4. Type System

### 4.1 Primitive Types

```
#     integer         exact whole number
%     float           64-bit floating point
"     string          unicode text
?     boolean         true | false
∅     null            absence of value
```

### 4.2 Compound Types

```
*T         collection of T
@T         reference to T
T|U        union type
(T→U)      function signature
```

### 4.3 Probabilistic Type

The `~` type wraps any value with a confidence score and is unique to AIL.

```
~0.92:"purchase"    — string with 92% confidence
~+:§0.label         — high-confidence field access
~?                  — uncertain boolean
```

Confidence propagates compositionally. Combining two `~` values compounds scores unless explicitly overridden.

### 4.4 Error Type

```
!T    — error wrapping type T
!"    — error with message
!∅    — untyped error
```

All fallible nodes implicitly return `T|!` unless wrapped in `[⊢:guard]`.

---

## 5. Scope and Encapsulation

### 5.1 Named Scopes

```
⟦ scope_name
  ... nodes ...
⟧
```

Scope names exist for debugging and cross-scope reference only.  
`§N` refs reset to `§0` inside each new scope.

### 5.2 Scope Invocation

```
[call] ← scope:@auth_check, input:§3 → §4
```

### 5.3 Recursion

Recursive scopes reference themselves by name:

```
[⟳:rec] ← scope:@my_scope, input:§2 → §3
```

---

## 6. Design Decisions (locked v0.1)

| Question | Decision | Rationale |
|----------|----------|-----------|
| `§` ref scope | Per `⟦⟧` block | Enables independent scope optimization |
| Recursion reference | By scope name | Survives graph restructuring |
| Binary format | Custom length-prefixed (Protobuf-compatible) | Ecosystem + tooling without schema complexity |
| `~` encoding | Float 0–1 with named anchors | Maximal expressiveness + ergonomic common cases |
| Streaming | `§∞N` lazy edge sigil | No new node types needed |

---

## 7. Binary Wire Format

Each node encodes to:
```
[1B: node_type_id] [1B: input_count] [inputs...] [1B: output_count] [outputs...]
```

Each input:
```
[1B: role_id] [ref...]
```

Each ref:
```
[1B: tag] [payload...]
```

Node type IDs and role IDs are defined in `transpiler/serializer.py`.

**Observed compression:** ~49% size reduction vs UTF-8 AIL text.  
**Round-trip:** encode → decode produces byte-identical AIL output.

---

## 8. What Does NOT Exist in AIL

| Absent | Reason |
|--------|--------|
| Variable names | `§N` refs are sufficient and unambiguous |
| Comments | Scope names carry semantic context |
| Whitespace rules | Graph topology encodes structure |
| Import statements | Dependencies declared as `@ref` inputs |
| Classes / OOP | Scopes + typed refs subsume object patterns |
| Exceptions | `!` type propagation handles errors structurally |
| For / while loops | `[⟳:loop]` and `[⟳:rec]` are declarative |

---

## 9. Roadmap

| Version | Milestone |
|---------|-----------|
| v0.1 | Spec, type system, transpiler, binary serializer ✅ |
| v0.2 | Execution semantics, uncertainty propagation math |
| v0.3 | Cost model, model selection hints, caching strategy |
| v0.4 | Hash-based node IDs for large graph scalability |
| v0.5 | AIL Optimizer — node collapsing, redundancy elimination |
| v1.0 | Runtime + multi-agent communication layer |

---

*AIL Specification v0.1*  
*Status: Published*  
*Author: [@engjellhas](https://github.com/engjellhas)*
