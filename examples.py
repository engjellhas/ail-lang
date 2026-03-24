"""
AIL Examples
Annotated usage of the AIL transpiler and type system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "transpiler"))

from transpiler import transpile
from serializer import encode, decode
from ail_types import (
    AILProgram, AILScope, AILNode,
    InputBinding, InlineExpr, GraphRef, ScopeRef
)

SEP = "─" * 60


def show(label: str, source: str):
    print(f"\n{SEP}")
    print(f"EXAMPLE: {label}")
    print(SEP)
    print("PYTHON:")
    print(source.strip())
    prog = transpile(source)
    ail = prog.to_ail()
    binary = encode(prog)
    print("\nAIL:")
    print(ail)
    print(f"\nBinary: {len(binary)} bytes  |  Text: {len(ail.encode())} bytes  |  Compression: {round((1-len(binary)/len(ail.encode()))*100)}%")


# ── 1. Filter + sort + limit ──────────────────────────────────────────────────

show("Filter, sort, limit", """
async def get_top_users(db, min_score=0.8):
    users = await db.query("SELECT * FROM users")
    scored = [u for u in users if u['score'] >= min_score]
    scored.sort(key=lambda u: u['score'], reverse=True)
    return scored[:10]
""")

# ── 2. Aggregation ────────────────────────────────────────────────────────────

show("Aggregation", """
def user_stats(users):
    total = len(users)
    avg_score = sum(u['score'] for u in users) / total
    return avg_score
""")

# ── 3. Conditional classification ─────────────────────────────────────────────

show("Conditional classification (if/elif/else)", """
def classify_user(user):
    if user['score'] > 0.9:
        return 'premium'
    elif user['score'] > 0.5:
        return 'standard'
    else:
        return 'basic'
""")

# ── 4. Data pipeline ──────────────────────────────────────────────────────────

show("Data pipeline", """
def top_categories(transactions):
    seen = set()
    unique = [t for t in transactions if t['id'] not in seen and not seen.add(t['id'])]
    filtered = [t for t in unique if t['amount'] >= 100]
    groups = {}
    for t in filtered:
        cat = t['category']
        groups[cat] = groups.get(cat, 0) + t['amount']
    sorted_cats = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    return sorted_cats[:5]
""")

# ── 5. Agent recommendation ───────────────────────────────────────────────────

show("Agent recommendation (async)", """
async def recommend_products(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""")

# ── 6. Manual AIL construction (AI-native nodes) ──────────────────────────────

print(f"\n{SEP}")
print("EXAMPLE: Manual AIL — AI-native probabilistic pipeline")
print(SEP)

scope = AILScope("intent_router")
r0 = scope.next_ref()
r1 = scope.next_ref()
r2 = scope.next_ref()
r3 = scope.next_ref()
r4 = scope.next_ref()

# Take raw user input
scope.add_node(AILNode("io:in",
    inputs=[InputBinding("param", '"user_message"')],
    outputs=[r0]))

# Classify intent with probabilistic output
scope.add_node(AILNode("~:intent",
    inputs=[InputBinding("val", r0)],
    outputs=[r1]))

# Gate on high confidence only
scope.add_node(AILNode("~:threshold",
    inputs=[
        InputBinding("val", r1),
        InputBinding("min", "~+")   # ~+ = >0.9 confidence anchor
    ],
    outputs=[r2]))

# Probabilistic branch based on intent
scope.add_node(AILNode("⊢:prob",
    inputs=[
        InputBinding("cond", r2),
        InputBinding("true", ScopeRef("handle_purchase")),
        InputBinding("false", ScopeRef("handle_general")),
    ],
    outputs=[r3]))

scope.add_node(AILNode("io:out",
    inputs=[InputBinding("val", r3)],
    outputs=[]))

prog = AILProgram()
prog.add_scope(scope)
print("AIL:")
print(prog.to_ail())
binary = encode(prog)
recovered = decode(binary)
print(f"\nBinary: {len(binary)} bytes")
print(f"Round-trip OK: {prog.to_ail() == recovered.to_ail()}")

# ── 7. Round-trip verification ────────────────────────────────────────────────

print(f"\n{SEP}")
print("EXAMPLE: Binary round-trip verification")
print(SEP)

source = """
def process(data, threshold=0.5):
    filtered = [x for x in data if x['value'] > threshold]
    return sorted(filtered, key=lambda x: x['value'], reverse=True)
"""

prog = transpile(source)
original = prog.to_ail()
binary = encode(prog)
recovered = decode(binary)
restored = recovered.to_ail()

print(f"Original AIL:\n{original}")
print(f"\nBinary ({len(binary)} bytes): {binary[:32].hex()}...")
print(f"\nRestored AIL:\n{restored}")
print(f"\nIdentical: {original == restored} ✓" if original == restored else f"\nMISMATCH ✗")
