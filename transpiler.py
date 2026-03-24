"""
AIL Transpiler — Python → AIL
Walks Python AST and emits AIL graph nodes

Handles:
- Function definitions → named scopes
- For loops → [∿:map] or [⟳:loop]
- List comprehensions → [⊗:where] + [∿:map]
- If/else → [⊢:if]
- Return → [io:out]
- Function calls → [call] or IO nodes
- Assignments → graph ref bindings
- Sort/filter builtins → native AIL nodes
- Async functions → same structure, streaming-aware
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ail_types import (
    AILProgram, AILScope, AILNode, AILType, AILBaseType,
    GraphRef, ScopeRef, InputBinding, InlineExpr, GraphRef
)


# ─── Python builtin → AIL node mapping ───────────────────────────────────────

BUILTIN_MAP = {
    "len":      "Σ:count",
    "sum":      "Σ:sum",
    "max":      "Σ:max",
    "min":      "Σ:min",
    "sorted":   "∇:asc",
    "reversed": "∇:desc",
    "filter":   "⊗:where",
    "map":      "∿:map",
    "zip":      "⊕:zip",
    "enumerate":"⊕:zip",
    "list":     "∿:cast",
    "str":      "∿:cast",
    "int":      "∿:cast",
    "float":    "∿:cast",
    "set":      "⊗:dedup",
    "print":    "io:emit",
}

# DB/IO call patterns
IO_PATTERNS = {
    "query":    "io:read",
    "fetch":    "io:read",
    "select":   "io:read",
    "find":     "io:read",
    "get":      "io:read",
    "insert":   "io:write",
    "update":   "io:write",
    "delete":   "io:write",
    "save":     "io:write",
    "write":    "io:write",
    "emit":     "io:emit",
    "send":     "io:emit",
    "publish":  "io:emit",
}


# ─── Expression → Inline string ───────────────────────────────────────────────

class ExprSerializer(ast.NodeVisitor):
    """Converts a Python expression AST to compact AIL inline expression string"""

    def visit_Name(self, node):
        return node.id

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return f'"{node.value}"'
        elif isinstance(node.value, float):
            return f"%{node.value}"
        elif isinstance(node.value, int):
            return f"#{node.value}"
        elif isinstance(node.value, bool):
            return "?" + str(node.value).lower()
        elif node.value is None:
            return "∅"
        return str(node.value)

    def visit_Attribute(self, node):
        obj = self.visit(node.value)
        return f"{obj}.{node.attr}"

    def visit_Subscript(self, node):
        obj = self.visit(node.value)
        idx = self.visit(node.slice)
        return f"{obj}[{idx}]"

    def visit_Compare(self, node):
        left = self.visit(node.left)
        parts = [left]
        for op, comp in zip(node.ops, node.comparators):
            parts.append(self._op(op))
            parts.append(self.visit(comp))
        return " ".join(parts)

    def visit_BoolOp(self, node):
        op = "∧" if isinstance(node.op, ast.And) else "∨"
        return f" {op} ".join(self.visit(v) for v in node.values)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return f"¬{self.visit(node.operand)}"
        elif isinstance(node.op, ast.USub):
            return f"-{self.visit(node.operand)}"
        return self.visit(node.operand)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self._binop(node.op)
        return f"{left} {op} {right}"

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"{test} ? {body} : {orelse}"

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = ", ".join(self.visit(a) for a in node.args)
        return f"{func}({args})"

    def visit_Lambda(self, node):
        args = ", ".join(a.arg for a in node.args.args)
        body = self.visit(node.body)
        return f"({args}→{body})"

    def _op(self, op):
        return {
            ast.Eq: "=", ast.NotEq: "≠",
            ast.Lt: "<", ast.LtE: "≤",
            ast.Gt: ">", ast.GtE: "≥",
            ast.In: "∈", ast.NotIn: "∉",
            ast.Is: "≡", ast.IsNot: "≢",
        }.get(type(op), "?")

    def _binop(self, op):
        return {
            ast.Add: "+", ast.Sub: "-",
            ast.Mult: "×", ast.Div: "÷",
            ast.Mod: "%", ast.Pow: "^",
            ast.BitAnd: "∧", ast.BitOr: "∨",
            ast.FloorDiv: "÷⌊",
        }.get(type(op), "?")

    def generic_visit(self, node):
        return f"⟨?{type(node).__name__}⟩"


_expr = ExprSerializer()
def expr_to_inline(node: ast.expr) -> str:
    return _expr.visit(node)


# ─── Main Transpiler ──────────────────────────────────────────────────────────

class PythonToAIL(ast.NodeVisitor):

    def __init__(self):
        self.program = AILProgram()
        self._scope_stack: list[AILScope] = []
        self._binding: dict[str, GraphRef] = {}   # varname → §ref in current scope

    # ── Scope management ──────────────────────────────────────────────────────

    @property
    def scope(self) -> AILScope:
        return self._scope_stack[-1]

    def push_scope(self, name: str) -> AILScope:
        s = AILScope(name)
        self._scope_stack.append(s)
        self._binding = {}
        return s

    def pop_scope(self) -> AILScope:
        s = self._scope_stack.pop()
        self.program.add_scope(s)
        self._binding = {}
        return s

    def emit(self, node_type: str, inputs=None, outputs=None, **meta) -> GraphRef:
        inputs = inputs or []
        if outputs is None:
            out_ref = self.scope.next_ref()
            outputs = [out_ref]
        else:
            out_ref = outputs[0] if outputs else None
        self.scope.add_node(AILNode(node_type, inputs, outputs, meta))
        return out_ref

    def bind(self, name: str, ref: GraphRef):
        self._binding[name] = ref

    def resolve(self, name: str) -> GraphRef:
        return self._binding.get(name, GraphRef(-1))  # -1 = unresolved

    # ── Top-level visitors ────────────────────────────────────────────────────

    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._transpile_function(node, streaming=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._transpile_function(node, streaming=True)

    def _transpile_function(self, node, streaming: bool):
        scope = self.push_scope(node.name)

        # Emit input nodes for each parameter
        for i, arg in enumerate(node.args.args):
            ref = scope.next_ref(streaming and i == 0)
            self.emit("io:in",
                inputs=[InputBinding("param", f'"{arg.arg}"')],
                outputs=[ref])
            self.bind(arg.arg, ref)

        # Walk body
        for stmt in node.body:
            self.visit(stmt)

        self.pop_scope()

    # ── Statement visitors ────────────────────────────────────────────────────

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            self.emit("io:out", inputs=[InputBinding("val", "∅")], outputs=[])
            return
        val_ref = self._eval_expr(node.value)
        self.emit("io:out", inputs=[InputBinding("val", val_ref)], outputs=[])

    def visit_Assign(self, node: ast.Assign):
        val_ref = self._eval_expr(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.bind(target.id, val_ref)

    def visit_AugAssign(self, node: ast.AugAssign):
        # x += y → merge/transform
        left = self.resolve(node.target.id) if isinstance(node.target, ast.Name) else None
        right = self._eval_expr(node.value)
        if left and left.index >= 0:
            ref = self.emit("⊕:merge",
                inputs=[InputBinding("a", left), InputBinding("b", right)])
            if isinstance(node.target, ast.Name):
                self.bind(node.target.id, ref)

    def visit_For(self, node: ast.For):
        iter_ref = self._eval_expr(node.iter)

        # Collect body into inline or nested scope
        if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
            # Simple single-expression loop body → [∿:map]
            body_expr = expr_to_inline(node.body[0].value)
            ref = self.emit("∿:map",
                inputs=[
                    InputBinding("col", iter_ref),
                    InputBinding("fn", InlineExpr(body_expr))
                ])
        else:
            # Complex body → [⟳:loop] with nested scope
            loop_name = f"{self.scope.name}:loop"
            ref = self.emit("⟳:loop",
                inputs=[
                    InputBinding("col", iter_ref),
                    InputBinding("scope", ScopeRef(loop_name))
                ])

        if isinstance(node.target, ast.Name):
            self.bind(node.target.id, ref)

    def visit_If(self, node: ast.If):
        return self._emit_if_chain(node)

    def _emit_if_chain(self, node: ast.If) -> GraphRef:
        """Recursively unroll if/elif/else into nested [⊢:if] nodes"""
        cond_expr = expr_to_inline(node.test)
        true_expr = self._block_to_inline(node.body)
        # elif → single If node in orelse → recurse
        if node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            nested_ref = self._emit_if_chain(node.orelse[0])
            return self.emit("⊢:if",
                inputs=[
                    InputBinding("cond", InlineExpr(cond_expr)),
                    InputBinding("true", InlineExpr(true_expr)),
                    InputBinding("false", nested_ref),
                ])
        else:
            false_expr = self._block_to_inline(node.orelse) if node.orelse else "∅"
            return self.emit("⊢:if",
                inputs=[
                    InputBinding("cond", InlineExpr(cond_expr)),
                    InputBinding("true", InlineExpr(true_expr)),
                    InputBinding("false", InlineExpr(false_expr)),
                ])

    def visit_Expr(self, node: ast.Expr):
        self._eval_expr(node.value)

    # ── Expression evaluator ──────────────────────────────────────────────────

    def _eval_expr(self, node: ast.expr) -> GraphRef:
        """Evaluate an expression and return a graph ref to its output"""

        if isinstance(node, ast.Name):
            ref = self.resolve(node.id)
            if ref.index >= 0:
                return ref
            # Unresolved name — treat as scope reference
            return self.emit("io:in",
                inputs=[InputBinding("ref", ScopeRef(node.id))])

        elif isinstance(node, ast.Constant):
            inline = expr_to_inline(node)
            return self.emit("io:in",
                inputs=[InputBinding("val", InlineExpr(inline))])

        elif isinstance(node, ast.Call):
            return self._eval_call(node)

        elif isinstance(node, ast.ListComp):
            return self._eval_listcomp(node)

        elif isinstance(node, ast.DictComp):
            return self._eval_listcomp(node)  # same structure

        elif isinstance(node, ast.Attribute):
            obj_ref = self._eval_expr(node.value)
            return self.emit("io:in",
                inputs=[InputBinding("field",
                    InlineExpr(f"{obj_ref}.{node.attr}"))])

        elif isinstance(node, ast.Subscript):
            obj_ref = self._eval_expr(node.value)
            idx = expr_to_inline(node.slice)
            return self.emit("∂:slice",
                inputs=[InputBinding("col", obj_ref),
                        InputBinding("idx", InlineExpr(idx))]) \
                if idx.isdigit() or ":" in idx else obj_ref

        elif isinstance(node, ast.Await):
            # await expr → unwrap, mark output as streaming-aware
            return self._eval_expr(node.value)

        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            op_map = {
                ast.Add: "⊕:concat", ast.Sub: "⊕:merge",
                ast.Div: "∿:map", ast.Mult: "∿:map",
            }
            node_type = op_map.get(type(node.op), "⊕:merge")
            inline = expr_to_inline(node)
            return self.emit(node_type,
                inputs=[InputBinding("a", left),
                        InputBinding("b", right),
                        InputBinding("op", InlineExpr(inline))])

        elif isinstance(node, ast.IfExp):
            # Ternary: x if cond else y
            cond = expr_to_inline(node.test)
            true_val = expr_to_inline(node.body)
            false_val = expr_to_inline(node.orelse)
            return self.emit("⊢:if",
                inputs=[
                    InputBinding("cond", InlineExpr(cond)),
                    InputBinding("true", InlineExpr(true_val)),
                    InputBinding("false", InlineExpr(false_val)),
                ])

        elif isinstance(node, ast.GeneratorExp):
            # sum(x for x in col if cond) — treat like list comp
            return self._eval_listcomp(node)

        elif isinstance(node, ast.Dict):
            # {k: v, ...} → [⊕:merge] of key-value pairs
            if not node.keys:
                return self.emit("io:in", inputs=[InputBinding("val", "∅")])
            pairs = []
            for k, v in zip(node.keys, node.values):
                k_expr = expr_to_inline(k) if k else "∅"
                v_ref = self._eval_expr(v)
                pairs.append(InputBinding(k_expr, v_ref))
            return self.emit("⊕:merge", inputs=pairs)

        elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            # [a, b, c] → [⊕:concat] of elements
            if not node.elts:
                return self.emit("io:in", inputs=[InputBinding("val", "*∅")])
            refs = [self._eval_expr(e) for e in node.elts]
            inputs = [InputBinding(f"el{i}", r) for i, r in enumerate(refs)]
            return self.emit("⊕:concat", inputs=inputs)

        elif isinstance(node, ast.Set):
            # {a, b, c} → concat then dedup
            refs = [self._eval_expr(e) for e in node.elts]
            inputs = [InputBinding(f"el{i}", r) for i, r in enumerate(refs)]
            concat_ref = self.emit("⊕:concat", inputs=inputs)
            return self.emit("⊗:dedup", inputs=[InputBinding("col", concat_ref)])

        elif isinstance(node, ast.JoinedStr):
            # f-strings → format node
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(f'"{v.value}"')
                elif isinstance(v, ast.FormattedValue):
                    parts.append(expr_to_inline(v.value))
            return self.emit("∿:fmt",
                inputs=[InputBinding("tmpl", InlineExpr(" ⊕ ".join(parts)))])

        elif isinstance(node, ast.Starred):
            return self._eval_expr(node.value)

        else:
            # Fallback — emit as inline expression
            inline = expr_to_inline(node)
            return self.emit("io:in",
                inputs=[InputBinding("expr", InlineExpr(inline))])

    def _eval_call(self, node: ast.Call) -> GraphRef:
        """Map Python function calls to AIL nodes"""

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Check builtin map
        if func_name in BUILTIN_MAP:
            node_type = BUILTIN_MAP[func_name]
            inputs = []

            if node.args:
                col_ref = self._eval_expr(node.args[0])
                inputs.append(InputBinding("col", col_ref))

            # Handle sorted() keyword args — must process reverse BEFORE emitting
            for kw in node.keywords:
                if kw.arg == "reverse":
                    # ast.Constant(value=True) → flip to desc
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        node_type = "∇:desc"
                elif kw.arg == "key":
                    inputs.append(InputBinding("key", InlineExpr(expr_to_inline(kw.value))))

            # Handle filter(fn, col) signature
            if func_name == "filter" and len(node.args) >= 2:
                inputs = [
                    InputBinding("col", self._eval_expr(node.args[1])),
                    InputBinding("fn", InlineExpr(expr_to_inline(node.args[0]))),
                ]

            return self.emit(node_type, inputs=inputs)

        # Check IO patterns
        if func_name in IO_PATTERNS:
            node_type = IO_PATTERNS[func_name]
            inputs = []

            if isinstance(node.func, ast.Attribute):
                src_ref = self._eval_expr(node.func.value)
                inputs.append(InputBinding("src", src_ref))

            for arg in node.args:
                inputs.append(InputBinding("query", InlineExpr(expr_to_inline(arg))))

            for kw in node.keywords:
                inputs.append(InputBinding(kw.arg, InlineExpr(expr_to_inline(kw.value))))

            return self.emit(node_type, inputs=inputs)

        # Slice notation (list[n:m])
        if func_name == "slice":
            col_ref = self._eval_expr(node.args[0]) if node.args else GraphRef(-1)
            return col_ref

        # Generic call → [call] node referencing named scope
        inputs = [InputBinding("scope", ScopeRef(func_name))]
        for i, arg in enumerate(node.args):
            ref = self._eval_expr(arg)
            inputs.append(InputBinding(f"arg{i}", ref))

        return self.emit("call", inputs=inputs)

    def _eval_listcomp(self, node) -> GraphRef:
        """
        [x for x in col if cond]  →  [⊗:where] + [∿:map]
        GeneratorExp same structure, used inside sum()/max() etc.
        """
        generator = node.generators[0]
        iter_ref = self._eval_expr(generator.iter)
        current_ref = iter_ref

        # Apply each filter condition
        for cond in generator.ifs:
            cond_expr = expr_to_inline(cond)
            current_ref = self.emit("⊗:where",
                inputs=[
                    InputBinding("col", current_ref),
                    InputBinding("cond", InlineExpr(cond_expr))
                ])

        # Map element expression — skip if it's just the loop variable (identity)
        elt = getattr(node, "elt", None) or getattr(node, "key", None)
        if elt is not None:
            target_name = generator.target.id if isinstance(generator.target, ast.Name) else None
            elt_expr = expr_to_inline(elt)
            is_identity = (elt_expr == target_name)
            if not is_identity:
                current_ref = self.emit("∿:map",
                    inputs=[
                        InputBinding("col", current_ref),
                        InputBinding("fn", InlineExpr(f"({target_name or '§'}→{elt_expr})"))
                    ])

        return current_ref

    def _block_to_inline(self, stmts: list) -> str:
        """Convert a simple block to an inline expression string"""
        if not stmts:
            return "∅"
        if len(stmts) == 1:
            stmt = stmts[0]
            if isinstance(stmt, ast.Return) and stmt.value:
                return expr_to_inline(stmt.value)
            elif isinstance(stmt, ast.Expr):
                return expr_to_inline(stmt.value)
        return f"scope:{len(stmts)}_stmts"


# ─── Public API ───────────────────────────────────────────────────────────────

def transpile(python_source: str) -> AILProgram:
    tree = ast.parse(python_source)
    transpiler = PythonToAIL()
    transpiler.visit(tree)
    return transpiler.program


def transpile_file(path: str) -> AILProgram:
    return transpile(Path(path).read_text())


if __name__ == "__main__":
    test_cases = [
        # 1 — Basic filter + sort + limit
        ("""
async def get_top_users(db, min_score=0.8):
    users = await db.query("SELECT * FROM users")
    scored = [u for u in users if u['score'] >= min_score]
    scored.sort(key=lambda u: u['score'], reverse=True)
    return scored[:10]
""", "Filter + Sort + Limit"),

        # 2 — Aggregation
        ("""
def user_stats(users):
    total = len(users)
    avg_score = sum(u['score'] for u in users) / total
    return avg_score
""", "Aggregation"),

        # 3 — Conditional branching
        ("""
def classify_user(user):
    if user['score'] > 0.9:
        return 'premium'
    elif user['score'] > 0.5:
        return 'standard'
    else:
        return 'basic'
""", "Conditional branch"),
    ]

    for source, label in test_cases:
        print(f"\n{'='*60}")
        print(f"[{label}]")
        print(f"{'─'*60}")
        print("PYTHON:")
        print(source.strip())
        print(f"{'─'*60}")
        print("AIL OUTPUT:")
        prog = transpile(source)
        print(prog.to_ail())
