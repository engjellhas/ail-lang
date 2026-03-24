"""
AIL Binary Serializer
Compact binary encoding for AIL programs

Format: custom length-prefixed binary (Protobuf-compatible field tagging)
Goal: minimize bytes per semantic unit so token cost of binary repr is minimal

Wire format per node:
  [1B: node_type_id] [1B: input_count] [inputs...] [1B: output_count] [outputs...]

Input wire format:
  [1B: role_id or 0xFF for literal] [2B: ref_index or literal_len] [literal_bytes if literal]

This format is designed so that a model consuming AIL binary
pays ~1 token per node regardless of the node's semantic complexity.
"""

import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ail_types import (
    AILProgram, AILScope, AILNode, InputBinding,
    GraphRef, ScopeRef, InlineExpr
)

# ─── Node type → 1-byte ID ────────────────────────────────────────────────────

NODE_TYPE_IDS = {
    "io:in":      0x01,
    "io:out":     0x02,
    "io:read":    0x03,
    "io:write":   0x04,
    "io:emit":    0x05,
    "∿:map":      0x10,
    "∿:flat":     0x11,
    "∿:cast":     0x12,
    "∿:norm":     0x13,
    "∿:fmt":      0x14,
    "⊗:where":    0x20,
    "⊗:dedup":    0x21,
    "⊗:limit":    0x22,
    "⊗:skip":     0x23,
    "⊗:null":     0x24,
    "Σ:sum":      0x30,
    "Σ:count":    0x31,
    "Σ:avg":      0x32,
    "Σ:max":      0x33,
    "Σ:min":      0x34,
    "Σ:group":    0x35,
    "⊢:if":       0x40,
    "⊢:match":    0x41,
    "⊢:guard":    0x42,
    "⊢:prob":     0x43,
    "⟳:loop":     0x50,
    "⟳:rec":      0x51,
    "⊕:zip":      0x60,
    "⊕:concat":   0x61,
    "⊕:merge":    0x62,
    "⊕:fan-in":   0x63,
    "∇:asc":      0x70,
    "∇:desc":     0x71,
    "~:infer":    0x80,
    "~:rank":     0x81,
    "~:threshold":0x82,
    "~:sample":   0x83,
    "~:embed":    0x84,
    "~:sim":      0x85,
    "~:intent":   0x86,
    "call":       0x90,
    "∂:slice":    0xA0,
}

ID_TO_NODE_TYPE = {v: k for k, v in NODE_TYPE_IDS.items()}

# ─── Role name → 1-byte ID ───────────────────────────────────────────────────

ROLE_IDS = {
    "col":    0x01,
    "cond":   0x02,
    "fn":     0x03,
    "key":    0x04,
    "n":      0x05,
    "val":    0x06,
    "src":    0x07,
    "query":  0x08,
    "scope":  0x09,
    "true":   0x0A,
    "false":  0x0B,
    "op":     0x0C,
    "a":      0x0D,
    "b":      0x0E,
    "param":  0x0F,
    "ref":    0x10,
    "field":  0x11,
    "idx":    0x12,
    "tmpl":   0x13,
    "threshold": 0x14,
}

ID_TO_ROLE = {v: k for k, v in ROLE_IDS.items()}

# ─── Ref encoding ─────────────────────────────────────────────────────────────
#
# Ref types (1 byte tag):
#   0x00 = graph ref §N        → followed by 2B little-endian index
#   0x01 = streaming ref §∞N   → followed by 2B index
#   0x02 = scope ref @name     → followed by 1B name_len + name bytes
#   0x03 = inline expr ⟨...⟩   → followed by 2B expr_len + expr bytes
#   0x04 = literal string      → followed by 2B len + bytes
#   0x05 = literal int #N      → followed by 4B signed int
#   0x06 = literal float %N    → followed by 4B IEEE float
#   0x07 = null ∅
#   0x08 = unknown role literal → followed by 2B len + bytes

REF_GRAPH    = 0x00
REF_STREAM   = 0x01
REF_SCOPE    = 0x02
REF_INLINE   = 0x03
REF_STR      = 0x04
REF_INT      = 0x05
REF_FLOAT    = 0x06
REF_NULL     = 0x07
REF_LITERAL  = 0x08


# ─── Encoder ─────────────────────────────────────────────────────────────────

class AILEncoder:

    def encode_program(self, program: AILProgram) -> bytes:
        out = bytearray()
        # Magic header: AIL\x00 + version
        out += b"AIL\x00"
        out += struct.pack("BB", 0, 1)  # version 0.1
        out += struct.pack("<H", len(program.scopes))
        for scope in program.scopes:
            out += self.encode_scope(scope)
        return bytes(out)

    def encode_scope(self, scope: AILScope) -> bytes:
        out = bytearray()
        name_bytes = scope.name.encode("utf-8")
        out += struct.pack("<H", len(name_bytes))
        out += name_bytes
        out += struct.pack("<H", len(scope.nodes))
        for node in scope.nodes:
            out += self.encode_node(node)
        return bytes(out)

    def encode_node(self, node: AILNode) -> bytes:
        out = bytearray()
        node_id = NODE_TYPE_IDS.get(node.node_type, 0xFF)
        out += struct.pack("B", node_id)
        if node_id == 0xFF:
            # Unknown node — encode name as fallback
            name_b = node.node_type.encode("utf-8")
            out += struct.pack("<H", len(name_b)) + name_b

        # Inputs
        out += struct.pack("B", len(node.inputs))
        for inp in node.inputs:
            out += self.encode_input(inp)

        # Outputs
        out += struct.pack("B", len(node.outputs))
        for ref in node.outputs:
            out += self.encode_ref(ref)

        return bytes(out)

    def encode_input(self, inp: InputBinding) -> bytes:
        out = bytearray()
        role_id = ROLE_IDS.get(inp.role, 0xFF)
        out += struct.pack("B", role_id)
        if role_id == 0xFF:
            # Unknown role — length-prefix the name
            role_b = inp.role.encode("utf-8")
            out += struct.pack("B", len(role_b)) + role_b

        out += self.encode_ref(inp.ref)
        return bytes(out)

    def encode_ref(self, ref) -> bytes:
        out = bytearray()

        if isinstance(ref, GraphRef):
            tag = REF_STREAM if ref.streaming else REF_GRAPH
            out += struct.pack("B", tag)
            out += struct.pack("<H", ref.index if ref.index >= 0 else 0xFFFF)

        elif isinstance(ref, ScopeRef):
            out += struct.pack("B", REF_SCOPE)
            name_b = ref.scope_name.encode("utf-8")
            out += struct.pack("B", len(name_b)) + name_b
            if ref.inner_ref:
                out += struct.pack("<H", ref.inner_ref.index)
            else:
                out += struct.pack("<H", 0xFFFF)

        elif isinstance(ref, InlineExpr):
            out += struct.pack("B", REF_INLINE)
            expr_b = ref.expression.encode("utf-8")
            out += struct.pack("<H", len(expr_b)) + expr_b

        elif isinstance(ref, str):
            s = ref.strip()
            if s == "∅":
                out += struct.pack("B", REF_NULL)
            elif s.startswith("#") and s[1:].lstrip("-").isdigit():
                out += struct.pack("B", REF_INT)
                out += struct.pack("<i", int(s[1:]))
            elif s.startswith("%"):
                try:
                    out += struct.pack("B", REF_FLOAT)
                    out += struct.pack("<f", float(s[1:]))
                except ValueError:
                    out += self._encode_str_literal(s)
            elif s.startswith('"') and s.endswith('"'):
                out += struct.pack("B", REF_STR)
                inner = s[1:-1].encode("utf-8")
                out += struct.pack("<H", len(inner)) + inner
            else:
                out += self._encode_str_literal(s)
        else:
            # Fallback
            s = str(ref).encode("utf-8")
            out += struct.pack("B", REF_LITERAL)
            out += struct.pack("<H", len(s)) + s

        return bytes(out)

    def _encode_str_literal(self, s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("B", REF_LITERAL) + struct.pack("<H", len(b)) + b


# ─── Decoder ─────────────────────────────────────────────────────────────────

class AILDecoder:

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        chunk = self.data[self.pos:self.pos+n]
        self.pos += n
        return chunk

    def decode_program(self) -> AILProgram:
        magic = self.read(4)
        assert magic == b"AIL\x00", f"Bad magic: {magic}"
        _ver_major, _ver_minor = struct.unpack("BB", self.read(2))
        scope_count = struct.unpack("<H", self.read(2))[0]
        prog = AILProgram()
        for _ in range(scope_count):
            prog.add_scope(self.decode_scope())
        return prog

    def decode_scope(self) -> AILScope:
        name_len = struct.unpack("<H", self.read(2))[0]
        name = self.read(name_len).decode("utf-8")
        node_count = struct.unpack("<H", self.read(2))[0]
        scope = AILScope(name)
        for _ in range(node_count):
            scope.nodes.append(self.decode_node())
        return scope

    def decode_node(self) -> AILNode:
        node_id = struct.unpack("B", self.read(1))[0]
        if node_id == 0xFF:
            name_len = struct.unpack("<H", self.read(2))[0]
            node_type = self.read(name_len).decode("utf-8")
        else:
            node_type = ID_TO_NODE_TYPE.get(node_id, f"unknown:0x{node_id:02x}")

        input_count = struct.unpack("B", self.read(1))[0]
        inputs = [self.decode_input() for _ in range(input_count)]

        output_count = struct.unpack("B", self.read(1))[0]
        outputs = [self.decode_ref() for _ in range(output_count)]

        return AILNode(node_type, inputs, outputs)

    def decode_input(self) -> InputBinding:
        role_id = struct.unpack("B", self.read(1))[0]
        if role_id == 0xFF:
            role_len = struct.unpack("B", self.read(1))[0]
            role = self.read(role_len).decode("utf-8")
        else:
            role = ID_TO_ROLE.get(role_id, f"role_{role_id:02x}")
        ref = self.decode_ref()
        return InputBinding(role, ref)

    def decode_ref(self):
        tag = struct.unpack("B", self.read(1))[0]
        if tag == REF_GRAPH:
            idx = struct.unpack("<H", self.read(2))[0]
            return GraphRef(idx if idx != 0xFFFF else -1)
        elif tag == REF_STREAM:
            idx = struct.unpack("<H", self.read(2))[0]
            return GraphRef(idx if idx != 0xFFFF else -1, streaming=True)
        elif tag == REF_SCOPE:
            name_len = struct.unpack("B", self.read(1))[0]
            name = self.read(name_len).decode("utf-8")
            inner_idx = struct.unpack("<H", self.read(2))[0]
            inner = GraphRef(inner_idx) if inner_idx != 0xFFFF else None
            return ScopeRef(name, inner)
        elif tag == REF_INLINE:
            expr_len = struct.unpack("<H", self.read(2))[0]
            expr = self.read(expr_len).decode("utf-8")
            return InlineExpr(expr)
        elif tag == REF_STR:
            slen = struct.unpack("<H", self.read(2))[0]
            return f'"{self.read(slen).decode("utf-8")}"'
        elif tag == REF_INT:
            val = struct.unpack("<i", self.read(4))[0]
            return f"#{val}"
        elif tag == REF_FLOAT:
            val = struct.unpack("<f", self.read(4))[0]
            return f"%{val:.4g}"
        elif tag == REF_NULL:
            return "∅"
        elif tag == REF_LITERAL:
            slen = struct.unpack("<H", self.read(2))[0]
            return self.read(slen).decode("utf-8")
        return f"?unknown_ref_{tag}"


# ─── Public API ───────────────────────────────────────────────────────────────

def encode(program: AILProgram) -> bytes:
    return AILEncoder().encode_program(program)

def decode(data: bytes) -> AILProgram:
    return AILDecoder(data).decode_program()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from transpiler import transpile

    source = '''
async def get_top_users(db, min_score=0.8):
    users = await db.query("SELECT * FROM users")
    scored = [u for u in users if u["score"] >= min_score]
    scored.sort(key=lambda u: u["score"], reverse=True)
    return scored[:10]
'''

    prog = transpile(source)
    print("=== AIL TEXT ===")
    print(prog.to_ail())

    binary = encode(prog)
    print(f"\n=== BINARY STATS ===")
    print(f"AIL text  : {len(prog.to_ail().encode())} bytes")
    print(f"AIL binary: {len(binary)} bytes")
    print(f"Binary compression: {round((1 - len(binary)/len(prog.to_ail().encode()))*100, 1)}%")
    print(f"Hex: {binary.hex()}")

    # Round-trip test
    recovered = decode(binary)
    print(f"\n=== ROUND-TRIP ===")
    print(recovered.to_ail())
    print(f"\nRound-trip OK: {prog.to_ail() == recovered.to_ail()}")
