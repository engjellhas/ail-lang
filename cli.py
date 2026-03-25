#!/usr/bin/env python3
"""
AIL v0.4 -- CLI Toolchain
Usage:
    ail compile <file.py>              Transpile Python -> AIL text
    ail compile <file.py> --binary     Transpile Python -> AIL binary
    ail optimize <file.py>             Transpile + optimize, show report
    ail run <file.py>                  Transpile + execute (confidence trace)
    ail execute <file.py>              Transpile + run through runtime (real execution)
    ail hash <file.py>                 Transpile + show hash-based node IDs
    ail bench <file.py>                Transpile + benchmark (optimized vs unoptimized)
    ail price <file.py>                Transpile + show real pricing (Anthropic/OpenAI)
    ail viz <file.py>                  Transpile + generate interactive HTML graph
    ail info <file.ail.bin>            Decode binary and show stats

Options:
    --threshold <float>    Confidence pruning threshold (default: 0.1)
    --json                 Output as JSON
    --binary               Output binary format
    --quiet                Suppress banner
    --provider <str>       Pricing provider: anthropic|openai (default: anthropic)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure imports work from any directory
sys.path.insert(0, str(Path(__file__).parent))

from ail_types import AILProgram
from transpiler import transpile, transpile_file
from serializer import encode, decode
from executor import execute_program
from cost_model import analyze_program
from optimizer import optimize_program
from hash_ids import hash_program


# ─── Banner ──────────────────────────────────────────────────────────────────

BANNER = """
    _    ___ _
   / \\  |_ _| |
  / _ \\  | || |
 / ___ \\ | || |___
/_/   \\_\\___|_____|  v0.4

Artificial Intent Language
"""


def print_banner():
    print(BANNER, file=sys.stderr)


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_compile(args):
    """Transpile Python to AIL."""
    source = Path(args.file).read_text()
    prog = transpile(source)

    if args.binary:
        binary = encode(prog)
        out_path = Path(args.file).with_suffix(".ail.bin")
        if args.output:
            out_path = Path(args.output)
        out_path.write_bytes(binary)
        text_size = len(prog.to_ail().encode("utf-8"))
        print(f"Compiled: {args.file}")
        print(f"  Text size:   {text_size} bytes")
        print(f"  Binary size: {len(binary)} bytes")
        print(f"  Compression: {(1 - len(binary)/text_size)*100:.1f}%")
        print(f"  Output:      {out_path}")
    elif args.json:
        print(prog.to_json())
    else:
        print(prog.to_ail())


def cmd_optimize(args):
    """Transpile and optimize."""
    source = Path(args.file).read_text()
    prog = transpile(source)

    threshold = args.threshold or 0.1

    print("── BEFORE ─────────────────────────────────────")
    print(prog.to_ail())

    opt_prog, reports = optimize_program(prog, confidence_threshold=threshold)

    print("\n── AFTER ──────────────────────────────────────")
    print(opt_prog.to_ail())

    print()
    for report in reports:
        print(report.summary())

    if args.binary:
        binary = encode(opt_prog)
        out_path = Path(args.file).with_suffix(".opt.ail.bin")
        if args.output:
            out_path = Path(args.output)
        out_path.write_bytes(binary)
        print(f"\nBinary output: {out_path} ({len(binary)} bytes)")


def cmd_run(args):
    """Transpile and execute (confidence trace)."""
    source = Path(args.file).read_text()
    prog = transpile(source)

    # Optionally optimize first
    if args.optimize:
        threshold = args.threshold or 0.1
        prog, reports = optimize_program(prog, confidence_threshold=threshold)
        for r in reports:
            print(r.summary())
            print()

    results = execute_program(prog)
    for result in results:
        print(result.confidence_trace())
        print(f"\nExecution plan:\n{result.plan}")
        print()


def cmd_hash(args):
    """Show hash-based node IDs."""
    source = Path(args.file).read_text()
    prog = transpile(source)

    hashed_scopes = hash_program(prog)
    for hs in hashed_scopes:
        print(hs.to_ail())
        print()
        print(hs.summary())
        print()


def cmd_bench(args):
    """Benchmark optimized vs unoptimized."""
    source = Path(args.file).read_text()

    # Compile
    t0 = time.perf_counter()
    prog = transpile(source)
    compile_time = time.perf_counter() - t0

    # Cost analysis (unoptimized)
    costs_before = analyze_program(prog)

    # Optimize
    threshold = args.threshold or 0.1
    t0 = time.perf_counter()
    opt_prog, reports = optimize_program(prog, confidence_threshold=threshold)
    optimize_time = time.perf_counter() - t0

    # Cost analysis (optimized)
    costs_after = analyze_program(opt_prog)

    # Execute both
    t0 = time.perf_counter()
    results_before = execute_program(prog)
    exec_before_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results_after = execute_program(opt_prog)
    exec_after_time = time.perf_counter() - t0

    # Binary sizes
    binary_before = encode(prog)
    binary_after = encode(opt_prog)

    # Hash analysis
    hashed = hash_program(opt_prog)

    # Report
    print("╔═══════════════════════════════════════════════════════╗")
    print("║              AIL BENCHMARK REPORT                    ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print(f"║  Source: {args.file:<45s} ║")
    print("╠═══════════════════════════════════════════════════════╣")

    for i, (cb, ca, rb, ra, rep) in enumerate(zip(
        costs_before, costs_after, results_before, results_after, reports
    )):
        scope_name = cb.scope_name
        print(f"║  Scope: {scope_name:<45s} ║")
        print("║                                                       ║")
        print(f"║  Nodes:    {cb.total_nodes:>4d}  →  {ca.total_nodes:>4d}  "
              f"({rep.nodes_eliminated:+d}){' ' * 25}║")
        print(f"║  Tokens:   {cb.total_token_cost:>4d}  →  {ca.total_token_cost:>4d}  "
              f"(saved {rep.token_savings}){' ' * 22}║")
        print(f"║  LLM tier: {cb.max_model_tier.value:<10s} →  {ca.max_model_tier.value:<10s}"
              f"{' ' * 21}║")
        print(f"║  Cache:    {cb.cacheable_nodes}/{cb.total_nodes}     →  "
              f"{ca.cacheable_nodes}/{ca.total_nodes}{' ' * 31}║")
        print(f"║  Binary:   {len(binary_before):>4d}B   →  {len(binary_after):>4d}B  "
              f"({(1-len(binary_after)/max(len(binary_before),1))*100:+.1f}%){' ' * 18}║")
        print(f"║  Confidence: ~{rb.output_confidence:.4f} →  ~{ra.output_confidence:.4f}"
              f"{' ' * 23}║")

    print("║                                                       ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print(f"║  Compile time:  {compile_time*1000:>8.2f}ms{' ' * 33}║")
    print(f"║  Optimize time: {optimize_time*1000:>8.2f}ms{' ' * 33}║")
    print(f"║  Exec (before): {exec_before_time*1000:>8.2f}ms{' ' * 33}║")
    print(f"║  Exec (after):  {exec_after_time*1000:>8.2f}ms{' ' * 33}║")
    print("╚═══════════════════════════════════════════════════════╝")

    # Optimization details
    print()
    for rep in reports:
        print(rep.summary())


def cmd_execute(args):
    """Transpile and execute through the runtime (real execution)."""
    import asyncio
    from runtime import Runtime, MockBackend, AnthropicBackend

    source = Path(args.file).read_text()
    prog = transpile(source)

    if args.optimize:
        threshold = args.threshold or 0.1
        prog, reports = optimize_program(prog, confidence_threshold=threshold)
        for r in reports:
            print(r.summary())
            print()

    # Choose backend
    if args.backend == "anthropic":
        backend = AnthropicBackend()
    else:
        backend = MockBackend()

    rt = Runtime(backend=backend)

    async def run():
        return await rt.execute(prog)

    result = asyncio.run(run())
    print(result.summary())

    for sr in result.scope_results:
        print(sr.trace())


def cmd_price(args):
    """Show real pricing analysis."""
    from pricing import price_program, compare_pricing

    source = Path(args.file).read_text()
    prog = transpile(source)

    provider = getattr(args, "provider", "anthropic")
    reports = price_program(prog, provider)
    for r in reports:
        print(r.summary())

    # Also show optimized pricing
    if not getattr(args, "no_optimize", False):
        threshold = getattr(args, "threshold", 0.1) or 0.1
        opt_prog, _ = optimize_program(prog, confidence_threshold=threshold)
        print("\n── Optimized ──────────────────────────────────")
        opt_reports = price_program(opt_prog, provider)
        for r in opt_reports:
            print(r.summary())

        # Comparison
        for before_scope, after_scope in zip(prog.scopes, opt_prog.scopes):
            print()
            print(compare_pricing(before_scope, after_scope, provider))


def cmd_viz(args):
    """Generate interactive HTML graph visualization."""
    from visualizer import render_html, render_comparison, render_program

    source = Path(args.file).read_text()
    prog = transpile(source)

    provider = getattr(args, "provider", "anthropic")

    if args.optimize:
        opt_prog, _ = optimize_program(prog)
        # Render comparison for each scope
        for before, after in zip(prog.scopes, opt_prog.scopes):
            from visualizer import render_comparison
            html = render_comparison(before, after, pricing_provider=provider)
            out_path = Path(args.file).with_suffix(f".comparison.html")
            if args.output:
                out_path = Path(args.output)
            out_path.write_text(html)
            print(f"Comparison: {out_path}")
    else:
        html = render_program(prog, pricing_provider=provider)
        out_path = Path(args.file).with_suffix(".html")
        if args.output:
            out_path = Path(args.output)
        out_path.write_text(html)
        print(f"Graph: {out_path}")


def cmd_info(args):
    """Decode and inspect a binary AIL file."""
    data = Path(args.file).read_bytes()
    prog = decode(data)

    print(f"File:    {args.file}")
    print(f"Size:    {len(data)} bytes")
    print(f"Scopes:  {len(prog.scopes)}")
    print()

    for scope in prog.scopes:
        print(f"  Scope: {scope.name}")
        print(f"  Nodes: {len(scope.nodes)}")
        print()

    print("── Decoded AIL ────────────────────────────────")
    print(prog.to_ail())

    # Cost analysis
    costs = analyze_program(prog)
    print()
    for c in costs:
        print(c.summary())


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="ail",
        description="AIL — Artificial Intent Language toolchain",
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress banner")

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Shared parent for --quiet on subcommands too
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress banner")

    # compile
    p_compile = sub.add_parser("compile", parents=[shared], help="Transpile Python → AIL")
    p_compile.add_argument("file", help="Python source file")
    p_compile.add_argument("--binary", "-b", action="store_true",
                           help="Output binary format")
    p_compile.add_argument("--json", "-j", action="store_true",
                           help="Output JSON format")
    p_compile.add_argument("--output", "-o", help="Output file path")

    # optimize
    p_optimize = sub.add_parser("optimize", parents=[shared], help="Transpile + optimize")
    p_optimize.add_argument("file", help="Python source file")
    p_optimize.add_argument("--threshold", "-t", type=float, default=0.1,
                            help="Confidence pruning threshold")
    p_optimize.add_argument("--binary", "-b", action="store_true",
                            help="Also output binary")
    p_optimize.add_argument("--output", "-o", help="Binary output path")

    # run
    p_run = sub.add_parser("run", parents=[shared], help="Transpile + execute (confidence trace)")
    p_run.add_argument("file", help="Python source file")
    p_run.add_argument("--optimize", "-O", action="store_true",
                       help="Optimize before executing")
    p_run.add_argument("--threshold", "-t", type=float, default=0.1,
                       help="Confidence pruning threshold")

    # hash
    p_hash = sub.add_parser("hash", parents=[shared], help="Show hash-based node IDs")
    p_hash.add_argument("file", help="Python source file")

    # bench
    p_bench = sub.add_parser("bench", parents=[shared], help="Benchmark optimized vs unoptimized")
    p_bench.add_argument("file", help="Python source file")
    p_bench.add_argument("--threshold", "-t", type=float, default=0.1,
                         help="Confidence pruning threshold")

    # execute (runtime)
    p_exec = sub.add_parser("execute", parents=[shared], help="Run through runtime engine")
    p_exec.add_argument("file", help="Python source file")
    p_exec.add_argument("--optimize", "-O", action="store_true",
                        help="Optimize before executing")
    p_exec.add_argument("--threshold", "-t", type=float, default=0.1,
                        help="Confidence pruning threshold")
    p_exec.add_argument("--backend", choices=["mock", "anthropic"], default="mock",
                        help="Backend to use (default: mock)")

    # price
    p_price = sub.add_parser("price", parents=[shared], help="Show real pricing")
    p_price.add_argument("file", help="Python source file")
    p_price.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                         help="Pricing provider")
    p_price.add_argument("--threshold", "-t", type=float, default=0.1,
                         help="Confidence pruning threshold")
    p_price.add_argument("--no-optimize", action="store_true",
                         help="Skip optimization comparison")

    # viz
    p_viz = sub.add_parser("viz", parents=[shared], help="Generate HTML graph visualization")
    p_viz.add_argument("file", help="Python source file")
    p_viz.add_argument("--optimize", "-O", action="store_true",
                       help="Show before/after comparison")
    p_viz.add_argument("--output", "-o", help="Output HTML path")
    p_viz.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                       help="Pricing provider")

    # info
    p_info = sub.add_parser("info", parents=[shared], help="Inspect binary AIL file")
    p_info.add_argument("file", help="Binary AIL file (.ail.bin)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not args.quiet:
        print_banner()

    commands = {
        "compile": cmd_compile,
        "optimize": cmd_optimize,
        "run": cmd_run,
        "execute": cmd_execute,
        "hash": cmd_hash,
        "bench": cmd_bench,
        "price": cmd_price,
        "viz": cmd_viz,
        "info": cmd_info,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
