"""
AIL v0.4 -- Graph Visualizer
Renders AIL graphs as interactive HTML DAGs with cost/confidence annotations.

Generates a standalone HTML file with:
  - Nodes as colored boxes (color = model tier)
  - Edges showing data flow with confidence annotations
  - Cost overlay ($/node, total $/scope)
  - Click a node to see details (inputs, outputs, metadata)
  - Before/after optimization comparison view

Uses pure SVG + vanilla JS -- no dependencies.

Usage:
    from visualizer import render_html, render_comparison

    html = render_html(scope)                    # single graph
    html = render_comparison(before, after)      # before/after
    Path("graph.html").write_text(html)
"""

from __future__ import annotations

import json
import math
from typing import Optional

from ail_types import (
    AILNode, AILScope, AILProgram,
    GraphRef, ScopeRef, InputBinding, InlineExpr,
)
from executor import execute_scope, NODE_PROPAGATION, PropagationRule
from cost_model import analyze_scope, NODE_COSTS, ModelTier
from pricing import price_scope, PricingReport


# -- Layout Engine ---------------------------------------------------------

TIER_COLORS = {
    ModelTier.NONE:     "#4a5568",   # gray
    ModelTier.CHEAP:    "#38a169",   # green
    ModelTier.MID:      "#d69e2e",   # yellow
    ModelTier.POWERFUL: "#e53e3e",   # red
}

CATEGORY_COLORS = {
    "io":   "#63b3ed",   # blue
    "\u223f":   "#9f7aea",   # purple (transform)
    "\u2297":   "#48bb78",   # green (filter)
    "\u03a3":   "#ed8936",   # orange (aggregate)
    "\u22a2":   "#fc8181",   # red (control)
    "\u27f3":   "#f6ad55",   # warm orange (loop)
    "\u2295":   "#76e4f7",   # cyan (merge)
    "\u2207":   "#b794f4",   # light purple (sort)
    "~":    "#f687b3",   # pink (AI-native)
    "call": "#a0aec0",   # gray (call)
}


def _node_color(node_type: str) -> str:
    """Get color for a node based on its type category."""
    profile = NODE_COSTS.get(node_type)
    if profile and profile.model_tier != ModelTier.NONE:
        return TIER_COLORS[profile.model_tier]
    category = node_type.split(":")[0]
    return CATEGORY_COLORS.get(category, "#718096")


def _layout_nodes(scope: AILScope) -> list[dict]:
    """
    Compute x,y positions for each node using layered layout.
    Uses the execution plan levels for horizontal position.
    """
    from executor import schedule, build_dependency_graph

    plan = schedule(scope)
    exec_result = execute_scope(scope)
    cost_analysis = analyze_scope(scope)

    nodes = []
    x_spacing = 220
    y_spacing = 100
    x_offset = 80
    y_offset = 60

    for level_idx, level in enumerate(plan.levels):
        # Center nodes vertically within their level
        total_height = (len(level) - 1) * y_spacing
        y_start = y_offset + (400 - total_height) / 2  # center in 400px height

        for pos, node_idx in enumerate(level):
            node = scope.nodes[node_idx]
            ns = exec_result.node_states[node_idx]
            nc = cost_analysis.node_costs[node_idx]

            x = x_offset + level_idx * x_spacing
            y = y_start + pos * y_spacing

            nodes.append({
                "idx": node_idx,
                "type": node.node_type,
                "x": x,
                "y": y,
                "color": _node_color(node.node_type),
                "confidence": ns.confidence,
                "error": ns.error,
                "token_cost": nc.profile.token_cost,
                "model_tier": nc.profile.model_tier.value,
                "cacheable": nc.profile.cacheable,
                "cacheable_subgraph": nc.cacheable_subgraph,
                "inputs": [str(inp) for inp in node.inputs],
                "outputs": [str(out) for out in node.outputs],
                "metadata": node.metadata,
            })

    return nodes


def _layout_edges(scope: AILScope, node_positions: list[dict]) -> list[dict]:
    """Compute edges between nodes."""
    # Map §N -> node index
    ref_to_node: dict[int, int] = {}
    for i, node in enumerate(scope.nodes):
        for out in node.outputs:
            if isinstance(out, GraphRef):
                ref_to_node[out.index] = i

    # Build pos lookup
    pos_map = {n["idx"]: n for n in node_positions}

    edges = []
    for i, node in enumerate(scope.nodes):
        for inp in node.inputs:
            if isinstance(inp.ref, GraphRef) and inp.ref.index in ref_to_node:
                src_idx = ref_to_node[inp.ref.index]
                if src_idx in pos_map and i in pos_map:
                    src = pos_map[src_idx]
                    dst = pos_map[i]
                    edges.append({
                        "from": src_idx,
                        "to": i,
                        "x1": src["x"] + 160,  # right edge of node
                        "y1": src["y"] + 25,    # center of node
                        "x2": dst["x"],          # left edge of node
                        "y2": dst["y"] + 25,
                        "role": inp.role,
                        "confidence": src["confidence"],
                    })

    return edges


# -- HTML Generator --------------------------------------------------------

def render_html(
    scope: AILScope,
    title: Optional[str] = None,
    pricing_provider: str = "anthropic",
) -> str:
    """Render a scope as a standalone interactive HTML file."""
    nodes = _layout_nodes(scope)
    edges = _layout_edges(scope, nodes)
    pricing = price_scope(scope, pricing_provider)

    title = title or f"AIL Graph: {scope.name}"

    # Calculate SVG dimensions
    max_x = max((n["x"] for n in nodes), default=200) + 250
    max_y = max((n["y"] for n in nodes), default=200) + 150

    return _html_template(
        title=title,
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        scope_name=scope.name,
        total_cost=f"${pricing.total_cost_usd:.6f}",
        cached_cost=f"${pricing.cached_cost_usd:.6f}",
        total_nodes=len(scope.nodes),
        llm_nodes=pricing.llm_node_count,
        svg_width=max_x,
        svg_height=max_y,
    )


def render_comparison(
    before: AILScope,
    after: AILScope,
    title: Optional[str] = None,
    pricing_provider: str = "anthropic",
) -> str:
    """Render before/after optimization comparison."""
    nodes_before = _layout_nodes(before)
    edges_before = _layout_edges(before, nodes_before)
    nodes_after = _layout_nodes(after)
    edges_after = _layout_edges(after, nodes_after)
    pricing_before = price_scope(before, pricing_provider)
    pricing_after = price_scope(after, pricing_provider)

    title = title or f"AIL Optimization: {before.name}"

    max_x = max(
        max((n["x"] for n in nodes_before), default=200),
        max((n["x"] for n in nodes_after), default=200),
    ) + 250
    max_y = max(
        max((n["y"] for n in nodes_before), default=200),
        max((n["y"] for n in nodes_after), default=200),
    ) + 150

    return _comparison_template(
        title=title,
        before_nodes=json.dumps(nodes_before),
        before_edges=json.dumps(edges_before),
        after_nodes=json.dumps(nodes_after),
        after_edges=json.dumps(edges_after),
        before_name=before.name,
        after_name=after.name,
        before_cost=f"${pricing_before.total_cost_usd:.6f}",
        after_cost=f"${pricing_after.total_cost_usd:.6f}",
        savings=f"${pricing_before.total_cost_usd - pricing_after.total_cost_usd:.6f}",
        before_nodes_count=len(before.nodes),
        after_nodes_count=len(after.nodes),
        svg_width=max_x,
        svg_height=max_y,
    )


def render_program(program: AILProgram, pricing_provider: str = "anthropic") -> str:
    """Render all scopes in a program."""
    if len(program.scopes) == 1:
        return render_html(program.scopes[0], pricing_provider=pricing_provider)

    # Multi-scope: tabs
    scope_htmls = []
    for scope in program.scopes:
        nodes = _layout_nodes(scope)
        edges = _layout_edges(scope, nodes)
        pricing = price_scope(scope, pricing_provider)
        scope_htmls.append({
            "name": scope.name,
            "nodes": nodes,
            "edges": edges,
            "cost": f"${pricing.total_cost_usd:.6f}",
            "node_count": len(scope.nodes),
        })

    return _multi_scope_template(scope_htmls)


# -- HTML Templates --------------------------------------------------------

def _html_template(**kwargs) -> str:
    # Use string replacement to avoid f-string vs JS template literal conflicts
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
         background: #1a202c; color: #e2e8f0; }
  .header { padding: 16px 24px; background: #2d3748; border-bottom: 1px solid #4a5568;
             display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .stats { display: flex; gap: 24px; font-size: 13px; color: #a0aec0; }
  .stat-val { color: #63b3ed; font-weight: 600; }
  .graph-container { padding: 24px; overflow: auto; }
  svg { background: #1a202c; }
  .node { cursor: pointer; transition: filter 0.15s; }
  .node:hover { filter: brightness(1.2); }
  .node-rect { rx: 6; ry: 6; stroke-width: 2; }
  .node-label { font-size: 12px; font-weight: 600; fill: white;
                 text-anchor: middle; dominant-baseline: central; }
  .node-conf { font-size: 10px; fill: #a0aec0;
                text-anchor: middle; dominant-baseline: central; }
  .node-cost { font-size: 9px; fill: #f6ad55;
                text-anchor: middle; dominant-baseline: central; }
  .edge { stroke-width: 2; fill: none; }
  .edge-label { font-size: 9px; fill: #718096; }
  .tooltip { position: fixed; background: #2d3748; border: 1px solid #4a5568;
              border-radius: 8px; padding: 12px; font-size: 12px;
              max-width: 300px; display: none; z-index: 100; }
  .tooltip h3 { color: #63b3ed; margin-bottom: 8px; }
  .tooltip .row { display: flex; justify-content: space-between; margin: 2px 0; }
  .tooltip .label { color: #a0aec0; }
  .legend { padding: 8px 24px; display: flex; gap: 16px; font-size: 11px; color: #a0aec0; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
</style>
</head>
<body>
<div class="header">
  <h1>__SCOPE_NAME__</h1>
  <div class="stats">
    <span>Nodes: <span class="stat-val">__TOTAL_NODES__</span></span>
    <span>LLM nodes: <span class="stat-val">__LLM_NODES__</span></span>
    <span>Cost/exec: <span class="stat-val">__TOTAL_COST__</span></span>
    <span>Cached: <span class="stat-val">__CACHED_COST__</span></span>
  </div>
</div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#4a5568"></div> No LLM</div>
  <div class="legend-item"><div class="legend-dot" style="background:#38a169"></div> Cheap (Haiku)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#d69e2e"></div> Mid (Sonnet)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#e53e3e"></div> Powerful (Opus)</div>
</div>
<div class="graph-container">
  <svg id="graph" width="__SVG_WIDTH__" height="__SVG_HEIGHT__"></svg>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const nodes = __NODES_JSON__;
const edges = __EDGES_JSON__;

const svg = document.getElementById('graph');
const tooltip = document.getElementById('tooltip');
const ns = 'http://www.w3.org/2000/svg';

edges.forEach(e => {
  const dx = e.x2 - e.x1;
  const cx1 = e.x1 + dx * 0.4;
  const cx2 = e.x2 - dx * 0.4;
  const path = document.createElementNS(ns, 'path');
  const opacity = Math.max(0.3, e.confidence);
  path.setAttribute('d', `M${e.x1},${e.y1} C${cx1},${e.y1} ${cx2},${e.y2} ${e.x2},${e.y2}`);
  path.setAttribute('stroke', `rgba(99, 179, 237, ${opacity})`);
  path.classList.add('edge');
  svg.appendChild(path);

  const mx = (e.x1 + e.x2) / 2;
  const my = (e.y1 + e.y2) / 2 - 8;
  const text = document.createElementNS(ns, 'text');
  text.setAttribute('x', mx);
  text.setAttribute('y', my);
  text.classList.add('edge-label');
  text.textContent = `${e.role} ~${e.confidence.toFixed(2)}`;
  svg.appendChild(text);
});

nodes.forEach(n => {
  const g = document.createElementNS(ns, 'g');
  g.classList.add('node');
  g.setAttribute('transform', `translate(${n.x}, ${n.y})`);

  const rect = document.createElementNS(ns, 'rect');
  rect.setAttribute('width', 160);
  rect.setAttribute('height', 50);
  rect.setAttribute('fill', n.color);
  rect.setAttribute('stroke', n.cacheable_subgraph ? '#48bb78' : n.color);
  rect.classList.add('node-rect');
  if (n.cacheable_subgraph) rect.setAttribute('stroke-dasharray', '4,2');
  g.appendChild(rect);

  const label = document.createElementNS(ns, 'text');
  label.setAttribute('x', 80);
  label.setAttribute('y', 18);
  label.classList.add('node-label');
  label.textContent = `[${n.type}]`;
  g.appendChild(label);

  const conf = document.createElementNS(ns, 'text');
  conf.setAttribute('x', 80);
  conf.setAttribute('y', 33);
  conf.classList.add('node-conf');
  const confLabel = n.confidence >= 1 ? '~!' : n.confidence > 0.9 ? '~+' : n.confidence > 0.3 ? '~?' : '~-';
  conf.textContent = `~${n.confidence.toFixed(3)} (${confLabel})`;
  g.appendChild(conf);

  if (n.token_cost > 0) {
    const cost = document.createElementNS(ns, 'text');
    cost.setAttribute('x', 80);
    cost.setAttribute('y', 45);
    cost.classList.add('node-cost');
    cost.textContent = `${n.token_cost}tok [${n.model_tier}]`;
    g.appendChild(cost);
  }

  g.addEventListener('click', (ev) => {
    tooltip.innerHTML = `
      <h3>$` + '`' + `${n.idx} [${n.type}]</h3>
      <div class="row"><span class="label">Confidence:</span> ~${n.confidence.toFixed(4)}</div>
      <div class="row"><span class="label">Model tier:</span> ${n.model_tier}</div>
      <div class="row"><span class="label">Token cost:</span> ${n.token_cost}</div>
      <div class="row"><span class="label">Cacheable:</span> ${n.cacheable ? 'Yes' : 'No'}</div>
      <div class="row"><span class="label">Subgraph cache:</span> ${n.cacheable_subgraph ? 'Yes' : 'No'}</div>
      ${n.error ? '<div class="row"><span class="label">Error:</span> ' + n.error + '</div>' : ''}
      <div style="margin-top:8px; color:#a0aec0;">
        <div>Inputs: ${n.inputs.join(', ')}</div>
        <div>Outputs: ${n.outputs.join(', ')}</div>
      </div>
    `;
    tooltip.style.display = 'block';
    tooltip.style.left = ev.clientX + 10 + 'px';
    tooltip.style.top = ev.clientY + 10 + 'px';
  });

  svg.appendChild(g);
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.node')) tooltip.style.display = 'none';
});
</script>
</body>
</html>"""
    return (html
        .replace("__TITLE__", str(kwargs['title']))
        .replace("__SCOPE_NAME__", str(kwargs['scope_name']))
        .replace("__TOTAL_NODES__", str(kwargs['total_nodes']))
        .replace("__LLM_NODES__", str(kwargs['llm_nodes']))
        .replace("__TOTAL_COST__", str(kwargs['total_cost']))
        .replace("__CACHED_COST__", str(kwargs['cached_cost']))
        .replace("__SVG_WIDTH__", str(kwargs['svg_width']))
        .replace("__SVG_HEIGHT__", str(kwargs['svg_height']))
        .replace("__NODES_JSON__", kwargs['nodes_json'])
        .replace("__EDGES_JSON__", kwargs['edges_json'])
    )


def _comparison_template(**kwargs) -> str:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, monospace;
         background: #1a202c; color: #e2e8f0; }
  .header { padding: 16px 24px; background: #2d3748; text-align: center; }
  .header h1 { font-size: 20px; }
  .comparison { display: flex; }
  .panel { flex: 1; border-right: 1px solid #4a5568; }
  .panel:last-child { border-right: none; }
  .panel-header { padding: 12px; background: #2d3748; text-align: center;
                   border-bottom: 1px solid #4a5568; }
  .panel-header h2 { font-size: 14px; }
  .panel-header .cost { color: #63b3ed; font-size: 18px; font-weight: bold; }
  .savings { padding: 12px; background: #22543d; text-align: center;
              font-size: 16px; color: #48bb78; font-weight: bold; }
  svg { background: #1a202c; width: 100%; }
  .node { cursor: pointer; }
  .node:hover { filter: brightness(1.2); }
  .node-rect { rx: 6; ry: 6; stroke-width: 2; }
  .node-label { font-size: 11px; font-weight: 600; fill: white;
                 text-anchor: middle; dominant-baseline: central; }
  .node-conf { font-size: 9px; fill: #a0aec0;
                text-anchor: middle; dominant-baseline: central; }
  .edge { stroke-width: 2; fill: none; }
</style>
</head>
<body>
<div class="header">
  <h1>__TITLE__</h1>
</div>
<div class="savings">
  Saved __SAVINGS__/execution | Nodes: __BEFORE_COUNT__ -> __AFTER_COUNT__
</div>
<div class="comparison">
  <div class="panel">
    <div class="panel-header">
      <h2>BEFORE</h2>
      <div class="cost">__BEFORE_COST__/exec</div>
      <div>__BEFORE_COUNT__ nodes</div>
    </div>
    <svg id="before" width="__SVG_WIDTH__" height="__SVG_HEIGHT__"></svg>
  </div>
  <div class="panel">
    <div class="panel-header">
      <h2>AFTER (optimized)</h2>
      <div class="cost">__AFTER_COST__/exec</div>
      <div>__AFTER_COUNT__ nodes</div>
    </div>
    <svg id="after" width="__SVG_WIDTH__" height="__SVG_HEIGHT__"></svg>
  </div>
</div>
<script>
function renderGraph(svgId, nodes, edges) {
  const svg = document.getElementById(svgId);
  const ns = 'http://www.w3.org/2000/svg';

  edges.forEach(e => {
    const dx = e.x2 - e.x1; const cx1 = e.x1 + dx*0.4; const cx2 = e.x2 - dx*0.4;
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', `M${e.x1},${e.y1} C${cx1},${e.y1} ${cx2},${e.y2} ${e.x2},${e.y2}`);
    path.setAttribute('stroke', `rgba(99,179,237,${Math.max(0.3, e.confidence)})`);
    path.classList.add('edge');
    svg.appendChild(path);
  });

  nodes.forEach(n => {
    const g = document.createElementNS(ns, 'g');
    g.classList.add('node');
    g.setAttribute('transform', `translate(${n.x},${n.y})`);

    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('width', 160); rect.setAttribute('height', 44);
    rect.setAttribute('fill', n.color);
    rect.setAttribute('stroke', n.color);
    rect.classList.add('node-rect');
    g.appendChild(rect);

    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', 80); label.setAttribute('y', 16);
    label.classList.add('node-label');
    label.textContent = `[${n.type}]`;
    g.appendChild(label);

    const conf = document.createElementNS(ns, 'text');
    conf.setAttribute('x', 80); conf.setAttribute('y', 32);
    conf.classList.add('node-conf');
    conf.textContent = `~${n.confidence.toFixed(3)}`;
    g.appendChild(conf);

    svg.appendChild(g);
  });
}

renderGraph('before', __BEFORE_NODES__, __BEFORE_EDGES__);
renderGraph('after', __AFTER_NODES__, __AFTER_EDGES__);
</script>
</body>
</html>"""
    return (html
        .replace("__TITLE__", str(kwargs['title']))
        .replace("__SAVINGS__", str(kwargs['savings']))
        .replace("__BEFORE_COUNT__", str(kwargs['before_nodes_count']))
        .replace("__AFTER_COUNT__", str(kwargs['after_nodes_count']))
        .replace("__BEFORE_COST__", str(kwargs['before_cost']))
        .replace("__AFTER_COST__", str(kwargs['after_cost']))
        .replace("__SVG_WIDTH__", str(kwargs['svg_width']))
        .replace("__SVG_HEIGHT__", str(kwargs['svg_height']))
        .replace("__BEFORE_NODES__", kwargs['before_nodes'])
        .replace("__BEFORE_EDGES__", kwargs['before_edges'])
        .replace("__AFTER_NODES__", kwargs['after_nodes'])
        .replace("__AFTER_EDGES__", kwargs['after_edges'])
    )


def _multi_scope_template(scopes: list[dict]) -> str:
    """Generate multi-scope tabbed view."""
    tabs_html = "".join(
        f'<button class="tab" onclick="showScope({i})">{s["name"]} ({s["node_count"]} nodes, {s["cost"]})</button>'
        for i, s in enumerate(scopes)
    )
    scope_data = json.dumps(scopes, default=str)

    html = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>AIL Program</title>
<style>
  body { font-family: monospace; background: #1a202c; color: #e2e8f0; margin: 0; }
  .tabs { display: flex; background: #2d3748; border-bottom: 1px solid #4a5568; }
  .tab { padding: 12px 24px; background: none; border: none; color: #a0aec0;
          cursor: pointer; font-family: monospace; }
  .tab.active { color: #63b3ed; border-bottom: 2px solid #63b3ed; }
  svg { background: #1a202c; }
  .node-rect { rx: 6; ry: 6; stroke-width: 2; }
  .node-label { font-size: 11px; fill: white; text-anchor: middle; dominant-baseline: central; font-weight: 600; }
  .edge { stroke-width: 2; fill: none; }
</style>
</head>
<body>
<div class="tabs">__TABS__</div>
<div id="container"></div>
<script>
const scopes = __SCOPE_DATA__;
function showScope(i) {
  document.querySelectorAll('.tab').forEach((t,j) => t.classList.toggle('active', j===i));
  const s = scopes[i];
  const container = document.getElementById('container');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', 1200); svg.setAttribute('height', 500);
  container.innerHTML = '';
  container.appendChild(svg);
  const ns = 'http://www.w3.org/2000/svg';
  (s.edges||[]).forEach(e => {
    const p = document.createElementNS(ns,'path');
    const dx=e.x2-e.x1;
    p.setAttribute('d',`M${e.x1},${e.y1} C${e.x1+dx*0.4},${e.y1} ${e.x2-dx*0.4},${e.y2} ${e.x2},${e.y2}`);
    p.setAttribute('stroke',`rgba(99,179,237,0.5)`);
    p.classList.add('edge'); svg.appendChild(p);
  });
  (s.nodes||[]).forEach(n => {
    const g=document.createElementNS(ns,'g');
    g.setAttribute('transform',`translate(${n.x},${n.y})`);
    const r=document.createElementNS(ns,'rect');
    r.setAttribute('width',160);r.setAttribute('height',44);
    r.setAttribute('fill',n.color);r.setAttribute('stroke',n.color);
    r.classList.add('node-rect');g.appendChild(r);
    const t=document.createElementNS(ns,'text');
    t.setAttribute('x',80);t.setAttribute('y',22);
    t.classList.add('node-label');t.textContent=`[${n.type}]`;
    g.appendChild(t);svg.appendChild(g);
  });
}
showScope(0);
</script>
</body>
</html>"""
    return html.replace("__TABS__", tabs_html).replace("__SCOPE_DATA__", scope_data)


# -- Demo -----------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from optimizer import optimize_scope

    SEP = "-" * 60

    # -- Test 1: Single graph --
    print(f"\n{SEP}")
    print("TEST 1: Render single graph")
    print(SEP)

    scope = AILScope("demo_pipeline")
    refs = [scope.next_ref() for _ in range(7)]
    scope.add_node(AILNode("io:in",
        inputs=[InputBinding("param", '"query"')],
        outputs=[refs[0]]))
    scope.add_node(AILNode("~:embed",
        inputs=[InputBinding("val", refs[0])],
        outputs=[refs[1]]))
    scope.add_node(AILNode("io:read",
        inputs=[InputBinding("src", '"vector_db"')],
        outputs=[refs[2]]))
    scope.add_node(AILNode("~:sim",
        inputs=[InputBinding("a", refs[1]), InputBinding("b", refs[2])],
        outputs=[refs[3]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[3])],
        outputs=[refs[4]]))
    scope.add_node(AILNode("~:rank",
        inputs=[InputBinding("col", refs[4])],
        outputs=[refs[5]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[5])]))

    html = render_html(scope)
    out_path = Path("ail_graph.html")
    out_path.write_text(html)
    print(f"  Written: {out_path} ({len(html)} bytes)")
    assert len(html) > 1000, "HTML should be substantial"
    assert "<svg" in html, "Should contain SVG"
    print("PASSED: Single graph rendered")

    # -- Test 2: Before/after comparison --
    print(f"\n{SEP}")
    print("TEST 2: Render optimization comparison")
    print(SEP)

    scope = AILScope("redundant_ops")
    refs = [scope.next_ref() for _ in range(8)]
    scope.add_node(AILNode("io:in", outputs=[refs[0]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[0]), InputBinding("cond", InlineExpr("active = true"))],
        outputs=[refs[1]]))
    scope.add_node(AILNode("⊗:where",
        inputs=[InputBinding("col", refs[1]), InputBinding("cond", InlineExpr("score > 0.5"))],
        outputs=[refs[2]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[2])],
        outputs=[refs[3]]))
    scope.add_node(AILNode("~:infer",
        inputs=[InputBinding("val", refs[2])],
        outputs=[refs[4]]))
    scope.add_node(AILNode("~:rank",
        inputs=[InputBinding("col", refs[3])],
        outputs=[refs[5]]))
    scope.add_node(AILNode("io:out",
        inputs=[InputBinding("val", refs[5])]))

    opt_scope, report = optimize_scope(scope)
    html = render_comparison(scope, opt_scope)
    out_path = Path("ail_comparison.html")
    out_path.write_text(html)
    print(f"  Written: {out_path} ({len(html)} bytes)")
    assert "BEFORE" in html and "AFTER" in html
    print("PASSED: Comparison rendered")

    # -- Test 3: Transpiled pipeline --
    print(f"\n{SEP}")
    print("TEST 3: Transpiled pipeline visualization")
    print(SEP)

    from transpiler import transpile
    prog = transpile("""
async def recommend(catalog, query, min_confidence=0.75):
    results = await catalog.search(query)
    confident = [r for r in results if r['confidence'] >= min_confidence]
    ranked = sorted(confident, key=lambda r: r['confidence'], reverse=True)
    return ranked[:3]
""")

    html = render_program(prog)
    out_path = Path("ail_program.html")
    out_path.write_text(html)
    print(f"  Written: {out_path} ({len(html)} bytes)")
    print("PASSED: Program visualization rendered")

    print(f"\n{SEP}")
    print("ALL TESTS PASSED")
    print(SEP)
    print(f"\nOpen the HTML files in a browser:")
    print(f"  open ail_graph.html")
    print(f"  open ail_comparison.html")
    print(f"  open ail_program.html")
