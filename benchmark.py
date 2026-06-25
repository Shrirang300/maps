"""
benchmark_routing.py
====================
Benchmarks the hierarchical Dijkstra (dijkstra.py) against plain
NetworkX Dijkstra on a synthetic city-scale graph.

Tests
-----
1. Speed        — wall-clock time, hierarchical vs vanilla, across
                  short / medium / long / cross-city query buckets.
2. Correctness  — cost ratio  hierarchical_cost / nx_cost  for every
                  query pair.  Ideally ≈1.0; flags anything > 1.05.
3. Stress       — 200 random queries, latency percentiles (p50/p95/p99).
4. Node visits  — how many nodes each algorithm actually pops from the
                  heap (lower = more efficient search).

Synthetic graph
---------------
Builds a grid graph that mimics a real OSM MultiDiGraph:
  • N×N grid of intersections  (nodes have x, y attributes)
  • Directed edges both ways
  • travel_time  = random 5–60 s per edge
  • Node IDs are plain integers (matching OSM style)

Blocks / shortcuts are auto-generated using the same logic as
02_partitioner.py / 03_shortcut_builder.py so the benchmark is
self-contained — no real data files required.

Usage
-----
    python benchmark_routing.py [--grid N] [--block-grid B] [--queries Q]

    --grid        City grid dimension  (default 80  → 6 400 nodes)
    --block-grid  Block partition size (default 8   → 8×8 = 64 blocks)
    --queries     Random queries per distance bucket (default 40)
"""

import argparse
import heapq
import math
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# ── make dijkstra.py importable from the outputs dir ──────────────────────────
sys.path.insert(0, "/mnt/user-data/outputs")
from dijkstra import dijkstra as hier_dijkstra   # hierarchical version

random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_city_graph(grid: int) -> nx.MultiDiGraph:
    """
    N×N directed grid graph.  Node id = row*N + col (int).
    Each directed edge gets travel_time drawn from U[5, 60] seconds.
    """
    G = nx.MultiDiGraph()
    N = grid

    # nodes
    for r in range(N):
        for c in range(N):
            nid = r * N + c
            G.add_node(nid, x=float(c), y=float(r))

    # edges (cardinal directions, both ways)
    for r in range(N):
        for c in range(N):
            u = r * N + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    v = nr * N + nc
                    tt = random.uniform(5, 60)
                    G.add_edge(u, v, travel_time=round(tt, 3))

    return G


# ══════════════════════════════════════════════════════════════════════════════
# 2. BLOCK + SHORTCUT BUILDER  (mirrors 02_partitioner + 03_shortcut_builder)
# ══════════════════════════════════════════════════════════════════════════════

def build_blocks(G: nx.MultiDiGraph, city_grid: int, block_grid: int) -> dict:
    N, B = city_grid, block_grid
    min_lat, max_lat = 0.0, float(N - 1)
    min_lon, max_lon = 0.0, float(N - 1)
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    node_block: Dict[int, str] = {}
    for nid, data in G.nodes(data=True):
        lat, lon = float(data["y"]), float(data["x"])
        row = int((lat - min_lat) / lat_span * B)
        col = int((lon - min_lon) / lon_span * B)
        row = max(0, min(B - 1, row))
        col = max(0, min(B - 1, col))
        node_block[nid] = f"{row}_{col}"

    blocks: Dict[str, dict] = {}
    for nid, bid in node_block.items():
        if bid not in blocks:
            blocks[bid] = {"nodes": [], "gates": []}
        blocks[bid]["nodes"].append(str(nid))

    gate_set: Dict[str, Set[str]] = defaultdict(set)
    for u, v in G.edges():
        bu, bv = node_block.get(u), node_block.get(v)
        if bu and bv and bu != bv:
            gate_set[bu].add(str(u))
            gate_set[bv].add(str(v))

    for bid, gates in gate_set.items():
        blocks[bid]["gates"] = sorted(gates)

    return {
        "grid_size": B,
        "bounding_box": {
            "min_lat": min_lat, "max_lat": max_lat,
            "min_lon": min_lon, "max_lon": max_lon,
        },
        "blocks": blocks,
    }


def build_shortcuts(G: nx.MultiDiGraph, blocks_data: dict) -> dict:
    shortcuts = {}
    blocks = blocks_data["blocks"]

    for block_id, binfo in blocks.items():
        gates = binfo["gates"]
        if len(gates) < 2:
            continue
        node_set = set(int(n) for n in binfo["nodes"])
        subgraph = G.subgraph(node_set)
        block_sc: Dict[str, float] = {}

        for sg in gates:
            sg_int = int(sg)
            if sg_int not in subgraph:
                continue
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    subgraph, sg_int, weight="travel_time"
                )
            except nx.NetworkXError:
                continue
            for dg in gates:
                if dg == sg:
                    continue
                dg_int = int(dg)
                if dg_int in lengths:
                    block_sc[f"{sg}_{dg}"] = round(lengths[dg_int], 3)

        if block_sc:
            shortcuts[block_id] = block_sc

    return shortcuts


# ══════════════════════════════════════════════════════════════════════════════
# 3. VANILLA DIJKSTRA  (instrumented to count node visits)
# ══════════════════════════════════════════════════════════════════════════════

def vanilla_dijkstra(
    G: nx.MultiDiGraph, src: int, tgt: int, weight: str = "travel_time"
) -> Tuple[float, int]:
    """Returns (cost, nodes_visited)."""
    dist = {src: 0.0}
    heap = [(0.0, src)]
    visited: Set[int] = set()
    visits = 0

    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        visits += 1
        if u == tgt:
            return cost, visits
        for _, v, data in G.out_edges(u, data=True):
            tt = data.get(weight, 1.0)
            nc = cost + float(tt)
            if nc < dist.get(v, float("inf")):
                dist[v] = nc
                heapq.heappush(heap, (nc, v))

    return float("inf"), visits


# ══════════════════════════════════════════════════════════════════════════════
# 4. INSTRUMENTED HIERARCHICAL WRAPPER  (count heap pops)
# ══════════════════════════════════════════════════════════════════════════════

def hier_dijkstra_instrumented(
    G, src, tgt, shortcuts, blocks_data
) -> Tuple[float, List, int]:
    """Wraps hier_dijkstra and monkey-patches heapq.heappop to count pops."""
    pops = [0]
    original_pop = heapq.heappop

    def counting_pop(h):
        pops[0] += 1
        return original_pop(h)

    heapq.heappop = counting_pop
    try:
        cost, path = hier_dijkstra(
            G, src, tgt,
            weight="travel_time",
            shortcuts=shortcuts,
            blocks_data=blocks_data,
        )
    finally:
        heapq.heappop = original_pop

    return cost, path, pops[0]


# ══════════════════════════════════════════════════════════════════════════════
# 5. QUERY PAIR GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def grid_distance(a: int, b: int, N: int) -> int:
    ra, ca = divmod(a, N)
    rb, cb = divmod(b, N)
    return abs(ra - rb) + abs(ca - cb)


def sample_pairs(nodes: List[int], N: int, count: int, min_d: int, max_d: int):
    pairs, tries = [], 0
    while len(pairs) < count and tries < count * 200:
        tries += 1
        s = random.choice(nodes)
        t = random.choice(nodes)
        d = grid_distance(s, t, N)
        if min_d <= d <= max_d and s != t:
            pairs.append((s, t))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 6. RESULT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def hdr(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*64}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*64}{RESET}")

def row(label, *cols, color=RESET):
    label_col = f"{label:<28}"
    col_strs  = "  ".join(f"{str(c):>14}" for c in cols)
    print(f"  {color}{label_col}{col_strs}{RESET}")

def pct_color(pct):
    if pct >= 95:  return GREEN
    if pct >= 80:  return YELLOW
    return RED

def cost_color(ratio):
    if ratio <= 1.01: return GREEN
    if ratio <= 1.05: return YELLOW
    return RED


# ══════════════════════════════════════════════════════════════════════════════
# 7. BENCHMARK SUITES
# ══════════════════════════════════════════════════════════════════════════════

def bench_speed_and_correctness(G, blocks_data, shortcuts, nodes, N, Q):
    """
    Runs Q queries per distance bucket.
    Reports timing, correctness (cost ratio), and node-visit reduction.
    """
    hdr("TEST 1 — SPEED & CORRECTNESS  (per distance bucket)")
    row("Bucket", "Queries", "Vanilla ms", "Hier ms", "Speedup", "Cost ratio", color=BOLD)
    row("", "", "(avg)", "(avg)", "", "(avg)", color=BOLD)

    BUCKETS = [
        ("Short   (d ≤ 5)",    1,   5),
        ("Medium  (5 < d ≤ 20)", 6,  20),
        ("Long    (20 < d ≤ 50)", 21, 50),
        ("X-City  (d > 50)",   51, 9999),
    ]

    all_ok = True

    for label, lo, hi in BUCKETS:
        pairs = sample_pairs(nodes, N, Q, lo, hi)
        if not pairs:
            row(label, 0, "n/a", "n/a", "n/a", "n/a", color=YELLOW)
            continue

        v_times, h_times, ratios = [], [], []

        for src, tgt in pairs:
            t0 = time.perf_counter()
            v_cost, _ = vanilla_dijkstra(G, src, tgt)
            v_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            h_cost, _, _ = hier_dijkstra_instrumented(G, src, tgt, shortcuts, blocks_data)
            h_times.append((time.perf_counter() - t0) * 1000)

            if v_cost > 0 and h_cost < float("inf"):
                ratios.append(h_cost / v_cost)
            elif v_cost == float("inf") and h_cost == float("inf"):
                ratios.append(1.0)

        avg_v  = sum(v_times) / len(v_times)
        avg_h  = sum(h_times) / len(h_times)
        speedup = avg_v / avg_h if avg_h > 0 else float("inf")
        avg_r  = sum(ratios) / len(ratios) if ratios else float("nan")

        bad = sum(1 for r in ratios if r > 1.05)
        r_color = cost_color(avg_r)
        s_color = GREEN if speedup >= 1.5 else (YELLOW if speedup >= 1.0 else RED)

        if bad:
            all_ok = False

        row(
            label,
            len(pairs),
            f"{avg_v:.2f}",
            f"{avg_h:.2f}",
            f"{s_color}{speedup:.2f}×{RESET}",
            f"{r_color}{avg_r:.4f}{RESET}  {'⚠' if bad else '✓'}",
        )

    return all_ok


def bench_node_visits(G, blocks_data, shortcuts, nodes, N, Q):
    """Compares how many heap pops each algorithm needs."""
    hdr("TEST 2 — NODE VISITS  (heap pops, lower = more efficient)")
    row("Bucket", "Queries", "Vanilla", "Hier", "Reduction %", color=BOLD)

    BUCKETS = [
        ("Medium  (5 < d ≤ 20)", 6,  20),
        ("Long    (20 < d ≤ 50)", 21, 50),
        ("X-City  (d > 50)",   51, 9999),
    ]

    for label, lo, hi in BUCKETS:
        pairs = sample_pairs(nodes, N, Q, lo, hi)
        if not pairs:
            row(label, 0, "n/a", "n/a", "n/a", color=YELLOW)
            continue

        v_visits, h_visits = [], []

        for src, tgt in pairs:
            _, vv = vanilla_dijkstra(G, src, tgt)
            v_visits.append(vv)
            _, _, hv = hier_dijkstra_instrumented(G, src, tgt, shortcuts, blocks_data)
            h_visits.append(hv)

        avg_v = sum(v_visits) / len(v_visits)
        avg_h = sum(h_visits) / len(h_visits)
        reduction = (1 - avg_h / avg_v) * 100 if avg_v > 0 else 0
        r_color = GREEN if reduction > 30 else (YELLOW if reduction > 0 else RED)

        row(
            label,
            len(pairs),
            f"{avg_v:.0f}",
            f"{avg_h:.0f}",
            f"{r_color}{reduction:+.1f}%{RESET}",
        )


def bench_stress(G, blocks_data, shortcuts, nodes, N, Q_stress):
    """200 random queries → latency percentile table."""
    hdr("TEST 3 — STRESS  (latency percentiles, ms)")
    row("Algorithm", "p50", "p75", "p95", "p99", "max", color=BOLD)

    pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(Q_stress)]
    pairs = [(s, t) for s, t in pairs if s != t]

    v_lat, h_lat = [], []
    for src, tgt in pairs:
        t0 = time.perf_counter()
        vanilla_dijkstra(G, src, tgt)
        v_lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hier_dijkstra(G, src, tgt,
                      weight="travel_time",
                      shortcuts=shortcuts,
                      blocks_data=blocks_data)
        h_lat.append((time.perf_counter() - t0) * 1000)

    def percentile(data, p):
        s = sorted(data)
        idx = int(math.ceil(p / 100 * len(s))) - 1
        return s[max(0, idx)]

    for label, lat in [("Vanilla", v_lat), ("Hierarchical", h_lat)]:
        p50  = percentile(lat, 50)
        p75  = percentile(lat, 75)
        p95  = percentile(lat, 95)
        p99  = percentile(lat, 99)
        pmax = max(lat)
        row(label, f"{p50:.2f}", f"{p75:.2f}", f"{p95:.2f}",
            f"{p99:.2f}", f"{pmax:.2f}")


def bench_correctness_distribution(G, blocks_data, shortcuts, nodes, N, Q):
    """Full cost-ratio histogram to show correctness spread."""
    hdr("TEST 4 — CORRECTNESS DISTRIBUTION  (cost ratio histogram)")

    pairs = sample_pairs(nodes, N, Q * 3, 0, 9999)
    ratios = []
    unreachable_vanilla = 0
    unreachable_hier    = 0

    for src, tgt in pairs:
        v_cost, _ = vanilla_dijkstra(G, src, tgt)
        h_cost, _, _ = hier_dijkstra_instrumented(G, src, tgt, shortcuts, blocks_data)

        if v_cost == float("inf"):
            unreachable_vanilla += 1
            continue
        if h_cost == float("inf"):
            unreachable_hier += 1
            continue

        ratios.append(h_cost / v_cost)

    if not ratios:
        print("  No valid pairs found.")
        return

    buckets = [
        ("<0.99  (hier faster — impossible if correct)", lambda r: r < 0.99),
        ("0.99–1.00  (within 1%,  optimal)",             lambda r: 0.99 <= r < 1.00),
        ("1.00       (exact match)",                      lambda r: r == 1.00),
        ("1.00–1.01  (within 1%,  excellent)",            lambda r: 1.00 < r <= 1.01),
        ("1.01–1.05  (within 5%,  acceptable)",           lambda r: 1.01 < r <= 1.05),
        (">1.05      (> 5% worse — needs review)",        lambda r: r > 1.05),
    ]

    total = len(ratios)
    for label, fn in buckets:
        cnt  = sum(1 for r in ratios if fn(r))
        pct  = cnt / total * 100
        bar  = "█" * int(pct / 2)
        c    = RED if ">1.05" in label else (YELLOW if "1.01–1.05" in label else GREEN)
        print(f"  {label:<44}  {c}{bar:<25}{RESET}  {cnt:4d} / {total}  ({pct:5.1f}%)")

    avg_r = sum(ratios) / total
    max_r = max(ratios)
    print(f"\n  Average ratio : {avg_r:.5f}")
    print(f"  Max ratio     : {max_r:.5f}")
    if unreachable_hier:
        print(f"  {RED}⚠ Hierarchical failed to find path for {unreachable_hier} pairs "
              f"that vanilla solved.{RESET}")
    else:
        print(f"  {GREEN}✓ No paths missed by hierarchical that vanilla found.{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid",       type=int, default=80,
                        help="City grid size N (N×N nodes, default 80)")
    parser.add_argument("--block-grid", type=int, default=8,
                        help="Block partition size B (B×B blocks, default 8)")
    parser.add_argument("--queries",    type=int, default=40,
                        help="Queries per bucket in speed/correctness test (default 40)")
    args = parser.parse_args()

    N, B, Q = args.grid, args.block_grid, args.queries

    # ── build synthetic city ──────────────────────────────────────────────────
    print(f"\n{BOLD}Building synthetic {N}×{N} city graph  ({N*N:,} nodes)…{RESET}")
    t0 = time.perf_counter()
    G = nx.read_graphml("data/raw/surat_drive.graphml")
    print(f"  Graph built in {(time.perf_counter()-t0)*1000:.0f} ms  "
          f"({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")

    print(f"{BOLD}Partitioning into {B}×{B} blocks…{RESET}")
    t0 = time.perf_counter()
    blocks_data = build_blocks(G, N, B)
    n_blocks = len(blocks_data["blocks"])
    n_gates  = sum(len(b["gates"]) for b in blocks_data["blocks"].values())
    print(f"  {n_blocks} blocks, {n_gates:,} gate nodes  "
          f"({(time.perf_counter()-t0)*1000:.0f} ms)")

    print(f"{BOLD}Building shortcuts…{RESET}")
    t0 = time.perf_counter()
    shortcuts = build_shortcuts(G, blocks_data)
    n_sc = sum(len(v) for v in shortcuts.values())
    print(f"  {n_sc:,} shortcuts across {len(shortcuts)} blocks  "
          f"({(time.perf_counter()-t0)*1000:.0f} ms)")

    nodes = list(G.nodes())

    # ── run tests ─────────────────────────────────────────────────────────────
    bench_speed_and_correctness(G, blocks_data, shortcuts, nodes, N, Q)
    bench_node_visits(G, blocks_data, shortcuts, nodes, N, Q)
    bench_stress(G, blocks_data, shortcuts, nodes, N, Q_stress=200)
    bench_correctness_distribution(G, blocks_data, shortcuts, nodes, N, Q)

    hdr("SUMMARY")
    print(f"  Graph        : {N}×{N} grid  ({N*N:,} nodes, {G.number_of_edges():,} edges)")
    print(f"  Block grid   : {B}×{B} = {B*B} blocks  ({n_gates:,} gate nodes)")
    print(f"  Shortcuts    : {n_sc:,} pre-computed gate pairs")
    print(f"  Query budget : {Q} per distance bucket\n")


if __name__ == "__main__":
    main()