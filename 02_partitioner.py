"""
Phase 2a — Macro-Block Partitioner
===================================
Reads:  data/raw/surat_drive.graphml
Writes: data/processed/blocks.json

Overlays a grid on Surat's bounding box.
Tags every node with a block_id (row_col).
Identifies gate nodes: any node that has at least one edge
crossing into a different block.

Usage:
    python 02_partitioner.py [--grid-size N]

    --grid-size  Number of rows AND columns in the grid (default: 10)
                 Use 8 for sparser coverage, 12 for finer granularity.
"""

import argparse
import json
import os
import sys
import time

import networkx as nx

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw",       "surat_drive.graphml")
OUTPUT_PATH= os.path.join(BASE_DIR, "data", "processed", "blocks.json")


def load_graph(path: str) -> nx.MultiDiGraph:
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] GraphML file not found: {path}\n"
            "        Run 01_map_downloader.py first to generate it."
        )
    print(f"[1/4] Loading graph from {path} …", flush=True)
    t0 = time.time()
    G = nx.read_graphml(path)
    print(f"      Loaded {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges in {time.time()-t0:.1f}s")
    return G


def compute_bounding_box(G: nx.MultiDiGraph) -> tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon)."""
    lats = [float(d["y"]) for _, d in G.nodes(data=True)]
    lons = [float(d["x"]) for _, d in G.nodes(data=True)]
    return min(lats), max(lats), min(lons), max(lons)


def assign_block_ids(
    G: nx.MultiDiGraph,
    grid_size: int,
    min_lat: float, max_lat: float,
    min_lon: float, max_lon: float,
) -> dict[str, str]:
    """
    Returns {node_id: "row_col"} for every node.
    Clamps to [0, grid_size-1] so boundary nodes don't fall outside.
    """
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    node_block: dict[str, str] = {}
    for node_id, data in G.nodes(data=True):
        lat = float(data["y"])
        lon = float(data["x"])

        row = int((lat - min_lat) / lat_span * grid_size)
        col = int((lon - min_lon) / lon_span * grid_size)

        row = max(0, min(grid_size - 1, row))
        col = max(0, min(grid_size - 1, col))

        node_block[str(node_id)] = f"{row}_{col}"

    return node_block


def build_blocks(
    G: nx.MultiDiGraph,
    node_block: dict[str, str],
) -> dict[str, dict]:
    """
    Groups nodes by block and identifies gate nodes.

    A node is a gate if any of its edges (in or out) leads to a node
    in a different block.
    """
    # group nodes
    blocks: dict[str, dict] = {}
    for node_id, block_id in node_block.items():
        if block_id not in blocks:
            blocks[block_id] = {"nodes": [], "gates": []}
        blocks[block_id]["nodes"].append(node_id)

    # identify gates
    gate_set: dict[str, set] = {bid: set() for bid in blocks}

    for u, v in G.edges():
        u_str, v_str = str(u), str(v)
        bu = node_block.get(u_str)
        bv = node_block.get(v_str)
        if bu and bv and bu != bv:
            gate_set[bu].add(u_str)
            gate_set[bv].add(v_str)

    for block_id, gates in gate_set.items():
        blocks[block_id]["gates"] = sorted(gates)

    return blocks


def print_stats(blocks: dict, grid_size: int) -> None:
    total_gates = sum(len(b["gates"]) for b in blocks.values())
    sizes = [len(b["nodes"]) for b in blocks.values()]
    non_empty = len(sizes)
    print(f"      Grid: {grid_size}×{grid_size} = {grid_size**2} cells, "
          f"{non_empty} non-empty blocks")
    print(f"      Nodes per block — min: {min(sizes)}, "
          f"avg: {sum(sizes)//non_empty}, max: {max(sizes)}")
    print(f"      Total gate nodes: {total_gates:,}")


def main():
    parser = argparse.ArgumentParser(description="Macro-block partitioner for Surat graph")
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Grid dimension N for N×N partition (default: 10)")
    args = parser.parse_args()

    grid_size = args.grid_size
    if not (2 <= grid_size <= 30):
        sys.exit("[ERROR] --grid-size must be between 2 and 30")

    # ── 1. load ────────────────────────────────────────────────────────────────
    G = load_graph(INPUT_PATH)

    # ── 2. bounding box ────────────────────────────────────────────────────────
    print("[2/4] Computing bounding box …", flush=True)
    min_lat, max_lat, min_lon, max_lon = compute_bounding_box(G)
    print(f"      lat [{min_lat:.5f}, {max_lat:.5f}]  "
          f"lon [{min_lon:.5f}, {max_lon:.5f}]")

    # ── 3. assign block IDs ────────────────────────────────────────────────────
    print(f"[3/4] Assigning nodes to {grid_size}×{grid_size} grid …", flush=True)
    node_block = assign_block_ids(G, grid_size, min_lat, max_lat, min_lon, max_lon)

    # ── 4. build blocks & find gates ───────────────────────────────────────────
    print("[4/4] Building blocks and identifying gate nodes …", flush=True)
    blocks = build_blocks(G, node_block)
    print_stats(blocks, grid_size)

    # ── write output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {
        "grid_size": grid_size,
        "bounding_box": {
            "min_lat": min_lat, "max_lat": max_lat,
            "min_lon": min_lon, "max_lon": max_lon,
        },
        "blocks": blocks,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n✓ Saved blocks.json  ({size_kb:.0f} KB)  →  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
