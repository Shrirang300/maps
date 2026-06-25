"""
Phase 2b — Shortcut Builder
=============================
Reads:  data/raw/surat_drive.graphml
        data/processed/blocks.json
Writes: data/processed/shortcuts.json

For every block, runs Dijkstra from each gate node to every other gate
node WITHIN that block.  Stores the result as a flat lookup table so
Phase 3/4 can answer "how long to cross block B from gate G1 to G2?"
with a single dictionary access — O(1) instead of O(E log V).

Performance notes
-----------------
* Uses networkx.single_source_dijkstra with cutoff to avoid leaking
  into neighbouring blocks.
* Parallel processing via multiprocessing.Pool — one process per block.
* Progress bar printed every 5 % of blocks.
* Estimated 5–15 min for a 70 k-node Surat graph on a laptop.

Usage:
    python 03_shortcut_builder.py [--workers N] [--weight ATTR]

    --workers  Number of parallel processes (default: CPU count − 1, min 1)
    --weight   Edge attribute to use as cost (default: travel_time)
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from typing import Any
import shutil

import networkx as nx

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
GRAPHML_PATH    = os.path.join(BASE_DIR, "data", "raw",       "surat_drive.graphml")
BLOCKS_PATH     = os.path.join(BASE_DIR, "data", "processed", "blocks.json")
SHORTCUTS_PATH  = os.path.join(BASE_DIR, "data", "processed", "shortcuts.json")

# Global graph shared across worker processes (read-only after fork)
_G: nx.MultiDiGraph | None = None
_WEIGHT: str = "travel_time"


# ── worker initialiser (runs once per process) ─────────────────────────────────
def _init_worker(graphml_path: str, weight: str) -> None:
    global _G, _WEIGHT

    _G = nx.read_graphml(graphml_path)
    _WEIGHT = weight

    for _, _, data in _G.edges(data=True):
        if weight in data:
            try:
                data[weight] = float(data[weight])
            except (TypeError, ValueError):
                data[weight] = float("inf")


# ── per-block worker ──────────────────────────────────────────────────────────
def _process_block(args: tuple[str, dict]) -> tuple[str, dict[str, float]]:
    """
    Returns (block_id, {gate_src_gate_dst: cost_seconds, ...}).

    Uses single_source_dijkstra from each gate with a node-whitelist
    cutoff: only travels through nodes belonging to this block.
    """
    block_id, block_data = args
    gates: list[str] = block_data["gates"]
    node_set: set[str] = set(block_data["nodes"])

    if len(gates) < 2:
        return block_id, {}

    # Build a subgraph restricted to this block's nodes so Dijkstra
    # cannot leak into neighbours.
    subgraph = _G.subgraph(node_set)

    shortcuts: dict[str, float] = {}

    for src_gate in gates:
        if src_gate not in subgraph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(
                subgraph, src_gate, weight=_WEIGHT
            )
        except nx.NetworkXError:
            continue

        for dst_gate in gates:
            if dst_gate == src_gate:
                continue
            cost = lengths.get(dst_gate)
            if cost is not None:
                key = f"{src_gate}_{dst_gate}"
                shortcuts[key] = round(cost, 3)

    return block_id, shortcuts


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_blocks(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] blocks.json not found: {path}\n"
            "        Run 02_partitioner.py first."
        )
    with open(path) as f:
        return json.load(f)


def _progress(done: int, total: int, t0: float) -> None:
    pct = done / total * 100
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    bar_len = 30
    filled = int(bar_len * done / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r  [{bar}] {pct:5.1f}%  {done}/{total} blocks  "
        f"ETA {eta/60:.1f} min",
        end="", flush=True
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pre-compute intra-block shortcuts")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help="Parallel worker processes (default: cpu_count − 1)")
    parser.add_argument("--weight", type=str, default="travel_time",
                        help="Edge weight attribute (default: travel_time)")
    args = parser.parse_args()

    if not os.path.exists(GRAPHML_PATH):
        sys.exit(f"[ERROR] GraphML not found: {GRAPHML_PATH}\n"
                 "        Run 01_map_downloader.py first.")

    # ── load blocks ────────────────────────────────────────────────────────────
    print("[1/3] Loading blocks.json …", flush=True)
    data = _load_blocks(BLOCKS_PATH)
    blocks: dict[str, dict] = data["blocks"]

    # Only process blocks that have ≥2 gates (no shortcuts possible otherwise)
    workload = [(bid, bdata) for bid, bdata in blocks.items()
                if len(bdata.get("gates", [])) >= 2]

    total_gates = sum(len(b["gates"]) for b in blocks.values())
    total_pairs = sum(
        len(b["gates"]) * (len(b["gates"]) - 1)
        for b in blocks.values() if len(b["gates"]) >= 2
    )
    print(f"      {len(blocks)} blocks, {len(workload)} with ≥2 gates")
    print(f"      {total_gates:,} gate nodes  →  up to {total_pairs:,} shortcut pairs")

    # ── run Dijkstra ───────────────────────────────────────────────────────────
    print(f"[2/3] Running intra-block Dijkstra with {args.workers} worker(s) …")
    print(f"      Weight attribute: '{args.weight}'")

    t0 = time.time()
    temp_dir = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "_shortcut_tmp"
    )

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir)

    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(GRAPHML_PATH, args.weight),
    ) as pool:
        for i, (block_id, shortcuts) in enumerate(
            pool.imap_unordered(_process_block, workload, chunksize=4), start=1
        ):
            if shortcuts:

                tmp_path = os.path.join(
                    temp_dir,
                    f"{block_id}.json"
                )

                with open(tmp_path, "w") as f:
                    json.dump(
                        shortcuts,
                        f,
                        separators=(",", ":")
                    )

            if i % max(1, len(workload) // 40) == 0 or i == len(workload):
                _progress(i, len(workload), t0)

    print()  # newline after progress bar
    elapsed = time.time() - t0
    
    print("[3/3] Merging temporary files ...")

    total_shortcuts = 0

    with open(SHORTCUTS_PATH, "w") as out:

        out.write("{")

        first = True

        for filename in os.listdir(temp_dir):

            if not filename.endswith(".json"):
                continue

            block_id = filename[:-5]

            file_path = os.path.join(
                temp_dir,
                filename
            )

            with open(file_path, "r") as f:
                content = f.read()

            try:
                total_shortcuts += len(json.loads(content))
            except Exception:
                pass

            if not first:
                out.write(",")

            json.dump(block_id, out)
            out.write(":")
            out.write(content)

            first = False

        out.write("}")

    size_kb = os.path.getsize(SHORTCUTS_PATH) / 1024
    print(f"\n✓ Saved shortcuts.json  ({size_kb:.0f} KB)  →  {SHORTCUTS_PATH}")
    coverage = len(
        [f for f in os.listdir(temp_dir) if f.endswith(".json")]
    )

    print(
        f"  Coverage: {coverage}/{len(workload)} blocks have shortcuts"
    )
    shutil.rmtree(temp_dir)
    print("  Temporary files removed")


if __name__ == "__main__":
    # Required for multiprocessing on Windows / macOS spawn mode
    multiprocessing.freeze_support()
    main()
    # size_kb = os.path.getsize(SHORTCUTS_PATH) / 1024
