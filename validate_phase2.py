"""
Phase 2 — Smoke Test & Validator
==================================
Run after 02_partitioner.py and 03_shortcut_builder.py to verify
the output files are well-formed and internally consistent.

Usage:
    python validate_phase2.py

Exit code 0 = all checks passed.
Exit code 1 = at least one check failed.
"""

import json
import os
import sys

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
BLOCKS_PATH    = os.path.join(BASE_DIR, "data", "processed", "blocks.json")
SHORTCUTS_PATH = os.path.join(BASE_DIR, "data", "processed", "shortcuts.json")

PASS = "  ✓"
FAIL = "  ✗"
errors: list[str] = []


def check(condition: bool, msg_ok: str, msg_fail: str) -> None:
    if condition:
        print(f"{PASS} {msg_ok}")
    else:
        print(f"{FAIL} {msg_fail}")
        errors.append(msg_fail)


# ── blocks.json ───────────────────────────────────────────────────────────────
print("\n── blocks.json ──────────────────────────────────────")

if not os.path.exists(BLOCKS_PATH):
    print(f"{FAIL} File not found: {BLOCKS_PATH}")
    sys.exit(1)

with open(BLOCKS_PATH) as f:
    bdata = json.load(f)

check("grid_size" in bdata,        "top-level 'grid_size' key present", "missing 'grid_size'")
check("bounding_box" in bdata,     "top-level 'bounding_box' key present", "missing 'bounding_box'")
check("blocks" in bdata,           "top-level 'blocks' key present", "missing 'blocks'")

blocks = bdata.get("blocks", {})
grid_size = bdata.get("grid_size", 0)

check(len(blocks) > 0,             f"{len(blocks)} blocks found", "blocks dict is empty")
check(isinstance(grid_size, int) and grid_size >= 2,
                                   f"grid_size = {grid_size}",
                                   f"invalid grid_size: {grid_size}")

# every block has 'nodes' and 'gates'
malformed = [bid for bid, b in blocks.items()
             if "nodes" not in b or "gates" not in b]
check(len(malformed) == 0,
      "all blocks have 'nodes' and 'gates' keys",
      f"{len(malformed)} blocks missing 'nodes'/'gates': {malformed[:3]}")

# gates are a subset of nodes
gate_not_in_nodes = []
for bid, b in blocks.items():
    node_set = set(b.get("nodes", []))
    for g in b.get("gates", []):
        if g not in node_set:
            gate_not_in_nodes.append((bid, g))
check(len(gate_not_in_nodes) == 0,
      "all gate nodes are members of their own block",
      f"{len(gate_not_in_nodes)} gate nodes not in their block's node list")

total_nodes = sum(len(b["nodes"]) for b in blocks.values())
total_gates = sum(len(b["gates"]) for b in blocks.values())
print(f"{PASS} {total_nodes:,} total node-memberships across all blocks")
print(f"{PASS} {total_gates:,} gate node references across all blocks")

# ── shortcuts.json ────────────────────────────────────────────────────────────
print("\n── shortcuts.json ───────────────────────────────────")

if not os.path.exists(SHORTCUTS_PATH):
    print(f"{FAIL} File not found: {SHORTCUTS_PATH}")
    sys.exit(1)

with open(SHORTCUTS_PATH) as f:
    sdata = json.load(f)

check(isinstance(sdata, dict),     "top-level is a dict", "top-level is not a dict")

# spot-check: every key in shortcuts corresponds to a known block
unknown_blocks = [bid for bid in sdata if bid not in blocks]
check(len(unknown_blocks) == 0,
      "all shortcut block_ids exist in blocks.json",
      f"{len(unknown_blocks)} unknown block_ids in shortcuts: {unknown_blocks[:3]}")

# spot-check costs are positive numbers
bad_costs: list[tuple] = []
for bid, pairs in list(sdata.items())[:20]:   # sample first 20 blocks
    for key, cost in pairs.items():
        if not isinstance(cost, (int, float)) or cost <= 0:
            bad_costs.append((bid, key, cost))
check(len(bad_costs) == 0,
      "sampled shortcut costs are positive numbers",
      f"{len(bad_costs)} non-positive costs found: {bad_costs[:3]}")

# spot-check gate consistency: shortcut keys should be "nodeA_nodeB"
bad_keys: list[str] = []
for bid, pairs in list(sdata.items())[:5]:
    gate_set = set(blocks.get(bid, {}).get("gates", []))
    for key in list(pairs.keys())[:10]:
        parts = key.split("_", 1)
        if len(parts) != 2:
            bad_keys.append(key)
        elif parts[0] not in gate_set or parts[1] not in gate_set:
            bad_keys.append(key)
check(len(bad_keys) == 0,
      "sampled shortcut keys reference valid gate nodes",
      f"{len(bad_keys)} keys with non-gate endpoints: {bad_keys[:3]}")

total_shortcuts = sum(len(v) for v in sdata.values())
covered = len(sdata)
workable = sum(1 for b in blocks.values() if len(b["gates"]) >= 2)
print(f"{PASS} {total_shortcuts:,} shortcuts across {covered} blocks")
print(f"{PASS} Coverage: {covered}/{workable} blocks that have ≥2 gates")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n─────────────────────────────────────────────────────")
if errors:
    print(f"RESULT: {len(errors)} check(s) FAILED:\n")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print("RESULT: All checks passed ✓")
    sys.exit(0)
