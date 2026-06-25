"""
offline_pipeline/01_map_downloader.py
--------------------------------------
Phase 1: Data Acquisition

Downloads the road network for a target city using OSMnx and saves it
as a .graphml file for all subsequent pipeline stages.

Run:
    python offline_pipeline/01_map_downloader.py
"""

import osmnx as ox
import os
import time

# ── Configuration ────────────────────────────────────────────────────────────
CITY_NAME   = "Surat, Gujarat, India"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "surat_drive.graphml")

# Network type: 'drive' keeps only drivable roads (no footpaths / railways)
NETWORK_TYPE = "drive"
# ─────────────────────────────────────────────────────────────────────────────


def download_city_graph(city: str, network_type: str, output_path: str) -> None:
    """
    Fetch the road network for `city` and persist it to `output_path`.

    OSMnx returns a MultiDiGraph where:
      - Nodes  = intersections / dead-ends  (attributes: osmid, x, y, street_count)
      - Edges  = road segments              (attributes: length, maxspeed, highway, ...)
    """

    if os.path.exists(output_path):
        print(f"[SKIP] Graph already exists at: {output_path}")
        print("       Delete the file and re-run if you need a fresh download.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[INFO] Downloading '{city}' road network …")
    print(f"       Network type : {network_type}")
    print(f"       This may take 30–120 seconds on first run.\n")

    t0 = time.time()

    # retain_all=False drops isolated sub-graphs (keeps the largest connected component)
    G = ox.graph_from_place(
        city,
        network_type=network_type,
        retain_all=False,
        simplify=True,          # merge series edges into single edges
    )

    elapsed = time.time() - t0

    # ── Stats ────────────────────────────────────────────────────────────────
    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    print(f"[OK]  Download complete in {elapsed:.1f}s")
    print(f"      Nodes (intersections) : {n_nodes:,}")
    print(f"      Edges (road segments) : {n_edges:,}")

    # ── Enrich edges with a `travel_time` attribute ──────────────────────────
    # OSMnx can impute missing speed limits and derive travel time in seconds.
    # This becomes the base weight for Dijkstra (Phase 1) and temporal
    # multiplier logic (Phase 3).
    print("\n[INFO] Adding speed & travel-time attributes to edges …")
    G = ox.add_edge_speeds(G)          # fills missing maxspeed with heuristics
    G = ox.add_edge_travel_times(G)    # travel_time = length / speed

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Saving graph to {output_path} …")
    ox.save_graphml(G, filepath=output_path)
    size_mb = os.path.getsize(output_path) / 1_048_576
    print(f"[OK]  Saved ({size_mb:.1f} MB)\n")

    _print_summary(G)


def _print_summary(G) -> None:
    """Print a human-readable summary of key graph properties."""
    stats = ox.basic_stats(G)

    print("── Graph Summary ──────────────────────────────────────────")
    print(f"  Total road length : {stats['edge_length_total']/1000:.1f} km")
    print(f"  Avg node degree   : {stats['k_avg']:.2f}")
    print(f"  Intersection count: {stats.get('intersection_count', 'n/a')}")
    print("───────────────────────────────────────────────────────────\n")
    print("Next step → run:  python offline_pipeline/02_partitioner.py")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_city_graph(
        city=CITY_NAME,
        network_type=NETWORK_TYPE,
        output_path=OUTPUT_FILE,
    )
