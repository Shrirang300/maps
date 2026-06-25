import heapq
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set


def dijkstra(
    graph,
    source: Any,
    target: Any,
    weight: str = "travel_time",
    shortcuts: Optional[Dict] = None,
    blocks_data: Optional[Dict] = None,
) -> Tuple[float, List[Any]]:
    """
    Hierarchical Dijkstra — three-zone search.

    Zones
    -----
    SOURCE BLOCK  (LOCAL)
        Expand all real road edges.
        Gate nodes on the boundary also emit intra-block shortcuts so
        the search can enter HIGHWAY mode.

    HIGHWAY  (all blocks except source and target)
        Two kinds of relaxation:
          a) Intra-block shortcuts  — jump gate-to-gate within the same
             block in O(1), bypassing its interior road nodes.
          b) Cross-block real edges — but ONLY between gate nodes.
             This is what lets the search hop from block to block;
             pure intra-block shortcuts alone cannot do this because
             they never cross a block boundary.

    TARGET BLOCK  (LOCAL)
        Expand all real road edges so every interior node is reachable,
        not just the gate nodes.

    Why cross-block real edges are restricted to gates
    --------------------------------------------------
    A gate node is, by definition, one that has at least one edge
    crossing into a neighbouring block.  Allowing only gate→gate
    real edges in HIGHWAY mode keeps the search space small (we
    still skip the thousands of interior nodes) while guaranteeing
    connectivity between all blocks in the graph.
    """

    # ── 0. node-type coercion ─────────────────────────────────────────────────
    node_type = type(list(graph.nodes)[0]) if graph.nodes else str
    src, tgt = node_type(source), node_type(target)

    # ── 1. O(1) node→block index ──────────────────────────────────────────────
    node_to_block: Dict[Any, str] = {}
    block_gates:   Dict[str, Set[Any]] = {}
    all_gate_nodes: Set[Any] = set()

    if blocks_data:
        for block_id, binfo in blocks_data.get("blocks", {}).items():
            gates = {node_type(g) for g in binfo.get("gates", [])}
            block_gates[block_id] = gates
            all_gate_nodes |= gates
            for n in binfo.get("nodes", []):
                node_to_block[node_type(n)] = block_id

    src_block = node_to_block.get(src) if blocks_data else None
    tgt_block = node_to_block.get(tgt) if blocks_data else None

    # ── 2. intra-block shortcut adjacency ─────────────────────────────────────
    # gate_src → [(gate_dst, cost, block_id)]
    fast_shortcuts: Dict[Any, List[Tuple[Any, float, str]]] = {}

    if shortcuts and blocks_data:
        for block_id, pairs in shortcuts.items():
            if not isinstance(pairs, dict):
                continue
            gates = block_gates.get(block_id, set())
            for gate_pair, cost in pairs.items():
                parts = gate_pair.split("_", 1)
                if len(parts) != 2:
                    continue
                su, sv = node_type(parts[0]), node_type(parts[1])
                if su in gates and sv in gates:
                    fast_shortcuts.setdefault(su, []).append(
                        (sv, float(cost), block_id)
                    )

    # ── 3. Dijkstra ───────────────────────────────────────────────────────────
    dist: Dict[Any, float] = {src: 0.0}
    prev: Dict[Any, Optional[Any]] = {src: None}
    heap: List[Tuple[float, Any]] = [(0.0, src)]
    visited: Set[Any] = set()

    while heap:
        cost, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        if u == tgt:
            return cost, _reconstruct_path(prev, src, tgt, graph, weight)

        u_block = node_to_block.get(u) if blocks_data else None
        in_local_zone = (u_block in (src_block, tgt_block)) or (u_block is None)

        if in_local_zone:
            # ── LOCAL: full road-graph expansion ──────────────────────────────
            for _, v, data in graph.out_edges(u, data=True):
                if v in visited:
                    continue
                nc = cost + _get_edge_cost(data, weight)
                if nc < dist.get(v, float("inf")):
                    dist[v] = nc
                    prev[v] = u
                    heapq.heappush(heap, (nc, v))

            # Source-block gates also push shortcuts into highway
            if u_block == src_block and u in fast_shortcuts:
                for sv, sc, blk in fast_shortcuts[u]:
                    if sv in visited or blk == tgt_block:
                        continue
                    nc = cost + sc
                    if nc < dist.get(sv, float("inf")):
                        dist[sv] = nc
                        prev[sv] = u
                        heapq.heappush(heap, (nc, sv))

        else:
            # ── HIGHWAY: shortcuts + gate-only cross-block real edges ──────────

            # a) Intra-block shortcuts (skip interior of this block)
            if u in fast_shortcuts:
                for sv, sc, blk in fast_shortcuts[u]:
                    if sv in visited:
                        continue
                    nc = cost + sc
                    if nc < dist.get(sv, float("inf")):
                        dist[sv] = nc
                        prev[sv] = u
                        heapq.heappush(heap, (nc, sv))

            # b) Cross-block real edges, but only to gate nodes.
            #    This is the inter-block hop that connects the shortcut
            #    islands to each other and to the target block.
            for _, v, data in graph.out_edges(u, data=True):
                if v in visited:
                    continue
                v_block = node_to_block.get(v)
                # Allow: v is a gate of any block, OR v is in the target block
                if v not in all_gate_nodes and v_block != tgt_block:
                    continue
                nc = cost + _get_edge_cost(data, weight)
                if nc < dist.get(v, float("inf")):
                    dist[v] = nc
                    prev[v] = u
                    heapq.heappush(heap, (nc, v))

    return float("inf"), []


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_edge_cost(edge_data: dict, weight: str) -> float:
    raw = edge_data.get(weight)
    if raw is None:
        return float(edge_data.get("length", 1.0))
    if isinstance(raw, list):
        valid = [v for v in raw if v is not None]
        return float(sum(valid) / len(valid)) if valid else 1.0
    return float(raw)


def _reconstruct_path(
    prev: Dict[Any, Optional[Any]],
    source: Any,
    target: Any,
    graph: nx.MultiDiGraph,
    weight: str,
) -> List[Any]:
    path = []
    curr = target

    while curr is not None:
        path.append(curr)
        p = prev.get(curr)
        if p is not None and not graph.has_edge(p, curr):
            try:
                sub_path = nx.shortest_path(graph, p, curr, weight=weight)
                for intermediate in reversed(sub_path[1:-1]):
                    path.append(intermediate)
            except nx.NetworkXNoPath:
                pass
        curr = p

    path.reverse()
    return path if path and path[0] == source else []