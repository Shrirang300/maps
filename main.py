import json
import os
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dijkstra import dijkstra

app = FastAPI(title="Surat Hierarchical Routing API")

CITY_GRAPH: nx.MultiDiGraph = None
BLOCKS_DATA: dict = None
SHORTCUTS_CACHE: dict = None

class PathRequest(BaseModel):
    source_node: str
    target_node: str

class UpdateWeightRequest(BaseModel):
    block_id: str
    node_u: str
    node_v: str
    new_travel_time_seconds: float

@app.on_event("startup")
def load_routing_data():
    global CITY_GRAPH, BLOCKS_DATA, SHORTCUTS_CACHE
    CITY_GRAPH = nx.read_graphml("data/raw/surat_drive.graphml")

    with open("data/processed/blocks.json", "r") as f:
        BLOCKS_DATA = json.load(f)

    with open("data/processed/shortcuts.json", "r") as f:
        SHORTCUTS_CACHE = json.load(f)

@app.post("/find_path")
def find_fastest_path(req: PathRequest):
    node_type = type(list(CITY_GRAPH.nodes)[0])
    src, tgt = node_type(req.source_node), node_type(req.target_node)

    if src not in CITY_GRAPH or tgt not in CITY_GRAPH:
        raise HTTPException(status_code=404, detail="Nodes not found in graph.")

    # FIX: pass blocks_data so dijkstra can apply hierarchical block-level routing
    cost, path = dijkstra(
        CITY_GRAPH, src, tgt,
        weight="travel_time",
        shortcuts=SHORTCUTS_CACHE,
        blocks_data=BLOCKS_DATA,        # <-- new argument
    )

    if not path:
        return {"status": "failed", "message": "No path exists."}

    return {
        "status": "success",
        "total_time_seconds": cost,
        "path_node_count": len(path),
        "path": path,
    }

@app.post("/update_weight")
def update_road_traffic(req: UpdateWeightRequest):
    global CITY_GRAPH, SHORTCUTS_CACHE
    node_type = type(list(CITY_GRAPH.nodes)[0])
    u, v = node_type(req.node_u), node_type(req.node_v)

    if CITY_GRAPH.has_edge(u, v):
        CITY_GRAPH[u][v][0]['travel_time'] = req.new_travel_time_seconds
    else:
        raise HTTPException(status_code=404, detail="Edge not found.")

    if req.block_id not in BLOCKS_DATA["blocks"]:
        raise HTTPException(status_code=404, detail="Block ID not found.")

    block_info = BLOCKS_DATA["blocks"][req.block_id]
    gates = [node_type(g) for g in block_info["gates"]]
    node_set = {node_type(n) for n in block_info["nodes"]}

    subgraph = CITY_GRAPH.subgraph(node_set)
    new_block_shortcuts = {}

    for src_gate in gates:
        if src_gate not in subgraph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(
                subgraph, src_gate, weight="travel_time"
            )
        except nx.NetworkXError:
            continue

        for dst_gate in gates:
            if dst_gate != src_gate and dst_gate in lengths:
                key = f"{src_gate}_{dst_gate}"
                new_block_shortcuts[key] = round(lengths[dst_gate], 3)

    SHORTCUTS_CACHE[req.block_id] = new_block_shortcuts

    return {
        "status": "success",
        "shortcuts_updated": len(new_block_shortcuts),
    }