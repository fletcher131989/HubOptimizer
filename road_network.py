"""
Road-network distances from OpenStreetMap, used instead of straight-line
radii so that hub coverage and milk-round routing respect rivers, estuaries
and other things you can't drive across (e.g. the Thames is only crossable
at bridges/tunnels).

The drivable network is downloaded via osmnx (Overpass API) in small tiles,
one at a time, and cached to disk as compact numpy arrays in
`road_cache/tiles/` (see the Download / cache section for why). All
shortest-path work is done with scipy's compiled Dijkstra on a sparse
matrix, not networkx, so it stays fast and memory-light.

Two cost metrics are supported:
  - "distance": metres along the road network
  - "time":     seconds, from OSM speed limits (imputed per road type where
                untagged) divided by a traffic `speed_factor` (e.g. 0.6 means
                traffic averages 60% of the free-flow speed limit)

Postcode centroids and hub sites are "snapped" to their nearest road
junction; the straight-line hop from the point to that junction is added on
top as an access cost (at ACCESS_SPEED_KMH for the time metric).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import BallTree

ROAD_CACHE_DIR = Path("road_cache")
EARTH_RADIUS_M = 6_371_000.0
ACCESS_SPEED_KMH = 15.0          # point -> nearest junction hop, time metric
MAX_BBOX_SQ_DEG = 6.0            # ~ a large region; bigger = very slow download
_NO_SNAP_HIGHWAYS = {"motorway", "motorway_link"}
MIN_SNAP_COMPONENT_NODES = 100   # see RoadGraph.snap

_GRAPH_MEMO = {}                 # frozenset(tiles) -> RoadGraph (only the latest is kept)


# --------------------------------------------------
# Graph container
# --------------------------------------------------

@dataclass
class RoadGraph:
    node_lat: np.ndarray        # (n,) float64
    node_lon: np.ndarray        # (n,) float64
    edge_u: np.ndarray          # (m,) int32, deduplicated (one edge per u->v)
    edge_v: np.ndarray          # (m,) int32
    length_m: np.ndarray        # (m,) float64
    time_s: np.ndarray          # (m,) float64, free-flow
    snappable: np.ndarray       # (n,) bool, False for motorway-only junctions
    bbox: tuple                 # (south, west, north, east)
    missing_tiles: frozenset = frozenset()   # tiles with no local data (not downloaded)
    _csr_memo: dict = field(default_factory=dict, repr=False)
    _snap_tree: object = field(default=None, repr=False)
    _edge_keys: np.ndarray = field(default=None, repr=False)

    @property
    def n_nodes(self):
        return len(self.node_lat)

    def weights(self, metric, speed_factor=1.0):
        if metric == "distance":
            return self.length_m
        if metric == "time":
            if speed_factor <= 0:
                raise ValueError("Traffic speed factor must be greater than zero.")
            return self.time_s / float(speed_factor)
        raise ValueError(f"Unknown road metric: {metric!r}")

    def csr(self, metric, speed_factor=1.0):
        key = (metric, round(float(speed_factor), 4))
        if key not in self._csr_memo:
            w = np.maximum(self.weights(metric, speed_factor), 1e-3)
            self._csr_memo = {key: csr_matrix(
                (w, (self.edge_u, self.edge_v)), shape=(self.n_nodes, self.n_nodes)
            )}
        return self._csr_memo[key]

    def snap(self, lats, lons):
        """Nearest snappable junction for each point -> (node_idx, offset_m)."""
        if self._snap_tree is None:
            # Only snap onto junctions in a decent-sized strongly connected
            # network: tiny fragments (gated estates, one-way stubs at the area
            # edge) would make a postcode unreachable from every hub. The size
            # floor keeps genuine islands (e.g. Isle of Wight) as their own
            # networks, so island postcodes aren't snapped across the water.
            from scipy.sparse.csgraph import connected_components

            _, comp = connected_components(self.csr("distance"), directed=True, connection="strong")
            comp_size = np.bincount(comp)
            in_main = comp_size[comp] >= MIN_SNAP_COMPONENT_NODES
            snap_idx = np.where(self.snappable & in_main)[0]
            if len(snap_idx) == 0:
                snap_idx = np.where(self.snappable)[0]
            self._snap_idx = snap_idx
            self._snap_tree = BallTree(
                np.radians(np.column_stack([self.node_lat[snap_idx], self.node_lon[snap_idx]])),
                metric="haversine",
            )
        pts = np.radians(np.column_stack([np.asarray(lats, float), np.asarray(lons, float)]))
        dist, idx = self._snap_tree.query(pts, k=1)
        return self._snap_idx[idx[:, 0]], dist[:, 0] * EARTH_RADIUS_M

    @staticmethod
    def access_cost(offset_m, metric):
        offset_m = np.asarray(offset_m, dtype=np.float64)
        if metric == "distance":
            return offset_m
        return offset_m / (ACCESS_SPEED_KMH / 3.6)

    def edge_lengths_along(self, nodes):
        """Total length (m) of the consecutive node path `nodes`."""
        if len(nodes) < 2:
            return 0.0
        if self._edge_keys is None:
            keys = self.edge_u.astype(np.int64) * self.n_nodes + self.edge_v
            order = np.argsort(keys)
            self._edge_keys = keys[order]
            self._edge_len_sorted = self.length_m[order]
        nodes = np.asarray(nodes, dtype=np.int64)
        q = nodes[:-1] * self.n_nodes + nodes[1:]
        pos = np.searchsorted(self._edge_keys, q)
        return float(self._edge_len_sorted[pos].sum())


# --------------------------------------------------
# Download / cache (tiled, memory-bounded)
# --------------------------------------------------
#
# The network is downloaded in fixed TILE_DEG x TILE_DEG tiles on a global
# grid, one tile at a time: each tile is fetched, simplified, reduced to
# compact numpy arrays, saved to road_cache/tiles/, and freed before the next
# one starts. Peak memory is therefore set by a single tile rather than the
# whole area -- an all-at-once osmnx download of Greater London peaked at
# ~5 GB. Tiles are reused by any later area that overlaps them, and an
# interrupted download resumes where it left off.
#
# Stitching: junctions are stored by OSM node id, so tiles join wherever they
# share a node. To guarantee they do, every node on or next to a tile's edge
# (outside the tile, or one road segment away from outside) is kept as a
# junction during simplification, in both neighbouring tiles.

TILE_DEG = 0.1
TILE_DIR = ROAD_CACHE_DIR / "tiles"          # downloaded at runtime (fallback)
SHIPPED_TILE_DIR = Path("road_tiles")        # pre-built offline by build_road_tiles.py
_SHIPPED_COVERED = None                      # set of (i, j) the shipped build covers

# Free-flow speeds (km/h) for road types with no maxspeed tag. Fixed values
# (not osmnx's per-download average) so every tile is consistent.
UK_HWY_SPEEDS_KMH = {
    "motorway": 112, "motorway_link": 64, "trunk": 88, "trunk_link": 56,
    "primary": 64, "primary_link": 48, "secondary": 56, "secondary_link": 40,
    "tertiary": 48, "tertiary_link": 40, "unclassified": 48, "residential": 32,
    "living_street": 16, "service": 16, "road": 32, "busway": 32,
}
FALLBACK_SPEED_KMH = 32

# Public Overpass API mirrors, tried in order. The main server regularly
# returns "at capacity" (502/504/429) under load.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://overpass.kumi.systems/api",
]
OVERPASS_RETRY_WAITS_S = [0, 20, 60]


def _tile_path(ti, tj):
    return TILE_DIR / f"t_{ti}_{tj}.npz"


def _shipped_covered():
    """Tiles covered by the offline build (road_tiles/manifest.json), including
    empty sea tiles next to land. Empty set if nothing has been built."""
    global _SHIPPED_COVERED
    if _SHIPPED_COVERED is None:
        manifest = SHIPPED_TILE_DIR / "manifest.json"
        if manifest.exists():
            import json
            data = json.loads(manifest.read_text())
            _SHIPPED_COVERED = {tuple(t) for t in data.get("covered_tiles", [])}
        else:
            _SHIPPED_COVERED = set()
    return _SHIPPED_COVERED


def _online_download_allowed():
    """HUB_ALLOW_ROAD_DOWNLOAD=1/0 forces it; by default downloads are only
    allowed when no pre-built road_tiles/ are present (i.e. local dev)."""
    flag = os.environ.get("HUB_ALLOW_ROAD_DOWNLOAD", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return not (SHIPPED_TILE_DIR / "manifest.json").exists()


def tile_of(lat, lon):
    """(i, j) tile index arrays for points."""
    return (np.floor(np.asarray(lat) / TILE_DEG).astype(np.int64),
            np.floor(np.asarray(lon) / TILE_DEG).astype(np.int64))


def _resolve_tile(ti, tj):
    """Path of a tile's data, "empty" if it's known to have no roads, or None
    if it isn't available locally (needs downloading)."""
    shipped = SHIPPED_TILE_DIR / f"t_{ti}_{tj}.npz"
    if shipped.exists():
        return shipped
    if (ti, tj) in _shipped_covered():
        return "empty"
    cached = _tile_path(ti, tj)
    return cached if cached.exists() else None


def _tiles_for_bbox(south, west, north, east):
    i0, i1 = int(np.floor(south / TILE_DEG)), int(np.ceil(north / TILE_DEG))
    j0, j1 = int(np.floor(west / TILE_DEG)), int(np.ceil(east / TILE_DEG))
    return [(i, j) for i in range(i0, max(i1, i0 + 1)) for j in range(j0, max(j1, j0 + 1))]


def _is_empty_area_error(exc):
    msg = str(exc).lower()
    return "no data elements" in msg or "no graph nodes" in msg


def _overpass_tile_graph(core, progress=print):
    """Unsimplified drive graph for one tile, retrying across mirrors.
    Returns None if the tile genuinely has no roads (e.g. open sea)."""
    import time
    import osmnx as ox

    ox.settings.use_cache = False       # tiles are cached as compact npz instead
    ox.settings.log_console = False
    ox.settings.requests_timeout = 300

    s, w, n, e = core
    errors = []
    for wait_s in OVERPASS_RETRY_WAITS_S:
        if wait_s:
            progress(f"    Overpass busy; retrying in {wait_s}s...")
            time.sleep(wait_s)
        for url in OVERPASS_ENDPOINTS:
            ox.settings.overpass_url = url
            # osmnx's pre-request /status check gets an unrecognised reply from
            # these servers and then sleeps 60s before *every* request; retries
            # and backoff are handled by this loop instead.
            ox.settings.overpass_rate_limit = False
            try:
                return ox.graph_from_bbox(
                    (w, s, e, n), network_type="drive", simplify=False,
                    retain_all=True, truncate_by_edge=True,
                )
            except Exception as exc:  # empty area, network or server-capacity errors
                if _is_empty_area_error(exc):
                    return None
                errors.append(f"{url}: {str(exc).splitlines()[0][:200]}")
                progress(f"    {errors[-1]}")
    raise RuntimeError(
        "Could not download the road network from any OpenStreetMap Overpass server "
        "(they may be overloaded -- try again in a few minutes; tiles already "
        "downloaded are kept).\n" + "\n".join(errors[-3:])
    )


def _empty_tile_arrays():
    return dict(
        node_id=np.zeros(0, np.int64), node_lat=np.zeros(0), node_lon=np.zeros(0),
        node_nonmotorway=np.zeros(0, bool), edge_u_id=np.zeros(0, np.int64),
        edge_v_id=np.zeros(0, np.int64), length_m=np.zeros(0), time_s=np.zeros(0),
    )


def _download_tile(ti, tj, progress=print):
    import gc
    import osmnx as ox

    core = (ti * TILE_DEG, tj * TILE_DEG, (ti + 1) * TILE_DEG, (tj + 1) * TILE_DEG)
    G = _overpass_tile_graph(core, progress=progress)
    if G is None or G.number_of_nodes() == 0:
        arrays = _empty_tile_arrays()
    else:
        s, w, n, e = core
        outside = {
            nid for nid, d in G.nodes(data=True)
            if not (s <= d["y"] <= n and w <= d["x"] <= e)
        }
        keep = set(outside)
        for u, v in G.edges():
            if u in outside or v in outside:
                keep.add(u)
                keep.add(v)
        for nid in keep:
            G.nodes[nid]["tile_edge"] = True
        G = ox.simplify_graph(G, node_attrs_include=["tile_edge"])
        G = ox.routing.add_edge_speeds(G, hwy_speeds=UK_HWY_SPEEDS_KMH, fallback=FALLBACK_SPEED_KMH)
        G = ox.routing.add_edge_travel_times(G)

        node_ids = np.fromiter(G.nodes, dtype=np.int64, count=G.number_of_nodes())
        pos = {nid: k for k, nid in enumerate(node_ids)}
        node_lat = np.array([G.nodes[nid]["y"] for nid in node_ids], dtype=np.float64)
        node_lon = np.array([G.nodes[nid]["x"] for nid in node_ids], dtype=np.float64)
        m = G.number_of_edges()
        eu = np.empty(m, np.int64)
        ev = np.empty(m, np.int64)
        elen = np.empty(m, np.float64)
        etime = np.empty(m, np.float64)
        nonmotorway = np.zeros(len(node_ids), dtype=bool)
        for k, (u, v, data) in enumerate(G.edges(data=True)):
            eu[k], ev[k] = u, v
            elen[k] = float(data.get("length", 0.0))
            etime[k] = float(data.get("travel_time", 0.0))
            hwy = data.get("highway")
            hwys = set(hwy) if isinstance(hwy, list) else {hwy}
            if not hwys <= _NO_SNAP_HIGHWAYS:
                nonmotorway[pos[u]] = True
                nonmotorway[pos[v]] = True
        arrays = dict(
            node_id=node_ids, node_lat=node_lat, node_lon=node_lon,
            node_nonmotorway=nonmotorway, edge_u_id=eu, edge_v_id=ev,
            length_m=elen, time_s=etime,
        )
    del G
    gc.collect()

    TILE_DIR.mkdir(parents=True, exist_ok=True)
    final = _tile_path(ti, tj)
    tmp = final.with_name(final.stem + "_partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(final)  # never leave a half-written tile behind


def _merge_tiles(tiles, bbox):
    """Stitch cached tiles into one RoadGraph (nodes joined by OSM id)."""
    parts = []
    for t in tiles:
        path = _resolve_tile(*t)
        if path is None or path == "empty":
            continue
        with np.load(path) as z:
            parts.append({k: z[k] for k in z.files})
    if not parts:
        raise ValueError(
            "No road network is available for this area (the pre-built road tiles cover "
            "England, Scotland and Wales). Use 'Straight line' coverage here."
        )

    node_id = np.concatenate([p["node_id"] for p in parts])
    uniq_id, first_idx, inverse = np.unique(node_id, return_index=True, return_inverse=True)
    node_lat = np.concatenate([p["node_lat"] for p in parts]).astype(np.float64)[first_idx]
    node_lon = np.concatenate([p["node_lon"] for p in parts]).astype(np.float64)[first_idx]
    snappable = np.zeros(len(uniq_id), dtype=bool)
    np.logical_or.at(snappable, inverse, np.concatenate([p["node_nonmotorway"] for p in parts]))

    eu = np.searchsorted(uniq_id, np.concatenate([p["edge_u_id"] for p in parts])).astype(np.int32)
    ev = np.searchsorted(uniq_id, np.concatenate([p["edge_v_id"] for p in parts])).astype(np.int32)
    elen = np.concatenate([p["length_m"] for p in parts]).astype(np.float64)
    etime = np.concatenate([p["time_s"] for p in parts]).astype(np.float64)
    del parts, node_id, inverse

    # One edge per (u, v): keep the fastest (this also removes the copies of
    # boundary-crossing edges that both neighbouring tiles contain).
    keys = eu.astype(np.int64) * len(uniq_id) + ev
    order = np.lexsort((etime, keys))
    keys_sorted = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = keys_sorted[1:] != keys_sorted[:-1]
    keep = order[first]

    return RoadGraph(
        node_lat=node_lat, node_lon=node_lon,
        edge_u=eu[keep], edge_v=ev[keep], length_m=elen[keep], time_s=etime[keep],
        snappable=snappable, bbox=bbox,
    )


def load_road_graph(south, west, north, east, buffer_km=3.0, progress=print):
    """
    Road graph covering the given bbox plus `buffer_km` (roads just outside
    an area can still be the quickest way between two points inside it).
    Missing tiles are downloaded one at a time; cached tiles are reused.
    """
    lat_buf = buffer_km / 110.574
    lon_buf = buffer_km / (111.320 * np.cos(np.radians((south + north) / 2.0)))
    s, w, n, e = south - lat_buf, west - lon_buf, north + lat_buf, east + lon_buf

    area = (n - s) * (e - w)
    if area > MAX_BBOX_SQ_DEG:
        raise ValueError(
            f"Area is too large for road-network mode ({area:.1f} sq degrees, limit "
            f"{MAX_BBOX_SQ_DEG}). Draw a smaller area or use straight-line coverage."
        )

    tiles = _tiles_for_bbox(s, w, n, e)
    tile_set = frozenset(tiles)
    for memo_tiles, graph in _GRAPH_MEMO.items():
        if tile_set <= memo_tiles:
            return graph

    missing = [t for t in tiles if _resolve_tile(*t) is None]
    if missing and not _online_download_allowed():
        # Never block a hosted run on the public Overpass API: treat tiles
        # outside the pre-built network as roadless and let the caller report
        # any postcodes that fall in them (RoadGraph.missing_tiles).
        progress(f"{len(missing)} tile(s) are outside the pre-built road network; skipping.")
        _GRAPH_MEMO.clear()
        graph = _merge_tiles(tiles, None)
        graph.missing_tiles = frozenset(missing)
        _GRAPH_MEMO[tile_set] = graph
        return graph
    if missing:
        # Only reached outside the area pre-built by build_road_tiles.py.
        progress(
            f"Downloading OpenStreetMap roads: {len(missing)} of {len(tiles)} tile(s) "
            f"are outside the pre-built road tiles (slow, one-off)..."
        )
    for k, t in enumerate(missing, start=1):
        progress(f"  tile {k}/{len(missing)} {t}")
        _download_tile(*t, progress=progress)

    bbox = (
        min(t[0] for t in tiles) * TILE_DEG, min(t[1] for t in tiles) * TILE_DEG,
        (max(t[0] for t in tiles) + 1) * TILE_DEG, (max(t[1] for t in tiles) + 1) * TILE_DEG,
    )
    _GRAPH_MEMO.clear()  # hold at most one merged graph in memory
    graph = _merge_tiles(tiles, bbox)
    _GRAPH_MEMO[tile_set] = graph
    return graph


def load_road_graph_for_points(lats, lons, buffer_km=3.0, progress=print):
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    return load_road_graph(lats.min(), lons.min(), lats.max(), lons.max(),
                           buffer_km=buffer_km, progress=progress)


# --------------------------------------------------
# Coverage: which demand points a source can reach
# --------------------------------------------------

class RoadCoverage:
    """
    Answers "which demand points (postcodes) can a hub at (lat, lon) reach
    within `limit` along the road network?", in the same shape as a
    BallTree.query_radius result, so the optimisers can use it as a drop-in
    replacement for straight-line radius coverage.

    metric "distance": limit in metres. metric "time": limit in seconds.
    Demand row order must match the dataframe the optimiser is working on;
    use subset() when the optimiser narrows the demand set.
    """

    def __init__(self, graph, demand_lats, demand_lons, limit, metric="time",
                 speed_factor=1.0, batch_size=32):
        self.graph = graph
        self.limit = float(limit)
        self.metric = metric
        self.speed_factor = float(speed_factor)
        # Each Dijkstra in a batch holds a full (n_nodes,) float64 array, so on
        # a big network shrink the batch to keep that under ~64 MB.
        self.batch_size = int(max(1, min(batch_size, 64e6 // (8 * max(graph.n_nodes, 1)))))
        self.csr = graph.csr(metric, speed_factor)
        node, offset = graph.snap(demand_lats, demand_lons)
        self.demand_node = node
        self.demand_access = graph.access_cost(offset, metric)

    def __len__(self):
        return len(self.demand_node)

    def subset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        clone = object.__new__(RoadCoverage)
        clone.__dict__.update(self.__dict__)
        clone.demand_node = self.demand_node[mask]
        clone.demand_access = self.demand_access[mask]
        return clone

    def candidate_sites(self, lats, lons):
        """Snap candidate sites (e.g. a grid) onto unique road junctions, so
        optimised hubs always sit on the road network, never in a river/park."""
        node, _ = self.graph.snap(lats, lons)
        node = np.unique(node)
        return np.column_stack([self.graph.node_lat[node], self.graph.node_lon[node]]).astype(np.float32)

    def reach(self, lats, lons, return_costs=False):
        """
        For each source point, the demand indices reachable within the limit
        (and, optionally, their costs: metres or seconds).
        """
        src_node, src_off = self.graph.snap(lats, lons)
        src_access = self.graph.access_cost(src_off, self.metric)
        out = []
        for b0 in range(0, len(src_node), self.batch_size):
            b_nodes = src_node[b0:b0 + self.batch_size]
            b_access = src_access[b0:b0 + self.batch_size]
            dist = dijkstra(self.csr, directed=True, indices=b_nodes, limit=self.limit)
            if dist.ndim == 1:
                dist = dist[None, :]
            for row, acc in zip(dist, b_access):
                cost = row[self.demand_node] + self.demand_access + acc
                # int32 indices: these lists are held for every candidate site,
                # so halving them matters on memory-limited hosting.
                idx = np.nonzero(cost <= self.limit)[0].astype(np.int32)
                out.append((idx, cost[idx].astype(np.float32)) if return_costs else idx)
            del dist
        return out

    def cost_display(self, cost, radius_unit="miles"):
        """Convert raw costs to display units (minutes, or miles/km)."""
        cost = np.asarray(cost, dtype=np.float64)
        if self.metric == "time":
            return cost / 60.0
        km = cost / 1000.0
        return km / 1.60934 if radius_unit.lower() in ("mile", "miles", "mi") else km

    def cost_column(self, radius_unit="miles"):
        if self.metric == "time":
            return "Drive Time (min)"
        unit = "miles" if radius_unit.lower() in ("mile", "miles", "mi") else "km"
        return f"Road Distance ({unit})"


# --------------------------------------------------
# Point-to-point travel matrix (milk rounds)
# --------------------------------------------------

def road_travel_matrix(graph, lats, lons, speed_factor=1.0):
    """
    Fastest-route drive time (minutes) and that route's length (km) between
    every pair of points, plus each route's path as [(lat, lon), ...] for
    drawing. Unreachable pairs come back as inf -- callers decide what to do.

    Returns (dist_km, minutes, paths) with paths[(i, j)] -> list of (lat, lon).
    """
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    n = len(lats)
    node, offset = graph.snap(lats, lons)
    access_s = graph.access_cost(offset, "time")
    csr = graph.csr("time", speed_factor)

    dist_km = np.full((n, n), np.inf)
    minutes = np.full((n, n), np.inf)
    paths = {}
    for i in range(n):
        # One source at a time: holds two (n_nodes,) arrays rather than two
        # (n_points x n_nodes) matrices.
        t_i, p_i = dijkstra(csr, directed=True, indices=int(node[i]), return_predecessors=True)
        for j in range(n):
            if i == j:
                dist_km[i, j] = 0.0
                minutes[i, j] = 0.0
                continue
            t = t_i[node[j]]
            if not np.isfinite(t):
                continue
            seq = [node[j]]
            while seq[-1] != node[i]:
                p = p_i[seq[-1]]
                if p < 0:
                    break
                seq.append(p)
            seq.reverse()
            minutes[i, j] = (t + access_s[i] + access_s[j]) / 60.0
            dist_km[i, j] = (graph.edge_lengths_along(seq) + offset[i] + offset[j]) / 1000.0
            paths[(i, j)] = (
                [(float(lats[i]), float(lons[i]))]
                + [(float(graph.node_lat[k]), float(graph.node_lon[k])) for k in seq]
                + [(float(lats[j]), float(lons[j]))]
            )
    return dist_km, minutes, paths
