"""
Build the shipped road-network tiles (road_tiles/) from a Geofabrik
OpenStreetMap extract, offline, so the hosted app never has to download
roads from the (often overloaded) public Overpass API at runtime.

Usage (run locally, not on the host; needs `pip install osmium`, which is
deliberately not in requirements.txt since the hosted app never builds tiles):

    curl -L -o osm_source/great-britain-latest.osm.pbf \
        https://download.geofabrik.de/europe/great-britain-latest.osm.pbf
    python build_road_tiles.py osm_source/great-britain-latest.osm.pbf

Rebuild occasionally (e.g. quarterly) to pick up road changes.

Pipeline:
  1. Stream the extract once with pyosmium, keeping drivable ways only (same
     rules as osmnx's "drive" network) with node locations resolved from a
     disk-backed index (keeps RAM low).
  2. Simplify the WHOLE network in one go: junctions are nodes shared by 2+
     ways or at a way's ends; everything in between is collapsed into one
     edge with summed length/time. Doing this globally (before tiling) is
     what makes tiles stitch perfectly -- every tile agrees on the junctions.
  3. Cut into TILE_DEG tiles: each edge is stored in the tile(s) of its two
     end junctions; the loader joins tiles on OSM node id and de-duplicates.
  4. Write road_tiles/t_{i}_{j}.npz plus road_tiles/manifest.json, which
     lists every tile the build covers (including empty sea tiles next to
     land) so the app knows not to try downloading them.
"""

import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

from road_network import (
    SHIPPED_TILE_DIR, TILE_DEG, UK_HWY_SPEEDS_KMH, FALLBACK_SPEED_KMH, _NO_SNAP_HIGHWAYS,
)

# osmnx "drive" network filter, as exact-value sets.
EXCLUDED_HIGHWAYS = {
    "abandoned", "bridleway", "bus_guideway", "construction", "corridor", "cycleway",
    "elevator", "escalator", "footway", "no", "path", "pedestrian", "planned",
    "platform", "proposed", "raceway", "razed", "service", "steps", "track",
    "bus_stop", "rest_area", "services", "emergency_bay", "via_ferrata", "busway",
}
EXCLUDED_SERVICE = {"alley", "driveway", "emergency_access", "parking", "parking_aisle", "private"}

MPH = 1.609344
NSL_KMH = {
    "gb:nsl_single": 60 * MPH, "nsl_single": 60 * MPH, "national": 60 * MPH,
    "gb:nsl_dual": 70 * MPH, "nsl_dual": 70 * MPH, "gb:motorway": 70 * MPH,
    "uk:nsl_single": 60 * MPH, "uk:nsl_dual": 70 * MPH, "uk:motorway": 70 * MPH,
}
EARTH_RADIUS_M = 6_371_000.0


def is_drivable(tags):
    hwy = tags.get("highway")
    if not hwy or hwy in EXCLUDED_HIGHWAYS:
        return False
    if tags.get("area") == "yes":
        return False
    if "private" in (tags.get("access") or ""):
        return False
    if "no" in (tags.get("motor_vehicle") or "").split(";") or "no" in (tags.get("motorcar") or "").split(";"):
        return False
    if tags.get("service") in EXCLUDED_SERVICE:
        return False
    return True


def parse_speed_kmh(maxspeed, highway):
    default = UK_HWY_SPEEDS_KMH.get(highway, FALLBACK_SPEED_KMH)
    if not maxspeed:
        return default
    first = maxspeed.split(";")[0].strip().lower()
    if first in NSL_KMH:
        return 70 * MPH if highway in ("motorway", "motorway_link") and "single" in first else NSL_KMH[first]
    if first == "walk":
        return 6.0
    try:
        if first.endswith("mph"):
            return float(first[:-3].strip()) * MPH
        if first.endswith("km/h") or first.endswith("kmh"):
            return float(first.rstrip("km/h").strip())
        return float(first)  # bare number = km/h in OSM
    except ValueError:
        return default


def direction(tags):
    """(forward_allowed, backward_allowed) along the way's node order."""
    oneway = (tags.get("oneway") or "").lower()
    if oneway in ("yes", "true", "1"):
        return True, False
    if oneway in ("-1", "reverse"):
        return False, True
    if oneway == "no":
        return True, True
    if tags.get("junction") in ("roundabout", "circular"):
        return True, False
    return True, True


def read_ways(pbf_path):
    import osmium

    idx_path = Path(tempfile.gettempdir()) / f"hub_optimizer_nodeidx_{int(time.time())}.bin"
    fp = (
        osmium.FileProcessor(str(pbf_path))
        .with_locations(storage=f"sparse_file_array,{idx_path}")
        .with_filter(osmium.filter.EntityFilter(osmium.osm.NODE | osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("highway"))
    )

    ref_chunks, lat_chunks, lon_chunks, len_chunks = [], [], [], []
    speeds, fwds, bwds, motorway = [], [], [], []
    n_ways = 0
    t0 = time.time()
    try:
        for obj in fp:
            if not obj.is_way():
                continue
            tags = obj.tags
            if not is_drivable(tags):
                continue
            refs, lats, lons = [], [], []
            for nd in obj.nodes:
                loc = nd.location
                if loc.valid():
                    refs.append(nd.ref)
                    lats.append(loc.lat)
                    lons.append(loc.lon)
            if len(refs) < 2:
                continue
            ref_chunks.append(np.array(refs, dtype=np.int64))
            lat_chunks.append(np.array(lats, dtype=np.float64))
            lon_chunks.append(np.array(lons, dtype=np.float64))
            len_chunks.append(len(refs))
            hwy = tags.get("highway")
            speeds.append(min(max(parse_speed_kmh(tags.get("maxspeed"), hwy), 5.0), 130.0))
            f, b = direction(tags)
            fwds.append(f)
            bwds.append(b)
            motorway.append(hwy in _NO_SNAP_HIGHWAYS)
            n_ways += 1
            if n_ways % 250_000 == 0:
                print(f"  {n_ways:,} drivable ways read ({time.time() - t0:.0f}s)", flush=True)
    finally:
        try:
            idx_path.unlink()
        except OSError:
            pass

    print(f"Read {n_ways:,} drivable ways in {time.time() - t0:.0f}s", flush=True)
    return (
        np.concatenate(ref_chunks), np.concatenate(lat_chunks), np.concatenate(lon_chunks),
        np.array(len_chunks, dtype=np.int64), np.array(speeds), np.array(fwds),
        np.array(bwds), np.array(motorway),
    )


def build_graph(ref, lat, lon, way_len, speed_kmh, fwd, bwd, is_motorway):
    """Globally simplified directed graph: junction nodes + edges between them."""
    n_pos = len(ref)
    way_start = np.concatenate([[0], np.cumsum(way_len)[:-1]])
    way_end = way_start + way_len - 1
    way_of_pos = np.repeat(np.arange(len(way_len)), way_len)

    _, inverse, counts = np.unique(ref, return_inverse=True, return_counts=True)
    is_junction = counts[inverse] > 1
    is_junction[way_start] = True
    is_junction[way_end] = True

    # Segment i joins position i to i+1 (zero at the last position of each way).
    seg_len = np.zeros(n_pos)
    phi1, phi2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dphi = phi2 - phi1
    dlmb = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    seg_len[:-1] = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
    seg_len[way_end] = 0.0
    seg_time = seg_len / (speed_kmh[way_of_pos] / 3.6)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    cum_time = np.concatenate([[0.0], np.cumsum(seg_time)])

    jpos = np.nonzero(is_junction)[0]
    same_way = way_of_pos[jpos[:-1]] == way_of_pos[jpos[1:]]
    pa, pb = jpos[:-1][same_way], jpos[1:][same_way]
    e_len = cum_len[pb] - cum_len[pa]
    e_time = cum_time[pb] - cum_time[pa]
    e_way = way_of_pos[pa]
    u_ref, v_ref = ref[pa], ref[pb]
    keep = u_ref != v_ref
    pa, u_ref, v_ref, e_len, e_time, e_way = pa[keep], u_ref[keep], v_ref[keep], e_len[keep], e_time[keep], e_way[keep]

    f, b = fwd[e_way], bwd[e_way]
    eu = np.concatenate([u_ref[f], v_ref[b]])
    ev = np.concatenate([v_ref[f], u_ref[b]])
    elen = np.concatenate([e_len[f], e_len[b]])
    etime = np.concatenate([e_time[f], e_time[b]])
    emotor = np.concatenate([is_motorway[e_way][f], is_motorway[e_way][b]])

    # De-duplicate (u, v), keeping the fastest.
    keys = np.stack([eu, ev], axis=1)
    order = np.lexsort((etime, ev, eu))
    eu, ev, elen, etime, emotor = eu[order], ev[order], elen[order], etime[order], emotor[order]
    first = np.ones(len(eu), dtype=bool)
    first[1:] = (eu[1:] != eu[:-1]) | (ev[1:] != ev[:-1])
    eu, ev, elen, etime, emotor = eu[first], ev[first], elen[first], etime[first], emotor[first]
    del keys

    node_ref, first_pos = np.unique(ref[jpos], return_index=True)
    node_lat = lat[jpos][first_pos]
    node_lon = lon[jpos][first_pos]
    nonmotor = np.zeros(len(node_ref), dtype=bool)
    nonmotor[np.searchsorted(node_ref, eu[~emotor])] = True
    nonmotor[np.searchsorted(node_ref, ev[~emotor])] = True
    print(f"Simplified graph: {len(node_ref):,} junctions, {len(eu):,} directed edges", flush=True)
    return node_ref, node_lat, node_lon, nonmotor, eu, ev, elen, etime


def write_tiles(out_dir, node_ref, node_lat, node_lon, nonmotor, eu, ev, elen, etime, source_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("t_*.npz"):
        old.unlink()

    ti = np.floor(node_lat / TILE_DEG).astype(np.int64)
    tj = np.floor(node_lon / TILE_DEG).astype(np.int64)
    node_key = (ti + 10_000) * 100_000 + (tj + 50_000)

    ui = np.searchsorted(node_ref, eu)
    vi = np.searchsorted(node_ref, ev)
    ku, kv = node_key[ui], node_key[vi]
    edge_idx = np.concatenate([np.arange(len(eu)), np.nonzero(kv != ku)[0]])
    edge_key = np.concatenate([ku, kv[kv != ku]])
    order = np.argsort(edge_key, kind="stable")
    edge_idx, edge_key = edge_idx[order], edge_key[order]
    bounds = np.nonzero(np.diff(edge_key))[0] + 1
    groups = np.split(edge_idx, bounds)
    keys = edge_key[np.concatenate([[0], bounds])] if len(edge_key) else []

    total_bytes = 0
    tiles = []
    for key, eidx in zip(keys, groups):
        i, j = int(key // 100_000 - 10_000), int(key % 100_000 - 50_000)
        nidx = np.unique(np.concatenate([ui[eidx], vi[eidx]]))
        path = out_dir / f"t_{i}_{j}.npz"
        np.savez_compressed(
            path,
            node_id=node_ref[nidx], node_lat=node_lat[nidx].astype(np.float32),
            node_lon=node_lon[nidx].astype(np.float32), node_nonmotorway=nonmotor[nidx],
            edge_u_id=eu[eidx], edge_v_id=ev[eidx],
            length_m=elen[eidx].astype(np.float32), time_s=etime[eidx].astype(np.float32),
        )
        total_bytes += path.stat().st_size
        tiles.append((i, j))

    write_manifest(out_dir, tiles, source_name)
    print(f"Wrote {len(tiles):,} tiles ({total_bytes / 1e6:.0f} MB) + manifest to {out_dir}", flush=True)


def write_manifest(out_dir, tiles, source_name, sea_margin=2):
    """Covered = tiles with roads plus sea tiles up to `sea_margin` tiles away,
    so the app treats coastal sea as "known empty" rather than missing.
    (2 tiles ~ 11-22 km; keeps Northern Ireland, ~20+ km from Scotland, out.)"""
    covered = set()
    for i, j in tiles:
        for di in range(-sea_margin, sea_margin + 1):
            for dj in range(-sea_margin, sea_margin + 1):
                covered.add((i + di, j + dj))
    manifest = {
        "source": source_name,
        "built": date.today().isoformat(),
        "tile_deg": TILE_DEG,
        "tiles_with_roads": len(tiles),
        "covered_tiles": sorted([list(t) for t in covered]),
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))


def main(pbf_path, out_dir=SHIPPED_TILE_DIR):
    t0 = time.time()
    pbf_path = Path(pbf_path)
    print(f"Reading {pbf_path} ...", flush=True)
    graph_inputs = read_ways(pbf_path)
    graph = build_graph(*graph_inputs)
    del graph_inputs
    write_tiles(Path(out_dir), *graph, source_name=pbf_path.name)
    print(f"Done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python build_road_tiles.py <extract.osm.pbf> [out_dir]")
    main(*sys.argv[1:])
