from functools import lru_cache
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree
import folium
from folium.plugins import HeatMap

DATA_FOLDER = Path("postcode_data")
EARTH_RADIUS_KM = 6371.0
KM_PER_MILE = 1.60934
_PARQUET_COLS = ["postcode", "lat", "lon", "population", "households", "area_type"]
HUB_COLORS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "cadetblue", "darkgreen", "darkpurple",
    "lightred", "beige", "darkblue", "lightblue", "lightgreen",
    "gray", "black",
]


# --------------------------------------------------
# Distance calculations
# --------------------------------------------------

@lru_cache(maxsize=200_000)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def haversine_array(lat1, lon1, lats2, lons2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lats2 = np.radians(lats2)
    lons2 = np.radians(lons2)

    dlat = lats2 - lat1
    dlon = lons2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def convert_to_km(distance, unit):
    unit = unit.lower()

    if unit in ["km", "kilometers", "kilometres"]:
        return distance

    if unit in ["mile", "miles", "mi"]:
        return distance * KM_PER_MILE

    raise ValueError("Unit must be 'km' or 'miles'")


def _generate_grid_candidates(area_df, spacing_km):
    """Return a float32 (N, 2) array of lat/lon grid points covering area_df's bounding box."""
    min_lat = float(area_df["lat"].min())
    max_lat = float(area_df["lat"].max())
    min_lon = float(area_df["lon"].min())
    max_lon = float(area_df["lon"].max())
    mid_lat = (min_lat + max_lat) / 2.0
    lat_step = spacing_km / 110.574
    lon_step = spacing_km / (111.320 * np.cos(np.radians(mid_lat)))
    lats = np.arange(min_lat, max_lat + lat_step, lat_step, dtype=np.float32)
    lons = np.arange(min_lon, max_lon + lon_step, lon_step, dtype=np.float32)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    return np.column_stack([grid_lats.ravel(), grid_lons.ravel()])


# --------------------------------------------------
# Load postcode data
# --------------------------------------------------

def build_postcode_parquet_data(chunk_size=250_000):
    """
    Run locally to build deployable parquet files from raw CSV sources.

    Raw inputs:
      postcode_data/postcode_base.csv
      postcode_data/postcode_populations_2.csv
      postcode_data/postcode_populations.csv
      postcode_data/demographic_lookup.csv

    Output:
      postcode_parquet/postcodes_part_000.parquet
      postcode_parquet/postcodes_part_001.parquet
      ...
    """

    raw_folder = DATA_FOLDER
    parquet_folder = Path("postcode_parquet")
    parquet_folder.mkdir(exist_ok=True)

    base_file = raw_folder / "postcode_base.csv"
    population_file_1 = raw_folder / "postcode_populations_2.csv"
    population_file_2 = raw_folder / "postcode_populations.csv"
    demographic_file = raw_folder / "demographic_lookup.csv"

    if not base_file.exists():
        raise FileNotFoundError(f"Missing base postcode file: {base_file}")

    # Clear old parquet output
    for old_file in parquet_folder.glob("*.parquet"):
        old_file.unlink()

    # -----------------------------
    # Build population lookup
    # -----------------------------
    if population_file_1.exists():
        print(f"Loading {population_file_1.name}")

        pop = pd.read_csv(population_file_1, low_memory=False)
        pop.columns = pop.columns.str.strip()

        pop = pop[["Postcode", "Total", "Occupied_Households"]].copy()
        pop = pop.rename(columns={
            "Postcode": "postcode",
            "Total": "population",
            "Occupied_Households": "households"
        })

        pop["postcode"] = pop["postcode"].astype(str).str.strip().str.upper()
        pop["population"] = pd.to_numeric(pop["population"], errors="coerce").fillna(0)
        pop["households"] = pd.to_numeric(pop["households"], errors="coerce").fillna(0)

        pop = pop.groupby("postcode", as_index=False).agg({
            "population": "sum",
            "households": "sum"
        })

    elif population_file_2.exists():
        print(f"Loading {population_file_2.name}")

        pop_raw = pd.read_csv(population_file_2, low_memory=False)
        pop_raw.columns = pop_raw.columns.str.strip()

        pop_raw = pop_raw[["Postcode", "Count"]].copy()
        pop_raw = pop_raw.rename(columns={
            "Postcode": "postcode",
            "Count": "population"
        })

        pop_raw["postcode"] = pop_raw["postcode"].astype(str).str.strip().str.upper()
        pop_raw["population"] = pd.to_numeric(pop_raw["population"], errors="coerce").fillna(0)

        pop = pop_raw.groupby("postcode", as_index=False).agg({
            "population": "sum"
        })
        pop["households"] = 0

    else:
        print("No population file found. Population and households will be zero.")
        pop = pd.DataFrame(columns=["postcode", "population", "households"])

    # -----------------------------
    # Build demographic lookup
    # -----------------------------
    if demographic_file.exists():
        print(f"Loading {demographic_file.name}")

        demo = pd.read_csv(demographic_file, low_memory=False)
        demo.columns = demo.columns.str.strip()

        demo = demo[[
            "OAC_Subgroup_Code",
            "OAC_Goup_Name",
            "OAC_Subgroup_Name",
            "OAC_Supergroup_Name"
        ]].copy()

        demo = demo.rename(columns={
            "OAC_Subgroup_Code": "oac_subgroup_code",
            "OAC_Goup_Name": "oac_group_name",
            "OAC_Subgroup_Name": "area_type",
            "OAC_Supergroup_Name": "oac_supergroup_name"
        })

        demo["oac_subgroup_code"] = demo["oac_subgroup_code"].astype(str).str.strip().str.upper()

    else:
        print("No demographic lookup file found.")
        demo = pd.DataFrame(columns=[
            "oac_subgroup_code",
            "oac_group_name",
            "area_type",
            "oac_supergroup_name"
        ])

    # -----------------------------
    # Stream base postcodes in chunks
    # -----------------------------
    print(f"Reading {base_file.name} in chunks...")

    output_count = 0
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(base_file, chunksize=chunk_size, low_memory=False)):
        print(f"Processing chunk {i + 1}")

        chunk.columns = chunk.columns.str.strip()
        col_lookup = {c.lower(): c for c in chunk.columns}

        required = ["pcd", "lat", "long", "oac11"]
        missing = [c for c in required if c not in col_lookup]

        if missing:
            raise ValueError(
                f"{base_file.name} is missing required columns: {missing}. "
                f"Available columns are: {chunk.columns.tolist()}"
            )

        df = chunk[[
            col_lookup["pcd"],
            col_lookup["lat"],
            col_lookup["long"],
            col_lookup["oac11"]
        ]].copy()

        df = df.rename(columns={
            col_lookup["pcd"]: "postcode",
            col_lookup["lat"]: "lat",
            col_lookup["long"]: "lon",
            col_lookup["oac11"]: "oac_subgroup_code"
        })

        df["postcode"] = df["postcode"].astype(str).str.strip().str.upper()
        df["oac_subgroup_code"] = df["oac_subgroup_code"].astype(str).str.strip().str.upper()

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

        df = df.merge(pop, on="postcode", how="left")
        df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)
        df["households"] = pd.to_numeric(df["households"], errors="coerce").fillna(0)

        df = df.merge(demo, on="oac_subgroup_code", how="left")

        df["oac_group_name"] = df["oac_group_name"].fillna("Other")
        df["area_type"] = df["area_type"].fillna("Other")
        df["oac_supergroup_name"] = df["oac_supergroup_name"].fillna("Other")

        df = df.dropna(subset=["lat", "lon"])
        df = df[
            df["lat"].between(49, 61) &
            df["lon"].between(-8, 2)
        ]

        output_file = parquet_folder / f"postcodes_part_{i:03d}.parquet"
        df.to_parquet(output_file, index=False)

        total_rows += len(df)
        output_count += 1

    print(f"Created {output_count} parquet files in {parquet_folder}")
    print(f"Total rows written: {total_rows:,}")

# if __name__ == "__main__":
#     build_postcode_parquet_data()



#@st.cache_data(show_spinner=False)
def load_postcode_data():
    parquet_folder = Path("postcode_parquet")
    files = sorted(parquet_folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {parquet_folder}. "
            "Run build_postcode_parquet_data() locally first."
        )

    df = pd.concat(
        [pd.read_parquet(f, columns=_PARQUET_COLS) for f in files],
        ignore_index=True,
    )

    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(49, 61) & df["lon"].between(-8, 2)]

    print(f"Loaded {len(df):,} postcodes from {len(files)} parquet files")
    return df


# --------------------------------------------------
# Filter city radius
# --------------------------------------------------

def filter_city(df, centre_lat, centre_lon, radius_km):
    distances = haversine_array(
        centre_lat,
        centre_lon,
        df["lat"].to_numpy(),
        df["lon"].to_numpy()
    )

    city_df = df.loc[distances <= radius_km].copy()

    print(f"Rows inside city radius: {len(city_df):,}")

    return city_df


# --------------------------------------------------
# Polygon / path-based area support
# --------------------------------------------------

def validate_boundary_points(boundary_points):
    if not isinstance(boundary_points, (list, tuple)):
        raise ValueError("boundary_points must be a list or tuple of (lat, lon) pairs.")

    if len(boundary_points) < 3:
        raise ValueError("At least 3 boundary points are required to define a polygon.")

    cleaned = []

    for i, pt in enumerate(boundary_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"Boundary point {i} is invalid. Each point must be (lat, lon).")

        lat, lon = pt

        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            raise ValueError(f"Boundary point {i} contains non-numeric values.")

        if not (-90 <= lat <= 90):
            raise ValueError(f"Boundary point {i} has invalid latitude: {lat}")

        if not (-180 <= lon <= 180):
            raise ValueError(f"Boundary point {i} has invalid longitude: {lon}")

        cleaned.append((lat, lon))

    deduped = [cleaned[0]]
    for pt in cleaned[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)

    if len(deduped) < 3:
        raise ValueError("Boundary points collapse to fewer than 3 unique points.")

    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])

    if len(deduped) < 4:
        raise ValueError("Polygon must contain at least 3 unique boundary points.")

    area_proxy = polygon_area_proxy(deduped)
    if abs(area_proxy) < 1e-10:
        raise ValueError("Boundary points do not form a sensible polygon (area is effectively zero).")

    if polygon_self_intersects(deduped):
        raise ValueError("Boundary points form a self-intersecting polygon, which is not supported.")

    return deduped


def polygon_area_proxy(points):
    area = 0.0
    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]
        area += lon1 * lat2 - lon2 * lat1
    return area / 2.0


def orientation(a, b, c):
    ay, ax = a
    by, bx = b
    cy, cx = c
    val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def on_segment(a, b, c):
    ay, ax = a
    by, bx = b
    cy, cx = c

    return (
        min(ax, cx) <= bx <= max(ax, cx) and
        min(ay, cy) <= by <= max(ay, cy)
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def polygon_self_intersects(points):
    n = len(points) - 1

    for i in range(n):
        p1 = points[i]
        q1 = points[i + 1]

        for j in range(i + 1, n):
            p2 = points[j]
            q2 = points[j + 1]

            if abs(i - j) <= 1:
                continue

            if i == 0 and j == n - 1:
                continue

            if segments_intersect(p1, q1, p2, q2):
                return True

    return False


def point_in_polygon(lat, lon, polygon_points):
    inside = False

    for i in range(len(polygon_points) - 1):
        lat1, lon1 = polygon_points[i]
        lat2, lon2 = polygon_points[i + 1]

        intersects = ((lat1 > lat) != (lat2 > lat))
        if intersects:
            lon_intersection = lon1 + (lon2 - lon1) * (lat - lat1) / (lat2 - lat1)
            if lon < lon_intersection:
                inside = not inside

    return inside


def points_in_polygon(test_lats, test_lons, polygon_points):
    """Vectorized ray-casting point-in-polygon test using numpy."""
    poly_lats = np.array([pt[0] for pt in polygon_points])
    poly_lons = np.array([pt[1] for pt in polygon_points])

    n_edges = len(polygon_points) - 1
    inside = np.zeros(len(test_lats), dtype=bool)

    for i in range(n_edges):
        lat1, lon1 = poly_lats[i], poly_lons[i]
        lat2, lon2 = poly_lats[i + 1], poly_lons[i + 1]

        crosses = (lat1 > test_lats) != (lat2 > test_lats)
        if not crosses.any():
            continue
        lon_intersect = lon1 + (lon2 - lon1) * (test_lats[crosses] - lat1) / (lat2 - lat1)
        inside[crosses] ^= (test_lons[crosses] < lon_intersect)

    return inside


def filter_polygon(df, boundary_points):
    polygon = validate_boundary_points(boundary_points)

    lats = np.array([pt[0] for pt in polygon])
    lons = np.array([pt[1] for pt in polygon])

    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()

    bbox_df = df[
        df["lat"].between(min_lat, max_lat) &
        df["lon"].between(min_lon, max_lon)
    ]

    print(f"Rows inside polygon bounding box: {len(bbox_df):,}")

    mask = points_in_polygon(
        bbox_df["lat"].to_numpy(),
        bbox_df["lon"].to_numpy(),
        polygon,
    )

    polygon_df = bbox_df.loc[mask].copy()

    print(f"Rows inside polygon: {len(polygon_df):,}")

    return polygon_df, polygon


# --------------------------------------------------
# Fixed-hub input validation
# --------------------------------------------------

def validate_fixed_hubs(hubs):
    if not isinstance(hubs, (list, tuple)):
        raise ValueError("hubs must be a list or tuple of (name, lat, lon) items.")

    if len(hubs) == 0:
        raise ValueError("At least one hub must be supplied.")

    cleaned = []

    for i, hub in enumerate(hubs):
        if not isinstance(hub, (list, tuple)) or len(hub) != 3:
            raise ValueError(f"Hub {i} is invalid. Each hub must be (name, lat, lon).")

        name, lat, lon = hub

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Hub {i} has an invalid name.")

        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            raise ValueError(f"Hub {i} has non-numeric latitude/longitude.")

        if not (-90 <= lat <= 90):
            raise ValueError(f"Hub {i} has invalid latitude: {lat}")

        if not (-180 <= lon <= 180):
            raise ValueError(f"Hub {i} has invalid longitude: {lon}")

        cleaned.append((name.strip(), lat, lon))

    return cleaned


# --------------------------------------------------
# Fixed-hub coverage calculation
# --------------------------------------------------

def evaluate_fixed_hubs(df, hubs, hub_radius_km):
    if df.empty:
        raise ValueError("No postcode rows found in the selected area.")

    hubs = validate_fixed_hubs(hubs)

    demand_df = df.reset_index(drop=True)
    populations = demand_df["population"].to_numpy(dtype=np.float32)
    households = demand_df["households"].to_numpy(dtype=np.float32)
    demand_coords_rad = np.radians(demand_df[["lat", "lon"]].to_numpy(dtype=np.float32))

    tree = BallTree(demand_coords_rad, metric="haversine")
    radius_rad = hub_radius_km / EARTH_RADIUS_KM

    covered_mask = np.zeros(len(demand_df), dtype=bool)
    hub_results = []

    for hub_num, (hub_name, hub_lat, hub_lon) in enumerate(hubs, start=1):
        hub_coord_rad = np.radians([[hub_lat, hub_lon]])
        full_cover_idx = tree.query_radius(hub_coord_rad, r=radius_rad)[0]

        new_cover_idx = full_cover_idx[~covered_mask[full_cover_idx]]

        full_population = float(populations[full_cover_idx].sum())
        new_population = float(populations[new_cover_idx].sum())
        full_households = float(households[full_cover_idx].sum())
        new_households = float(households[new_cover_idx].sum())

        overlap_population = full_population - new_population
        overlap_households = full_households - new_households

        covered_mask[new_cover_idx] = True

        covered_df = demand_df.iloc[new_cover_idx] if len(new_cover_idx) > 0 else demand_df.iloc[[]]

        hub_postcode = find_nearest_postcode(hub_lat, hub_lon, demand_df)

        top_area_types = covered_df["area_type"].value_counts().head(5).to_dict()

        hub_results.append({
            "hub_number": hub_num,
            "hub_name": hub_name,
            "hub_postcode": hub_postcode,
            "lat": float(hub_lat),
            "lon": float(hub_lon),
            "postcodes": int(len(new_cover_idx)),
            "population": float(new_population),
            "households": float(new_households),
            "potential_postcodes": int(len(full_cover_idx)),
            "potential_population": float(full_population),
            "potential_households": float(full_households),
            "overlap_population": float(overlap_population),
            "overlap_households": float(overlap_households),
            "top_area_types": top_area_types
        })

        print(
            f"Evaluated hub {hub_num}: {hub_name} | "
            f"new coverage {new_population:,.0f} people | "
            f"overlap {overlap_population:,.0f}"
        )

    covered_postcodes = set(demand_df.loc[covered_mask, "postcode"].tolist())

    return hub_results, covered_postcodes


# --------------------------------------------------
# Unified map output
# --------------------------------------------------

def create_hub_map(
    hub_radius_km,
    hubs,
    unit="km",
    output_file="Hub_Map.html",
    boundary_points=None,
    centre_lat=None,
    centre_lon=None,
    city_radius_km=None,
):
    """Create an interactive Folium map showing hub placements.

    Supports both polygon-boundary and city-circle modes:
      - Polygon mode: pass boundary_points (centre is derived automatically).
      - Circle mode:  pass centre_lat, centre_lon, and city_radius_km.
    """
    if boundary_points is not None:
        polygon = validate_boundary_points(boundary_points)
        lats = [pt[0] for pt in polygon]
        lons = [pt[1] for pt in polygon]
        centre_lat = sum(lats) / len(lats)
        centre_lon = sum(lons) / len(lons)
    else:
        polygon = None

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=11, control_scale=True)

    if unit.lower() in ["mile", "miles", "mi"]:
        hub_radius_display = hub_radius_km / KM_PER_MILE
    else:
        hub_radius_display = hub_radius_km

    if polygon is not None:
        folium.Polygon(
            locations=polygon,
            color="black",
            weight=2,
            fill=True,
            fill_opacity=0.08,
            popup="Boundary polygon",
        ).add_to(m)
    elif city_radius_km is not None:
        city_radius_display = (
            city_radius_km / KM_PER_MILE
            if unit.lower() in ["mile", "miles", "mi"]
            else city_radius_km
        )
        folium.Marker([centre_lat, centre_lon], popup="City Centre").add_to(m)
        folium.Circle(
            [centre_lat, centre_lon],
            radius=city_radius_km * 1000,
            color="black",
            fill=False,
            popup=f"City Radius: {city_radius_display:.1f} {unit}",
        ).add_to(m)

    for i, hub in enumerate(hubs):
        color = HUB_COLORS[i % len(HUB_COLORS)]
        hub_label = hub.get("hub_name", f"Hub {hub['hub_number']}")

        popup_parts = [
            f"<b>{hub_label}</b>",
            f"Hub #{hub['hub_number']}",
            f"Postcode: {hub.get('hub_postcode', '')}",
            f"Lat/Lon: {float(hub['lat']):.6f}, {float(hub['lon']):.6f}",
            f"Postcodes: {int(hub['postcodes']):,}",
            f"Population: {int(hub['population']):,}",
            f"Households: {int(hub['households']):,}",
        ]
        if "potential_population" in hub:
            popup_parts.append(f"Potential population: {int(hub['potential_population']):,}")
        if "overlap_population" in hub:
            popup_parts.append(f"Overlap population: {int(hub['overlap_population']):,}")

        folium.Marker(
            [float(hub["lat"]), float(hub["lon"])],
            popup="<br>".join(popup_parts),
            tooltip=hub_label,
            icon=folium.Icon(color=color),
        ).add_to(m)

        folium.Circle(
            [float(hub["lat"]), float(hub["lon"])],
            radius=hub_radius_km * 1000,
            color=color,
            fill=True,
            fill_opacity=0.18,
            popup=f"{hub_label} radius: {hub_radius_display:.1f} {unit}",
        ).add_to(m)

    m.save(output_file)
    print(f"\nMap saved to {output_file}")


# --------------------------------------------------
# Shared result printing
# --------------------------------------------------

def print_hub_results(hubs, covered_population, total_population, coverage_pct, title="HUB RESULTS"):
    print(f"\n================ {title} ================\n")

    for hub in hubs:
        label = hub.get("hub_name", f"Hub {hub['hub_number']}")
        print(label)
        print("-" * 60)
        print(f"Location:              {float(hub['lat']):.6f}, {float(hub['lon']):.6f}")
        print(f"Hub Postcode:          {hub.get('hub_postcode', '')}")
        print(f"Postcodes:             {int(hub['postcodes']):,}")
        print(f"Population:            {int(hub['population']):,}")
        print(f"Households:            {int(hub['households']):,}")

        if "potential_population" in hub:
            print(f"Potential Population:  {int(hub['potential_population']):,}")
        if "overlap_population" in hub:
            print(f"Overlap Population:    {int(hub['overlap_population']):,}")
            pot = hub.get("potential_population", hub["population"])
            overlap_pct = 100.0 * hub["overlap_population"] / pot if pot > 0 else 0.0
            print(f"Overlap %:             {overlap_pct:.2f}%")

        print("\nTop Area Types:")
        if hub.get("top_area_types"):
            for area, count in hub["top_area_types"].items():
                print(f"   {area:<40} {count:,}")
        else:
            print("   No net-new coverage")

        print()

    print("OVERALL COVERAGE")
    print(f"Covered population: {covered_population:,.0f} / {total_population:,.0f}")
    print(f"Coverage: {coverage_pct:.2f}%")


# --------------------------------------------------
# Fixed hubs runner: polygon mode
# --------------------------------------------------

def run_fixed_hub_coverage_polygon(
    boundary_points,
    hubs,
    hub_radius,
    radius_unit="km",
    create_map_output=True,
    map_filename="Fixed_Hub_Map_Polygon.html"
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)

    df = load_postcode_data()

    area_df, cleaned_polygon = filter_polygon(df, boundary_points)

    if area_df.empty:
        raise ValueError(
            "No postcode data found inside the polygon boundary. "
            "Check that the points define a sensible area."
        )

    hub_results, covered = evaluate_fixed_hubs(
        area_df,
        hubs,
        hub_radius_km
    )

    total_population = float(area_df["population"].sum())
    covered_population = float(
        area_df.loc[area_df["postcode"].isin(covered), "population"].sum()
    )
    coverage_pct = 0.0 if total_population == 0 else 100.0 * covered_population / total_population

    print_hub_results(hub_results, covered_population, total_population, coverage_pct, "FIXED HUB RESULTS")

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hub_results,
            unit=radius_unit,
            output_file=map_filename,
            boundary_points=cleaned_polygon,
        )

    multi_df, single_df = build_postcode_hub_mappings(
        area_df, hub_results, hub_radius_km, radius_unit
    )

    return {
        "hubs": hub_results,
        "covered_postcodes": covered,
        "total_population": total_population,
        "covered_population": covered_population,
        "coverage_pct": coverage_pct,
        "boundary_points": cleaned_polygon,
        "radius_unit": radius_unit,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


# --------------------------------------------------
# Hybrid optimisation runner
# --------------------------------------------------

def run_hybrid_optimisation_polygon(
    boundary_points,
    fixed_hubs,
    num_free_hubs,
    hub_radius,
    radius_unit="km",
    grid_spacing_km=1.0,
    map_filename="Hybrid_Hub_Map_Polygon.html",
):
    """
    Evaluate a set of user-supplied fixed hubs, then greedily optimise
    `num_free_hubs` additional locations on whatever demand remains uncovered.
    """
    hub_radius_km = convert_to_km(hub_radius, radius_unit)

    df = load_postcode_data()
    area_df, cleaned_polygon = filter_polygon(df, boundary_points)

    if area_df.empty:
        raise ValueError(
            "No postcode data found inside the polygon boundary. "
            "Check that the points define a sensible area."
        )

    # --- Stage 1: evaluate fixed hubs ---
    fixed_results, covered_postcodes = evaluate_fixed_hubs(
        area_df, fixed_hubs, hub_radius_km
    )

    print(f"\nFixed hubs evaluated. Covered postcodes so far: {len(covered_postcodes):,}")

    # --- Stage 2: optimise free hubs on remaining demand ---
    free_results = []

    if num_free_hubs > 0:
        remaining_df = area_df[~area_df["postcode"].isin(covered_postcodes)].copy()

        if remaining_df.empty:
            print("All demand already covered by fixed hubs; no free hubs placed.")
        else:
            print(f"\nOptimising {num_free_hubs} free hub(s) on "
                  f"{len(remaining_df):,} remaining demand rows…")

            raw_free, free_covered = optimise_hubs_fast_refined(
                remaining_df,
                num_free_hubs,
                hub_radius_km,
                grid_spacing_km=grid_spacing_km,
                jostle_radius_km=2.0,
                refine_passes=3,
            )

            offset = len(fixed_results)
            for i, h in enumerate(raw_free):
                h["hub_number"] = offset + i + 1
                h["hub_name"]   = f"Optimized Hub {i + 1}"
                h.setdefault("potential_population", h["population"])
                h.setdefault("potential_households", h["households"])
                h.setdefault("overlap_population",   0.0)
                h.setdefault("overlap_households",   0.0)

            free_results = raw_free
            covered_postcodes.update(free_covered)

    all_hubs = fixed_results + free_results

    total_population = float(area_df["population"].sum())
    covered_population = float(
        area_df.loc[area_df["postcode"].isin(covered_postcodes), "population"].sum()
    )
    coverage_pct = (
        0.0 if total_population == 0
        else 100.0 * covered_population / total_population
    )

    print_hub_results(all_hubs, covered_population, total_population, coverage_pct, "HYBRID HUB RESULTS")

    create_hub_map(
        hub_radius_km=hub_radius_km,
        hubs=all_hubs,
        unit=radius_unit,
        output_file=map_filename,
        boundary_points=cleaned_polygon,
    )

    multi_df, single_df = build_postcode_hub_mappings(
        area_df, all_hubs, hub_radius_km, radius_unit
    )

    return {
        "hubs":               all_hubs,
        "covered_postcodes":  covered_postcodes,
        "total_population":   total_population,
        "covered_population": covered_population,
        "coverage_pct":       coverage_pct,
        "boundary_points":    cleaned_polygon,
        "radius_unit": radius_unit,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


# --------------------------------------------------
# OLD BRUTE FORCE METHOD
# --------------------------------------------------

def optimise_hubs_bruteforce(df, num_hubs, hub_radius_km):

    remaining = df.copy()
    hubs = []
    covered_postcodes = set()

    for i in range(num_hubs):

        best_score = -1
        best_location = None
        best_cover = None

        print(f"Selecting hub {i+1} (brute force)...")

        for _, candidate in remaining.iterrows():

            distances = remaining.apply(
                lambda r: haversine(candidate["lat"], candidate["lon"], r["lat"], r["lon"]),
                axis=1
            )

            covered = remaining.loc[distances <= hub_radius_km]
            score = covered["population"].sum()

            if score > best_score:
                best_score = score
                best_location = candidate
                best_cover = covered

        if best_location is None:
            break

        hub_postcode = find_nearest_postcode(best_location["lat"], best_location["lon"], df)

        hubs.append({
            "hub_number": i + 1,
            "hub_postcode": hub_postcode,
            "lat": best_location["lat"],
            "lon": best_location["lon"],
            "postcodes": len(best_cover),
            "population": best_cover["population"].sum(),
            "households": best_cover["households"].sum(),
            "top_area_types": best_cover["area_type"].value_counts().head(5).to_dict()
        })

        covered_postcodes.update(best_cover["postcode"])
        remaining = remaining[~remaining["postcode"].isin(best_cover["postcode"])]

        print(f"Placed hub {i+1}: {best_cover['population'].sum():,.0f} population")

    return hubs, covered_postcodes


def run_fixed_hub_coverage(
    centre_lat,
    centre_lon,
    hubs,
    hub_radius,
    city_radius,
    radius_unit="km",
    create_map_output=True,
    map_filename="Fixed_Hub_Map.html"
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)
    city_radius_km = convert_to_km(city_radius, radius_unit)

    df = load_postcode_data()

    city_df = filter_city(df, centre_lat, centre_lon, city_radius_km)

    if city_df.empty:
        raise ValueError(
            "No postcode data found inside the city radius. "
            "Check the centre point and radius."
        )

    hub_results, covered = evaluate_fixed_hubs(
        city_df,
        hubs,
        hub_radius_km
    )

    total_population = float(city_df["population"].sum())
    covered_population = float(
        city_df.loc[city_df["postcode"].isin(covered), "population"].sum()
    )
    coverage_pct = 0.0 if total_population == 0 else 100.0 * covered_population / total_population

    print_hub_results(hub_results, covered_population, total_population, coverage_pct, "FIXED HUB RESULTS")

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hub_results,
            unit=radius_unit,
            output_file=map_filename,
            centre_lat=centre_lat,
            centre_lon=centre_lon,
            city_radius_km=city_radius_km,
        )

    return {
        "hubs": hub_results,
        "covered_postcodes": covered,
        "total_population": total_population,
        "covered_population": covered_population,
        "coverage_pct": coverage_pct
    }


# --------------------------------------------------
# FAST OPTIMIZED METHOD
# --------------------------------------------------

def optimise_hubs_fast(df, num_hubs, hub_radius_km, candidate_stride=1):

    demand_df = df.reset_index(drop=True)

    if candidate_stride > 1:
        candidate_df = demand_df.iloc[::candidate_stride].reset_index(drop=True)
    else:
        candidate_df = demand_df.copy()

    demand_coords = np.radians(demand_df[["lat", "lon"]].to_numpy(dtype=np.float32))
    candidate_coords = np.radians(candidate_df[["lat", "lon"]].to_numpy(dtype=np.float32))

    tree = BallTree(demand_coords, metric="haversine")
    radius_rad = hub_radius_km / EARTH_RADIUS_KM

    print("Precomputing coverage...")
    neighbor_indices = tree.query_radius(candidate_coords, r=radius_rad)

    populations = demand_df["population"].to_numpy(dtype=np.float32)
    households = demand_df["households"].to_numpy(dtype=np.float32)

    covered_mask = np.zeros(len(demand_df), dtype=bool)
    hubs = []

    for hub_num in range(1, num_hubs + 1):

        best_idx = None
        best_gain = -1
        best_cover = None

        print(f"Selecting hub {hub_num} (optimized)...")

        for idx, cover in enumerate(neighbor_indices):

            uncovered = cover[~covered_mask[cover]]
            gain = populations[uncovered].sum()

            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_cover = uncovered

        if best_idx is None:
            break

        covered_mask[best_cover] = True

        hub_row = candidate_df.iloc[best_idx]
        covered_df = demand_df.iloc[best_cover]

        hub_postcode = find_nearest_postcode(hub_row["lat"], hub_row["lon"], demand_df)

        hubs.append({
            "hub_number": hub_num,
            "hub_postcode": hub_postcode,
            "lat": hub_row["lat"],
            "lon": hub_row["lon"],
            "postcodes": len(best_cover),
            "population": populations[best_cover].sum(),
            "households": households[best_cover].sum(),
            "top_area_types": covered_df["area_type"].value_counts().head(5).to_dict()
        })

        print(f"Placed hub {hub_num}: {populations[best_cover].sum():,.0f} population")

    covered_postcodes = set(demand_df.loc[covered_mask, "postcode"])

    return hubs, covered_postcodes


def optimise_hubs_fast_refined(
    df,
    num_hubs,
    hub_radius_km,
    grid_spacing_km=1.0,
    jostle_radius_km=2.0,
    refine_passes=3,
    min_improvement_population=1.0
):
    if df.empty:
        raise ValueError("No postcode rows found inside the search area.")

    demand_df = df.reset_index(drop=True)

    candidate_latlons = _generate_grid_candidates(demand_df, grid_spacing_km)

    demand_coords = np.radians(demand_df[["lat", "lon"]].to_numpy(dtype=np.float32))
    candidate_coords = np.radians(candidate_latlons)

    demand_tree = BallTree(demand_coords, metric="haversine")
    candidate_tree = BallTree(candidate_coords, metric="haversine")

    hub_radius_rad = hub_radius_km / EARTH_RADIUS_KM
    jostle_radius_rad = jostle_radius_km / EARTH_RADIUS_KM

    print(f"Precomputing hub coverage for {len(candidate_latlons):,} grid candidates ({grid_spacing_km} km spacing)...")
    neighbor_indices = demand_tree.query_radius(candidate_coords, r=hub_radius_rad)

    populations = demand_df["population"].to_numpy(dtype=np.float32)
    households = demand_df["households"].to_numpy(dtype=np.float32)

    # --- Stage 1: Greedy seed solution ---

    covered_mask = np.zeros(len(demand_df), dtype=bool)
    selected_candidate_indices = []

    for hub_num in range(1, num_hubs + 1):
        best_idx = None
        best_gain = -1.0
        best_cover = None

        print(f"Selecting hub {hub_num} (greedy seed)...")

        for idx, cover in enumerate(neighbor_indices):
            uncovered = cover[~covered_mask[cover]]
            gain = populations[uncovered].sum()

            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_cover = uncovered

        if best_idx is None or best_cover is None or len(best_cover) == 0:
            print(f"No further useful hub placement found after {hub_num - 1} hub(s).")
            break

        selected_candidate_indices.append(best_idx)
        covered_mask[best_cover] = True

        print(f"Placed seed hub {hub_num}: {populations[best_cover].sum():,.0f} population")

    if not selected_candidate_indices:
        return [], set()

    # --- Helper: summarize chosen hubs ---

    def summarize_selection(selected_indices):
        overall_mask = np.zeros(len(demand_df), dtype=bool)
        hubs = []

        for hub_num, candidate_idx in enumerate(selected_indices, start=1):
            hub_lat = float(candidate_latlons[candidate_idx][0])
            hub_lon = float(candidate_latlons[candidate_idx][1])
            full_cover = neighbor_indices[candidate_idx]
            hub_postcode = find_nearest_postcode(hub_lat, hub_lon, demand_df)

            other_indices = [idx for i, idx in enumerate(selected_indices) if (i + 1) != hub_num]
            others_mask = np.zeros(len(demand_df), dtype=bool)
            for other_idx in other_indices:
                others_mask[neighbor_indices[other_idx]] = True

            net_new_cover = full_cover[~others_mask[full_cover]]
            covered_df = demand_df.iloc[net_new_cover]

            potential_population = float(populations[full_cover].sum())
            net_population = float(populations[net_new_cover].sum())
            potential_households = float(households[full_cover].sum())
            net_households = float(households[net_new_cover].sum())

            overlap_population = potential_population - net_population
            overlap_households = potential_households - net_households

            overall_mask[full_cover] = True

            hubs.append({
                "hub_number": hub_num,
                "hub_postcode": hub_postcode,
                "lat": hub_lat,
                "lon": hub_lon,
                "postcodes": int(len(net_new_cover)),
                "population": float(net_population),
                "households": float(net_households),
                "potential_postcodes": int(len(full_cover)),
                "potential_population": float(potential_population),
                "potential_households": float(potential_households),
                "overlap_population": float(overlap_population),
                "overlap_households": float(overlap_households),
                "top_area_types": covered_df["area_type"].value_counts().head(5).to_dict()
            })

        covered_postcodes = set(demand_df.loc[overall_mask, "postcode"])
        return hubs, covered_postcodes

    # --- Helper: total unique covered population ---

    def total_unique_population(selected_indices):
        mask = np.zeros(len(demand_df), dtype=bool)
        for idx in selected_indices:
            mask[neighbor_indices[idx]] = True
        return float(populations[mask].sum())

    # --- Stage 2: Local refinement / jostling ---

    current_total = total_unique_population(selected_candidate_indices)
    print(f"\nInitial greedy unique covered population: {current_total:,.0f}")

    for refine_pass in range(1, refine_passes + 1):
        improved_this_pass = False
        print(f"\nRefinement pass {refine_pass}/{refine_passes}...")

        for hub_pos in range(len(selected_candidate_indices)):
            current_idx = selected_candidate_indices[hub_pos]

            others_mask = np.zeros(len(demand_df), dtype=bool)
            for j, idx in enumerate(selected_candidate_indices):
                if j != hub_pos:
                    others_mask[neighbor_indices[idx]] = True

            current_net_cover = neighbor_indices[current_idx][~others_mask[neighbor_indices[current_idx]]]
            current_net_gain = float(populations[current_net_cover].sum())

            nearby_candidate_indices = candidate_tree.query_radius(
                candidate_coords[current_idx:current_idx + 1],
                r=jostle_radius_rad
            )[0]

            best_local_idx = current_idx
            best_local_gain = current_net_gain

            for candidate_idx in nearby_candidate_indices:
                candidate_net_cover = neighbor_indices[candidate_idx][~others_mask[neighbor_indices[candidate_idx]]]
                candidate_net_gain = float(populations[candidate_net_cover].sum())

                if candidate_net_gain > best_local_gain + min_improvement_population:
                    best_local_gain = candidate_net_gain
                    best_local_idx = candidate_idx

            if best_local_idx != current_idx:
                trial_selection = selected_candidate_indices.copy()
                trial_selection[hub_pos] = best_local_idx

                trial_total = total_unique_population(trial_selection)

                if trial_total > current_total + min_improvement_population:
                    print(
                        f"Hub {hub_pos + 1} moved: "
                        f"{current_total:,.0f} -> {trial_total:,.0f} "
                        f"(+{trial_total - current_total:,.0f})"
                    )
                    selected_candidate_indices = trial_selection
                    current_total = trial_total
                    improved_this_pass = True

        if not improved_this_pass:
            print("No improvements found in this pass.")
            break

    print(f"\nFinal refined unique covered population: {current_total:,.0f}")

    hubs, covered_postcodes = summarize_selection(selected_candidate_indices)
    return hubs, covered_postcodes


# --------------------------------------------------
# COVERAGE-TARGET OPTIMISER
# --------------------------------------------------

def optimise_hubs_by_coverage(
    df,
    target_coverage_pct,
    hub_radius_km,
    max_hubs=50,
    grid_spacing_km=1.0,
    jostle_radius_km=2.0,
    refine_passes=3,
    min_improvement_population=1.0,
):
    if df.empty:
        raise ValueError("No postcode rows found inside the search area.")

    demand_df = df.reset_index(drop=True)
    total_population = float(demand_df["population"].sum())

    if total_population == 0:
        raise ValueError("Total population in area is zero.")

    target_population = total_population * target_coverage_pct / 100.0

    candidate_latlons = _generate_grid_candidates(demand_df, grid_spacing_km)

    demand_coords = np.radians(demand_df[["lat", "lon"]].to_numpy(dtype=np.float32))
    candidate_coords = np.radians(candidate_latlons)

    demand_tree = BallTree(demand_coords, metric="haversine")
    candidate_tree = BallTree(candidate_coords, metric="haversine")

    hub_radius_rad = hub_radius_km / EARTH_RADIUS_KM
    jostle_radius_rad = jostle_radius_km / EARTH_RADIUS_KM

    print(f"Precomputing hub coverage for {len(candidate_latlons):,} grid candidates ({grid_spacing_km} km spacing)...")
    neighbor_indices = demand_tree.query_radius(candidate_coords, r=hub_radius_rad)

    populations = demand_df["population"].to_numpy(dtype=np.float32)
    households = demand_df["households"].to_numpy(dtype=np.float32)

    # --- Stage 1: Greedy seed until target coverage reached ---
    covered_mask = np.zeros(len(demand_df), dtype=bool)
    selected_candidate_indices = []

    for hub_num in range(1, max_hubs + 1):
        best_idx = None
        best_gain = -1.0
        best_cover = None

        print(f"Selecting hub {hub_num} (greedy seed)...")

        for idx, cover in enumerate(neighbor_indices):
            uncovered = cover[~covered_mask[cover]]
            gain = populations[uncovered].sum()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_cover = uncovered

        if best_idx is None or best_cover is None or len(best_cover) == 0:
            print(f"No further useful hub placement found after {hub_num - 1} hub(s).")
            break

        selected_candidate_indices.append(best_idx)
        covered_mask[best_cover] = True

        covered_pop = float(populations[covered_mask].sum())
        pct_achieved = 100.0 * covered_pop / total_population
        print(f"Placed seed hub {hub_num}: {populations[best_cover].sum():,.0f} population | Coverage: {pct_achieved:.1f}%")

        if covered_pop >= target_population:
            print(f"Target coverage of {target_coverage_pct:.1f}% reached with {hub_num} hub(s).")
            break

    if not selected_candidate_indices:
        return [], set()

    # --- Helper: summarize chosen hubs ---
    def summarize_selection(selected_indices):
        overall_mask = np.zeros(len(demand_df), dtype=bool)
        hubs = []

        for h_num, candidate_idx in enumerate(selected_indices, start=1):
            hub_lat = float(candidate_latlons[candidate_idx][0])
            hub_lon = float(candidate_latlons[candidate_idx][1])
            full_cover = neighbor_indices[candidate_idx]
            hub_postcode = find_nearest_postcode(hub_lat, hub_lon, demand_df)

            other_indices = [idx for i, idx in enumerate(selected_indices) if (i + 1) != h_num]
            others_mask = np.zeros(len(demand_df), dtype=bool)
            for other_idx in other_indices:
                others_mask[neighbor_indices[other_idx]] = True

            net_new_cover = full_cover[~others_mask[full_cover]]
            covered_df = demand_df.iloc[net_new_cover]

            potential_population = float(populations[full_cover].sum())
            net_population = float(populations[net_new_cover].sum())
            potential_households = float(households[full_cover].sum())
            net_households = float(households[net_new_cover].sum())

            overlap_population = potential_population - net_population
            overlap_households = potential_households - net_households

            overall_mask[full_cover] = True

            hubs.append({
                "hub_number": h_num,
                "hub_postcode": hub_postcode,
                "lat": hub_lat,
                "lon": hub_lon,
                "postcodes": int(len(net_new_cover)),
                "population": float(net_population),
                "households": float(net_households),
                "potential_postcodes": int(len(full_cover)),
                "potential_population": float(potential_population),
                "potential_households": float(potential_households),
                "overlap_population": float(overlap_population),
                "overlap_households": float(overlap_households),
                "top_area_types": covered_df["area_type"].value_counts().head(5).to_dict(),
            })

        covered_postcodes = set(demand_df.loc[overall_mask, "postcode"])
        return hubs, covered_postcodes

    # --- Helper: total unique covered population ---
    def total_unique_population(selected_indices):
        mask = np.zeros(len(demand_df), dtype=bool)
        for idx in selected_indices:
            mask[neighbor_indices[idx]] = True
        return float(populations[mask].sum())

    # --- Stage 2: Local refinement / jostling ---
    current_total = total_unique_population(selected_candidate_indices)
    print(f"\nInitial greedy unique covered population: {current_total:,.0f}")

    for refine_pass in range(1, refine_passes + 1):
        improved_this_pass = False
        print(f"\nRefinement pass {refine_pass}/{refine_passes}...")

        for hub_pos in range(len(selected_candidate_indices)):
            current_idx = selected_candidate_indices[hub_pos]

            others_mask = np.zeros(len(demand_df), dtype=bool)
            for j, idx in enumerate(selected_candidate_indices):
                if j != hub_pos:
                    others_mask[neighbor_indices[idx]] = True

            current_net_cover = neighbor_indices[current_idx][~others_mask[neighbor_indices[current_idx]]]
            current_net_gain = float(populations[current_net_cover].sum())

            nearby_candidate_indices = candidate_tree.query_radius(
                candidate_coords[current_idx:current_idx + 1],
                r=jostle_radius_rad,
            )[0]

            best_local_idx = current_idx
            best_local_gain = current_net_gain

            for candidate_idx in nearby_candidate_indices:
                candidate_net_cover = neighbor_indices[candidate_idx][~others_mask[neighbor_indices[candidate_idx]]]
                candidate_net_gain = float(populations[candidate_net_cover].sum())

                if candidate_net_gain > best_local_gain + min_improvement_population:
                    best_local_gain = candidate_net_gain
                    best_local_idx = candidate_idx

            if best_local_idx != current_idx:
                trial_selection = selected_candidate_indices.copy()
                trial_selection[hub_pos] = best_local_idx
                trial_total = total_unique_population(trial_selection)

                if trial_total > current_total + min_improvement_population:
                    print(
                        f"Hub {hub_pos + 1} moved: "
                        f"{current_total:,.0f} -> {trial_total:,.0f} "
                        f"(+{trial_total - current_total:,.0f})"
                    )
                    selected_candidate_indices = trial_selection
                    current_total = trial_total
                    improved_this_pass = True

        if not improved_this_pass:
            print("No improvements found in this pass.")
            break

    print(f"\nFinal refined unique covered population: {current_total:,.0f}")

    hubs, covered_postcodes = summarize_selection(selected_candidate_indices)
    return hubs, covered_postcodes


def run_hub_optimisation_polygon_by_coverage(
    boundary_points,
    target_coverage_pct,
    hub_radius,
    radius_unit="km",
    max_hubs=50,
    grid_spacing_km=1.0,
    create_map_output=True,
    map_filename="Hub_Map_Polygon_Coverage.html",
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)

    df = load_postcode_data()

    area_df, cleaned_polygon = filter_polygon(df, boundary_points)

    if area_df.empty:
        raise ValueError(
            "No postcode data found inside the polygon boundary. "
            "Check that the points are in the right order and cover a sensible area."
        )

    hubs, covered = optimise_hubs_by_coverage(
        area_df,
        target_coverage_pct=target_coverage_pct,
        hub_radius_km=hub_radius_km,
        max_hubs=max_hubs,
        grid_spacing_km=grid_spacing_km,
        jostle_radius_km=2.0,
        refine_passes=3,
    )

    total_population = area_df["population"].sum()
    covered_population = area_df.loc[
        area_df["postcode"].isin(covered), "population"
    ].sum()

    coverage_pct = 0 if total_population == 0 else 100 * covered_population / total_population

    print_hub_results(hubs, covered_population, total_population, coverage_pct, "COVERAGE TARGET RESULTS")

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hubs,
            unit=radius_unit,
            output_file=map_filename,
            boundary_points=cleaned_polygon,
        )

    multi_df, single_df = build_postcode_hub_mappings(
        area_df, hubs, hub_radius_km, radius_unit
    )

    return {
        "hubs": hubs,
        "covered_postcodes": covered,
        "total_population": float(total_population),
        "covered_population": float(covered_population),
        "coverage_pct": float(coverage_pct),
        "target_coverage_pct": target_coverage_pct,
        "boundary_points": cleaned_polygon,
        "radius_unit": radius_unit,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


# --------------------------------------------------
# RUNNER
# --------------------------------------------------

def run_hub_optimisation(
    centre_lat,
    centre_lon,
    num_hubs,
    hub_radius,
    city_radius,
    radius_unit="km",
    use_optimized=True,
    grid_spacing_km=1.0,
    create_map_output=True
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)
    city_radius_km = convert_to_km(city_radius, radius_unit)

    df = load_postcode_data()

    city_df = filter_city(df, centre_lat, centre_lon, city_radius_km)

    if use_optimized:
        hubs, covered = optimise_hubs_fast_refined(
            city_df,
            num_hubs,
            hub_radius_km,
            grid_spacing_km=grid_spacing_km,
            jostle_radius_km=3.0,
            refine_passes=5
        )
    else:
        hubs, covered = optimise_hubs_bruteforce(
            city_df,
            num_hubs,
            hub_radius_km
        )

    total_population = city_df["population"].sum()
    covered_population = city_df.loc[
        city_df["postcode"].isin(covered),
        "population"
    ].sum()

    coverage_pct = 100 * covered_population / total_population

    print_hub_results(hubs, covered_population, total_population, coverage_pct)

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hubs,
            unit=radius_unit,
            centre_lat=centre_lat,
            centre_lon=centre_lon,
            city_radius_km=city_radius_km,
        )


def run_hub_optimisation_polygon(
    boundary_points,
    num_hubs,
    hub_radius,
    radius_unit="km",
    use_optimized=True,
    grid_spacing_km=1.0,
    create_map_output=True,
    map_filename="Hub_Map_Polygon.html"
):
    hub_radius_km = convert_to_km(hub_radius, radius_unit)

    df = load_postcode_data()

    area_df, cleaned_polygon = filter_polygon(df, boundary_points)

    if area_df.empty:
        raise ValueError(
            "No postcode data found inside the polygon boundary. "
            "Check that the points are in the right order and cover a sensible area."
        )

    if use_optimized:
        hubs, covered = optimise_hubs_fast_refined(
            area_df,
            num_hubs,
            hub_radius_km,
            grid_spacing_km=grid_spacing_km,
            jostle_radius_km=2.0,
            refine_passes=3
        )
    else:
        hubs, covered = optimise_hubs_bruteforce(
            area_df,
            num_hubs,
            hub_radius_km
        )

    total_population = area_df["population"].sum()
    covered_population = area_df.loc[
        area_df["postcode"].isin(covered),
        "population"
    ].sum()

    coverage_pct = 0 if total_population == 0 else 100 * covered_population / total_population

    print_hub_results(hubs, covered_population, total_population, coverage_pct, "POLYGON HUB RESULTS")

    if create_map_output:
        create_hub_map(
            hub_radius_km=hub_radius_km,
            hubs=hubs,
            unit=radius_unit,
            output_file=map_filename,
            boundary_points=cleaned_polygon,
        )

    multi_df, single_df = build_postcode_hub_mappings(
        area_df, hubs, hub_radius_km, radius_unit
    )

    return {
        "hubs": hubs,
        "covered_postcodes": covered,
        "total_population": float(total_population),
        "covered_population": float(covered_population),
        "coverage_pct": float(coverage_pct),
        "boundary_points": cleaned_polygon,
        "radius_unit": radius_unit,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


def find_nearest_postcode(lat, lon, df):
    distances = haversine_array(
        lat,
        lon,
        df["lat"].to_numpy(),
        df["lon"].to_numpy()
    )

    nearest_idx = int(np.argmin(distances))
    return df.iloc[nearest_idx]["postcode"]


def build_postcode_hub_mappings(area_df, hub_results, hub_radius_km, radius_unit="miles"):
    """
    Returns two DataFrames:
      multi_df  – one row per (postcode, hub) pair where the hub covers that postcode.
      single_df – one row per postcode, assigned to its nearest covering hub
                  (tie-broken by hub_number, i.e. order placed during optimisation).
    """
    dist_col = f"Distance ({radius_unit})"
    cols = ["Postcode", "Lat", "Lon", "Hub", "Hub Number", dist_col]

    if area_df.empty or not hub_results:
        empty = pd.DataFrame(columns=cols)
        return empty, empty.copy()

    pc_lats = area_df["lat"].to_numpy(dtype=np.float32)
    pc_lons = area_df["lon"].to_numpy(dtype=np.float32)
    use_miles = radius_unit.lower() in ("mile", "miles", "mi")

    multi_dfs = []
    # df_idx -> (dist_km, hub_number, hub_name, dist_display)
    best: dict[int, tuple] = {}

    for hub in hub_results:
        hub_lat    = hub["lat"]
        hub_lon    = hub["lon"]
        hub_name   = hub.get("hub_name", f"Hub {hub['hub_number']}")
        hub_number = hub["hub_number"]

        dists_km = haversine_array(hub_lat, hub_lon, pc_lats, pc_lons)
        in_range = np.where(dists_km <= hub_radius_km)[0]

        if len(in_range) == 0:
            continue

        dists_display = dists_km[in_range] / KM_PER_MILE if use_miles else dists_km[in_range]

        batch = area_df.iloc[in_range][["postcode", "lat", "lon"]].copy()
        batch.columns = ["Postcode", "Lat", "Lon"]
        batch["Lat"] = batch["Lat"].round(5)
        batch["Lon"] = batch["Lon"].round(5)
        batch["Hub"] = hub_name
        batch["Hub Number"] = hub_number
        batch[dist_col] = np.round(dists_display, 4)
        multi_dfs.append(batch[cols])

        # nearest-hub tracking
        for arr_pos, df_idx in enumerate(in_range):
            d_km  = float(dists_km[df_idx])
            d_dis = float(dists_display[arr_pos])
            prev  = best.get(df_idx)
            if prev is None or d_km < prev[0] or (d_km == prev[0] and hub_number < prev[1]):
                best[df_idx] = (d_km, hub_number, hub_name, d_dis)

    multi_df = (
        pd.concat(multi_dfs, ignore_index=True)
        .sort_values(["Hub Number", "Postcode"])
        .reset_index(drop=True)
        if multi_dfs else pd.DataFrame(columns=cols)
    )

    if best:
        best_indices = list(best.keys())
        best_vals    = list(best.values())
        single_df = area_df.iloc[best_indices][["postcode", "lat", "lon"]].copy()
        single_df.columns = ["Postcode", "Lat", "Lon"]
        single_df["Lat"]        = single_df["Lat"].round(5)
        single_df["Lon"]        = single_df["Lon"].round(5)
        single_df["Hub Number"] = [v[1] for v in best_vals]
        single_df["Hub"]        = [v[2] for v in best_vals]
        single_df[dist_col]     = [round(v[3], 4) for v in best_vals]
        single_df = (
            single_df[cols]
            .sort_values(["Hub Number", "Postcode"])
            .reset_index(drop=True)
        )
    else:
        single_df = pd.DataFrame(columns=cols)

    return multi_df, single_df



# --------------------------------------------------
# Cross-border (airport catchment) optimisation
# --------------------------------------------------

AIRPORTS_CSV = Path("Airports.csv")


def load_uk_airports():
    """
    Load UK airport reference data from Airports.csv:
    name, IATA code, lat/lon, and passenger/freight importance
    ("Major" / "Regional" / "Limited" / "Specialist/business").
    """
    if not AIRPORTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing airport reference file: {AIRPORTS_CSV}. "
            "Expected columns: Airport, IATA, Latitude, Longitude, "
            "Passenger importance, Freight importance."
        )

    raw = pd.read_csv(AIRPORTS_CSV)
    raw.columns = raw.columns.str.strip()

    airports = []
    for _, row in raw.iterrows():
        airports.append({
            "code": str(row["IATA"]).strip(),
            "name": str(row["Airport"]).strip(),
            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),
            "passenger_importance": str(row["Passenger importance"]).strip(),
            "freight_importance": str(row["Freight importance"]).strip(),
        })

    return airports


def circle_area(radius):
    """Area of a circle of the given radius, in that radius's squared unit."""
    return float(np.pi * radius ** 2)


def local_population_window(lats, lons, populations, query_idx, window_radius_km):
    """
    Approximate, for each of the given query points, the total population within
    an axis-aligned square window (side ~2 * window_radius_km) centred on that
    point, using a fixed spatial grid + 2D prefix sum (integral image).

    This is an O(N + grid cells) approximation to an exact circular radius
    query. An exact per-point BallTree radius search is O(N * neighbours),
    which is unbounded in dense urban centres -- a 5-mile petal-run circle in
    central London can contain hundreds of thousands of postcodes, so summing
    that for every eligible postcode can exhaust tens of GB of RAM. The grid
    here is sized off the window radius, not the number of points, so memory
    stays flat regardless of local postcode density.
    """
    n = len(lats)
    if n == 0 or len(query_idx) == 0:
        return np.zeros(len(query_idx), dtype=np.float64)

    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    x_km = (lons - lon0) * 111.320 * np.cos(np.radians(lat0))
    y_km = (lats - lat0) * 110.574

    # Grid resolution: fine enough that the square window is a reasonable
    # stand-in for the circle, but independent of how many points fall in it.
    cell_km = max(window_radius_km / 8.0, 0.05)

    min_x, min_y = x_km.min(), y_km.min()
    col_idx = np.floor((x_km - min_x) / cell_km).astype(np.int64)
    row_idx = np.floor((y_km - min_y) / cell_km).astype(np.int64)

    n_cols = int(col_idx.max()) + 1
    n_rows = int(row_idx.max()) + 1

    flat_idx = row_idx * n_cols + col_idx
    grid_pop = np.bincount(
        flat_idx, weights=populations.astype(np.float64), minlength=n_rows * n_cols
    ).reshape(n_rows, n_cols)

    # Prefix sum padded with a leading zero row/col for O(1) rectangle sums.
    prefix = np.zeros((n_rows + 1, n_cols + 1), dtype=np.float64)
    prefix[1:, 1:] = np.cumsum(np.cumsum(grid_pop, axis=0), axis=1)

    half_cells = int(np.ceil(window_radius_km / cell_km))

    q_row = row_idx[query_idx]
    q_col = col_idx[query_idx]

    r0 = np.clip(q_row - half_cells, 0, n_rows)
    r1 = np.clip(q_row + half_cells + 1, 0, n_rows)
    c0 = np.clip(q_col - half_cells, 0, n_cols)
    c1 = np.clip(q_col + half_cells + 1, 0, n_cols)

    return (
        prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]
    )


def _empty_cross_border_result(airport):
    return {
        "airport_code": airport["code"],
        "airport_name": airport["name"],
        "lat": float(airport["lat"]),
        "lon": float(airport["lon"]),
        "eligible_postcodes": 0,
        "eligible_population": 0.0,
        "postcodes": 0,
        "population": 0.0,
        "potential_postcodes": 0,
        "potential_population": 0.0,
        "overlap_population": 0.0,
        "excluded_by_min_population": 0,
        "excluded_by_max_density": 0,
        "excluded_by_island": 0,
        "extended_postcodes": 0,
        "extended_population": 0.0,
    }


def island_filter_mask(
    lats, lons, populations, link_radius_km, min_cluster_postcodes=1,
    min_cluster_population=0.0, core_mask=None,
):
    """
    Group points into clusters, where two points are linked (transitively) if
    they're within roughly link_radius_km of each other, then keep only
    points whose cluster has at least min_cluster_postcodes members and at
    least min_cluster_population total population.

    This removes "islands" -- small pockets of postcodes that individually
    clear a density/population threshold but sit isolated from the rest of
    the qualifying area, which would mean an inefficient dead-leg trip to
    reach them rather than an efficient local delivery run.

    If `core_mask` is given (a boolean array aligned with lats/lons), a
    cluster is additionally required to contain at least one core point to
    be kept. This is how dense clusters are allowed to extend past a search
    radius: points beyond the radius are passed in alongside the in-radius
    ("core") points, and only clusters actually anchored inside the radius
    survive -- a wholly separate dense pocket entirely beyond the radius
    still gets dropped.

    Clustering is done on a fixed spatial grid (cell size = link_radius_km,
    connecting each occupied cell to its 8 neighbours) rather than an exact
    point-to-point BallTree radius query. An exact query is O(N x
    neighbours-per-point), which is unbounded in dense city centres -- the
    same failure mode already hit and fixed for the petal-run density calc.
    The grid/union-find approach here is bounded by the number of distinct
    grid cells (i.e. the geographic extent of the data), independent of how
    many postcodes fall in any one of them.
    """
    n = len(lats)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if link_radius_km <= 0:
        return np.ones(n, dtype=bool)

    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    x_km = (lons - lon0) * 111.320 * np.cos(np.radians(lat0))
    y_km = (lats - lat0) * 110.574

    cell_km = link_radius_km
    col_idx = np.floor(x_km / cell_km).astype(np.int64)
    row_idx = np.floor(y_km / cell_km).astype(np.int64)

    # Union-find over distinct occupied (row, col) cells only.
    occupied_cells = {}
    for r, c in zip(row_idx, col_idx):
        key = (int(r), int(c))
        if key not in occupied_cells:
            occupied_cells[key] = len(occupied_cells)

    n_cells = len(occupied_cells)
    parent = list(range(n_cells))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (r, c), cell_local_id in occupied_cells.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor_local_id = occupied_cells.get((r + dr, c + dc))
                if neighbor_local_id is not None:
                    union(cell_local_id, neighbor_local_id)

    cell_root = np.array([find(i) for i in range(n_cells)])
    point_cell_local_id = np.fromiter(
        (occupied_cells[(int(r), int(c))] for r, c in zip(row_idx, col_idx)),
        dtype=np.int64, count=n,
    )
    point_root = cell_root[point_cell_local_id]

    _, cluster_labels = np.unique(point_root, return_inverse=True)
    n_clusters = int(cluster_labels.max()) + 1 if n else 0

    cluster_postcode_counts = np.bincount(cluster_labels, minlength=n_clusters)
    cluster_populations = np.bincount(cluster_labels, weights=populations, minlength=n_clusters)

    cluster_ok = (cluster_postcode_counts >= min_cluster_postcodes) & (
        cluster_populations >= min_cluster_population
    )
    if core_mask is not None:
        core_mask = np.asarray(core_mask, dtype=bool)
        cluster_has_core = np.zeros(n_clusters, dtype=bool)
        cluster_has_core[np.unique(cluster_labels[core_mask])] = True
        cluster_ok &= cluster_has_core
    return cluster_ok[cluster_labels]


def evaluate_cross_border_airports(
    df,
    airports,
    outer_radius_km,
    circle_radius_km,
    circle_radius_display,
    density_threshold,
    radius_unit="miles",
    min_population_per_postcode=None,
    max_density_per_postcode=None,
    cluster_link_radius_km=None,
    min_cluster_postcodes=1,
    min_cluster_population=0.0,
    extension_radius_km=0.0,
):
    """
    For each airport, find postcodes within `outer_radius_km` of it whose local
    population density -- population within `circle_radius_km` of the postcode
    itself, divided by the circle's area -- meets `density_threshold`. A
    postcode's own neighbourhood can extend up to `circle_radius_km` beyond the
    outer radius, so the true search reach is outer + circle.

    Density (and the optional per-postcode population/density caps) is
    expressed per unit of `circle_radius_display`'s squared unit, e.g. people
    per square mile when the UI is set to miles.

    If `extension_radius_km` is set (and `cluster_link_radius_km` is also
    set, since extension only makes sense in terms of a cluster), a cluster
    that is dense/connected enough to survive the island filter and that
    reaches the outer radius may extend past it by up to
    `extension_radius_km`, provided the extra postcodes are themselves
    linked (within `cluster_link_radius_km`) and still meet the density/
    population filters. This runs after the initial radius+island filtering
    and lets a hub pick up the parts of a neighbouring dense pocket a local
    delivery run would naturally reach, rather than hard-cutting at the
    catchment radius. A dense pocket that never reaches the outer radius at
    all still doesn't qualify on its own.

    Postcodes are attributed to the first airport (in `airports` order) that
    covers them, mirroring the net-new/overlap accounting used for hubs.

    Returns (airport_results, covered_postcodes, multi_df, single_df, total_eligible_population).
    """
    area = circle_area(circle_radius_display)
    if area <= 0:
        raise ValueError("Circle radius must be greater than zero.")

    use_miles = radius_unit.lower() in ("mile", "miles", "mi")
    dist_col = f"Distance to Airport ({radius_unit})"
    density_col = f"Local Density (per sq {'mi' if use_miles else 'km'})"
    map_cols = ["Postcode", "Lat", "Lon", "Population", density_col, "Airport", "Airport Code", dist_col]

    extend_km = float(extension_radius_km) if (extension_radius_km and cluster_link_radius_km) else 0.0
    candidate_radius_km = outer_radius_km + extend_km
    search_radius_km = candidate_radius_km + circle_radius_km

    airport_results = []
    covered_postcodes = set()
    eligible_population_by_postcode = {}
    multi_dfs = []
    # postcode -> (dist_km, airport_code, airport_name, population, density, dist_display)
    best = {}

    for airport in airports:
        lat, lon = airport["lat"], airport["lon"]

        nearby_df = filter_city(df, lat, lon, search_radius_km).reset_index(drop=True)

        if nearby_df.empty:
            airport_results.append(_empty_cross_border_result(airport))
            continue

        dist_to_airport_km = haversine_array(
            lat, lon, nearby_df["lat"].to_numpy(), nearby_df["lon"].to_numpy()
        )
        eligible_idx = np.where(dist_to_airport_km <= outer_radius_km)[0]

        if len(eligible_idx) == 0:
            airport_results.append(_empty_cross_border_result(airport))
            continue

        candidate_idx = (
            np.where(dist_to_airport_km <= candidate_radius_km)[0] if extend_km > 0 else eligible_idx
        )

        populations = nearby_df["population"].to_numpy(dtype=np.float32)

        for pc, pop in zip(
            nearby_df.iloc[eligible_idx]["postcode"], populations[eligible_idx]
        ):
            eligible_population_by_postcode[pc] = float(pop)

        local_population = local_population_window(
            nearby_df["lat"].to_numpy(dtype=np.float64),
            nearby_df["lon"].to_numpy(dtype=np.float64),
            populations,
            candidate_idx,
            circle_radius_km,
        )
        local_density = local_population / area

        qualifies = local_density >= density_threshold
        candidate_own_population = populations[candidate_idx]

        excluded_min_pop = 0
        if min_population_per_postcode is not None:
            fails = qualifies & (candidate_own_population < min_population_per_postcode)
            excluded_min_pop = int(fails.sum())
            qualifies &= ~fails

        excluded_max_density = 0
        if max_density_per_postcode is not None:
            fails = qualifies & (local_density > max_density_per_postcode)
            excluded_max_density = int(fails.sum())
            qualifies &= ~fails

        qualifying_idx = candidate_idx[qualifies]

        excluded_by_island = 0
        if cluster_link_radius_km and cluster_link_radius_km > 0 and len(qualifying_idx) > 0:
            core_mask = dist_to_airport_km[qualifying_idx] <= outer_radius_km
            keep_mask = island_filter_mask(
                nearby_df["lat"].to_numpy(dtype=np.float64)[qualifying_idx],
                nearby_df["lon"].to_numpy(dtype=np.float64)[qualifying_idx],
                populations[qualifying_idx],
                link_radius_km=cluster_link_radius_km,
                min_cluster_postcodes=min_cluster_postcodes,
                min_cluster_population=min_cluster_population,
                core_mask=core_mask,
            )
            excluded_by_island = int((~keep_mask).sum())
            if excluded_by_island > 0:
                # Fold the island exclusion back into `qualifies` so it stays
                # aligned with local_density/candidate_idx for everything below.
                qualifying_positions = np.where(qualifies)[0]
                qualifies[qualifying_positions[~keep_mask]] = False
                qualifying_idx = candidate_idx[qualifies]

        extended_mask = dist_to_airport_km[qualifying_idx] > outer_radius_km
        extended_postcodes = int(extended_mask.sum())
        extended_population = float(populations[qualifying_idx][extended_mask].sum())

        potential_population = float(populations[qualifying_idx].sum())
        potential_postcodes = int(len(qualifying_idx))

        if potential_postcodes > 0:
            batch = nearby_df.iloc[qualifying_idx][["postcode", "lat", "lon", "population"]].copy()
            batch.columns = ["Postcode", "Lat", "Lon", "Population"]
            batch["Lat"] = batch["Lat"].round(5)
            batch["Lon"] = batch["Lon"].round(5)

            qualifying_density = local_density[qualifies]
            dists_km = dist_to_airport_km[qualifying_idx]
            dists_display = dists_km / KM_PER_MILE if use_miles else dists_km

            batch[density_col] = np.round(qualifying_density, 2)
            batch["Airport"] = airport["name"]
            batch["Airport Code"] = airport["code"]
            batch[dist_col] = np.round(dists_display, 4)
            multi_dfs.append(batch[map_cols])

            is_new = ~batch["Postcode"].isin(covered_postcodes)
            new_postcode_count = int(is_new.sum())
            new_population = float(batch.loc[is_new, "Population"].sum())
            covered_postcodes.update(batch["Postcode"])

            for pc, d_km, d_dis, pop, dens in zip(
                batch["Postcode"], dists_km, dists_display, batch["Population"], qualifying_density
            ):
                prev = best.get(pc)
                if prev is None or d_km < prev[0]:
                    best[pc] = (float(d_km), airport["code"], airport["name"], float(pop), float(dens), float(d_dis))
        else:
            new_postcode_count = 0
            new_population = 0.0

        airport_results.append({
            "airport_code": airport["code"],
            "airport_name": airport["name"],
            "lat": float(lat),
            "lon": float(lon),
            "eligible_postcodes": int(len(eligible_idx)),
            "eligible_population": float(populations[eligible_idx].sum()),
            "postcodes": new_postcode_count,
            "population": new_population,
            "potential_postcodes": potential_postcodes,
            "potential_population": potential_population,
            "overlap_population": potential_population - new_population,
            "excluded_by_min_population": excluded_min_pop,
            "excluded_by_max_density": excluded_max_density,
            "excluded_by_island": excluded_by_island,
            "extended_postcodes": extended_postcodes,
            "extended_population": extended_population,
        })

    multi_df = (
        pd.concat(multi_dfs, ignore_index=True)
        .sort_values(["Airport Code", "Postcode"])
        .reset_index(drop=True)
        if multi_dfs else pd.DataFrame(columns=map_cols)
    )

    if best:
        rows = []
        for pc, (d_km, code, name, pop, dens, d_dis) in best.items():
            rows.append((pc, code, name, pop, dens, d_dis))
        single_df = pd.DataFrame(
            rows, columns=["Postcode", "Airport Code", "Airport", "Population", density_col, dist_col]
        )
        latlon_lookup = multi_df.drop_duplicates("Postcode").set_index("Postcode")[["Lat", "Lon"]]
        single_df = single_df.join(latlon_lookup, on="Postcode")
        single_df = single_df[map_cols].sort_values(["Airport Code", "Postcode"]).reset_index(drop=True)
    else:
        single_df = pd.DataFrame(columns=map_cols)

    total_eligible_population = float(sum(eligible_population_by_postcode.values()))

    return airport_results, covered_postcodes, multi_df, single_df, total_eligible_population


def _aggregate_points_for_heatmap(df, max_points=20_000):
    """
    Collapse a (lat, lon, population) dataframe onto a coarse grid so a
    Folium HeatMap layer stays a manageable size. Plotting one point per raw
    postcode can mean millions of entries for a large cross-border run
    (e.g. several "Major" freight airports at once), producing a many-tens-
    of-MB HTML file that hangs or fails to render in the browser. Coarsening
    to whatever grid resolution keeps the point count under `max_points`
    fixes that while still giving a faithful visual heatmap.
    """
    if df.empty:
        return df

    for cell_deg in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        lat_bin = (df["lat"] / cell_deg).round().astype(int)
        lon_bin = (df["lon"] / cell_deg).round().astype(int)
        grouped = (
            df.assign(_lat_bin=lat_bin, _lon_bin=lon_bin)
            .groupby(["_lat_bin", "_lon_bin"], as_index=False)
            .agg(lat=("lat", "mean"), lon=("lon", "mean"), population=("population", "sum"))
        )
        if len(grouped) <= max_points:
            break

    return grouped[["lat", "lon", "population"]]


def create_cross_border_map(
    airports,
    outer_radius_km,
    unit,
    covered_df,
    output_file="Cross_Border_Map.html",
    extension_radius_km=0.0,
):
    if not airports:
        raise ValueError("At least one airport must be selected.")

    centre_lat = float(np.mean([a["lat"] for a in airports]))
    centre_lon = float(np.mean([a["lon"] for a in airports]))

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=7, control_scale=True)

    use_miles = unit.lower() in ("mile", "miles", "mi")
    outer_radius_display = outer_radius_km / KM_PER_MILE if use_miles else outer_radius_km
    max_reach_km = outer_radius_km + extension_radius_km
    max_reach_display = max_reach_km / KM_PER_MILE if use_miles else max_reach_km

    for i, airport in enumerate(airports):
        color = HUB_COLORS[i % len(HUB_COLORS)]

        folium.Marker(
            [airport["lat"], airport["lon"]],
            popup=f"<b>{airport['name']} ({airport['code']})</b>",
            tooltip=f"{airport['name']} ({airport['code']})",
            icon=folium.Icon(color=color, icon="plane", prefix="fa"),
        ).add_to(m)

        folium.Circle(
            [airport["lat"], airport["lon"]],
            radius=outer_radius_km * 1000,
            color=color,
            fill=False,
            weight=2,
            popup=f"{airport['name']} catchment radius: {outer_radius_display:.1f} {unit}",
        ).add_to(m)

        if extension_radius_km > 0:
            folium.Circle(
                [airport["lat"], airport["lon"]],
                radius=max_reach_km * 1000,
                color=color,
                fill=False,
                weight=1,
                dash_array="6, 6",
                popup=(
                    f"{airport['name']} max reach with cluster extension: "
                    f"{max_reach_display:.1f} {unit}"
                ),
            ).add_to(m)

    if covered_df is not None and not covered_df.empty:
        heat_df = _aggregate_points_for_heatmap(covered_df)
        heat_data = heat_df[["lat", "lon", "population"]].values.tolist()
        HeatMap(
            heat_data, radius=8, blur=6, max_zoom=11, name="Covered postcodes (by population)"
        ).add_to(m)
        folium.LayerControl().add_to(m)

    m.save(output_file)
    print(f"\nCross-border map saved to {output_file}")


def run_cross_border_optimisation(
    airport_codes,
    outer_radius,
    circle_radius,
    density_threshold,
    radius_unit="miles",
    min_population_per_postcode=None,
    max_density_per_postcode=None,
    cluster_link_radius=None,
    min_cluster_postcodes=1,
    min_cluster_population=0.0,
    extension_radius=0.0,
    create_map_output=True,
    map_filename="Cross_Border_Map.html",
):
    all_airports = load_uk_airports()
    airports = [a for a in all_airports if a["code"] in airport_codes]
    if not airports:
        raise ValueError("At least one airport must be selected.")

    outer_radius_km = convert_to_km(outer_radius, radius_unit)
    circle_radius_km = convert_to_km(circle_radius, radius_unit)
    cluster_link_radius_km = (
        convert_to_km(cluster_link_radius, radius_unit) if cluster_link_radius else None
    )
    extension_radius_km = convert_to_km(extension_radius, radius_unit) if extension_radius else 0.0

    df = load_postcode_data()

    airport_results, covered_postcodes, multi_df, single_df, total_eligible_population = (
        evaluate_cross_border_airports(
            df,
            airports,
            outer_radius_km=outer_radius_km,
            circle_radius_km=circle_radius_km,
            circle_radius_display=circle_radius,
            density_threshold=density_threshold,
            radius_unit=radius_unit,
            min_population_per_postcode=min_population_per_postcode,
            max_density_per_postcode=max_density_per_postcode,
            cluster_link_radius_km=cluster_link_radius_km,
            min_cluster_postcodes=min_cluster_postcodes,
            min_cluster_population=min_cluster_population,
            extension_radius_km=extension_radius_km,
        )
    )

    covered_population = float(single_df["Population"].sum()) if not single_df.empty else 0.0
    coverage_pct = (
        0.0 if total_eligible_population == 0
        else 100.0 * covered_population / total_eligible_population
    )

    if create_map_output:
        if not single_df.empty:
            covered_map_df = single_df.rename(
                columns={"Lat": "lat", "Lon": "lon", "Population": "population"}
            )
        else:
            covered_map_df = pd.DataFrame(columns=["lat", "lon", "population"])

        create_cross_border_map(
            airports,
            outer_radius_km=outer_radius_km,
            unit=radius_unit,
            covered_df=covered_map_df,
            output_file=map_filename,
            extension_radius_km=extension_radius_km,
        )

    return {
        "mode": "cross_border",
        "airports": airport_results,
        "outer_radius": outer_radius,
        "circle_radius": circle_radius,
        "radius_unit": radius_unit,
        "density_threshold": density_threshold,
        "min_population_per_postcode": min_population_per_postcode,
        "max_density_per_postcode": max_density_per_postcode,
        "cluster_link_radius": cluster_link_radius,
        "min_cluster_postcodes": min_cluster_postcodes,
        "min_cluster_population": min_cluster_population,
        "extension_radius": extension_radius,
        "total_population": total_eligible_population,
        "covered_population": covered_population,
        "coverage_pct": coverage_pct,
        "covered_postcodes": covered_postcodes,
        "multi_hub_df": multi_df,
        "single_hub_df": single_df,
    }


if __name__ == "__main__":
    mode = 'Hub Input Radius'  # 'Hub Input Radius', 'Hub Input' OR 'Polygon'

    m25_boundary = [
        (51.572431815210564, 0.28512828001075263),
        (51.58104317801602, 0.27922642344330484),
        # ... rest of your boundary points
    ]

    m60_boundary = [
        (53.4092, -2.1742),
        (53.4055, -2.1921),
        # ... rest of your boundary points
    ]

    fixed_hubs = [
        ("Mothership - B7 5RD", 52.497709, -1.864495),
        ("Oldbury - B69 1DT", 52.492876, -2.023987),
        ("Small Heath - B10 0EU", 52.460967, -1.849933),
        ("Selly Oak - B29 7ES", 52.448578, -1.911831)
    ]

    if mode == "Polygon":
        run_hub_optimisation_polygon(
            boundary_points=m25_boundary,
            num_hubs=15,
            hub_radius=5,
            radius_unit="miles",
            use_optimized=True,
            grid_spacing_km=1.0,
            create_map_output=True,
            map_filename="London-M25-20_hubs-5_mile_radius.html"
        )

    elif mode == 'Hub Input':
        run_fixed_hub_coverage_polygon(
            boundary_points=m25_boundary,
            hubs=fixed_hubs,
            hub_radius=3,
            radius_unit="miles",
            create_map_output=True,
            map_filename="Fixed_Hubs-3_mile_Polygon_Map.html"
        )

    elif mode == "Hub Input Radius":
        run_fixed_hub_coverage(
            centre_lat=52.508502,
            centre_lon=-1.980584,
            hubs=fixed_hubs,
            hub_radius=5,
            city_radius=10,
            radius_unit="miles",
            create_map_output=True,
            map_filename="Birmignham-FixedHubs-5Mile.html"
        )

    else:
        run_hub_optimisation(
            centre_lat=53.479092,
            centre_lon=-2.243147,
            num_hubs=4,
            hub_radius=5,
            city_radius=10,
            radius_unit="miles",
            use_optimized=True,
            grid_spacing_km=1.0,
            create_map_output=True
        )