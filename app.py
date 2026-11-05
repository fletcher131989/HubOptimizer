import json
from pathlib import Path

import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from main import (
    run_hub_optimisation_polygon,
    run_fixed_hub_coverage_polygon,
    run_hybrid_optimisation_polygon,
    run_hub_optimisation_polygon_by_coverage,
    run_cross_border_optimisation,
    load_uk_airports,
    add_milk_rounds_to_result,
    convert_to_km,
    KM_PER_MILE,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def download_buttons(multi_df, single_df, key_prefix="top"):
    """Render the two postcode download buttons side-by-side."""
    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="⬇️ Download covered postcodes (all hubs)",
            data=multi_df.to_csv(index=False),
            file_name="covered_postcodes_all_hubs.csv",
            mime="text/csv",
            key=f"{key_prefix}_multi",
            help="Each postcode appears once per hub that covers it.",
            width='stretch',
        )

    with c2:
        st.download_button(
            label="⬇️ Download covered postcodes (nearest hub only)",
            data=single_df.to_csv(index=False),
            file_name="covered_postcodes_nearest_hub.csv",
            mime="text/csv",
            key=f"{key_prefix}_single",
            help="Each postcode is assigned exclusively to its nearest hub.",
            width='stretch',
        )


def geojson_polygon_to_latlon_list(geojson_geometry):
    if not geojson_geometry:
        raise ValueError("No geometry supplied.")
    if geojson_geometry["type"] != "Polygon":
        raise ValueError("Only Polygon geometries are supported.")
    ring = geojson_geometry["coordinates"][0]
    return [(lat, lon) for lon, lat in ring]


def show_overlay(placeholder, message="Optimizing…", subtext="This may take a moment."):
    placeholder.markdown(f"""
    <style>
    .hub-overlay {{
        position: fixed;
        inset: 0;
        background: rgba(14, 17, 23, 0.78);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .hub-overlay-box {{
        background: #1e2130;
        border: 1px solid #3a3f5c;
        border-radius: 16px;
        padding: 2.5rem 3.5rem;
        text-align: center;
        color: #f0f2f6;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        min-width: 280px;
    }}
    .hub-spinner {{
        width: 52px;
        height: 52px;
        border: 5px solid #3a3f5c;
        border-top-color: #e05c5c;
        border-radius: 50%;
        animation: hub-spin 0.85s linear infinite;
        margin: 0 auto 1.4rem;
    }}
    @keyframes hub-spin {{
        to {{ transform: rotate(360deg); }}
    }}
    .hub-overlay-box h3 {{
        margin: 0 0 0.4rem;
        font-size: 1.25rem;
        font-weight: 600;
    }}
    .hub-overlay-box p {{
        margin: 0;
        opacity: 0.65;
        font-size: 0.9rem;
    }}
    </style>
    <div class="hub-overlay">
        <div class="hub-overlay-box">
            <div class="hub-spinner"></div>
            <h3>{message}</h3>
            <p>{subtext}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_results_df(hubs):
    rows = []
    for h in hubs:
        pot_pop = h.get("potential_population") or h["population"]
        overlap = h.get("overlap_population", 0.0)
        overlap_pct = (100.0 * overlap / pot_pop) if pot_pop > 0 else 0.0
        rows.append({
            "Hub":            h.get("hub_name", f"Hub {h['hub_number']}"),
            "Role":           "Mothership" if h.get("is_mothership") else "Hub",
            "Postcode":       h.get("hub_postcode", ""),
            "Latitude":       round(h["lat"], 5),
            "Longitude":      round(h["lon"], 5),
            "New Postcodes":  int(h["postcodes"]),
            "New Population": int(h["population"]),
            "New Households": int(h["households"]),
            "Potential Pop.": int(pot_pop),
            "Overlap Pop.":   int(overlap),
            "Overlap %":      f"{overlap_pct:.1f}%",
        })
    return pd.DataFrame(rows)


def render_results(result, map_filename="user_polygon_result.html"):
    st.markdown("---")
    st.subheader("Results")

    target = result.get("target_coverage_pct")
    n_hubs = len(result["hubs"])

    if target is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Population (area)", f"{result['total_population']:,.0f}")
        c2.metric("Covered Population",      f"{result['covered_population']:,.0f}")
        c3.metric("Coverage Achieved",       f"{result['coverage_pct']:.1f}%",
                  delta=f"{result['coverage_pct'] - target:+.1f}% vs {target:.0f}% target")
        c4.metric("Hubs Required",           str(n_hubs))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Population (area)", f"{result['total_population']:,.0f}")
        c2.metric("Covered Population",      f"{result['covered_population']:,.0f}")
        c3.metric("Coverage",                f"{result['coverage_pct']:.1f}%")
        c4.metric("Hubs Placed",             str(n_hubs))

    for note in result.get("notes", []):
        st.info(note)

    # ← NEW: top-level download buttons
    multi_df  = result.get("multi_hub_df",  pd.DataFrame())
    single_df = result.get("single_hub_df", pd.DataFrame())
    if not multi_df.empty:
        st.markdown("#### Download Covered Postcodes")
        download_buttons(multi_df, single_df, key_prefix="top")

    st.markdown("#### Hub Summary")
    st.dataframe(build_results_df(result["hubs"]), width='stretch', hide_index=True)

    if result.get("milk_rounds_error"):
        st.warning(f"Milk rounds not planned: {result['milk_rounds_error']}")
    if result.get("milk_rounds"):
        render_milk_rounds(result["milk_rounds"])

    st.markdown("#### Top Area Types per Hub")
    for h in result["hubs"]:
        hub_label = h.get("hub_name", f"Hub {h['hub_number']}")
        postcode  = h.get("hub_postcode", "")
        hub_num   = h["hub_number"]

        with st.expander(f"{hub_label}  —  {postcode}"):
            area_types = h.get("top_area_types", {})
            if area_types:
                st.dataframe(
                    pd.DataFrame(list(area_types.items()), columns=["Area Type", "Postcodes"]),
                    hide_index=True,
                    width='stretch'
                )
            else:
                st.write("No net-new coverage for this hub.")

            if not multi_df.empty:
                hub_multi  = multi_df[multi_df["Hub Number"] == hub_num]
                hub_single = single_df[single_df["Hub Number"] == hub_num]

                st.markdown("**Download postcodes for this hub**")
                download_buttons(
                    hub_multi,
                    hub_single,
                    key_prefix=f"hub_{hub_num}",
                )

    st.caption("Map output saved to `user_polygon_result.html`.")
    st.markdown("#### Coverage Map")
    if result.get("coverage_mode", "straight") != "straight":
        st.caption(
            f"Coverage measured along the road network ({result.get('coverage_label', '')}). "
            "Shaded squares show where each hub's covered postcodes are."
        )
    if Path(map_filename).exists():
        st.iframe(map_filename, height=550)
    else:
        st.warning("Map file not found — it may not have been generated.")


def render_milk_rounds(mr):
    st.markdown("#### Milk Rounds")
    s = mr["settings"]
    unit = s["distance_unit"]
    speed_display = s["avg_speed_kmh"] / KM_PER_MILE if unit == "miles" else s["avg_speed_kmh"]
    speed_unit = "mph" if unit == "miles" else "km/h"
    mode_txt = (
        f"up to {s['num_rounds']} round(s)" if s["num_rounds"]
        else "auto (fewest rounds that hit the window)"
    )
    if s.get("travel_source") == "road network":
        travel_txt = (
            f"travel: fastest **road-network** routes at speed limits × "
            f"{s.get('speed_factor', 0.6):.2f}"
        )
    else:
        travel_txt = (
            f"**estimated** travel: straight line × {s['road_factor']:.2f} at "
            f"{speed_display:.0f} {speed_unit}"
        )
    st.caption(
        f"Mode: {mode_txt} · window {s['time_window_min']:.0f} min to reach each hub · "
        f"{s['unload_min']:.0f} min unload per hub · {travel_txt}. Return leg is shown "
        f"but not counted against the window."
    )
    if s.get("unreachable_pairs"):
        st.warning(
            f"{s['unreachable_pairs']} hub-to-hub leg(s) had no road route (a hub may sit on a "
            "disconnected part of the network) and were estimated instead."
        )

    if not mr["rounds"]:
        st.info("No non-mothership hubs to deliver to.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rounds Used", str(mr["rounds_used"]))
    m2.metric("Outbound Drive Time", f"{mr['total_drive_min'] / 60:.1f} h")
    m3.metric(f"Outbound Distance ({unit})", f"{mr['total_drive_distance']:,.1f}")
    m4.metric("Hubs Arriving Late", str(mr["late_stops"]))
    if mr["late_stops"]:
        st.warning(
            f"{mr['late_stops']} hub(s) can't be reached within the "
            f"{s['time_window_min']:.0f}-minute window with these settings — "
            + ("try more milk rounds, or " if s["num_rounds"] else "")
            + "check the speed/window settings (some hubs may be too far from any mothership)."
        )

    st.dataframe(mr["rounds_df"], width="stretch", hide_index=True)
    with st.expander("Stop-by-stop schedule"):
        st.dataframe(mr["stops_df"], width="stretch", hide_index=True)
    st.download_button(
        label="⬇️ Download milk round schedule",
        data=mr["stops_df"].to_csv(index=False),
        file_name="milk_rounds.csv",
        mime="text/csv",
        key="milk_rounds_csv",
    )


def build_cross_border_results_df(airport_results):
    rows = []
    for a in airport_results:
        pot_pop = a["potential_population"] or a["population"]
        overlap_pct = (100.0 * a["overlap_population"] / pot_pop) if pot_pop > 0 else 0.0
        rows.append({
            "Airport":                a["airport_name"],
            "Code":                   a["airport_code"],
            "Relay Of":               a.get("relay_parent_name") or "",
            "Radius":                 a.get("outer_radius", 0.0),
            "Petal Radius":           a.get("circle_radius", 0.0),
            "Overridden":             bool(a.get("is_overridden", False)),
            "New Postcodes":          int(a["postcodes"]),
            "New Population":         int(a["population"]),
            "Eligible Postcodes":     int(a["eligible_postcodes"]),
            "Eligible Population":    int(a["eligible_population"]),
            "Qualifying Postcodes":   int(a["potential_postcodes"]),
            "Overlap Pop.":           int(a["overlap_population"]),
            "Overlap %":              f"{overlap_pct:.1f}%",
            "Excluded (min pop)":     int(a["excluded_by_min_population"]),
            "Excluded (max density)": int(a["excluded_by_max_density"]),
            "Excluded (island)":      int(a.get("excluded_by_island", 0)),
            "Extended Postcodes":     int(a.get("extended_postcodes", 0)),
            "Extended Pop.":          int(a.get("extended_population", 0)),
        })
    return pd.DataFrame(rows)


def render_cross_border_results(result, map_filename="Cross_Border_Map.html"):
    st.markdown("---")
    st.subheader("Results")

    relay_count = sum(1 for a in result["airports"] if a.get("relay_parent_name"))
    airport_count = len(result["airports"]) - relay_count

    cols = st.columns(5 if relay_count else 4)
    c1, c2, c3, c4 = cols[:4]
    c1.metric("Eligible Population (in radius)", f"{result['total_population']:,.0f}")
    c2.metric("Qualifying Population",            f"{result['covered_population']:,.0f}")
    c3.metric("Qualifying %",                     f"{result['coverage_pct']:.1f}%")
    c4.metric("Airports Used",                    str(airport_count))
    if relay_count:
        cols[4].metric("Relay Hubs Used", str(relay_count))

    if "england_scotland_population" in result:
        c5, c6 = st.columns(2)
        c5.metric(
            "England & Scotland Population",
            f"{result['england_scotland_population']:,.0f}",
            help="Total population across every postcode classified as England or "
                 "Scotland (Northern Ireland and Wales excluded), regardless of "
                 "distance from any selected entry point.",
        )
        c6.metric(
            "% of England & Scotland Covered",
            f"{result['coverage_pct_england_scotland']:.2f}%",
            help="Qualifying population as a share of the England & Scotland total above.",
        )

    multi_df = result.get("multi_hub_df_export", result.get("multi_hub_df", pd.DataFrame()))
    single_df = result.get("single_hub_df_export", result.get("single_hub_df", pd.DataFrame()))
    if not multi_df.empty:
        st.markdown("#### Download Covered Postcodes")
        if result.get("relay_hubs"):
            st.caption("Relay hubs' postcodes are reported under their parent airport in these downloads.")
        download_buttons(multi_df, single_df, key_prefix="cb_top")

    st.markdown("#### Airport Summary")
    st.dataframe(build_cross_border_results_df(result["airports"]), width='stretch', hide_index=True)

    st.caption(f"Map output saved to `{map_filename}`.")
    st.markdown("#### Coverage Map")
    if Path(map_filename).exists():
        st.iframe(map_filename, height=550)
    else:
        st.warning("Map file not found — it may not have been generated.")


# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(page_title="Hub Optimizer", layout="wide")
st.title("Hub Optimizer")

tab_hub, tab_cross = st.tabs(["Hub Optimizer", "Cross Border"])

with tab_hub:

    # --------------------------------------------------
    # Sidebar — optimization settings
    # --------------------------------------------------

    st.sidebar.header("Optimization Settings")

    opt_mode = st.sidebar.radio(
        "Optimization mode",
        ["Number of hubs", "Target coverage %"],
        index=0,
        help="Specify the number of hubs directly, or let the optimizer find how many are needed to hit a coverage target.",
    )

    if opt_mode == "Number of hubs":
        num_hubs = st.sidebar.number_input("Total number of hubs", min_value=1, max_value=50, value=4)
    else:
        target_coverage = st.sidebar.slider(
            "Target coverage (%)",
            min_value=10,
            max_value=99,
            value=80,
            step=1,
            help="The optimizer will keep adding hubs until this population coverage percentage is reached.",
        )
        max_hubs = st.sidebar.number_input(
            "Max hubs cap",
            min_value=1,
            max_value=100,
            value=20,
            help="Safety limit — optimization stops here even if the target coverage has not been reached.",
        )

    coverage_mode_label = st.sidebar.selectbox(
        "Coverage measured by",
        ["Straight line", "Road distance", "Drive time"],
        index=0,
        help="Straight line: a simple radius around each hub (ignores rivers etc.).\n\n"
             "Road distance / Drive time: measured along the OpenStreetMap road network, "
             "so e.g. the Thames can only be crossed at bridges and tunnels. The first run "
             "in a new area downloads the road network (can take several minutes); it's "
             "cached for later runs.",
        key="coverage_mode",
    )
    coverage_mode = {
        "Straight line": "straight", "Road distance": "road_distance", "Drive time": "drive_time",
    }[coverage_mode_label]

    max_drive_time = None
    speed_factor = None
    if coverage_mode == "drive_time":
        hub_radius = 5.0  # unused for drive-time coverage
        max_drive_time = st.sidebar.number_input(
            "Max drive time from hub (min)", min_value=1.0, max_value=240.0, value=15.0, step=1.0,
            key="max_drive_time",
        )
    else:
        hub_radius = st.sidebar.number_input(
            "Hub radius" + (" (along roads)" if coverage_mode == "road_distance" else ""),
            min_value=0.1, value=5.0,
        )
    radius_unit = st.sidebar.selectbox("Radius unit", ["miles", "km"], index=0)

    def traffic_speed_factor_input():
        return st.sidebar.number_input(
            "Traffic speed factor",
            min_value=0.1, max_value=1.5, value=0.6, step=0.05,
            help="Drive times use OpenStreetMap speed limits multiplied by this factor. "
                 "~0.5–0.6 is realistic for urban London traffic; ~0.8–0.9 for regional roads.",
            key="road_speed_factor",
        )

    if coverage_mode == "drive_time":
        speed_factor = traffic_speed_factor_input()
    grid_spacing_km = st.sidebar.number_input(
        "Grid spacing (km)",
        min_value=0.25,
        max_value=5.0,
        value=1.0,
        step=0.25,
        help="Distance between candidate hub locations. Larger = faster and less memory. 1 km is a good default.",
    )

    # --------------------------------------------------
    # Sidebar — fixed hubs (number of hubs mode only)
    # --------------------------------------------------

    fixed_hubs_input = []
    mothership_flags = []

    if opt_mode == "Number of hubs":
        st.sidebar.markdown("---")
        st.sidebar.header("Fixed Hubs (optional)")

        max_fixed = int(num_hubs)
        num_fixed = int(st.sidebar.number_input(
            "Number of fixed hubs",
            min_value=0,
            max_value=max_fixed,
            value=0,
            help=f"Pin up to {max_fixed} location(s). Remaining hubs will be optimized automatically.",
        ))

        for i in range(num_fixed):
            st.sidebar.markdown(f"**Fixed Hub {i + 1}**")
            name = st.sidebar.text_input("Name", key=f"fh_name_{i}", value=f"Fixed Hub {i + 1}")
            col_a, col_b = st.sidebar.columns(2)
            lat = col_a.number_input("Lat", key=f"fh_lat_{i}", value=52.4800, format="%.5f", step=0.001)
            lon = col_b.number_input("Lon", key=f"fh_lon_{i}", value=-1.8900, format="%.5f", step=0.001)
            is_ms = st.sidebar.checkbox(
                "Mothership",
                key=f"fh_ms_{i}",
                help="A mothership supplies the other hubs via milk rounds. It still "
                     "covers the population around it like any other hub.",
            )
            fixed_hubs_input.append((name.strip(), float(lat), float(lon)))
            mothership_flags.append(bool(is_ms))

        if num_fixed > 0:
            num_free = int(num_hubs) - num_fixed
            if num_free > 0:
                st.sidebar.caption(f"↳ {num_free} hub(s) will be optimized automatically.")
            else:
                st.sidebar.caption("↳ All hubs are fixed — no optimization will run.")

    mothership_hub_numbers = [i + 1 for i, flag in enumerate(mothership_flags) if flag]
    milk_round_kwargs = None

    if mothership_hub_numbers:
        st.sidebar.markdown("---")
        st.sidebar.header("Milk Rounds")
        st.sidebar.caption(
            "Routes from the mothership(s) to every other hub. Travel times are "
            "estimated for now (straight line × road factor at an average speed)."
        )
        num_rounds = st.sidebar.number_input(
            "Number of milk rounds (0 = auto)",
            min_value=0,
            max_value=50,
            value=0,
            help="Maximum number of rounds (routes) available, shared across all motherships. "
                 "0 = use the fewest rounds that reach every hub within the time window.",
            key="mr_num_rounds",
        )
        mr_c1, mr_c2 = st.sidebar.columns(2)
        time_window = mr_c1.number_input(
            "Window (min)", min_value=5, max_value=1440, value=120, step=15,
            help="Every hub should be reached within this many minutes of leaving the mothership "
                 "(including unloading at earlier hubs on the round).",
            key="mr_time_window",
        )
        unload_time = mr_c2.number_input(
            "Unload (min)", min_value=0, max_value=240, value=15, step=5,
            help="Time spent unloading at each hub.",
            key="mr_unload",
        )
        travel_source = st.sidebar.radio(
            "Travel times from",
            ["Road network", "Estimate"],
            index=0,
            horizontal=True,
            help="Road network: fastest routes along OpenStreetMap roads (respects the "
                 "Thames etc.), drawn on the map. Estimate: straight line × road factor at "
                 "an average speed.",
            key="mr_travel_source",
        )
        speed_unit = "mph" if radius_unit == "miles" else "km/h"
        avg_speed = 20.0 if radius_unit == "miles" else 32.0
        road_factor = 1.3
        if travel_source == "Estimate":
            mr_c3, mr_c4 = st.sidebar.columns(2)
            avg_speed = mr_c3.number_input(
                f"Avg speed ({speed_unit})", min_value=1.0, max_value=80.0,
                value=avg_speed, step=1.0,
                help="Average driving speed used to estimate travel time. ~15–20 mph is typical "
                     "for urban London; higher for regional/motorway routes.",
                key="mr_speed",
            )
            road_factor = mr_c4.number_input(
                "Road factor", min_value=1.0, max_value=3.0, value=1.3, step=0.05,
                help="Road distance ÷ straight-line distance. 1.3 is a common UK average.",
                key="mr_road_factor",
            )
        elif speed_factor is None:
            speed_factor = traffic_speed_factor_input()
        milk_round_kwargs = dict(
            num_rounds=int(num_rounds) or None,
            time_window_min=float(time_window),
            unload_min=float(unload_time),
            avg_speed_kmh=convert_to_km(float(avg_speed), radius_unit),
            road_factor=float(road_factor),
            use_road_network=travel_source == "Road network",
            speed_factor=float(speed_factor) if speed_factor else 0.6,
        )

    # --------------------------------------------------
    # Map
    # --------------------------------------------------

    base_map = folium.Map(location=[52.48, -1.89], zoom_start=10, control_scale=True)
    Draw(
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(base_map)

    st.write("Draw a polygon or rectangle on the map to define your search area.")
    map_data = st_folium(base_map, width=1000, height=600, key=f"map_{st.session_state.get('map_key', 0)}")

    # --------------------------------------------------
    # Extract geometry — no st.stop() used here
    # --------------------------------------------------

    raw_drawings = map_data.get("all_drawings") or []
    drawn_features = [f for f in raw_drawings if f is not None]
    geometry = drawn_features[-1].get("geometry") if drawn_features else None

    if not geometry:
        st.info("No polygon drawn yet. Use the drawing tools on the left side of the map.")

    else:
        try:
            boundary_points = geojson_polygon_to_latlon_list(geometry)
        except Exception as e:
            st.error(str(e))
            boundary_points = None

        if boundary_points is not None:
            n_pts = len(boundary_points)
            st.success(f"✅ Area defined — {n_pts} boundary point{'s' if n_pts != 1 else ''}.")

            with st.expander("View boundary points"):
                preview = (
                    boundary_points[:4] + [["...", "..."]] + boundary_points[-4:]
                    if n_pts > 8 else boundary_points
                )
                st.code(json.dumps(preview, indent=2), language="json")

            if opt_mode == "Number of hubs":
                num_free = int(num_hubs) - len(fixed_hubs_input)
                if not fixed_hubs_input:
                    run_label = "🚀 Run Optimization"
                elif num_free == 0:
                    run_label = f"🗺️ Compute Coverage  ({len(fixed_hubs_input)} fixed hub{'s' if len(fixed_hubs_input) != 1 else ''})"
                else:
                    run_label = f"🚀 Run Optimization  ({len(fixed_hubs_input)} fixed · {num_free} optimized)"
            else:
                run_label = f"🚀 Find Hubs for {target_coverage}% Coverage"

            if st.button(run_label, type="primary"):
                overlay = st.empty()
                uses_road = coverage_mode != "straight" or (
                    milk_round_kwargs is not None and milk_round_kwargs["use_road_network"]
                )
                road_note = (
                    " Road network: the first run in a new area downloads OpenStreetMap "
                    "roads, which can take several minutes."
                    if uses_road else ""
                )

                if opt_mode == "Target coverage %":
                    show_overlay(
                        overlay,
                        message=f"Finding hubs for {target_coverage}% coverage…",
                        subtext=f"Adding hubs until {target_coverage}% of the population is covered (max {int(max_hubs)})." + road_note,
                    )
                elif fixed_hubs_input and num_free == 0:
                    show_overlay(
                        overlay,
                        message="Computing coverage…",
                        subtext=f"Evaluating {len(fixed_hubs_input)} fixed hub(s) across {n_pts} boundary points." + road_note,
                    )
                else:
                    show_overlay(
                        overlay,
                        message="Optimizing…",
                        subtext=f"Placing {int(num_hubs)} hub(s) across {n_pts} boundary points." + road_note,
                    )

                try:
                    if opt_mode == "Target coverage %":
                        result = run_hub_optimisation_polygon_by_coverage(
                            boundary_points=boundary_points,
                            target_coverage_pct=float(target_coverage),
                            hub_radius=hub_radius,
                            radius_unit=radius_unit,
                            max_hubs=int(max_hubs),
                            grid_spacing_km=float(grid_spacing_km),
                            create_map_output=True,
                            map_filename="user_polygon_result.html",
                            coverage_mode=coverage_mode,
                            max_drive_time_min=max_drive_time,
                            speed_factor=float(speed_factor) if speed_factor else 0.6,
                        )
                    elif fixed_hubs_input and num_free > 0:
                        result = run_hybrid_optimisation_polygon(
                            boundary_points=boundary_points,
                            fixed_hubs=fixed_hubs_input,
                            num_free_hubs=num_free,
                            hub_radius=hub_radius,
                            radius_unit=radius_unit,
                            grid_spacing_km=float(grid_spacing_km),
                            map_filename="user_polygon_result.html",
                            coverage_mode=coverage_mode,
                            max_drive_time_min=max_drive_time,
                            speed_factor=float(speed_factor) if speed_factor else 0.6,
                        )
                    elif fixed_hubs_input and num_free == 0:
                        result = run_fixed_hub_coverage_polygon(
                            boundary_points=boundary_points,
                            hubs=fixed_hubs_input,
                            hub_radius=hub_radius,
                            radius_unit=radius_unit,
                            create_map_output=True,
                            map_filename="user_polygon_result.html",
                            coverage_mode=coverage_mode,
                            max_drive_time_min=max_drive_time,
                            speed_factor=float(speed_factor) if speed_factor else 0.6,
                        )
                    else:
                        result = run_hub_optimisation_polygon(
                            boundary_points=boundary_points,
                            num_hubs=int(num_hubs),
                            hub_radius=hub_radius,
                            radius_unit=radius_unit,
                            use_optimized=True,
                            grid_spacing_km=float(grid_spacing_km),
                            create_map_output=True,
                            map_filename="user_polygon_result.html",
                            coverage_mode=coverage_mode,
                            max_drive_time_min=max_drive_time,
                            speed_factor=float(speed_factor) if speed_factor else 0.6,
                        )

                    if milk_round_kwargs and fixed_hubs_input:
                        try:
                            add_milk_rounds_to_result(
                                result,
                                mothership_hub_numbers,
                                hub_radius_km=convert_to_km(hub_radius, radius_unit),
                                map_filename="user_polygon_result.html",
                                **milk_round_kwargs,
                            )
                        except Exception as mr_err:
                            result["milk_rounds_error"] = str(mr_err)

                    overlay.empty()
                    st.session_state["result"] = result
                    st.session_state["map_key"] = st.session_state.get("map_key", 0) + 1
                    st.rerun()
                except Exception as e:
                    overlay.empty()
                    st.error(str(e))

    if "result" in st.session_state:
        render_results(st.session_state["result"])


with tab_cross:

    st.write(
        "Find postcodes within reach of UK air/sea points of entry, for fulfilment "
        "into dense postcodes. Set a catchment radius around each entry point, plus "
        "a local delivery ('petal run') radius used to judge postcode density."
    )

    airports = load_uk_airports()
    freight_levels = ["Major", "Regional", "Limited", "Specialist/business"]
    present_levels = [lvl for lvl in freight_levels if any(a["freight_importance"] == lvl for a in airports)]

    st.markdown("#### Entry Points")

    quick_filter = st.multiselect(
        "Quick-select by freight importance",
        options=present_levels,
        default=["Major"] if "Major" in present_levels else present_levels[:1],
        help="Airports/ports used for freight import. Click 'Apply' to load these into the selection below — "
             "you can still add or remove individual airports afterwards.",
        key="cb_freight_filter",
    )

    airport_options = [
        f'{a["name"]} ({a["code"]}) — Freight: {a["freight_importance"]}, Passenger: {a["passenger_importance"]}'
        for a in airports
    ]
    name_by_option = {opt: a["name"] for opt, a in zip(airport_options, airports)}
    option_by_name = {a["name"]: opt for opt, a in zip(airport_options, airports)}

    # The multiselect below owns "cb_selected_options" via key=. We only ever
    # write to that session_state entry *before* the widget is instantiated
    # (seeding it here, or from the quick-filter button below) -- never after,
    # and never alongside a `default=` on the same widget. Doing both (as this
    # code used to) is a documented Streamlit anti-pattern that desyncs the
    # frontend/backend state and makes every add/remove need two clicks.
    default_excluded_codes = {"BFS", "EDI"}  # Belfast International, Edinburgh
    if "cb_selected_options" not in st.session_state:
        st.session_state["cb_selected_options"] = [
            option_by_name[a["name"]] for a in airports
            if a["freight_importance"] in quick_filter and a["code"] not in default_excluded_codes
        ]

    if st.button("Apply freight filter to selection"):
        st.session_state["cb_selected_options"] = [
            option_by_name[a["name"]] for a in airports if a["freight_importance"] in quick_filter
        ]

    selected_options = st.multiselect(
        "Selected entry points",
        options=airport_options,
        key="cb_selected_options",
        help="Currently UK airports only — ports will be added in future.",
    )
    selected_names = [name_by_option[o] for o in selected_options]

    st.markdown("#### Catchment Settings")

    cb_c1, cb_c2, cb_c3 = st.columns(3)
    with cb_c1:
        cb_radius_unit = st.selectbox("Radius unit", ["miles", "km"], index=0, key="cb_radius_unit")
    with cb_c2:
        outer_radius = st.number_input(
            "Radius around entry point",
            min_value=0.1,
            value=30.0,
            step=1.0,
            help="How far from the airport a postcode can be to be considered at all.",
            key="cb_outer_radius",
        )
    with cb_c3:
        petal_radius = st.number_input(
            "Petal run radius",
            min_value=0.1,
            value=5.0,
            step=0.5,
            help="Local last-mile delivery radius used to judge each postcode's surrounding density.",
            key="cb_petal_radius",
        )

    total_radius = outer_radius + petal_radius
    st.caption(
        f"Total search reach from each entry point: **{total_radius:.1f} {cb_radius_unit}** "
        f"({outer_radius:.1f} catchment + {petal_radius:.1f} petal run)."
    )

    airport_overrides = {}
    with st.expander("Per-airport catchment overrides (optional)", expanded=True):
        st.caption(
            "Give an individual entry point a bigger or smaller catchment / petal run "
            "radius than the settings above. Leave a cell blank to use the default. "
            "Only the catchment radii are overridable — density and route-efficiency "
            "filters stay global for every entry point."
        )
        if selected_names:
            name_to_code = {a["name"]: a["code"] for a in airports}
            override_editor_df = pd.DataFrame({
                "Airport": selected_names,
                "Radius Override": [None] * len(selected_names),
                "Petal Radius Override": [None] * len(selected_names),
            })
            edited_overrides_df = st.data_editor(
                override_editor_df,
                column_config={
                    "Airport": st.column_config.TextColumn("Entry point", disabled=True),
                    "Radius Override": st.column_config.NumberColumn(
                        f"Radius override ({cb_radius_unit})", min_value=0.1, step=1.0
                    ),
                    "Petal Radius Override": st.column_config.NumberColumn(
                        f"Petal radius override ({cb_radius_unit})", min_value=0.1, step=0.5
                    ),
                },
                hide_index=True,
                width="stretch",
                key="cb_airport_overrides_editor",
            )
            for _, row in edited_overrides_df.iterrows():
                entry = {}
                if pd.notna(row["Radius Override"]):
                    entry["outer_radius"] = float(row["Radius Override"])
                if pd.notna(row["Petal Radius Override"]):
                    entry["circle_radius"] = float(row["Petal Radius Override"])
                if entry:
                    airport_overrides[name_to_code[row["Airport"]]] = entry
        else:
            st.caption("Select at least one entry point above to set overrides.")

    relay_hubs = []
    with st.expander("Relay hubs (optional)", expanded=True):
        st.caption(
            "Add an extra entry point that's evaluated exactly like a selected airport "
            "above — its own catchment radius, density and route-efficiency filters, its "
            "own marker on the map — for example a hub placed between two airports to "
            "reach the area in between. The only difference: when you download the "
            "covered-postcode mapping, a relay's postcodes are reported under its parent "
            "airport instead of the relay itself, since the relay is a fulfilment detail, "
            "not a separate customs/entry point."
        )
        if selected_names:
            relay_editor_df = pd.DataFrame({
                "Name": pd.Series(dtype="str"),
                "Lat": pd.Series(dtype="float"),
                "Lon": pd.Series(dtype="float"),
                "Parent Airport": pd.Series(dtype="str"),
            })
            edited_relays_df = st.data_editor(
                relay_editor_df,
                column_config={
                    "Name": st.column_config.TextColumn("Relay name", required=True),
                    "Lat": st.column_config.NumberColumn(
                        "Latitude", min_value=49.0, max_value=61.0, format="%.5f", required=True
                    ),
                    "Lon": st.column_config.NumberColumn(
                        "Longitude", min_value=-8.0, max_value=2.0, format="%.5f", required=True
                    ),
                    "Parent Airport": st.column_config.SelectboxColumn(
                        "Parent airport", options=selected_names, required=True
                    ),
                },
                hide_index=True,
                width="stretch",
                num_rows="dynamic",
                key="cb_relay_editor",
            )
            name_to_code_relay = {a["name"]: a["code"] for a in airports}
            for _, row in edited_relays_df.iterrows():
                if not str(row["Name"] or "").strip():
                    continue
                if pd.isna(row["Lat"]) or pd.isna(row["Lon"]) or not row["Parent Airport"]:
                    continue
                relay_hubs.append({
                    "name": str(row["Name"]).strip(),
                    "lat": float(row["Lat"]),
                    "lon": float(row["Lon"]),
                    "parent_code": name_to_code_relay[row["Parent Airport"]],
                })
        else:
            st.caption("Select at least one entry point above to add relay hubs.")

    st.markdown("#### Density / Volume Filters")

    cb_d1, cb_d2 = st.columns(2)
    with cb_d1:
        min_population_per_postcode = st.number_input(
            "Minimum population per postcode",
            min_value=0.0,
            value=50.0,
            step=10.0,
            help="Postcodes with fewer people than this are excluded. 0 = no filter.",
            key="cb_min_population",
        )
    with cb_d2:
        density_threshold = st.number_input(
            f"Minimum local density (people per sq {cb_radius_unit})",
            min_value=0.0,
            value=200.0,
            step=50.0,
            help=(
                f"A postcode's local density is the population within the petal run radius of it, "
                f"divided by the petal run circle's area. Postcodes below this density are excluded. 0 = no filter."
            ),
            key="cb_density_threshold",
        )

    st.markdown("#### Route Efficiency (remove isolated postcodes)")
    st.caption(
        "Qualifying postcodes are grouped into clusters — two postcodes are linked "
        "(transitively) if they're within the link distance below of each other. "
        "Clusters smaller than the minimums are dropped entirely, even if their "
        "postcodes individually passed the filters above — this stops small, "
        "disconnected pockets from forcing an inefficient dead-leg trip."
    )

    cb_e1, cb_e2, cb_e3 = st.columns(3)
    with cb_e1:
        cluster_link_radius = st.number_input(
            f"Cluster link distance ({cb_radius_unit})",
            min_value=0.0,
            value=0.5,
            step=0.5,
            help="Two qualifying postcodes are treated as part of the same cluster if "
                 "within this distance of each other. 0 = don't check connectivity "
                 "(island filter off). A common starting point is the petal run radius.",
            key="cb_cluster_link_radius",
        )
    with cb_e2:
        min_cluster_postcodes = st.number_input(
            "Minimum postcodes per cluster",
            min_value=1,
            value=11,
            step=5,
            help="Clusters with fewer postcodes than this are dropped entirely. "
                 "1 = keep every cluster, including single isolated postcodes.",
            key="cb_min_cluster_postcodes",
        )
    with cb_e3:
        min_cluster_population = st.number_input(
            "Minimum population per cluster",
            min_value=0.0,
            value=2000.0,
            step=100.0,
            help="Clusters with less total population than this are dropped. 0 = no minimum.",
            key="cb_min_cluster_population",
        )

    cb_ext1, cb_ext2 = st.columns(2)
    with cb_ext1:
        extension_radius = st.number_input(
            f"Extend dense clusters beyond radius, up to ({cb_radius_unit})",
            min_value=0.0,
            value=10.0,
            step=1.0,
            help="A cluster that survives the filters above and reaches the outer radius "
                 "may keep growing past it, through postcodes that are still linked "
                 "(within the cluster link distance) and still meet the density/population "
                 "filters, up to this extra distance. A dense pocket that never reaches the "
                 "outer radius still doesn't qualify on its own. 0 = don't extend "
                 "(hard cutoff at the outer radius). Requires cluster link distance above 0.",
            key="cb_extension_radius",
        )
    with cb_ext2:
        extension_link_radius = st.number_input(
            f"Extension link distance (stricter, {cb_radius_unit})",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="A stricter, usually smaller version of the cluster link distance above, "
                 "used only for connections beyond the outer radius: postcodes inside the "
                 "radius still link at the normal distance, but reaching out into (or moving "
                 "around within) the extension zone requires postcodes to be this close "
                 "together instead. Lets you push the extension distance further to reach a "
                 "genuinely tightly-packed neighbouring area (e.g. another city) without also "
                 "bridging through a loosely-scattered corridor along the way. 0 = extension "
                 "uses the same link distance as the core (no extra strictness).",
            key="cb_extension_link_radius",
        )
    if extension_radius > 0 and cluster_link_radius <= 0:
        st.warning("Extension has no effect while cluster link distance is 0 (island filter off).")
    elif extension_radius > 0:
        if 0 < extension_link_radius < cluster_link_radius:
            ext_note = (
                f" Beyond the outer radius, postcodes must be within "
                f"**{extension_link_radius:.2f} {cb_radius_unit}** of each other to stay linked "
                f"(vs {cluster_link_radius:.2f} inside it)."
            )
        elif extension_link_radius >= cluster_link_radius > 0:
            ext_note = " ⚠️ Extension link distance isn't smaller than the base cluster link distance, so it won't add any extra strictness."
        else:
            ext_note = ""
        st.caption(
            f"Dense, connected clusters may reach up to "
            f"**{outer_radius + extension_radius:.1f} {cb_radius_unit}** from the entry point."
            + ext_note
        )

    run_disabled = len(selected_names) == 0
    if run_disabled:
        st.info("Select at least one entry point to run.")

    if st.button("🚀 Compute Cross-Border Coverage", type="primary", disabled=run_disabled):
        overlay = st.empty()
        show_overlay(
            overlay,
            message="Computing cross-border coverage…",
            subtext=f"Evaluating {len(selected_names)} entry point(s) with a "
                    f"{total_radius:.1f} {cb_radius_unit} total reach.",
        )

        try:
            airport_codes = [
                a["code"] for a in airports if a["name"] in selected_names
            ]
            cb_result = run_cross_border_optimisation(
                airport_codes=airport_codes,
                outer_radius=float(outer_radius),
                circle_radius=float(petal_radius),
                density_threshold=float(density_threshold),
                radius_unit=cb_radius_unit,
                min_population_per_postcode=float(min_population_per_postcode),
                cluster_link_radius=float(cluster_link_radius) if cluster_link_radius > 0 else None,
                min_cluster_postcodes=int(min_cluster_postcodes),
                min_cluster_population=float(min_cluster_population),
                extension_radius=float(extension_radius),
                extension_link_radius=(
                    float(extension_link_radius) if extension_link_radius > 0 else None
                ),
                airport_overrides=airport_overrides,
                relay_hubs=relay_hubs,
                create_map_output=True,
                map_filename="Cross_Border_Map.html",
            )
            overlay.empty()
            st.session_state["cb_result"] = cb_result
            st.rerun()
        except Exception as e:
            overlay.empty()
            st.error(str(e))

    if "cb_result" in st.session_state:
        render_cross_border_results(st.session_state["cb_result"])
