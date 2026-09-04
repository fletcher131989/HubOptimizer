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

    # ← NEW: top-level download buttons
    multi_df  = result.get("multi_hub_df",  pd.DataFrame())
    single_df = result.get("single_hub_df", pd.DataFrame())
    if not multi_df.empty:
        st.markdown("#### Download Covered Postcodes")
        download_buttons(multi_df, single_df, key_prefix="top")

    st.markdown("#### Hub Summary")
    st.dataframe(build_results_df(result["hubs"]), width='stretch', hide_index=True)

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
    if Path(map_filename).exists():
        st.iframe(map_filename, height=550)
    else:
        st.warning("Map file not found — it may not have been generated.")


def build_cross_border_results_df(airport_results):
    rows = []
    for a in airport_results:
        pot_pop = a["potential_population"] or a["population"]
        overlap_pct = (100.0 * a["overlap_population"] / pot_pop) if pot_pop > 0 else 0.0
        rows.append({
            "Airport":                a["airport_name"],
            "Code":                   a["airport_code"],
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
        })
    return pd.DataFrame(rows)


def render_cross_border_results(result, map_filename="Cross_Border_Map.html"):
    st.markdown("---")
    st.subheader("Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eligible Population (in radius)", f"{result['total_population']:,.0f}")
    c2.metric("Qualifying Population",            f"{result['covered_population']:,.0f}")
    c3.metric("Qualifying %",                     f"{result['coverage_pct']:.1f}%")
    c4.metric("Airports Used",                    str(len(result["airports"])))

    multi_df = result.get("multi_hub_df", pd.DataFrame())
    single_df = result.get("single_hub_df", pd.DataFrame())
    if not multi_df.empty:
        st.markdown("#### Download Covered Postcodes")
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

    hub_radius = st.sidebar.number_input("Hub radius", min_value=0.1, value=5.0)
    radius_unit = st.sidebar.selectbox("Radius unit", ["miles", "km"], index=0)
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
            fixed_hubs_input.append((name.strip(), float(lat), float(lon)))

        if num_fixed > 0:
            num_free = int(num_hubs) - num_fixed
            if num_free > 0:
                st.sidebar.caption(f"↳ {num_free} hub(s) will be optimized automatically.")
            else:
                st.sidebar.caption("↳ All hubs are fixed — no optimization will run.")

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

                if opt_mode == "Target coverage %":
                    show_overlay(
                        overlay,
                        message=f"Finding hubs for {target_coverage}% coverage…",
                        subtext=f"Adding hubs until {target_coverage}% of the population is covered (max {int(max_hubs)}).",
                    )
                elif fixed_hubs_input and num_free == 0:
                    show_overlay(
                        overlay,
                        message="Computing coverage…",
                        subtext=f"Evaluating {len(fixed_hubs_input)} fixed hub(s) across {n_pts} boundary points.",
                    )
                else:
                    show_overlay(
                        overlay,
                        message="Optimizing…",
                        subtext=f"Placing {int(num_hubs)} hub(s) across {n_pts} boundary points.",
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
                        )
                    elif fixed_hubs_input and num_free == 0:
                        result = run_fixed_hub_coverage_polygon(
                            boundary_points=boundary_points,
                            hubs=fixed_hubs_input,
                            hub_radius=hub_radius,
                            radius_unit=radius_unit,
                            create_map_output=True,
                            map_filename="user_polygon_result.html",
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
                        )
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
    if "cb_selected_options" not in st.session_state:
        st.session_state["cb_selected_options"] = [
            option_by_name[a["name"]] for a in airports if a["freight_importance"] in quick_filter
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

    st.markdown("#### Density / Volume Filters")

    cb_d1, cb_d2 = st.columns(2)
    with cb_d1:
        min_population_per_postcode = st.number_input(
            "Minimum population per postcode",
            min_value=0.0,
            value=0.0,
            step=10.0,
            help="Postcodes with fewer people than this are excluded. 0 = no filter.",
            key="cb_min_population",
        )
    with cb_d2:
        density_threshold = st.number_input(
            f"Minimum local density (people per sq {cb_radius_unit})",
            min_value=0.0,
            value=0.0,
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
            value=6,
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
