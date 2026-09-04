"""
Regenerate postcode_parquet/*.parquet directly from Redshift, replacing the
CSV-based build_postcode_parquet_data() pipeline in main.py.

Requires a .env file (gitignored, never committed) with:
    REDSHIFT_USER
    REDSHIFT_HOST
    REDSHIFT_PORT
    REDSHIFT_PASSWORD
    REDSHIFT_DBNAME

Run directly: python populate_parquet_files.py
"""

import os
from pathlib import Path

import pandas as pd
import pg8000.dbapi
from dotenv import load_dotenv

load_dotenv()

QUERY = os.environ.get(
    "QUERY", "SELECT * FROM reporting.uk_postcode_and_demographics"
)

OUTPUT_FOLDER = Path("postcode_parquet")
CHUNK_SIZE = 250_000

# reporting.uk_postcode_and_demographics -> the schema main.py's
# load_postcode_data() / build_postcode_parquet_data() expect.
COLUMN_MAP = {
    "pcd": "postcode",
    "x_latitude": "lat",
    "y_longitude": "lon",
    "total_persons": "population",
    "occupied_households": "households",
    "oac_subgroup_code": "oac_subgroup_code",
    "oac_goup_name": "oac_group_name",  # typo in the source table, kept on the Redshift side only
    "oac_subgroup_name": "area_type",
    "oac_supergroup_name": "oac_supergroup_name",
}

OUTPUT_COLUMNS = [
    "postcode", "lat", "lon", "oac_subgroup_code",
    "population", "households", "oac_group_name", "area_type", "oac_supergroup_name",
]


def get_connection():
    return pg8000.dbapi.connect(
        user=os.environ["REDSHIFT_USER"],
        password=os.environ["REDSHIFT_PASSWORD"],
        host=os.environ["REDSHIFT_HOST"],
        port=int(os.environ.get("REDSHIFT_PORT", 5439)),
        database=os.environ["REDSHIFT_DBNAME"],
        ssl_context=True,
    )


def _clean_chunk(df):
    df = df.rename(columns=COLUMN_MAP)
    df = df[[c for c in OUTPUT_COLUMNS if c in df.columns]].copy()

    # The source pcd column has no space (e.g. "AB101AJ"); reinsert the
    # standard UK postcode space before the 3-character inward code so
    # postcodes match the existing data's formatting (e.g. "AB10 1AJ").
    postcode = df["postcode"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    has_inward = postcode.str.len() > 3
    df["postcode"] = postcode.where(~has_inward, postcode.str[:-3] + " " + postcode.str[-3:])

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)
    df["households"] = pd.to_numeric(df["households"], errors="coerce").fillna(0)

    for col in ("oac_subgroup_code", "oac_group_name", "area_type", "oac_supergroup_name"):
        if col in df.columns:
            df[col] = df[col].fillna("Other")
            if col != "oac_subgroup_code":
                df[col] = df[col].replace("", "Other")

    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(49, 61) & df["lon"].between(-8, 2)]

    return df[[c for c in OUTPUT_COLUMNS if c in df.columns]]


def populate_parquet_files(chunk_size=CHUNK_SIZE, query=QUERY):
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    for old_file in OUTPUT_FOLDER.glob("*.parquet"):
        old_file.unlink()

    conn = get_connection()
    total_rows = 0
    part = 0

    try:
        cursor = conn.cursor()
        print(f"Running query against Redshift:\n  {query}")
        cursor.execute(query)
        columns = [d[0] for d in cursor.description]

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break

            df = pd.DataFrame(rows, columns=columns)
            df = _clean_chunk(df)

            output_file = OUTPUT_FOLDER / f"postcodes_part_{part:03d}.parquet"
            df.to_parquet(output_file, index=False)

            total_rows += len(df)
            print(f"Wrote {output_file.name}: {len(df):,} rows (running total {total_rows:,})")
            part += 1
    finally:
        conn.close()

    print(f"\nDone. {part} parquet file(s), {total_rows:,} rows written to {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    populate_parquet_files()
