import io
import logging
from pathlib import Path
import zipfile
import pandas as pd
import geopandas as gdp

# ================================================================================================================================
# Logging
# ================================================================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

# ================================================================================================================================
# Add suburb column
# ================================================================================================================================

# Read files
sites_gdf = gdp.read_file("data/raw/traffic_site/traffic_site_metadata.geojson")
suburbs = gdp.read_file("data/raw/traffic_site/suburbs.zip")
suburbs = suburbs.to_crs(sites_gdf.crs)

# Map a suburb to each site
sites_gdf = gdp.sjoin(sites_gdf, 
                      suburbs[["SAL_NAME21", "geometry"]], 
                      how="left", 
                      predicate="within")

# Rename column names
sites_gdf.rename(columns={
                    "SITE_NO": "site_id",
                    "SITE_NAME": "site_name",
                    "SAL_NAME21": "suburb"}, 
                 inplace=True)

# Drop '(Vic. )' from end of suburb strings
sites_gdf["suburb"] = sites_gdf["suburb"].str.replace(r"\s*\(Vic\.\)$", "", regex=True)

# Sort table by suburb and site id
sites_gdf.sort_values(by=["suburb", "site_id"], inplace=True)

# Save processed file
sites_gdf[["site_id", "site_name", "suburb"]].to_csv("data/processed/site_suburb.csv", index=False)

# ================================================================================================================================
# Process traffic volume files
# ================================================================================================================================

DATA_DIR = Path("data/raw/traffic_volume")
SITE_INFO_PATH = Path("data/processed/site_suburb.csv")
OUTPUT_DIR = Path("data/processed")

CSV_DTYPES = {
    "NB_SCATS_SITE": "int16",
    "NB_DETECTOR": "int8",
    "CT_RECORDS": "int8"}

SELECTED_SUBURBS = ["East Melbourne", "Richmond", "Cremorne", "Jolimont", "Melbourne", 
                    "South Yarra", "Southbank", "South Melbourne", "Fitzroy", "Collingwood"]

def map_suburb_2_site(site_info_path) -> pd.Series:
    """ Load traffic site metadata and return selected site IDS. """

    site_df = pd.read_csv(site_info_path)

    selected_sites = (site_df.loc[site_df["suburb"].isin(SELECTED_SUBURBS), "site_id"]).astype("int16")

    logger.info("Selected %d traffic sites", len(selected_sites))

    return selected_sites

def process_csv_file(csv_file: pd.DataFrame, selected_sites: list) -> pd.DataFrame:
    """ Clean and transform a daily traffic volume CSV file. """

    # Read file
    df = pd.read_csv(csv_file, dtype=CSV_DTYPES)

    # Drop irrelevant columns
    df.drop(columns=["NM_REGION", "QT_VOLUME_24HOUR", "CT_ALARM_24HOUR"],
            inplace=True)
    
    # Rename columns
    df.rename(columns={
                "NB_SCATS_SITE": "site_id",
                "QT_INTERVAL_COUNT": "datetime",
                "NB_DETECTOR": "detector_id",
                "CT_RECORDS": "working_period_count"},
              inplace=True)
    
    # Filter suburbs of interest
    df = df[df["site_id"].isin(selected_sites)]

    volume_cols = [c for c in df.columns if c.startswith("V")]

    # Remove rows with no positive volumes
    df = df[(df[volume_cols] > 0).any(axis=1)]

    # Replace NaN and negative volumes with zeros
    df = df.fillna(0)

    neg_mask = df[volume_cols] < 0
    df["working_period_count"] -= neg_mask.sum(axis=1)
    df[volume_cols] = df[volume_cols].mask(neg_mask, 0)

    # Convert datatypes
    df["datetime"] = pd.to_datetime(df["datetime"])
    df[volume_cols] = df[volume_cols].astype("int16")

    # Aggregate 15-minute volumes to hourly
    hourly_vol = {f"{h+1}": df[volume_cols[h*4 : (h+1)*4]].sum(axis=1) for h in range(24)}
    hourly_df = pd.DataFrame(hourly_vol, index=df.index)
    df = pd.concat([df.drop(columns=volume_cols), hourly_df], axis=1)

    # Transform to long format
    df = df.melt(id_vars=["site_id", "datetime", "detector_id", "working_period_count"],
            value_vars=hourly_df.columns,
            var_name="hour",
            value_name="volume")
    
    return df[["datetime", "hour", "site_id", "detector_id", "volume", "working_period_count"]]

def process_zip_file(zip_path: Path, selected_sites: list) -> pd.DataFrame:
    """ Process a yearly traffic volume ZIP file. """

    dfs = []

    with zipfile.ZipFile(zip_path) as parent_zip:
        for child_zip_name in sorted(parent_zip.namelist()):
            with parent_zip.open(child_zip_name) as child_zip:

                child_bytes = io.BytesIO(child_zip.read())

                with zipfile.ZipFile(child_bytes) as child_zip:
                    for csv_name in sorted(child_zip.namelist()):
                        if csv_name.endswith(".csv"):
                            with child_zip.open(csv_name) as csv_file:

                                df = process_csv_file(csv_file, selected_sites)
                                logger.info("Processed %s", csv_name)

                                dfs.append(df)

    # Combine all CSVs
    return pd.concat(dfs, ignore_index=True)

def main() -> None:

    selected_sites = map_suburb_2_site(SITE_INFO_PATH)

    # Read each ZIP file in directory
    for zip_path in sorted(DATA_DIR.glob("*.zip")):
        
        year = zip_path.stem[-4:]

        df = process_zip_file(zip_path, selected_sites)

        output_path = OUTPUT_DIR / f"traffic_volume_{year}.parquet"

        # Saved processed yearly file
        df.to_parquet(output_path, index=False)
        logger.info("Saved %s", output_path)

if __name__ == "__main__":
    main()