import io
import logging
from pathlib import Path
import zipfile
import geopandas as gdp
from geopy.distance import geodesic
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------
SITE_DATA_DIR = Path("data/raw/traffic_site/traffic_site_metadata.geojson")
VOLUME_DATA_DIR = Path("data/raw/traffic_volume")
SITE_OUTPUT_DIR = Path("data/processed/traffic_site/top_100_sites.csv")
VOLUME_OUTPUT_DIR = Path("data/processed/traffic_volume")

# ---------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------
def process_csv_file(csv_file: pd.DataFrame, selected_sites: pd.DataFrame) -> pd.DataFrame:
    """ Clean and transform a daily traffic volume CSV file. """

    # Read file and change datatypes
    df = pd.read_csv(csv_file)

    # Drop irrelevant columns
    df.drop(columns=["NB_DETECTOR", 
                     "NM_REGION", 
                     "CT_RECORDS",
                     "QT_VOLUME_24HOUR", 
                     "CT_ALARM_24HOUR"], 
            inplace=True)
    
    # Rename columns
    df.rename(columns={"NB_SCATS_SITE": "site_id", 
                       "QT_INTERVAL_COUNT": "date"}, 
              inplace=True)
    
    # Filter top 100 closest sites
    df = df[df["site_id"].isin(selected_sites["site_id"])]

    # Replace NaN and negative volumes with zeros
    volume_cols = [c for c in df.columns if c.startswith("V")]
    df = df.fillna(0)
    df[volume_cols] = df[volume_cols].clip(lower=0)

    # Remove rows with no positive volumes
    df = df[(df[volume_cols] > 0).any(axis=1)]

    # Sum volume by site ID
    volume_df = df[["site_id"] + volume_cols].groupby("site_id", as_index=False).sum()
    date_df = df[["site_id", "date"]].drop_duplicates(subset="site_id")
    df = pd.merge(volume_df, date_df, on="site_id", how="left")
    df = df[["site_id", "date"] + volume_cols]

    # Sum 15-minute volumes by hour
    volume_by_hour = {h+1: df[volume_cols[h*4 : (h+1)*4]].sum(axis=1) for h in range(24)}
    volume_df = pd.DataFrame(volume_by_hour, index=df.index)
    df = pd.concat([df.drop(columns=volume_cols), volume_df], axis=1)
    volume_cols = volume_df.columns

    # Transform to long format
    df = df.melt(id_vars=["site_id", "date"],
                 value_vars=volume_cols,
                 var_name="hour",
                 value_name="volume")
    
    # Change datatypes
    df["site_id"] = df["site_id"].astype("int16")
    df["hour"] = df["hour"].astype("int8")
    df["volume"] = df["volume"].astype("int64")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df[["date", "hour", "site_id", "volume"]]

def process_zip_file(zip_path: Path, selected_sites: pd.DataFrame) -> pd.DataFrame:
    """ Process a yearly traffic volume ZIP file. """

    dfs = []

    with zipfile.ZipFile(zip_path) as parent_zip:
        for child_zip_name in sorted(parent_zip.namelist()):
            with parent_zip.open(child_zip_name) as child_zip:
                with zipfile.ZipFile(io.BytesIO(child_zip.read())) as child_zip:
                    for csv_name in sorted(child_zip.namelist()):
                        if csv_name.endswith(".csv"):
                            with child_zip.open(csv_name) as csv_file:

                                df = process_csv_file(csv_file, selected_sites)
                                dfs.append(df)
                                logger.info("Processed %s", csv_name)

    # Combine all CSVs
    return pd.concat(dfs, ignore_index=True)

# ---------------------------------------------------------------------------------------------------------------
# SITE SELECTION
# ---------------------------------------------------------------------------------------------------------------
site_gdf = gdp.read_file(SITE_DATA_DIR)
site_gdf.rename(columns={"SITE_NO": "site_id", "SITE_NAME": "site_name"}, inplace=True)
site_gdf = site_gdf[["site_id", "site_name", "geometry"]]
site_gdf = site_gdf.to_crs(epsg=4326)

# Calculate road distance between traffic site and destination of interest
DESTINATION = (-37.8231, 144.9820) # Melbourne Park
site_gdf["distance_to"] = site_gdf.geometry.apply(lambda geom: geodesic((geom.y, geom.x), DESTINATION).km)

# Select top 100 closest traffic sites
site_gdf.sort_values(by="distance_to", inplace=True)
top_100_sites = site_gdf.iloc[:100, :3]
top_100_sites.to_csv(SITE_OUTPUT_DIR)

# ---------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------------------------------------------
def main() -> None:

    # Read each ZIP file in directory
    for zip_path in sorted(VOLUME_DATA_DIR.glob("*.zip")):

        year = zip_path.stem[-4:]
        output_path = VOLUME_OUTPUT_DIR / f"traffic_volume_{year}.parquet"

        df = process_zip_file(zip_path, top_100_sites)

        # Saved processed yearly file
        df.to_parquet(output_path,
                      index=False)
        logger.info("Saved %s", output_path)

if __name__ == "__main__":
    main()