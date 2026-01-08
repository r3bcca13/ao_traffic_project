import pandas as pd
import geopandas as gdp

# Add suburb column

sites_gdf = gdp.read_file("data/raw/traffic_site_metadata.geojson")

suburbs = gdp.read_file("data/raw/suburbs.zip")

suburbs = suburbs.to_crs(sites_gdf.crs)

# Map a suburb to each site
sites_gdf = gdp.sjoin(sites_gdf, 
                      suburbs[["SAL_NAME21", "geometry"]], 
                      how="left", 
                      predicate="within")

# Rename column names
sites_gdf.rename(columns={"SITE_NO": "site_id",
                          "SITE_NAME": "site_name",
                          "SAL_NAME21": "suburb"}, 
                 inplace=True)

# Drop (Vic. ) from end of suburb strings
sites_gdf["suburb"] = sites_gdf["suburb"].str.replace(r"\s*\(Vic\.\)$", "", regex=True)

sites_gdf.sort_values(by=["suburb", "site_id"], inplace=True)

# Save processed file
sites_gdf[["site_id", "site_name", "suburb"]].to_csv("data/processed/site_suburb.csv", index=False)