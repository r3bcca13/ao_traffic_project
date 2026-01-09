from datetime import date
from itertools import chain
import io
import zipfile
import holidays
import pandas as pd
from pyspark.sql import SparkSession, functions as F, Column
from pyspark.sql.functions import udf
from pyspark.sql.types import *

AO_DATES = {
    2020: (date(2020, 1, 20), date(2020, 2, 2)),
    2021: (date(2021, 2, 8), date(2021, 2, 21)),
    2022: (date(2022, 1, 17), date(2022, 1, 30)),
    2023: (date(2023, 1, 16), date(2023, 1, 29)),
    2024: (date(2024, 1, 14), date(2024, 1, 28)),
    2025: (date(2025, 1, 12), date(2025, 1, 26))
}

COVID_LOCKDOWN_DATES = {
    1: (date(2020, 3, 26), date(2020, 5, 12)),
    2: (date(2020, 8, 7), date(2020, 10, 27)),
    3: (date(2021, 2, 12), date(2021, 2, 17)),
    4: (date(2021, 5, 27), date(2021, 6, 10)),
    5: (date(2021, 7, 15), date(2021, 7, 27)),
    6: (date(2021, 8, 5), date(2021, 10, 21))}

SCHOOL_HOLIDAY_DATES = {
    2020: [
        (date(2020, 3, 28), date(2020, 4, 13)),
        (date(2020, 6, 27), date(2020, 7, 12)),
        (date(2020, 9, 19), date(2020, 9, 30)),
        (date(2020, 12, 19), date(2021, 1, 26))],
    2021: [
        (date(2021, 4, 2), date(2021, 4, 18)),
        (date(2021, 6, 26), date(2021, 7, 11)),
        (date(2021, 9, 18), date(2021, 10, 3)),
        (date(2021, 12, 17), date(2022, 1, 26))],
    2022: [
        (date(2022, 4, 9), date(2022, 4, 25)),
        (date(2022, 6, 25), date(2022, 7, 10)),
        (date(2022, 9, 16), date(2022, 10, 2)),
        (date(2022, 12, 20), date(2023, 1, 26))],
    2023: [
        (date(2023, 4, 7), date(2023, 4, 23)),
        (date(2023, 6, 24), date(2023, 7, 9)),
        (date(2023, 9, 16), date(2023, 10, 1)),
        (date(2023, 12, 20), date(2024, 1, 27))],
    2024: [
        (date(2024, 3, 29), date(2024, 4, 14)),
        (date(2024, 6, 29), date(2024, 7, 14)),
        (date(2024, 9, 21), date(2024, 10, 6)),
        (date(2024, 12, 20), date(2025, 1, 26))],
    2025: [
        (date(2025, 4, 5), date(2025, 4, 21)),
        (date(2025, 7, 5), date(2025, 7, 20)),
        (date(2025, 9, 20), date(2025, 10, 5)),
        (date(2025, 12, 20), date(2026, 1, 26))]
}

def is_during_period(dates_dict: dict) -> Column:
    """ Return a Spark Column of booleans indicating whether date is within any of the given date ranges. """

    column_val = None
    date_ranges = dates_dict.values()
    
    if dates_dict == SCHOOL_HOLIDAY_DATES:
        date_ranges = list(chain.from_iterable(date_ranges))

    for start, end in date_ranges:
        
        condition = F.col("datetime").between(start, end)
        column_val = condition if column_val is None else column_val | condition

    return column_val

def find_ao_day_num(date: date) -> int:
    """ Find AO day number based on date. """

    ao_dates = AO_DATES[date.year]

    if ao_dates[0] <= date <= ao_dates[1]:
        difference = date - ao_dates[0]
        return difference.days + 1
    else:
        return 0

# ================================================================================================================================
# Process other data tables
# ================================================================================================================================

# Overseas visitor data 

overseas_visitors_df = pd.read_excel("data/raw/other/overseas_visitor_data.xlsx", sheet_name="Data1")

overseas_visitors_df = overseas_visitors_df.iloc[9:, [0, 2]]

overseas_visitors_df.rename(columns={"Unnamed: 0": "date", 
                             "Number of movements ;  Short-term Visitors arriving ;  Vic ;": "overseas_visitor_count"},
                            inplace=True)

# Extract separate date columns
overseas_visitors_df["date"] = pd.to_datetime(overseas_visitors_df["date"], errors="coerce")
overseas_visitors_df["year"] = overseas_visitors_df["date"].dt.year
overseas_visitors_df["month"] = overseas_visitors_df["date"].dt.month

# Filter counts after 2020
overseas_visitors_df = overseas_visitors_df[overseas_visitors_df["year"] >= 2020]

# =================================================================================================================

# Site lane count data

outer_zip_path = "data/raw/traffic_volume/traffic_signal_volume_data_2025.zip"

with zipfile.ZipFile(outer_zip_path) as outer_zip:

    first_inner_zip_name = sorted(outer_zip.namelist())[0]

    with outer_zip.open(first_inner_zip_name) as inner_zip_bytes:
        with zipfile.ZipFile(io.BytesIO(inner_zip_bytes.read())) as inner_zip:

            first_csv_name = sorted(inner_zip.namelist())[0]
            
            with inner_zip.open(first_csv_name) as csv_file:

                lane_count_df = pd.read_csv(csv_file)

                # Remove rows with no positive volumes
                volume_cols = [c for c in lane_count_df.columns if c.startswith("V")]
                lane_count_df = lane_count_df[(lane_count_df[volume_cols] > 0).any(axis=1)]

                lane_count_df.rename(columns={"NB_SCATS_SITE": "site_id", "NB_DETECTOR": "lane_count"}, inplace=True)

                # Count lanes per traffic site
                lane_count_df = lane_count_df.groupby(by="site_id", as_index=False).count()

# ================================================================================================================================
# Read processed files in Spark
# ================================================================================================================================

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("data/processed/traffic_volume")

# Change schema 

new_schema = StructType([
    StructField("datetime", DateType(), True),
    StructField("hour", IntegerType(), True),
    StructField("site_id", IntegerType(), True),
    StructField("detector_id", IntegerType(), True),
    StructField("volume", IntegerType(), True),
    StructField("working_period_count", IntegerType(), True)
])

# Reconcile DataFrame to new schema
df = df.to(new_schema)

# ================================================================================================================================
# Feature engineering
# ================================================================================================================================

vic_holidays = list(holidays.Australia(years=[2020, 2021, 2022, 2023, 2024, 2025], state='VIC').keys())
ao_day_udf = udf(find_ao_day_num, IntegerType())

# Create date columns
df = df.withColumn("year", F.year(F.col("datetime")))
df = df.withColumn("month", F.month(F.col("datetime")))
df = df.withColumn("day", F.day(F.col("datetime")))
df = df.withColumn("day_of_week", F.weekday(F.col("datetime")))
df = df.withColumn("is_weekend", F.col("day_of_week") > 4)
df = df.withColumn("day_of_year", F.dayofyear(F.col("datetime")))
df = df.withColumn("week_of_year", F.weekofyear(F.col("datetime")))
df = df.withColumn("traffic_period",
   F.when((F.col("hour") >= 5) & (F.col("hour") < 9), "morning peak")
    .when((F.col("hour") >= 9) & (F.col("hour") < 12), "late morning")
    .when((F.col("hour") >= 12) & (F.col("hour") < 16), "afternoon")
    .when((F.col("hour") >= 16) & (F.col("hour") < 19), "evening peak")
    .when((F.col("hour") >= 19) & (F.col("hour") < 22), "evening")
    .otherwise("night"))
df = df.withColumn("is_during_ao", is_during_period(AO_DATES))
df = df.withColumn("is_during_lockdown", is_during_period(COVID_LOCKDOWN_DATES))
df = df.withColumn("is_during_school_holiday", is_during_period(SCHOOL_HOLIDAY_DATES))
df = df.withColumn("is_public_holiday", F.col("datetime").isin(vic_holidays))
df = df.withColumn("ao_day_num", ao_day_udf(F.col("datetime")))

# Create overseas visitor count column
overseas_visitors_sdf = spark.createDataFrame(overseas_visitors_df[["year", "month", "overseas_visitor_count"]])
df = df.join(overseas_visitors_sdf, on=["year", "month"], how="left")

# Create lane count column
lane_count_sdf = spark.createDataFrame(lane_count_df[["site_id", "lane_count"]])
df = df.join(lane_count_sdf, on=["site_id"], how="left")

df.show(10)
df.printSchema()
print(f"Length of table: {df.count()}")

# dist_to_ao
# during_road_closure