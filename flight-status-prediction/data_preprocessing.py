import os
from functools import reduce
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Set default bucket name if not provided in environment variables
BUCKET_NAME = os.environ.get('bucket_name', 'example-cluster-00')

# Initialize Spark session
spark = (
    SparkSession.builder
    .appName("Data Processing")
    .getOrCreate()
)

# Define the range of years for the dataset
years = [2018 + i for i in range(5)]

# Read each year's Parquet file, add a "year" column, and store it in a dictionary
year_wise_data = {
    year: spark.read.parquet(f"gs://{BUCKET_NAME}/data/Combined_Flights_{year}.parquet")
            .withColumn("year", F.lit(year))
    for year in years
}

# Combine all DataFrames into a single DataFrame using unionByName
combined_df = reduce(DataFrame.unionByName, year_wise_data.values())

# Display the first few rows of the combined DataFrame to verify
combined_df.show()

# Write the combined DataFrame to a single Parquet file in GCS
output_path = f"gs://{BUCKET_NAME}/data/flights.parquet"

(
    combined_df.coalesce(1)
    .write.mode("overwrite")
    .parquet(output_path)
)

print(f"Combined dataset written to {output_path}")
