# Databricks notebook source
import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Script for preparing data for ML")

# Add arguments
parser.add_argument("--bucket", required=True, type=str, help="Bucket Name", default="flight-status-prediction")  # Positional argument

# Parse arguments
args = parser.parse_args()
bucket_name = args.bucket

# Access arguments
print(f"Bucket name: {bucket_name}")

base_dir = f'gs://{bucket_name}/'
data_dir = os.path.join(base_dir, 'cleaned_data')
ml_data_dir = os.path.join(base_dir, 'ml_data')

# Creating Spark Application
spark = (
    SparkSession.builder
    .appName("Flight Status Prediction")
    .getOrCreate()
)

data = spark.read.parquet(data_dir)

# Keep the records where the Departure Delay is <= 10 Hours
data = data.where('DepDelayMinutes <=600')

# Convert the time columns to minutes
data = data.withColumns({
    'CRSDepTimeMinutes': (
        substring(col('CRSDepTime'), 1, 2).cast('int') * 60
        + substring(col('CRSDepTime'), 4, 2).cast('int')
        ),
    'CRSArrTimeMinutes': (
        substring(col('CRSArrTime'), 1, 2).cast('int') * 60
        + substring(col('CRSArrTime'), 4, 2).cast('int')
    )
})

# Change the DataType of columns that shouldn't be treated as numerical variables
data = data.withColumns({
    'Year': col('Year').cast(StringType()),
    'Month': col('Month').cast(StringType()),
    'DayOfMonth': col('DayOfMonth').cast(StringType()),
    'DayOfWeek': col('DayOfWeek').cast(StringType()),
    'DOT_ID_Operating_Airline': col('DOT_ID_Operating_Airline').cast(StringType()),

    # Remove the states from city
    'OriginCityName': split(col('OriginCityName'), ',')[0],
    'DestCityName': split(col('DestCityName'), ',')[0]
})

# Rename a few columns
data = data.withColumnRenamed('Origin', 'OriginAirport')
data = data.withColumnRenamed('Dest', 'DestAirport')

# The list of columns that needs to be dropped
cols_to_drop = [
    'FlightDate',                                   # The information already captured by Month, Day, Year columns
    'Operated_or_Branded_Code_Share_Partners',      # Too many values in the column
    'DOT_ID_Operating_Airline',                     # Too many values in the column
    'OriginCityName',                               # This information is already present in the Origin Airport
    'DestCityName',                                 # This information is already present in the Destination Airport
    'DepTime',                                      # Calculated from target variable
    'DepDelay',                                     # Calculated from target variable
    'DepartureDelayGroups',                         # Calculated from target variable
    'TaxiOut',                                      # Will not be available at inference time
    'TaxiIn',                                       # Will not be available at inference time
    'ArrTime',                                      # Will not be available at inference time
    'ArrDelay',                                     # Will not be available at inference time
    'ArrDelayMinutes',                              # Will not be available at inference time
    'ArrivalDelayGroups',                           # Will not be available at inference time
    'Cancelled',                                    # There is just one unique value in the column i.e. 0.0
    'ActualElapsedTime',                            # Will not be available at inference time
    'AirTime',                                      # Will not be available at inference time
    'Flights',                                      # There is just one unique value in the column i.e. 1.0
    'DivAirportLandings',                           # I might include this column?
    'Duplicate',                                    # There is just one unique value in the column i.e. N
    'CRSDepTime',                                   # This time is converted into minutes to be used as a numerical column
    'CRSArrTime'                                    # This time is converted into minutes to be used as a numerical column
]

data = data.drop(*cols_to_drop)

# Split the data & save it for modeling
for fraction in [0.25, 0.5, 0.75]:
    sample_data = data.sample(fraction=fraction, seed=42)
    train, test = sample_data.randomSplit([0.8, 0.2], seed=42)
    sampled_data_dir = os.path.join(ml_data_dir, f'{int(fraction * 100)}')
    sample_data.coalesce(1).write.mode('overwrite').parquet(f'{sampled_data_dir}/data')
    train.coalesce(1).write.mode('overwrite').parquet(f'{sampled_data_dir}/train')
    test.coalesce(1).write.mode('overwrite').parquet(f'{sampled_data_dir}/test')

train, test = data.randomSplit([0.8, 0.2], seed=42)
data.coalesce(1).write.mode('overwrite').parquet(f'{ml_data_dir}/100/data')
train.coalesce(1).write.mode('overwrite').parquet(f'{ml_data_dir}/100/train')
test.coalesce(1).write.mode('overwrite').parquet(f'{ml_data_dir}/100/test')
