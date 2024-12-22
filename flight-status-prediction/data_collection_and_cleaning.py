import os
from functools import reduce
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit, col, when, count, mean, format_string, substring

BUCKET_NAME = 'flight-status-prediction'
DATA_DIR = 'data'

spark = (
    SparkSession.builder
    .appName("Data")
    .getOrCreate()
)

# Define the range of years for the dataset
years = [2018 + i for i in range(5)]
constants = [i + 1 for i in range(12)]

combined_df = None
dataframes = []

# Iterate over the years and constants to read files
for year in years:
    for constant in constants:
        file_path = f"gs://{BUCKET_NAME}/data/Flights_{year}_{constant}.csv"
        
        # Check if the file exists before attempting to read it
        try:
            df = (
                spark.read.csv(file_path, inferSchema=True, header=True)
                .withColumn("year", lit(year))
            )
            
            dataframes.append(df)
        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")


# Combine all DataFrames into a single DataFrame using unionByName if there are valid DataFrames
if dataframes:
    combined_df = reduce(DataFrame.unionByName, dataframes)
    print(f"Total records: {combined_df.count()}")
else:
    print("No valid files were found to process.")

airline_df = spark.read.csv(f"gs://{BUCKET_NAME}/data/Airlines.csv", inferSchema=True, header=True)

# Clean column names to remove spaces or special characters
combined_df = combined_df.select(
    [col(c).alias(c.strip().replace(" ", "_")) for c in combined_df.columns]
)

# Re-check column names
print("Cleaned Columns:", combined_df.columns)

# Perform the left join with necessary columns
result_df = combined_df.join(
    airline_df.select("Code", "Description"),
    combined_df["Operating_Airline"] == airline_df["Code"],
    "left"
).drop("Code").withColumnRenamed("Description", "Airline")

threshold = 0.8

# Calculate null counts and percentages in a single pass
null_counts = result_df.selectExpr(
    *[f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}" for c in result_df.columns]
).first()

# Calculate total rows
total_rows = result_df.count()

# Identify columns to drop based on the threshold
columns_to_drop = [
    c for c in result_df.columns if null_counts[c] / total_rows > threshold
]

# Drop columns with high null percentages
df_cleaned = result_df.drop(*columns_to_drop)

# Output the number of remaining columns
print(f"Number of remaining columns: {len(df_cleaned.columns)}")

df_cleaned = df_cleaned.drop(*["Diverted", "Quarter", "Marketing_Airline_Network", "Marketing_Airline_Network", "DOT_ID_Marketing_Airline", "IATA_Code_Marketing_Airline", "Flight_Number_Marketing_Airline", "Operating_Airline", "IATA_Code_Operating_Airline", "Tail_Number", "Flight_Number_Operating_Airline", "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID", "OriginStateFips", "OriginStateName", "OriginWac", "DestAirportID", "DestAirportSeqID", "DestCityMarketID", "DestStateFips", "DestStateName", "DestWac", "DepDel15", "DepTimeBlk", "WheelsOff", "WheelsOn", "ArrDel15", "ArrTimeBlk", "DistanceGroup", "__index_level_0__"])


# Calculate the total row count once
total_count = df_cleaned.count()

# Compute null counts for all columns in a single pass
null_counts = df_cleaned.select([
    (count(when(col(c).isNull(), c)) / total_count * 100).alias(c) for c in df_cleaned.columns
])

# Collect the results to the driver
null_columns = [
    (col_name, percentage) 
    for col_name, percentage in null_counts.first().asDict().items() 
    if percentage > 0
]

# Print columns with null values
if null_columns:
    for column, percentage in null_columns:
        print(f"Column '{column}' has {percentage:.2f}% null values.")
else:
    print("No columns have null values.")


df_cleaned = df_cleaned.drop(*["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"])


# drop rows that have null values
df_cleaned = df_cleaned.na.drop()


df_cleaned = df_cleaned.withColumn("DepTime",
                   format_string("%02d:%02d",
                                   (col("DepTime") / 100).cast("int"),  # Extract hours
                                   (col("DepTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

df_cleaned = df_cleaned.withColumn("CRSDepTime",
                   format_string("%02d:%02d",
                                   (col("CRSDepTime") / 100).cast("int"),  # Extract hours
                                   (col("CRSDepTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

df_cleaned = df_cleaned.withColumn("ArrTime",
                   format_string("%02d:%02d",
                                   (col("ArrTime") / 100).cast("int"),  # Extract hours
                                   (col("ArrTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

df_cleaned = df_cleaned.withColumn("CRSArrTime",
                   format_string("%02d:%02d",
                                   (col("CRSArrTime") / 100).cast("int"),  # Extract hours
                                   (col("CRSArrTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

# Show the result
df_cleaned = df_cleaned.withColumn("flight_hour", substring(col("DepTime"), 1, 2).cast("int"))

# Classify the time into Early Morning, Morning, Afternoon, or Night
df_cleaned = df_cleaned.withColumn("time_of_day",
                   when((col("flight_hour") >= 0) & (col("flight_hour") < 6), "Early Morning")
                    .when((col("flight_hour") >= 6) & (col("flight_hour") < 12), "Morning")
                    .when((col("flight_hour") >= 12) & (col("flight_hour") < 16), "Afternoon")
                    .when((col("flight_hour") >= 16) & (col("flight_hour") < 20), "Evening")
                    .when((col("flight_hour") >= 20) & (col("flight_hour") < 24), "Night")
                    .otherwise("Unknown"))

# Drop the temporary flight_hour column if not needed
df_cleaned = df_cleaned.drop("flight_hour")

# Calculate the total row count once
total_count = df_cleaned.count()

# Compute null counts for all columns in a single pass
null_counts = df_cleaned.select([
    (count(when(col(c).isNull(), c)) / total_count * 100).alias(c) for c in df_cleaned.columns
])

# Collect the results to the driver
null_columns = [
    (col_name, percentage) 
    for col_name, percentage in null_counts.first().asDict().items() 
    if percentage > 0
]

# Print columns with null values
if null_columns:
    for column, percentage in null_columns:
        print(f"Column '{column}' has {percentage:.2f}% null values.")
else:
    print("No columns have null values.")

output_path = f"gs://{BUCKET_NAME}/cleaned_data"
df_cleaned.coalesce(1).write.mode('overwrite').parquet(output_path)
