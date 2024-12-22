#!/bin/bash

# Default bucket name
DEFAULT_BUCKET_NAME="example-cluster-00"

# Assign bucket name from input argument or use default
BUCKET_NAME=${1:-$DEFAULT_BUCKET_NAME}

echo "Using bucket: gs://$BUCKET_NAME"

# Step 1: Create temporary directory and download data from Kaggle
echo "Creating temporary directory and downloading data..."
mkdir -p ~/tmp
curl -L -o ~/tmp/data.zip https://www.kaggle.com/api/v1/datasets/download/robikscube/flight-delay-dataset-20182022

# Step 2: Unzip the data
echo "Unzipping data..."
unzip ~/tmp/data.zip -d /tmp/data

# Step 3: Copy the unzipped data to the specified GCP bucket
echo "Copying data to GCS bucket: gs://$BUCKET_NAME/data/"
gsutil cp -r /tmp/data/*.parquet gs://"$BUCKET_NAME"/data/
gsutil cp -r /tmp/data/Airlines.csv gs://"$BUCKET_NAME"/data/

# Step 4: Clean up by removing the temporary data from VM
echo "Cleaning up temporary files..."
rm -r ~/tmp /tmp/data

echo "Data uploaded successfully to gs://$BUCKET_NAME/data/ and temporary files cleaned up."
gsutil ls gs://"$BUCKET_NAME"/data/
