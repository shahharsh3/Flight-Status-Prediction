This README provides instructions to execute the Machine Learning (ML) pipeline using Spark. The pipeline consists of two steps: data cleaning and model training. Follow the steps below to run the pipeline on a GCP cluster.

---

## Prerequisites
1. Verify that the required Python scripts (`ml_data_cleaning.py` and `ml.py`) are present and accessible.
2. Replace placeholders such as `<BUCKET-NAME>` and other parameters with actual values.

---

## Step 1: Data Collection and Cleaning
The first step involves cleaning the data stored in a specified bucket.

### Command:
```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --driver-memory 4G \
  --executor-memory 2G \
  --num-executors 8 \
  data_collection_and_cleaning.py \
  --bucket <BUCKET-NAME>
```

### Description:
- `--master yarn`: Specifies the YARN cluster as the Spark master.
- `--deploy-mode client`: Specifies the deployment mode of spark.
- `--driver-memory`: Allocates 4 GB of memory for the driver.
- `--executor-memory`: Allocates 2 GB of memory for each executor.
- `--num-executors`: Configures the number of executors to 8.
- `ml_data_cleaning.py`: The script to clean the data.
- `--bucket`: Specifies the bucket name containing the raw data.

## Step 2: Data Cleaning (Part II)

### Command:
```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --driver-memory 4G \
  --executor-memory 2G \
  --num-executors 8 \
  ml_data_cleaning.py \
  --bucket <BUCKET-NAME>
```

### Description:
- `--master yarn`: Specifies the YARN cluster as the Spark master.
- `--deploy-mode client`: Specifies the deployment mode of spark.
- `--driver-memory`: Allocates 4 GB of memory for the driver.
- `--executor-memory`: Allocates 2 GB of memory for each executor.
- `--num-executors`: Configures the number of executors to 8.
- `ml_data_cleaning.py`: The script to clean the data.
- `--bucket`: Specifies the bucket name containing the raw data.

## Step 2: Model Training

### Command:
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 8G \
  --executor-memory 6G \
  --num-executors <NUM-EXECUTORS> \
  --conf spark.sql.shuffle.partitions=32 \
  ml.py \
  --bucket <BUCKET-NAME> \
  --frac <DATA-FRAC> \
  --n_exec <NUM-EXECUTORS>
```

### Description:
- `--master yarn`: Specifies the YARN cluster as the Spark master.
- `--deploy-mode cluster`: Runs the driver program in the cluster.
- `--driver-memory`: Allocates 8 GB of memory for the driver.
- `--executor-memory`: Allocates 6 GB of memory for each executor.
- `--num-executors`: Configures the number of executors (replace `<NUM-EXECUTORS>` with the desired number).
- `--conf spark.sql.shuffle.partitions=32`: Sets the number of partitions for shuffle operations to 32.
- `ml.py`: The script to train the machine learning model.
- `--bucket`: Specifies the bucket name containing the cleaned data.
- `--frac`: Determines the fraction of the data to use for training (replace `<DATA-FRAC>` with a value  from this list: [25, 50, 75, 100]).
- `--n_exec`: Specifies the number of executors (same as `<NUM-EXECUTORS>`).

