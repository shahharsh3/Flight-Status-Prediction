# Databricks notebook source
import os
import json
import gcsfs
import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
    MinMaxScaler
)

from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Script for training ML Models")

# Add arguments
parser.add_argument("--bucket", required=True, type=str, help="Bucket Name", default="flight-status-prediction")
parser.add_argument("--frac", required=True, type=str, help="Data Size", default="25")
parser.add_argument("--n_exec", required=True, type=str, help="Number of Executors", default="2")

# Parse arguments
args = parser.parse_args()
bucket_name = args.bucket
sample_size = args.frac
num_executors = args.n_exec

# Access arguments
print(f"Bucket name: {bucket_name}")
print(f"Fraction of Total Data Used: {sample_size}")
print(f"Number of Executors: {num_executors}")

base_dir = f'gs://{bucket_name}/'
ml_data_dir = os.path.join(base_dir, 'ml_data')
results_file = os.path.join(base_dir, 'results.json')

spark = (
    SparkSession.builder
    .appName("ML Pipeline")
    .getOrCreate()
)

# Load the train & test data
train = spark.read.parquet(os.path.join(ml_data_dir, f'{sample_size}/train'))
test = spark.read.parquet(os.path.join(ml_data_dir, f'{sample_size}/test'))

# Cache the train data for good performance
train.cache()


# Machine Learning Pipeline:
def calculate_evaluation_metrics(evaluator, predictions):
    """
    Calculate and return evaluation metrics for regression model predictions.

    Parameters:
    -----------
    evaluator : pyspark.ml.evaluation.RegressionEvaluator
        An instance of the RegressionEvaluator used to compute the metrics.

    predictions : pyspark.sql.DataFrame
        The DataFrame containing the predicted and actual values.

    Returns:
    --------
    dict
        A dictionary containing the following evaluation metrics:
        - 'rmse': Root Mean Squared Error
        - 'mse': Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'r2': R-squared value
        - 'explained_variance': Explained Variance

    Notes:
    ------
    This function assumes that the evaluator's `labelCol` and `predictionCol` 
    are correctly set to match the column names in the predictions DataFrame.

    Example:
    --------
    >>> from pyspark.ml.evaluation import RegressionEvaluator
    >>> evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
    >>> metrics = calculate_evaluation_metrics(evaluator, predictions)
    >>> print(metrics)
    {'rmse': 1.23, 'mse': 1.5, 'mae': 0.9, 'r2': 0.85, 'explained_variance': 0.86}
    """
    # Calculate metrics using the evaluator
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    explained_variance = evaluator.evaluate(predictions, {evaluator.metricName: "var"})

    # Construct and return the results dictionary
    results = {
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_variance
    }

    return results


def save_models(model_dir, cv_model):
    """
    Save all submodels and the best model from a CrossValidatorModel to the specified directory.

    Parameters:
    -----------
    model_dir : str
        Directory path where models will be saved.
    
    cv_model : pyspark.ml.tuning.CrossValidatorModel
        The CrossValidatorModel containing submodels and the best model.

    Returns:
    --------
    None
        Saves the models to the specified paths and prints confirmation messages.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save all submodels
    for fold_idx, fold_models in enumerate(cv_model.subModels):
        for model_idx, model in enumerate(fold_models):
            model_path = os.path.join(model_dir, f'fold{fold_idx}_model{model_idx}')
            model.write().save(model_path)

    print(f"All models saved to {model_dir}")

    # Save the best model separately
    best_model_path = os.path.join(model_dir, f'best_model')
    cv_model.bestModel.save(best_model_path)
    print(f"Best model saved to {best_model_path}")


def update_json_file(gcs_path, key, value):
    """
    Update a JSON file stored in Google Cloud Storage (GCS) using gcsfs.

    Parameters:
    -----------
    gcs_path : str
        Full GCS path to the JSON file (e.g., 'gs://bucket_name/file.json').
    key : str
        Key to update in the JSON file.
    value : any
        Value to associate with the key in the JSON file.

    Returns:
    --------
    None
        Updates the JSON file in GCS.
    """
    fs = gcsfs.GCSFileSystem()

    # Read the existing file or create a new one
    try:
        with fs.open(gcs_path, "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        json_data = {}

    # Update the JSON data
    json_data[key] = value

    # Write back the updated file
    with fs.open(gcs_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Updated '{gcs_path}' with key '{key}'.")


target_col = 'DepDelayMinutes'

categorical_cols = [
    field.name for field in train.schema.fields if isinstance(field.dataType, StringType)
]

continuous_cols = [
    field.name for field in train.schema.fields if isinstance(
        field.dataType, (IntegerType, LongType, DoubleType)
    ) and field.name != target_col
]

indexed_cols = [f'{col}_i' for col in categorical_cols]
onehot_cols = [f'{col}_onehot' for col in categorical_cols]

# Linear Regression Pipeline
model_dir = os.path.join(base_dir, f'models/lr/{sample_size}/{num_executors}')

continuous_assembler = VectorAssembler(
    inputCols=continuous_cols,
    outputCol='continuous'
)

continuous_scaler = MinMaxScaler(
    inputCol='continuous',
    outputCol='continuous_scaled'
)

string_indexer = StringIndexer(
    inputCols=categorical_cols,
    outputCols=indexed_cols,
    handleInvalid='keep'
)

onehot_encoder = OneHotEncoder(
    inputCols=indexed_cols,
    outputCols=onehot_cols
)

lr_assembler = VectorAssembler(
    inputCols=onehot_cols + ['continuous_scaled'],
    outputCol='features'
)

lr = LinearRegression(
    featuresCol='features',
    labelCol=target_col,
    predictionCol='prediction'
)

lr_pipeline = Pipeline(
    stages=[
        continuous_assembler,
        continuous_scaler,
        string_indexer,
        onehot_encoder,
        lr_assembler,
        lr
    ]
)

lr_param_grid = (
    ParamGridBuilder()
    .addGrid(lr.elasticNetParam, [0.0, 1.0])
    .addGrid(lr.regParam, [0.1, 0.5, 1.0])
    .build()
)

evaluator = RegressionEvaluator(
    labelCol='DepDelayMinutes',
    predictionCol='prediction',
    metricName='rmse'
)

lr_cv = CrossValidator(
    estimator=lr_pipeline,
    estimatorParamMaps=lr_param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42,
    collectSubModels=True,
)

# Record the time it takes to train the model
start_time = datetime.now()  # Record start time
lr_cv_model = lr_cv.fit(train)
end_time = datetime.now()  # Record end time
train_time = (end_time - start_time).total_seconds()

# Record the time for inference
start_time = datetime.now()  # Record start time
best_lr_model = lr_cv_model.bestModel
lr_predictions = best_lr_model.transform(test)
end_time = datetime.now()  # Record end time
inference_time = (end_time - start_time).total_seconds()

# Calculate and save the evaluation metrics
lr_results = calculate_evaluation_metrics(evaluator, lr_predictions)
lr_results['train_time'] = train_time
lr_results['inference_time'] = inference_time

lr_result_key = f'lr-{sample_size}-{num_executors}'
update_json_file(results_file, lr_result_key, lr_results)
save_models(model_dir, lr_cv_model)


# Random Forest Pipeline
model_dir = os.path.join(base_dir, f'models/rf/{sample_size}/{num_executors}')

rf_assembler = VectorAssembler(
    inputCols=indexed_cols + continuous_cols,
    outputCol='features'
)

rf = RandomForestRegressor(
    featuresCol='features',
    labelCol=target_col,
    predictionCol='prediction'
)

rf_pipeline = Pipeline(
    stages=[
        string_indexer,
        rf_assembler,
        rf
    ]
)

# Create a ParamGridBuilder
rf_param_grid = (
    ParamGridBuilder()
    .addGrid(rf.maxDepth, [5, 10])  # .addGrid(rf.maxDepth, [5, 10, 15])
    .addGrid(rf.numTrees, [5, 10])  # .addGrid(rf.numTrees, [5, 10, 15])
    .addGrid(rf.maxBins, [388, 400])
    .build()
)

rf_cv = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42,
    collectSubModels=True,
)

# Record the time it takes to train the model
start_time = datetime.now()  # Record start time
rf_cv_model = rf_cv.fit(train)
end_time = datetime.now()  # Record end time
train_time = (end_time - start_time).total_seconds()

# Record the time for inference
start_time = datetime.now()  # Record start time
best_rf_model = rf_cv_model.bestModel
rf_predictions = best_rf_model.transform(test)
end_time = datetime.now()  # Record end time
inference_time = (end_time - start_time).total_seconds()

# Calculate and save the evaluation metrics
rf_results = calculate_evaluation_metrics(evaluator, rf_predictions)
rf_results['train_time'] = train_time
rf_results['inference_time'] = inference_time

rf_result_key = f'rf-{sample_size}-{num_executors}'
update_json_file(results_file, rf_result_key, rf_results)
save_models(model_dir, rf_cv_model)

# Gradient Boosting Tree Pipeline
model_dir = os.path.join(base_dir, f'models/gbt/{sample_size}/{num_executors}')

gbt_assembler = VectorAssembler(
    inputCols=indexed_cols + continuous_cols,
    outputCol='features'
)

gbt = GBTRegressor(
    featuresCol='features',
    labelCol=target_col,
    predictionCol='prediction'
)

gbt_pipeline = Pipeline(
    stages=[
        string_indexer,
        gbt_assembler,
        gbt
    ]
)

# Create a ParamGridBuilder
gbt_param_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [5, 10])
    .addGrid(gbt.maxBins, [388, 400])
    .build()
)

gbt_cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42,
    collectSubModels=True,
)

# Record the time it takes to train the model
start_time = datetime.now()  # Record start time
gbt_cv_model = gbt_cv.fit(train)
end_time = datetime.now()  # Record end time
train_time = (end_time - start_time).total_seconds()

# Record the time for inference
start_time = datetime.now()  # Record start time
best_gbt_model = gbt_cv_model.bestModel
gbt_predictions = best_gbt_model.transform(test)
end_time = datetime.now()  # Record end time
inference_time = (end_time - start_time).total_seconds()

# Calculate and save the evaluation metrics
gbt_results = calculate_evaluation_metrics(evaluator, gbt_predictions)
gbt_results['train_time'] = train_time
gbt_results['inference_time'] = inference_time

gbt_result_key = f'gbt-{sample_size}-{num_executors}'
update_json_file(results_file, gbt_result_key, gbt_results)
save_models(model_dir, gbt_cv_model)
