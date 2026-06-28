import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array


def prepare_features(spark):
    # Load gold core profile
    df_core = spark.read.parquet("datamart/gold/feature_store/gold_core_profile_*.parquet")
    df_core_clean = df_core.drop("Type_of_Loan", "Occupation")

    # Encode categorical columns
    cols = ["Credit_Mix", "Payment_Behaviour", "Payment_of_Min_Amount"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx") for c in cols]
    encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_enc", dropLast=True) for c in cols]
    pipeline = Pipeline(stages=indexers + encoders)

    pipeline_model = pipeline.fit(df_core_clean)
    df_core_clean = pipeline_model.transform(df_core_clean)
    df_core_clean = df_core_clean.drop(*cols, *[c+"_idx" for c in cols])

    for i, c in enumerate(cols):
        labels = pipeline_model.stages[i].labels
        labels_kept = labels[:-1]
        df_core_clean = df_core_clean.withColumn(c+"_enc", vector_to_array(col(c+"_enc")))
        for j, label in enumerate(labels_kept):
            df_core_clean = df_core_clean.withColumn(f"{c}_{label}", col(c+"_enc")[j].cast(IntegerType()))
        df_core_clean = df_core_clean.drop(c+"_enc")

    # Load and clean labels
    df_labels = spark.read.parquet("datamart/gold/label_store/gold_label_store_*.parquet")
    df_labels_clean = df_labels.drop("label_def")

    # Join
    df_model = df_labels_clean.join(df_core_clean, on="Customer_ID", how="inner")

    return df_model.toPandas()


def train_model():
    import pyspark.sql as ps
    spark = ps.SparkSession.builder.appName("ml_training").master("local[*]").getOrCreate()

    # Prepare features
    df_pandas = prepare_features(spark)
    df_pandas["snapshot_date"] = pd.to_datetime(df_pandas["snapshot_date"])

    # Time based split
    df_current = df_pandas[df_pandas["snapshot_date"] < "2024-07-01"]
    
    X = df_current.drop(columns=["loan_id", "Customer_ID", "label", "snapshot_date"])
    y = df_current["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train
    model = XGBClassifier(
        random_state=42, eval_metric="auc",
        max_depth=3, n_estimators=100, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    print(f"Train AUC: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.4f}")
    print(f"Val AUC:   {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.4f}")
    print(f"Test AUC:  {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    # Save model
    os.makedirs("model/artefacts", exist_ok=True)
    joblib.dump(model, "model/artefacts/best_model.pkl")
    print("Model saved!")
    # Add this at the end of train_model() in data_processing_ml_table.py
    train_baseline_preds = model.predict_proba(pd.concat([X_train, X_val]))[:, 1]
    pd.DataFrame({"predicted_prob": train_baseline_preds}).to_parquet("model/artefacts/train_preds.parquet", index=False)


def run_inference(snapshot_date_str):
    import pyspark.sql as ps
    spark = ps.SparkSession.builder.appName("ml_inference").master("local[*]").getOrCreate()

    # Skip if not OOT period
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    if snapshot_date < datetime(2024, 7, 1):
        print(f"Skipping inference for {snapshot_date_str} - not in OOT period")
        return

    # Load model
    model = joblib.load("model/artefacts/best_model.pkl")

    # Prepare features for this month
    df_pandas = prepare_features(spark)
    df_pandas["snapshot_date"] = pd.to_datetime(df_pandas["snapshot_date"])
    df_month = df_pandas[df_pandas["snapshot_date"] == snapshot_date_str]

    customer_ids = df_month["Customer_ID"]
    loan_ids = df_month["loan_id"]
    y_actual = df_month["label"]

    X_month = df_month.drop(columns=["loan_id", "Customer_ID", "label", "snapshot_date"])

    # Run inference
    preds = model.predict_proba(X_month)[:, 1]

    # Save predictions
    df_preds = pd.DataFrame({
        "loan_id": loan_ids,
        "Customer_ID": customer_ids,
        "snapshot_date": snapshot_date_str,
        "predicted_prob": preds,
        "predicted_label": (preds >= 0.5).astype(int),
        "actual_label": y_actual
    })

    os.makedirs("datamart/gold/model_predictions", exist_ok=True)
    df_preds.to_parquet(f"datamart/gold/model_predictions/predictions_{snapshot_date_str.replace('-', '_')}.parquet", index=False)
    print(f"Inference saved for {snapshot_date_str}")