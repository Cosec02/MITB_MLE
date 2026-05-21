import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_feature_gold_tables(snapshot_date_str, silver_fa_directory, silver_ff_directory, silver_fc_directory, gold_feature_store_directory, spark):
    
    partition_suffix = "_" + snapshot_date_str.replace('-','_') + '.parquet'
    
    # Updated variables to match the function parameters
    df_fa = spark.read.parquet(silver_fa_directory + "silver_features_attributes" + partition_suffix)
    df_ff = spark.read.parquet(silver_ff_directory + "silver_features_financials" + partition_suffix)
    df_fc = spark.read.parquet(silver_fc_directory + "silver_feature_clickstream" + partition_suffix)

    df_gold_core = df_fa.join(df_ff, on="Customer_ID", how="inner")
    
    cols_to_drop = [c for c in ["Name", "SSN", "snapshot_date"] if c in df_gold_core.columns]
    df_gold_core = df_gold_core.drop(*cols_to_drop)

    clickstream_cols = [c for c in df_fc.columns if c.startswith('fe_')]
    agg_exprs = [F.sum(c).alias(f"Total_{c}") for c in clickstream_cols]
    df_gold_clickstream = df_fc.groupBy("Customer_ID").agg(*agg_exprs)

    # Updated variable for the save paths
    core_path = gold_feature_store_directory + "gold_core_profile" + partition_suffix
    df_gold_core.write.mode("overwrite").parquet(core_path)
    print('Saved Core Profile to:', core_path)

    click_path = gold_feature_store_directory + "gold_clickstream_profile" + partition_suffix
    df_gold_clickstream.write.mode("overwrite").parquet(click_path)
    print('Saved Clickstream Profile to:', click_path)
    
    return df_gold_core, df_gold_clickstream