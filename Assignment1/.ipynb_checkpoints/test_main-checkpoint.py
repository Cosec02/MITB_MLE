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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)

bronze_feature_directory = "datamart/bronze/feature_clickstream/"

if not os.path.exists(bronze_feature_directory):
    os.makedirs(bronze_feature_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_feature_table("feature_clickstream",date_str, bronze_feature_directory, spark)

bronze_feature_directory = "datamart/bronze/features_attributes/"

if not os.path.exists(bronze_feature_directory):
    os.makedirs(bronze_feature_directory)
    
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_feature_table("features_attributes",date_str, bronze_feature_directory, spark)

bronze_feature_directory = "datamart/bronze/features_financials/"

if not os.path.exists(bronze_feature_directory):
    os.makedirs(bronze_feature_directory)
    
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_feature_table("features_financials",date_str, bronze_feature_directory, spark)
    
# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# Silver Feature Attributes
silver_features_attributes_directory = "datamart/silver/features_attributes/"
bronze_features_directory = "datamart/bronze/features_attributes/"

if not os.path.exists(silver_features_attributes_directory):
    os.makedirs(silver_features_attributes_directory)
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_features_attributes_silver_table(date_str,  bronze_features_directory, silver_features_attributes_directory, spark)
    
# Silver Feature Financials
silver_features_financials_directory = "datamart/silver/features_financials/"
bronze_features_directory = "datamart/bronze/features_financials/"

if not os.path.exists(silver_features_financials_directory):
    os.makedirs(silver_features_financials_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_features_financials_silver_table(date_str,  bronze_features_directory, silver_features_financials_directory, spark)

# Silver feature clickstream
silver_feature_clickstream_directory = "datamart/silver/feature_clickstream/"
bronze_features_directory = "datamart/bronze/feature_clickstream/"

if not os.path.exists(silver_feature_clickstream_directory):
    os.makedirs(silver_feature_clickstream_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_feature_clickstream_silver_table(date_str, bronze_features_directory, silver_feature_clickstream_directory, spark)

# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)


folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

#Gold Feature store
silver_fa_directory="datamart/silver/features_attributes/"
silver_ff_directory="datamart/silver/features_financials/"
silver_fc_directory="datamart/silver/feature_clickstream/"

gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_feature_gold_tables(date_str, silver_fa_directory, silver_ff_directory, silver_fc_directory, gold_feature_store_directory, spark)

core_wildcard_path = gold_feature_store_directory + "gold_core_profile_*.parquet"
df_gold_core = spark.read.parquet(core_wildcard_path)
print("Core Profile Row Count:", df_gold_core.count())
display(df_gold_core.limit(20).toPandas())

click_wildcard_path = gold_feature_store_directory + "gold_clickstream_profile_*.parquet"
df_gold_click = spark.read.parquet(click_wildcard_path)
print("\nClickstream Profile Row Count:", df_gold_click.count())
display(df_gold_click.limit(20).toPandas())
