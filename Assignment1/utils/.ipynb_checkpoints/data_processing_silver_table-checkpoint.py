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

#Process lms
def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


#Process Features attributes
def process_features_attributes_silver_table(snapshot_date_str, bronze_features_directory, silver_features_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_features_directory + partition_name
    df_fa = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df_fa.count())
  
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Name": StringType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df_fa = df_fa.withColumn(column, col(column).cast(new_type))

    #Median impute incorrect age values
    df_fa = df_fa.withColumn("Age", F.regexp_replace(F.col("Age"), "_", ""))
    df_fa = df_fa.withColumn("Age", F.col("Age").cast(IntegerType()))
    
    median_age = int(df_fa.select(F.percentile_approx("Age", 0.5)).collect()[0][0])
    
    df_fa = df_fa.withColumn(
        "Age", 
        F.when((F.col("Age") < 18) | (F.col("Age") > 100) | F.col("Age").isNull(), F.lit(median_age))
        .otherwise(F.col("Age"))
    )
    
    #Specify occupation as unknown where required
    df_fa = df_fa.withColumn(
        "Occupation",
        F.when(F.col("Occupation").rlike("^_+$") | F.col("Occupation").isNull(), "Unknown")
        .otherwise(F.col("Occupation"))
    )
    
    ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    
    # Keep the SSN if it matches the pattern perfectly, otherwise turn it to null
    df_fa = df_fa.withColumn(
        "SSN",
        F.when(F.col("SSN").rlike(ssn_pattern), F.col("SSN"))
        .otherwise(F.lit(None))
    )

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_attributes_directory + partition_name
    df_fa.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df_fa

#Process Features Financials
def process_features_financials_silver_table(snapshot_date_str, bronze_features_directory, silver_features_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_features_directory + partition_name
    df_ff = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df_ff.count())

    df_ff = df_ff.fillna({"Type_of_Loan": "Not Specified"})

    df_ff = df_ff.withColumn("Credit_Mix", F.when(F.col("Credit_Mix") == "_", "Unknown").otherwise(F.col("Credit_Mix")))

    df_ff = df_ff.withColumn("Payment_Behaviour",F.when(F.col("Payment_Behaviour") == "!@9#%8", "Unknown").otherwise(F.col("Payment_Behaviour")))

    df_ff = df_ff.withColumn("Annual_Income", F.regexp_replace(F.col("Annual_Income"), "_", "").cast(FloatType()))

    df_ff = df_ff.withColumn("Outstanding_Debt", F.regexp_replace(F.col("Outstanding_Debt"), "_", "").cast(FloatType()))

    df = df_ff.filter((F.col("Num_Bank_Accounts") >= 0) & (F.col("Num_Bank_Accounts") <= 10))
    true_median_accounts = int(df.select(F.percentile_approx("Num_Bank_Accounts", 0.5)).collect()[0][0])
    df_ff = df_ff.withColumn("Num_Bank_Accounts", F.when((F.col("Num_Bank_Accounts") < 0) | (F.col("Num_Bank_Accounts") > 10), F.lit(true_median_accounts)).otherwise(F.col("Num_Bank_Accounts")))

    df = df_ff.filter((F.col("Num_Credit_Card") >= 0) & (F.col("Num_Credit_Card") <= 10))
    true_median_credit_cards = int(df.select(F.percentile_approx("Num_Credit_Card", 0.5)).collect()[0][0])
    df_ff = df_ff.withColumn("Num_Credit_Card", F.when((F.col("Num_Credit_Card") < 0) | (F.col("Num_Credit_Card") > 10), F.lit(true_median_credit_cards)).otherwise(F.col("Num_Credit_Card")))

    df = df_ff.filter((F.col("Interest_Rate") >= 0) & (F.col("Interest_Rate") <= 40))
    true_median_interest_rate = df.select(F.percentile_approx("Interest_Rate", 0.5)).collect()[0][0]
    df_ff = df_ff.withColumn("Interest_Rate", F.when((F.col("Interest_Rate") < 0) | (F.col("Interest_Rate") > 40), F.lit(true_median_interest_rate)).otherwise(F.col("Interest_Rate")))

    df_ff = df_ff.withColumn("Num_of_Loan", F.regexp_replace(F.col("Num_of_Loan"), "_", "").cast("integer"))
    df = df_ff.filter((F.col("Num_of_Loan") >= 0) & (F.col("Num_of_Loan") <= 10))
    true_median_Num_of_Loan = int(df.select(F.percentile_approx("Num_of_Loan", 0.5)).collect()[0][0])
    df_ff = df_ff.withColumn("Num_of_Loan", F.when((F.col("Num_of_Loan") < 0) | (F.col("Num_of_Loan") > 10), F.lit(true_median_Num_of_Loan)).otherwise(F.col("Num_of_Loan")))

    df = df_ff.filter((F.col("Delay_from_due_date") >= 0))
    true_median_Delay = int(df.select(F.percentile_approx("Delay_from_due_date", 0.5)).collect()[0][0])
    df_ff = df_ff.withColumn("Delay_from_due_date", F.when((F.col("Delay_from_due_date") < 0), F.lit(true_median_Delay)).otherwise(F.col("Delay_from_due_date")))
         
    df_ff = df_ff.withColumn("Num_of_Delayed_Payment", F.regexp_replace(F.col("Num_of_Delayed_Payment"), "_", "").cast("integer"))
    df = df_ff.filter((F.col("Num_of_Delayed_Payment") >= 0) & (F.col("Num_of_Delayed_Payment") <= 30))
    true_median_Num_of_Delayed = int(df.select(F.percentile_approx("Num_of_Delayed_Payment", 0.5)).collect()[0][0])
    df_ff = df_ff.withColumn("Num_of_Delayed_Payment", 
        F.when((F.col("Num_of_Delayed_Payment") < 0) | (F.col("Num_of_Delayed_Payment") > 30), F.lit(true_median_Num_of_Delayed))
        .otherwise(F.col("Num_of_Delayed_Payment")))

    df_ff = df_ff.withColumn("Changed_Credit_Limit", F.col("Changed_Credit_Limit").cast(FloatType()))
    df_ff = df_ff.fillna({"Changed_Credit_Limit": 0.0})

    df_ff = df_ff.withColumn("Num_Credit_Inquiries", F.col("Num_Credit_Inquiries").cast("integer"))
    df = df_ff.filter((F.col("Num_Credit_Inquiries") >= 0) & (F.col("Num_Credit_Inquiries") <= 30))
    true_median_Num_Credit_Inquiries = int(df.select(F.percentile_approx("Num_Credit_Inquiries", 0.5)).collect()[0][0])
    df_ff = df_ff.withColumn("Num_Credit_Inquiries", 
        F.when((F.col("Num_Credit_Inquiries") < 0) | (F.col("Num_Credit_Inquiries") > 30) | F.col("Num_Credit_Inquiries").isNull(), F.lit(true_median_Num_Credit_Inquiries))
        .otherwise(F.col("Num_Credit_Inquiries")))

    years = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s*Years", 1).cast(IntegerType())
    months = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s*Months", 1).cast(IntegerType())
    safe_years = F.coalesce(years, F.lit(0))
    safe_months = F.coalesce(months, F.lit(0))
    df_ff = df_ff.withColumn("Credit_History_Age_Months", (safe_years * 12) + safe_months)
    df_ff = df_ff.drop("Credit_History_Age")

    df_ff = df_ff.withColumn("Total_EMI_per_month", F.col("Total_EMI_per_month").cast(FloatType()))
    df = df_ff.filter((F.col("Total_EMI_per_month") >= 0) & (F.col("Total_EMI_per_month") <= 10000))
    true_median_emi = df.select(F.percentile_approx("Total_EMI_per_month", 0.5)).collect()[0][0]
    df_ff = df_ff.withColumn(
        "Total_EMI_per_month", 
        F.when((F.col("Total_EMI_per_month") < 0) | (F.col("Total_EMI_per_month") > 10000) | F.col("Total_EMI_per_month").isNull(), F.lit(true_median_emi))
        .otherwise(F.col("Total_EMI_per_month"))
    )

    df_ff = df_ff.withColumn("Amount_invested_monthly", F.regexp_replace(F.col("Amount_invested_monthly"), "_", "").cast(FloatType()))
    df = df_ff.filter((F.col("Amount_invested_monthly") >= 0) & (F.col("Amount_invested_monthly") < 10000))
    true_median_invested = df.select(F.percentile_approx("Amount_invested_monthly", 0.5)).collect()[0][0]
    df_ff = df_ff.withColumn(
        "Amount_invested_monthly", 
        F.when((F.col("Amount_invested_monthly") < 0) | (F.col("Amount_invested_monthly") >= 10000) | F.col("Amount_invested_monthly").isNull(), F.lit(true_median_invested))
        .otherwise(F.col("Amount_invested_monthly"))
    )

    df_ff = df_ff.withColumn("Monthly_Balance", F.regexp_replace(F.col("Monthly_Balance"), "_", "").cast(FloatType()))
    df = df_ff.filter((F.col("Monthly_Balance") >= -10000) & (F.col("Monthly_Balance") <= 100000))
    true_median_balance = df.select(F.percentile_approx("Monthly_Balance", 0.5)).collect()[0][0]
    df_ff = df_ff.withColumn(
        "Monthly_Balance", 
        F.when((F.col("Monthly_Balance") < -10000) | (F.col("Monthly_Balance") > 100000) | F.col("Monthly_Balance").isNull(), F.lit(true_median_balance))
        .otherwise(F.col("Monthly_Balance"))
    )

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_financials_directory + partition_name
    df_ff.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df_ff

#Process Feature clickstream
def process_feature_clickstream_silver_table(snapshot_date_str, bronze_features_directory, silver_feature_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_features_directory + partition_name
    df_fc = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df_fc.count())
  
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df_fc = df_fc.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_feature_clickstream_directory + partition_name
    df_fc.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df_fc