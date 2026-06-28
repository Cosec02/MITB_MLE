from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {"owner": "airflow", "start_date": datetime(2023, 1, 1)}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    schedule_interval="0 0 1 * *",
    catchup=True
) as dag:

    def bronze_lms(snapshot_date_str, **kwargs):
        import pyspark.sql as ps
        spark = ps.SparkSession.builder.appName("pipeline").master("local[*]").getOrCreate()
        from utils.data_processing_bronze_table import process_bronze_table
        process_bronze_table(snapshot_date_str, "datamart/bronze/lms/", spark)

    def bronze_features(snapshot_date_str, **kwargs):
        import pyspark.sql as ps
        spark = ps.SparkSession.builder.appName("pipeline").master("local[*]").getOrCreate()
        from utils.data_processing_bronze_table import process_bronze_feature_table
        process_bronze_feature_table("feature_clickstream", snapshot_date_str, "datamart/bronze/feature_clickstream/", spark)
        process_bronze_feature_table("features_attributes", snapshot_date_str, "datamart/bronze/features_attributes/", spark)
        process_bronze_feature_table("features_financials", snapshot_date_str, "datamart/bronze/features_financials/", spark)

    def silver_lms(snapshot_date_str, **kwargs):
        import pyspark.sql as ps
        spark = ps.SparkSession.builder.appName("pipeline").master("local[*]").getOrCreate()
        from utils.data_processing_silver_table import process_silver_table
        process_silver_table(snapshot_date_str, "datamart/bronze/lms/", "datamart/silver/loan_daily/", spark)

    def silver_features(snapshot_date_str, **kwargs):
        import pyspark.sql as ps
        spark = ps.SparkSession.builder.appName("pipeline").master("local[*]").getOrCreate()
        from utils.data_processing_silver_table import process_features_attributes_silver_table, process_features_financials_silver_table, process_feature_clickstream_silver_table
        process_features_attributes_silver_table(snapshot_date_str, "datamart/bronze/features_attributes/", "datamart/silver/features_attributes/", spark)
        process_features_financials_silver_table(snapshot_date_str, "datamart/bronze/features_financials/", "datamart/silver/features_financials/", spark)
        process_feature_clickstream_silver_table(snapshot_date_str, "datamart/bronze/feature_clickstream/", "datamart/silver/feature_clickstream/", spark)

    def gold_tables(snapshot_date_str, **kwargs):
        import pyspark.sql as ps
        spark = ps.SparkSession.builder.appName("pipeline").master("local[*]").getOrCreate()
        from utils.data_processing_gold_table import process_labels_gold_table, process_feature_gold_tables
        process_labels_gold_table(snapshot_date_str, "datamart/silver/loan_daily/", "datamart/gold/label_store/", spark, dpd=30, mob=6)
        process_feature_gold_tables(snapshot_date_str, "datamart/silver/features_attributes/", "datamart/silver/features_financials/", "datamart/silver/feature_clickstream/", "datamart/gold/feature_store/", spark)

    def ml_train(**kwargs):
        from utils.data_processing_ml_table import train_model
        train_model()

    def ml_inference(snapshot_date_str, **kwargs):
        from utils.data_processing_ml_table import run_inference
        run_inference(snapshot_date_str)

    def ml_monitoring(snapshot_date_str, **kwargs):
        from utils.data_processing_monitoring import monitor_model
        monitor_model(snapshot_date_str)

    t1 = PythonOperator(task_id="bronze_lms", python_callable=bronze_lms, op_kwargs={"snapshot_date_str": "{{ ds }}"})
    t2 = PythonOperator(task_id="bronze_features", python_callable=bronze_features, op_kwargs={"snapshot_date_str": "{{ ds }}"})
    t3 = PythonOperator(task_id="silver_lms", python_callable=silver_lms, op_kwargs={"snapshot_date_str": "{{ ds }}"})
    t4 = PythonOperator(task_id="silver_features", python_callable=silver_features, op_kwargs={"snapshot_date_str": "{{ ds }}"})
    t5 = PythonOperator(task_id="gold_tables", python_callable=gold_tables, op_kwargs={"snapshot_date_str": "{{ ds }}"})
    t6 = PythonOperator(task_id="ml_train", python_callable=ml_train)
    t7 = PythonOperator(task_id="ml_inference", python_callable=ml_inference, op_kwargs={"snapshot_date_str": "{{ ds }}"})
    t8 = PythonOperator(task_id="ml_monitoring", python_callable=ml_monitoring, op_kwargs={"snapshot_date_str": "{{ ds }}"})

    t1 >> t3
    t2 >> t4
    [t3, t4] >> t5 >> t6 >> t7 >> t8