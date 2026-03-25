
import os
import sys
# Spark ga tizimdagi asosiy Python ni ko'rsatamiz
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import isnan, lit, substring, trim, col, when, regexp_replace, concat, length
from pyspark.storagelevel import StorageLevel
import time


# 1. Spark sessiyasini drayver spark ichidagi jarsga joylashtirilgan postgresql drayveri bilan yaratamiz
spark = SparkSession.builder \
    .appName("PostgresFinalTest") \
    .getOrCreate()


# 1. Real loyihada ishlatiladigan professional sxema
# Titanic uchun ustun nomlarini to'g'rilab chiqamiz
titanic_schema = StructType([
    StructField("PassengerId", IntegerType(), False),
    StructField("Survived", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", StringType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),
    StructField("Embarked", StringType(), True)
])


# Ulanish sozlamalari postgresql ga(Boya o'rnatgan 123 paroli bilan)
url = "jdbc:postgresql://localhost:5432/spark_db"
properties = {"user": "postgres", "password": "123", "driver": "org.postgresql.Driver","batchsize": "10000","numPartitions": "4","stringtype": "unspecified"}


def run_titanic_pipeline(source_url, target_table):
    try:
        start_time = time.time()
        print(f"\n{'='*40}")
        print(f"[START] ETL Pipeline boshlandi: {time.strftime('%H:%M:%S')}")
        print(f"{'='*40}")

    
        # --- 1. EXTRACT
        pandas_raw = pd.read_csv(source_url)
        df_spark = spark.createDataFrame(pandas_raw, schema=titanic_schema)
        print(f"[INFO] 1. Ma'lumot yuklandi. Qatorlar: {df_spark.count()}")

        # --- 2. TRANSFORM
        # Ma'lumotlarni tozalash va transformatsiya qilish  
        df_transformed = df_spark \
            .fillna({"Age":29.7, "Cabin":"Unknown", "Embarked":"Unknown"}) \
            .withColumn("Family_Size", F.col("SibSp") + F.col("Parch") + 1) \
            .withColumn("Is_Alone", F.when(F.col("Family_Size") == 1, 1).otherwise(0)) \
            .withColumn("Load_Timestamp", F.current_timestamp())
        print("[INFO] 2. Transformatsiya va boyitish yakunlandi.")
        # Transformatsiyadan so'ng qatorlar sonini tekshirish
        current_rows = df_transformed.count()
        # Agar qatorlar soni 100 dan kam bo'lsa, xatolik beramiz, chunki bu transformatsiya jarayonida muammo borligini ko'rsatadi
        if current_rows < 100:
            raise Exception(f"Qatorlar soni juda kam: {current_rows}. Transformatsiya jarayonida xatolik yuz berishi mumkin.")  
        
        # biz onlinedan olingan ma'lumotni tekshirish uchun schema check qo'shamiz
        # 1. Biz kutayotgan (bazaga mos) ustunlar ro'yxati
        expected_columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Family_Size", "Is_Alone", "Processing_Time"]

        # 2. DataFrame ichidagi haqiqiy ustunlarni olamiz
        actual_columns = df_spark.columns + ["Family_Size", "Is_Alone", "Processing_Time"]  # Transformatsiya jarayonida qo'shilgan ustunlarni ham qo'shamiz

        # 3. Solishtirish (Schema Check)
        # 'set' orqali ustunlar farqini topamiz
        missing_columns = set(expected_columns) - set(actual_columns)

        if missing_columns:
        # Agar biron bir ustun yetishmasa - XATO beramiz!
            raise Exception(f"Sxema mos kelmadi! Yetishmayotgan ustunlar: {missing_columns}")

        # --- 3. LOAD (Siz yozgan ulanishlar bilan) ---
        # 'truncate': 'true' - jadval strukturasi va indekslarini saqlab qoladi
        print(f"[INFO] 3. PostgreSQL-ga yozilmoqda...")
        df_transformed.write.jdbc(
            url=url, 
            table=target_table, 
            mode="overwrite", 
            properties={**properties, "truncate": "true"},
        )

        end_time = time.time()
        print(f"\n{'='*40}")
        print(f"[END] ETL Pipeline yakunlandi: {time.strftime('%H:%M:%S')}")
        duration= end_time - start_time
        print(f"[INFO] Bajarilgan vaqt: {duration:.2f} sekund")
        print(f"{'='*40}")


        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{current_time}] STATUS: SUCCESS | ROWS: {current_rows} | DURATION: {duration}s\n"

        # 2. Faylga yozish ('a' - append mode, ya'ni eski loglarni o'chirmasdan davomidan yozadi)
        with open("etl_history.log", "a") as log_file:
            log_file.write(log_entry)

        print("[INFO] Hisobot 'etl_history.log' fayliga muhrlandi.")

    except Exception as e:
         current_time = time.strftime('%Y-%m-%d %H:%M:%S')
         log_entry = f"[{current_time}] STATUS: ERROR | MESSAGE: {str(e)}\n"
         with open("etl_history.log", "a") as log_file:
            log_file.write(log_entry)   
            print(f"[ERROR] ETL Pipeline xatolik yuz berdi: {str(e)}")    
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
run_titanic_pipeline(data_url, "jumazaroff_titanic")
run_titanic_pipeline(data_url, "jumazarov_titanic")