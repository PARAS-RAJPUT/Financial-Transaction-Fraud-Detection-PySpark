
# Financial Transaction Fraud Detection using PySpark MLlib
### Distributed Machine Learning • Apache Spark • Big Data Analytics • Anomaly Detection
**Course Code:** CSE494 — Capstone Project  
**Platform:** Apache Spark 3.5.3 (PySpark) • Python 3.10 • Spark MLlib  
**Dataset:** European Cardholders Credit Card Fraud Dataset (284,807 transactions, 492 frauds, 0.172% imbalance)

---

## Executive Summary
Financial fraud poses multi-billion dollar threats to the global banking ecosystem. Detecting fraud requires:
1. High-throughput distributed processing capable of handling millions of real-time transactions.
2. Robust handling of extreme class imbalance (99.83% legitimate vs 0.17% fraud).
3. Hybrid screening combining supervised classification with unsupervised anomaly detection.

In this project, we architect and evaluate an end-to-end distributed machine learning system using **PySpark SQL** and **MLlib**, comparing **Logistic Regression**, **Decision Trees**, **Linear Support Vector Machines (LinearSVC)**, and **K-Means Anomaly Scoring**.



---
## 1. Environment Setup & Distributed SparkSession Initialization
We configure a local multi-threaded Apache Spark session with memory allocation and partition tuning.


**In [1]:**
```python
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure PySpark workers bind to the active Python binary
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

# Initialize SparkSession with tuned memory and partition configs
spark = SparkSession.builder \
    .appName("CreditCardFraudDetection_PySpark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Apache Spark Cluster Details:")
print(" Spark Version:       ", spark.version)
print(" Master URL:          ", spark.sparkContext.master)
print(" Default Parallelism: ", spark.sparkContext.defaultParallelism)
print(" App Name:            ", spark.sparkContext.appName)
```

**Out [1]:**
```text
Apache Spark Cluster Details:
 Spark Version:        3.5.3
 Master URL:           local[*]
 Default Parallelism:  12
 App Name:             CreditCardFraudDetection_PySpark
```


---
## 2. Distributed Data Ingestion & Schema Inspection
We ingest the Kaggle Credit Card Fraud dataset into a distributed PySpark DataFrame. The dataset contains 284,807 European credit card transactions with 30 continuous numerical features (Time, PCA components V1 to V28, and Amount) and a binary target label `Class` (0 = Legitimate, 1 = Fraud).


**In [2]:**
```python
csv_path = os.path.join("..", "data", "creditcard.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join("data", "creditcard.csv")

t0 = time.time()
df = spark.read.csv(csv_path, header=True, inferSchema=True)
load_duration = time.time() - t0

total_rows = df.count()
total_cols = len(df.columns)

print(f"Data Ingestion Complete: {total_rows:,} records across {total_cols} columns loaded in {load_duration:.2f} seconds.")
df.printSchema()
df.show(5, truncate=False)
```

**Out [2]:**
```text
Data Ingestion Complete: 284,807 records across 31 columns loaded in 13.59 seconds.
root
 |-- Time: double (nullable = true)
 |-- V1: double (nullable = true)
 |-- V2: double (nullable = true)
 |-- V3: double (nullable = true)
 |-- V4: double (nullable = true)
 |-- V5: double (nullable = true)
 |-- V6: double (nullable = true)
 |-- V7: double (nullable = true)
 |-- V8: double (nullable = true)
 |-- V9: double (nullable = true)
 |-- V10: double (nullable = true)
 |-- V11: double (nullable = true)
 |-- V12: double (nullable = true)
 |-- V13: double (nullable = true)
 |-- V14: double (nullable = true)
 |-- V15: double (nullable = true)
 |-- V16: double (nullable = true)
 |-- V17: double (nullable = true)
 |-- V18: double (nullable = true)
 |-- V19: double (nullable = true)
 |-- V20: double (nullable = true)
 |-- V21: double (nullable = true)
 |-- V22: double (nullable = true)
 |-- V23: double (nullable = true)
 |-- V24: double (nullable = true)
 |-- V25: double (nullable = true)
 |-- V26: double (nullable = true)
 |-- V27: double (nullable = true)
 |-- V28: double (nullable = true)
 |-- Amount: double (nullable = true)
 |-- Class: integer (nullable = true)
```
```text
+----+------------------+-------------------+----------------+------------------+-------------------+-------------------+-------------------+------------------+------------------+-------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------+------------------+-------------------+--------------------+-------------------+------------------+------------------+------------------+------------------+--------------------+-------------------+------+-----+
|Time|V1                |V2                 |V3              |V4                |V5                 |V6                 |V7                 |V8                |V9                |V10                |V11               |V12               |V13               |V14               |V15               |V16               |V17               |V18                |V19               |V20                |V21                 |V22                |V23               |V24               |V25               |V26               |V27                 |V28                |Amount|Class|
+----+------------------+-------------------+----------------+------------------+-------------------+-------------------+-------------------+------------------+------------------+-------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------+------------------+-------------------+--------------------+-------------------+------------------+------------------+------------------+------------------+--------------------+-------------------+------+-----+
|0.0 |-1.3598071336738  |-0.0727811733098497|2.53634673796914|1.37815522427443  |-0.338320769942518 |0.462387777762292  |0.239598554061257  |0.0986979012610507|0.363786969611213 |0.0907941719789316 |-0.551599533260813|-0.617800855762348|-0.991389847235408|-0.311169353699879|1.46817697209427  |-0.470400525259478|0.207971241929242 |0.0257905801985591 |0.403992960255733 |0.251412098239705  |-0.018306777944153  |0.277837575558899  |-0.110473910188767|0.0669280749146731|0.128539358273528 |-0.189114843888824|0.133558376740387   |-0.0210530534538215|149.62|0    |
|0.0 |1.19185711131486  |0.26615071205963   |0.16648011335321|0.448154078460911 |0.0600176492822243 |-0.0823608088155687|-0.0788029833323113|0.0851016549148104|-0.255425128109186|-0.166974414004614 |1.61272666105479  |1.06523531137287  |0.48909501589608  |-0.143772296441519|0.635558093258208 |0.463917041022171 |-0.114804663102346|-0.183361270123994 |-0.145783041325259|-0.0690831352230203|-0.225775248033138  |-0.638671952771851 |0.101288021253234 |-0.339846475529127|0.167170404418143 |0.125894532368176 |-0.00898309914322813|0.0147241691924927 |2.69  |0    |
|1.0 |-1.35835406159823 |-1.34016307473609  |1.77320934263119|0.379779593034328 |-0.503198133318193 |1.80049938079263   |0.791460956450422  |0.247675786588991 |-1.51465432260583 |0.207642865216696  |0.624501459424895 |0.066083685268831 |0.717292731410831 |-0.165945922763554|2.34586494901581  |-2.89008319444231 |1.10996937869599  |-0.121359313195888 |-2.26185709530414 |0.524979725224404  |0.247998153469754   |0.771679401917229  |0.909412262347719 |-0.689280956490685|-0.327641833735251|-0.139096571514147|-0.0553527940384261 |-0.0597518405929204|378.66|0    |
|1.0 |-0.966271711572087|-0.185226008082898 |1.79299333957872|-0.863291275036453|-0.0103088796030823|1.24720316752486   |0.23760893977178   |0.377435874652262 |-1.38702406270197 |-0.0549519224713749|-0.226487263835401|0.178228225877303 |0.507756869957169 |-0.28792374549456 |-0.631418117709045|-1.0596472454325  |-0.684092786345479|1.96577500349538   |-1.2326219700892  |-0.208037781160366 |-0.108300452035545  |0.00527359678253453|-0.190320518742841|-1.17557533186321 |0.647376034602038 |-0.221928844458407|0.0627228487293033  |0.0614576285006353 |123.5 |0    |
|2.0 |-1.15823309349523 |0.877736754848451  |1.548717846511  |0.403033933955121 |-0.407193377311653 |0.0959214624684256 |0.592940745385545  |-0.270532677192282|0.817739308235294 |0.753074431976354  |-0.822842877946363|0.53819555014995  |1.3458515932154   |-1.11966983471731 |0.175121130008994 |-0.451449182813529|-0.237033239362776|-0.0381947870352842|0.803486924960175 |0.408542360392758  |-0.00943069713232919|0.79827849458971   |-0.137458079619063|0.141266983824769 |-0.206009587619756|0.502292224181569 |0.219422229513348   |0.215153147499206  |69.99 |0    |
+----+------------------+-------------------+----------------+------------------+-------------------+-------------------+-------------------+------------------+------------------+-------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------+------------------+-------------------+--------------------+-------------------+------------------+------------------+------------------+------------------+--------------------+-------------------+------+-----+
only showing top 5 rows
```


---
## 3. Exploratory Data Analysis & Extreme Imbalance Quantification
Evaluating the class balance and verifying data hygiene (checking for missing/null values).


**In [3]:**
```python
# Quantify Class Distribution
class_counts = df.groupBy("Class").count().toPandas()
print("Class Distribution:")
print(class_counts)

legit_count = int(class_counts[class_counts["Class"] == 0]["count"].iloc[0])
fraud_count = int(class_counts[class_counts["Class"] == 1]["count"].iloc[0])
fraud_pct = (fraud_count / total_rows) * 100
imbalance_ratio = legit_count / fraud_count

print(f"\nSummary:")
print(f" - Legitimate Transactions (0): {legit_count:,} ({100-fraud_pct:.4f}%)")
print(f" - Fraudulent Transactions (1): {fraud_count:,} ({fraud_pct:.4f}%)")
print(f" - Imbalance Ratio: 1 fraud per {imbalance_ratio:.1f} normal transactions")

# Check for Null / NaN values across all columns
null_counts = df.select([F.count(F.when(F.col(c).isNull() | F.isnan(c), c)).alias(c) for c in df.columns]).toPandas()
print("\nMissing/Null Values Count across features:")
print(null_counts.T.rename(columns={0: "Null_Count"}).head(10))
```

**Out [3]:**
```text
Class Distribution:
   Class   count
0      0  284315
1      1     492

Summary:
 - Legitimate Transactions (0): 284,315 (99.8273%)
 - Fraudulent Transactions (1): 492 (0.1727%)
 - Imbalance Ratio: 1 fraud per 577.9 normal transactions
```
```text
Missing/Null Values Count across features:
      Null_Count
Time           0
V1             0
V2             0
V3             0
V4             0
V5             0
V6             0
V7             0
V8             0
V9             0
```

**In [4]:**
```python
# Transaction Amount Statistics by Class
print("Transaction Amount Distribution by Class:")
df.groupBy("Class").agg(
    F.count("Amount").alias("Count"),
    F.mean("Amount").alias("Mean_Amount"),
    F.stddev("Amount").alias("StdDev_Amount"),
    F.min("Amount").alias("Min_Amount"),
    F.max("Amount").alias("Max_Amount")
).show()
```

**Out [4]:**
```text
Transaction Amount Distribution by Class:
```
```text
+-----+------+-----------------+------------------+----------+----------+
|Class| Count|      Mean_Amount|     StdDev_Amount|Min_Amount|Max_Amount|
+-----+------+-----------------+------------------+----------+----------+
|    0|284315|88.29102242231271|250.10509222589235|       0.0|  25691.16|
|    1|   492|122.2113211382114| 256.6832882977121|       0.0|   2125.87|
+-----+------+-----------------+------------------+----------+----------+
```


---
## 4. Feature Engineering & Class Imbalance Mitigation
To address the severe 577:1 class imbalance without synthetic data distortion:
1. **Cost-Sensitive Weighting:** We assign inverse class frequency weights to the loss function (`weight_fraud = total / (2 * fraud_count)`), penalizing missed fraud 577x more heavily than false alarms.
2. **Feature Assembly (`VectorAssembler`):** Combining all 30 input columns `[V1..V28, Amount, Time]` into a feature vector.
3. **Feature Standardization (`StandardScaler`):** Normalizing all features to mean 0 and unit variance.


**In [5]:**
```python
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

# Calculate balanced class weights
weight_fraud = total_rows / (2.0 * fraud_count)
weight_legit = total_rows / (2.0 * legit_count)
print(f"Cost-Sensitive Loss Weights -> Legit (0): {weight_legit:.4f} | Fraud (1): {weight_fraud:.4f}")

df_weighted = df.withColumn(
    "weight",
    F.when(F.col("Class") == 1, weight_fraud).otherwise(weight_legit)
)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)

# Train-Test Partitioning (80% Train, 20% Test, Seed 42)
train_df, test_df = df_weighted.randomSplit([0.8, 0.2], seed=42)
train_df = train_df.cache()
test_df = test_df.cache()

print(f"Training Set:   {train_df.count():,} rows")
print(f"Testing Set:    {test_df.count():,} rows")

# Fit Feature Pipeline on Training Set only (preventing data leakage)
feature_pipeline = Pipeline(stages=[assembler, scaler])
feature_model = feature_pipeline.fit(train_df)

train_prepared = feature_model.transform(train_df).cache()
test_prepared = feature_model.transform(test_df).cache()
print("Feature preparation complete!")
```

**Out [5]:**
```text
Cost-Sensitive Loss Weights -> Legit (0): 0.5009 | Fraud (1): 289.4380
```
```text
Training Set:   228,225 rows
```
```text
Testing Set:    56,582 rows
```
```text
Feature preparation complete!
```


---
## 5. Supervised Model Training & Evaluation
We train three diverse classification paradigms on Apache Spark:
1. **Logistic Regression** (Linear baseline with cost-sensitive weighted loss).
2. **Decision Tree Classifier** (Non-linear recursive partitioning).
3. **Linear Support Vector Machine (LinearSVC)** (Max-margin hyper-plane with cost-sensitive weights).


**In [6]:**
```python
eval_roc = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
eval_pr = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

def compute_metrics(predictions, pred_col="prediction"):
    tp = predictions.filter(f"Class == 1 AND {pred_col} == 1.0").count()
    fp = predictions.filter(f"Class == 0 AND {pred_col} == 1.0").count()
    tn = predictions.filter(f"Class == 0 AND {pred_col} == 0.0").count()
    fn = predictions.filter(f"Class == 1 AND {pred_col} == 0.0").count()
    
    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

results = {}
```

**In [7]:**
```python
# 1. Logistic Regression
print("Fitting Logistic Regression (Weighted Loss)...")
lr = LogisticRegression(featuresCol="scaled_features", labelCol="Class", weightCol="weight", maxIter=30, regParam=0.01)
t0 = time.time()
lr_model = lr.fit(train_prepared)
lr_time = time.time() - t0

lr_preds = lr_model.transform(test_prepared).cache()
m_lr = compute_metrics(lr_preds)
m_lr["ROC_AUC"] = eval_roc.evaluate(lr_preds)
m_lr["PR_AUC"] = eval_pr.evaluate(lr_preds)
m_lr["Train_Time_s"] = lr_time
results["Logistic Regression"] = m_lr

print(f"Logistic Regression ({lr_time:.2f}s): Recall={m_lr['Recall']:.4f}, Precision={m_lr['Precision']:.4f}, F1={m_lr['F1']:.4f}, ROC-AUC={m_lr['ROC_AUC']:.4f}")
```

**Out [7]:**
```text
Fitting Logistic Regression (Weighted Loss)...
```
```text
Logistic Regression (15.57s): Recall=0.8252, Precision=0.1310, F1=0.2261, ROC-AUC=0.9750
```

**In [8]:**
```python
# 2. Decision Tree Classifier
print("Fitting Decision Tree Classifier...")
dt = DecisionTreeClassifier(featuresCol="scaled_features", labelCol="Class", maxDepth=7, seed=42)
t0 = time.time()
dt_model = dt.fit(train_prepared)
dt_time = time.time() - t0

dt_preds = dt_model.transform(test_prepared).cache()
m_dt = compute_metrics(dt_preds)
m_dt["ROC_AUC"] = eval_roc.evaluate(dt_preds)
m_dt["PR_AUC"] = eval_pr.evaluate(dt_preds)
m_dt["Train_Time_s"] = dt_time
results["Decision Tree"] = m_dt

print(f"Decision Tree ({dt_time:.2f}s): Recall={m_dt['Recall']:.4f}, Precision={m_dt['Precision']:.4f}, F1={m_dt['F1']:.4f}, ROC-AUC={m_dt['ROC_AUC']:.4f}")
```

**Out [8]:**
```text
Fitting Decision Tree Classifier...
```
```text
Decision Tree (4.88s): Recall=0.6990, Precision=0.8090, F1=0.7500, ROC-AUC=0.7472
```

**In [9]:**
```python
# 3. Linear Support Vector Machine (LinearSVC)
print("Fitting Linear Support Vector Machine (Weighted Loss)...")
lsvc = LinearSVC(featuresCol="scaled_features", labelCol="Class", weightCol="weight", maxIter=25, regParam=0.01)
t0 = time.time()
lsvc_model = lsvc.fit(train_prepared)
lsvc_time = time.time() - t0

lsvc_preds = lsvc_model.transform(test_prepared).cache()
m_lsvc = compute_metrics(lsvc_preds)
m_lsvc["ROC_AUC"] = eval_roc.evaluate(lsvc_preds)
m_lsvc["PR_AUC"] = eval_pr.evaluate(lsvc_preds)
m_lsvc["Train_Time_s"] = lsvc_time
results["Linear SVM"] = m_lsvc

print(f"Linear SVM ({lsvc_time:.2f}s): Recall={m_lsvc['Recall']:.4f}, Precision={m_lsvc['Precision']:.4f}, F1={m_lsvc['F1']:.4f}, ROC-AUC={m_lsvc['ROC_AUC']:.4f}")
```

**Out [9]:**
```text
Fitting Linear Support Vector Machine (Weighted Loss)...
```
```text
Linear SVM (11.52s): Recall=0.8252, Precision=0.1172, F1=0.2053, ROC-AUC=0.9759
```


---
## 6. Unsupervised Anomaly Exploration with K-Means
In cold-start environments or zero-day fraud scenarios where fraud labels are not yet available, unsupervised anomaly detection is essential.
We train a distributed **K-Means Clustering** model ($k=4$), compute the Euclidean distance of each transaction to its cluster centroid, and flag transactions exceeding the 98.5th percentile distance threshold as anomalous.


**In [10]:**
```python
print("Fitting K-Means Clustering for Anomaly Scoring...")
kmeans = KMeans(featuresCol="scaled_features", predictionCol="cluster", k=4, seed=42, maxIter=20)
t0 = time.time()
km_model = kmeans.fit(train_prepared)
km_time = time.time() - t0

centers = [np.array(c) for c in km_model.clusterCenters()]
centers_bc = spark.sparkContext.broadcast(centers)

def get_dist(cluster_id, vec):
    c = centers_bc.value[int(cluster_id)]
    return float(np.linalg.norm(np.array(vec.toArray()) - c))

dist_udf = F.udf(get_dist, DoubleType())

km_train = km_model.transform(train_prepared).withColumn("dist_to_center", dist_udf(F.col("cluster"), F.col("scaled_features")))
km_test = km_model.transform(test_prepared).withColumn("dist_to_center", dist_udf(F.col("cluster"), F.col("scaled_features"))).cache()

# Compute 98.5th percentile threshold on training baseline
threshold = km_train.stat.approxQuantile("dist_to_center", [0.985], 0.001)[0]
print(f"Unsupervised Anomaly Distance Cutoff (98.5%): {threshold:.4f}")

km_test = km_test.withColumn("anomaly_pred", F.when(F.col("dist_to_center") >= threshold, 1.0).otherwise(0.0)).cache()
m_km = compute_metrics(km_test, pred_col="anomaly_pred")

eval_roc_dist = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="dist_to_center", metricName="areaUnderROC")
eval_pr_dist = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="dist_to_center", metricName="areaUnderPR")

m_km["ROC_AUC"] = eval_roc_dist.evaluate(km_test)
m_km["PR_AUC"] = eval_pr_dist.evaluate(km_test)
m_km["Train_Time_s"] = km_time
results["K-Means Anomaly Detector"] = m_km

print(f"K-Means Anomaly ({km_time:.2f}s): Recall={m_km['Recall']:.4f}, Precision={m_km['Precision']:.4f}, F1={m_km['F1']:.4f}, ROC-AUC={m_km['ROC_AUC']:.4f}")
```

**Out [10]:**
```text
Fitting K-Means Clustering for Anomaly Scoring...
```
```text
Unsupervised Anomaly Distance Cutoff (98.5%): 12.1889
```
```text
K-Means Anomaly (8.03s): Recall=0.6699, Precision=0.0756, F1=0.1358, ROC-AUC=0.9407
```


---
## 7. Model Comparison Table & Benchmark Analysis
We compile the empirical benchmark results across all four models.


**In [11]:**
```python
comparison_table = pd.DataFrame(results).T[[
    "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "TP", "FP", "TN", "FN", "Train_Time_s"
]]
print("=== COMPREHENSIVE MODEL COMPARISON BENCHMARK ===")
display(comparison_table.style.highlight_max(color='#d4edda', subset=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]))
```

**Out [11]:**
```text
=== COMPREHENSIVE MODEL COMPARISON BENCHMARK ===
```
*(Interactive DataFrame/Table displayed)*


---
## 8. Visualizations & Analytical Interpretations
Below we display the generated publication-grade visualizations showing class imbalance, correlation structures, confusion matrices, ROC/PR curves, and feature importance.


**In [12]:**
```python
from IPython.display import Image, display

# Display generated charts
chart_files = [
    "visualizations/01_class_imbalance.png",
    "visualizations/02_correlation_heatmap.png",
    "visualizations/03_confusion_matrices.png",
    "visualizations/04_roc_pr_curves.png",
    "visualizations/05_model_comparison_bar.png",
    "visualizations/06_kmeans_anomaly_clusters.png",
    "visualizations/07_feature_importance.png"
]

for chart in chart_files:
    if os.path.exists(chart):
        print(f"\n--- {chart} ---")
        display(Image(filename=chart))
```

**Out [12]:**
```text
--- visualizations/01_class_imbalance.png ---
```
```text
--- visualizations/02_correlation_heatmap.png ---
```
```text
--- visualizations/03_confusion_matrices.png ---
```
```text
--- visualizations/04_roc_pr_curves.png ---
```
```text
--- visualizations/05_model_comparison_bar.png ---
```
```text
--- visualizations/06_kmeans_anomaly_clusters.png ---
```
```text
--- visualizations/07_feature_importance.png ---
```


---
## 9. Real-Time Transaction Screening Simulation
Demonstrating real-time batch transaction scoring using the trained PySpark MLlib models.


**In [13]:**
```python
# Sample 5 fraudulent transactions and 5 legitimate transactions from test set
sample_fraud = test_prepared.filter("Class == 1").limit(5)
sample_legit = test_prepared.filter("Class == 0").limit(5)
sample_stream = sample_fraud.union(sample_legit)

scored_stream = lr_model.transform(sample_stream).select(
    "Time", "Amount", "Class",
    vector_to_array("probability")[1].alias("Fraud_Probability"),
    "prediction"
).withColumn(
    "Decision",
    F.when(F.col("prediction") == 1.0, "ALERT: HIGH RISK FRAUD - BLOCK").otherwise("APPROVE TRANSACTION")
).toPandas()

print("Real-Time Transaction Scoring Output:")
display(scored_stream)
```

**Out [13]:**
```text
Real-Time Transaction Scoring Output:
```
*(Interactive DataFrame/Table displayed)*


---
## 10. Conclusion & Architectural Insights
1. **Recall vs Precision Trade-off:** In banking, missing a fraud transaction incurs substantial financial losses and chargebacks. Logistic Regression and Linear SVM with class-weighted loss achieve over **82.5% Recall** (ROC-AUC > 0.975), capturing the vast majority of frauds.
2. **Precision Leader:** Decision Tree achieves **80.9% Precision** with only 17 false alarms, making it well-suited when minimizing customer verification friction is the primary operational objective.
3. **Unsupervised Utility:** K-Means clustering successfully flags **67.0% of fraud transactions** purely by measuring geometric outlier distance in feature space, proving its value as a zero-day screening layer.
4. **PySpark Scalability:** Spark's distributed architecture enabled reading, preprocessing, scaling, and training 4 models on 284,807 transactions in seconds, validating its production suitability for high-velocity payment rails.
