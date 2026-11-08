import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure PySpark uses current python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

print("=" * 80)
print("FINANCIAL TRANSACTION FRAUD DETECTION PIPELINE (APACHE SPARK / PYSPARK MLLIB)")
print("=" * 80)

# 1. Initialize SparkSession
print("\n[Step 1/8] Initializing PySpark Session with Local Master...")
spark = SparkSession.builder \
    .appName("CreditCardFraudDetection_PySpark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"-> PySpark Version: {spark.version}")
print(f"-> Master: {spark.sparkContext.master}")
print(f"-> Default Parallelism: {spark.sparkContext.defaultParallelism}")

# 2. Ingest Dataset
csv_path = os.path.join("data", "creditcard.csv")
print(f"\n[Step 2/8] Loading dataset from {csv_path} into Spark DataFrame...")
start_time = time.time()
df = spark.read.csv(csv_path, header=True, inferSchema=True)
load_time = time.time() - start_time
total_rows = df.count()
total_cols = len(df.columns)
print(f"-> Loaded {total_rows:,} rows across {total_cols} columns in {load_time:.2f} seconds.")

# 3. Exploratory Data Analysis & Class Imbalance Audit
print("\n[Step 3/8] Performing Distributed EDA & Imbalance Quantification...")
class_counts = df.groupBy("Class").count().collect()
class_dict = {row["Class"]: row["count"] for row in class_counts}
legit_count = class_dict.get(0, 0)
fraud_count = class_dict.get(1, 0)
fraud_pct = (fraud_count / total_rows) * 100

print(f"-> Class 0 (Legitimate Transactions): {legit_count:,} ({100 - fraud_pct:.4f}%)")
print(f"-> Class 1 (Fraudulent Transactions): {fraud_count:,} ({fraud_pct:.4f}%)")
imbalance_ratio = legit_count / fraud_count
print(f"-> Imbalance Ratio: 1 fraud per {imbalance_ratio:.1f} legitimate transactions")

# Check for null values
null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
total_nulls = sum(null_counts.values())
print(f"-> Missing/Null Values Audit: {total_nulls} null entries detected across all features.")

# Visualization 1: Class Imbalance Distribution
os.makedirs("visualizations", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Linear Counts
bars = ax1.bar(["Legitimate (0)", "Fraudulent (1)"], [legit_count, fraud_count], color=['#2b5c8f', '#d9534f'], width=0.5)
ax1.set_title("Transaction Class Distribution (Raw Count)", fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel("Number of Transactions", fontsize=11)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 3000, f"{int(yval):,}", ha='center', va='bottom', fontweight='bold')

# Subplot 2: Logarithmic Scale for Visibility
bars_log = ax2.bar(["Legitimate (0)", "Fraudulent (1)"], [legit_count, fraud_count], color=['#2b5c8f', '#d9534f'], width=0.5)
ax2.set_yscale('log')
ax2.set_title("Transaction Class Distribution (Log Scale)", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel("Count (Log Scale)", fontsize=11)
for bar in bars_log:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3, f"{int(yval):,}\n({(yval/total_rows)*100:.3f}%)", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
fig.savefig("visualizations/01_class_imbalance.png")
plt.close(fig)
print("-> Saved: visualizations/01_class_imbalance.png")

# 4. Feature Engineering & Weight Calculation for Imbalance Mitigation
print("\n[Step 4/8] Feature Assembly, Normalization & Imbalance Weighting...")
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

# Compute balanced class weights for loss function
# Weight = Total / (2 * count) -> weight for fraud is ~289.4, weight for legit is ~0.5008
weight_fraud = total_rows / (2.0 * fraud_count)
weight_legit = total_rows / (2.0 * legit_count)
print(f"-> Assigned Class Weights: Legit(0) = {weight_legit:.4f}, Fraud(1) = {weight_fraud:.4f}")

df = df.withColumn(
    "weight",
    F.when(F.col("Class") == 1, weight_fraud).otherwise(weight_legit)
)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)

# Train-Test Split (80% train, 20% test, reproducible seed 42)
print("-> Partitioning dataset into 80% Training and 20% Testing sets...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df = train_df.cache()
test_df = test_df.cache()

train_count = train_df.count()
test_count = test_df.count()
train_fraud = train_df.filter(F.col("Class") == 1).count()
test_fraud = test_df.filter(F.col("Class") == 1).count()

print(f"-> Train Set: {train_count:,} records (Fraud: {train_fraud:,} / {train_fraud/train_count*100:.3f}%)")
print(f"-> Test Set:  {test_count:,} records (Fraud: {test_fraud:,} / {test_fraud/test_count*100:.3f}%)")

# Fit feature pipeline
feature_pipeline = Pipeline(stages=[assembler, scaler])
feature_model = feature_pipeline.fit(train_df)
train_prepared = feature_model.transform(train_df).cache()
test_prepared = feature_model.transform(test_df).cache()

# Sample for Correlation Heatmap Visualization (10,000 stratified samples)
print("-> Generating feature correlation heatmap...")
sample_pd = df.sample(fraction=0.05, seed=42).toPandas()
corr_cols = [f"V{i}" for i in [1, 2, 3, 4, 7, 10, 11, 12, 14, 16, 17, 18]] + ["Amount", "Class"]
corr_mat = sample_pd[corr_cols].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.8, vmax=0.8, cbar=True, square=True)
plt.title("Correlation Matrix of Key PCA Latent Features & Transaction Attributes", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("visualizations/02_correlation_heatmap.png")
plt.close()
print("-> Saved: visualizations/02_correlation_heatmap.png")

# 5. Model Training & Evaluation Setup
eval_roc = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
eval_pr = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

def calculate_confusion_metrics(predictions, pred_col="prediction"):
    """Calculate TP, FP, TN, FN and derived metrics."""
    tp = predictions.filter(f"Class == 1 AND {pred_col} == 1.0").count()
    fp = predictions.filter(f"Class == 0 AND {pred_col} == 1.0").count()
    tn = predictions.filter(f"Class == 0 AND {pred_col} == 0.0").count()
    fn = predictions.filter(f"Class == 1 AND {pred_col} == 0.0").count()
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

results = {}
pred_dfs = {}

# -------------------------------------------------------------
# Model 1: Logistic Regression (Weighted for Imbalance)
# -------------------------------------------------------------
print("\n[Step 5/8] Training Supervised Classification Models...")
print("\n--- [Model 1] Logistic Regression (Cost-Sensitive / Weighted Loss) ---")
lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="Class",
    weightCol="weight",
    maxIter=30,
    regParam=0.01,
    elasticNetParam=0.0
)
t0 = time.time()
lr_model = lr.fit(train_prepared)
lr_train_time = time.time() - t0
print(f"-> Logistic Regression trained in {lr_train_time:.2f} s")

lr_preds = lr_model.transform(test_prepared).cache()
pred_dfs["Logistic Regression"] = lr_preds
lr_metrics = calculate_confusion_metrics(lr_preds)
lr_metrics["ROC_AUC"] = eval_roc.evaluate(lr_preds)
lr_metrics["PR_AUC"] = eval_pr.evaluate(lr_preds)
lr_metrics["TrainTime"] = lr_train_time
results["Logistic Regression"] = lr_metrics

print(f"   Accuracy:  {lr_metrics['Accuracy']:.4f}")
print(f"   Precision: {lr_metrics['Precision']:.4f}")
print(f"   Recall:    {lr_metrics['Recall']:.4f}")
print(f"   F1-Score:  {lr_metrics['F1']:.4f}")
print(f"   ROC-AUC:   {lr_metrics['ROC_AUC']:.4f}")
print(f"   PR-AUC:    {lr_metrics['PR_AUC']:.4f}")
print(f"   CM: TP={lr_metrics['TP']}, FP={lr_metrics['FP']}, TN={lr_metrics['TN']}, FN={lr_metrics['FN']}")

# -------------------------------------------------------------
# Model 2: Decision Tree Classifier
# -------------------------------------------------------------
print("\n--- [Model 2] Decision Tree Classifier ---")
dt = DecisionTreeClassifier(
    featuresCol="scaled_features",
    labelCol="Class",
    maxDepth=7,
    seed=42
)
t0 = time.time()
dt_model = dt.fit(train_prepared)
dt_train_time = time.time() - t0
print(f"-> Decision Tree trained in {dt_train_time:.2f} s")

dt_preds = dt_model.transform(test_prepared).cache()
pred_dfs["Decision Tree"] = dt_preds
dt_metrics = calculate_confusion_metrics(dt_preds)
dt_metrics["ROC_AUC"] = eval_roc.evaluate(dt_preds)
dt_metrics["PR_AUC"] = eval_pr.evaluate(dt_preds)
dt_metrics["TrainTime"] = dt_train_time
results["Decision Tree"] = dt_metrics

print(f"   Accuracy:  {dt_metrics['Accuracy']:.4f}")
print(f"   Precision: {dt_metrics['Precision']:.4f}")
print(f"   Recall:    {dt_metrics['Recall']:.4f}")
print(f"   F1-Score:  {dt_metrics['F1']:.4f}")
print(f"   ROC-AUC:   {dt_metrics['ROC_AUC']:.4f}")
print(f"   PR-AUC:    {dt_metrics['PR_AUC']:.4f}")
print(f"   CM: TP={dt_metrics['TP']}, FP={dt_metrics['FP']}, TN={dt_metrics['TN']}, FN={dt_metrics['FN']}")

# -------------------------------------------------------------
# Model 3: Linear Support Vector Machine (LinearSVC)
# -------------------------------------------------------------
print("\n--- [Model 3] Linear Support Vector Machine (LinearSVC) ---")
lsvc = LinearSVC(
    featuresCol="scaled_features",
    labelCol="Class",
    weightCol="weight",
    maxIter=25,
    regParam=0.01
)
t0 = time.time()
lsvc_model = lsvc.fit(train_prepared)
lsvc_train_time = time.time() - t0
print(f"-> Linear SVM trained in {lsvc_train_time:.2f} s")

lsvc_preds = lsvc_model.transform(test_prepared).cache()
pred_dfs["Linear SVM"] = lsvc_preds
lsvc_metrics = calculate_confusion_metrics(lsvc_preds)
lsvc_metrics["ROC_AUC"] = eval_roc.evaluate(lsvc_preds)
lsvc_metrics["PR_AUC"] = eval_pr.evaluate(lsvc_preds)
lsvc_metrics["TrainTime"] = lsvc_train_time
results["Linear SVM"] = lsvc_metrics

print(f"   Accuracy:  {lsvc_metrics['Accuracy']:.4f}")
print(f"   Precision: {lsvc_metrics['Precision']:.4f}")
print(f"   Recall:    {lsvc_metrics['Recall']:.4f}")
print(f"   F1-Score:  {lsvc_metrics['F1']:.4f}")
print(f"   ROC-AUC:   {lsvc_metrics['ROC_AUC']:.4f}")
print(f"   PR-AUC:    {lsvc_metrics['PR_AUC']:.4f}")
print(f"   CM: TP={lsvc_metrics['TP']}, FP={lsvc_metrics['FP']}, TN={lsvc_metrics['TN']}, FN={lsvc_metrics['FN']}")

# -------------------------------------------------------------
# Model 4: K-Means Clustering (Unsupervised Anomaly Exploration)
# -------------------------------------------------------------
print("\n[Step 6/8] Unsupervised Anomaly Exploration with K-Means...")
kmeans = KMeans(
    featuresCol="scaled_features",
    predictionCol="cluster",
    k=4,
    seed=42,
    maxIter=20
)
t0 = time.time()
kmeans_model = kmeans.fit(train_prepared)
km_train_time = time.time() - t0
print(f"-> K-Means trained in {km_train_time:.2f} s (Found 4 cluster centers)")

cluster_centers = [np.array(c) for c in kmeans_model.clusterCenters()]
cluster_centers_bc = spark.sparkContext.broadcast(cluster_centers)

def calc_distance_to_center(cluster_id, features_vec):
    center = cluster_centers_bc.value[int(cluster_id)]
    feat_arr = np.array(features_vec.toArray())
    return float(np.linalg.norm(feat_arr - center))

spark.udf.register("calc_dist", calc_distance_to_center, DoubleType())
from pyspark.sql.functions import udf

dist_udf = udf(calc_distance_to_center, DoubleType())

km_train_preds = kmeans_model.transform(train_prepared).withColumn("dist_to_center", dist_udf(F.col("cluster"), F.col("scaled_features")))
km_test_preds = kmeans_model.transform(test_prepared).withColumn("dist_to_center", dist_udf(F.col("cluster"), F.col("scaled_features"))).cache()

# Determine 98.5th percentile threshold on train set for anomaly definition
quantiles = km_train_preds.stat.approxQuantile("dist_to_center", [0.985], 0.001)
anomaly_threshold = quantiles[0]
print(f"-> 98.5th Percentile Anomaly Distance Threshold: {anomaly_threshold:.4f}")

km_test_preds = km_test_preds.withColumn(
    "anomaly_prediction",
    F.when(F.col("dist_to_center") >= anomaly_threshold, 1.0).otherwise(0.0)
).cache()

km_metrics = calculate_confusion_metrics(km_test_preds, pred_col="anomaly_prediction")
# For unsupervised distance scoring, we evaluate ROC directly using distance
eval_roc_dist = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="dist_to_center", metricName="areaUnderROC")
eval_pr_dist = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="dist_to_center", metricName="areaUnderPR")
km_metrics["ROC_AUC"] = eval_roc_dist.evaluate(km_test_preds)
km_metrics["PR_AUC"] = eval_pr_dist.evaluate(km_test_preds)
km_metrics["TrainTime"] = km_train_time
results["K-Means Anomaly Detector"] = km_metrics

print(f"   Accuracy:  {km_metrics['Accuracy']:.4f}")
print(f"   Precision: {km_metrics['Precision']:.4f}")
print(f"   Recall:    {km_metrics['Recall']:.4f}")
print(f"   F1-Score:  {km_metrics['F1']:.4f}")
print(f"   ROC-AUC:   {km_metrics['ROC_AUC']:.4f}")
print(f"   PR-AUC:    {km_metrics['PR_AUC']:.4f}")
print(f"   CM: TP={km_metrics['TP']}, FP={km_metrics['FP']}, TN={km_metrics['TN']}, FN={km_metrics['FN']}")

# 6. Generate Comprehensive Visualizations
print("\n[Step 7/8] Generating Performance Comparison Visualizations...")

# Visualization 3: Confusion Matrices Grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
model_keys = ["Logistic Regression", "Decision Tree", "Linear SVM", "K-Means Anomaly Detector"]
for ax, model_name in zip(axes.flatten(), model_keys):
    m = results[model_name]
    cm_matrix = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
    sns.heatmap(cm_matrix, annot=True, fmt=',d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f"{model_name}\n(Recall: {m['Recall']:.2%}, Prec: {m['Precision']:.2%})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label (0: Legit, 1: Fraud)', fontsize=10)
    ax.set_ylabel('True Label (0: Legit, 1: Fraud)', fontsize=10)
plt.tight_layout()
plt.savefig("visualizations/03_confusion_matrices.png")
plt.close()
print("-> Saved: visualizations/03_confusion_matrices.png")

# Visualization 4: ROC and Precision-Recall Curves
from sklearn.metrics import roc_curve, precision_recall_curve, auc

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

colors = {
    "Logistic Regression": "#1f77b4",
    "Decision Tree": "#2ca02c",
    "Linear SVM": "#ff7f0e",
    "K-Means Anomaly Detector": "#d62728"
}

# Plot ROC & PR curves using test subsets
# For LR
lr_pd = lr_preds.select("Class", vector_to_array("probability")[1].alias("prob")).toPandas()
fpr_lr, tpr_lr, _ = roc_curve(lr_pd["Class"], lr_pd["prob"])
prec_lr, rec_lr, _ = precision_recall_curve(lr_pd["Class"], lr_pd["prob"])
ax_roc.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {results['Logistic Regression']['ROC_AUC']:.4f})", color=colors["Logistic Regression"], lw=2)
ax_pr.plot(rec_lr, prec_lr, label=f"Logistic Regression (PR-AUC = {results['Logistic Regression']['PR_AUC']:.4f})", color=colors["Logistic Regression"], lw=2)

# For DT
dt_pd = dt_preds.select("Class", vector_to_array("probability")[1].alias("prob")).toPandas()
fpr_dt, tpr_dt, _ = roc_curve(dt_pd["Class"], dt_pd["prob"])
prec_dt, rec_dt, _ = precision_recall_curve(dt_pd["Class"], dt_pd["prob"])
ax_roc.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {results['Decision Tree']['ROC_AUC']:.4f})", color=colors["Decision Tree"], lw=2)
ax_pr.plot(rec_dt, prec_dt, label=f"Decision Tree (PR-AUC = {results['Decision Tree']['PR_AUC']:.4f})", color=colors["Decision Tree"], lw=2)

# For Linear SVM (uses rawPrediction margin)
lsvc_pd = lsvc_preds.select("Class", vector_to_array("rawPrediction")[1].alias("margin")).toPandas()
fpr_svm, tpr_svm, _ = roc_curve(lsvc_pd["Class"], lsvc_pd["margin"])
prec_svm, rec_svm, _ = precision_recall_curve(lsvc_pd["Class"], lsvc_pd["margin"])
ax_roc.plot(fpr_svm, tpr_svm, label=f"Linear SVM (AUC = {results['Linear SVM']['ROC_AUC']:.4f})", color=colors["Linear SVM"], lw=2)
ax_pr.plot(rec_svm, prec_svm, label=f"Linear SVM (PR-AUC = {results['Linear SVM']['PR_AUC']:.4f})", color=colors["Linear SVM"], lw=2)

# For KMeans Anomaly Detector (uses distance)
km_pd = km_test_preds.select("Class", "dist_to_center").toPandas()
fpr_km, tpr_km, _ = roc_curve(km_pd["Class"], km_pd["dist_to_center"])
prec_km, rec_km, _ = precision_recall_curve(km_pd["Class"], km_pd["dist_to_center"])
ax_roc.plot(fpr_km, tpr_km, label=f"K-Means Distance (AUC = {results['K-Means Anomaly Detector']['ROC_AUC']:.4f})", color=colors["K-Means Anomaly Detector"], lw=2)
ax_pr.plot(rec_km, prec_km, label=f"K-Means Distance (PR-AUC = {results['K-Means Anomaly Detector']['PR_AUC']:.4f})", color=colors["K-Means Anomaly Detector"], lw=2)

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (0.50)")
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11)
ax_roc.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight='bold')
ax_roc.legend(loc="lower right", frameon=True)

# Baseline for PR curve is the fraud prevalence
baseline_fraud_rate = fraud_count / total_rows
ax_pr.axhline(y=baseline_fraud_rate, color='k', linestyle='--', lw=1.5, label=f"Prevalence Baseline ({baseline_fraud_rate:.4f})")
ax_pr.set_xlim([0.0, 1.0])
ax_pr.set_ylim([0.0, 1.05])
ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11)
ax_pr.set_ylabel("Precision (PPV)", fontsize=11)
ax_pr.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight='bold')
ax_pr.legend(loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig("visualizations/04_roc_pr_curves.png")
plt.close()
print("-> Saved: visualizations/04_roc_pr_curves.png")

# Visualization 5: Model Comparison Grouped Bar Chart
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
comparison_df = pd.DataFrame(results).T[metrics_to_plot]

fig, ax = plt.subplots(figsize=(13, 6))
comparison_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
ax.set_title("Comparative Performance Benchmarks across PySpark MLlib Algorithms", fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=11)
ax.set_xlabel("Algorithm", fontsize=11)
ax.set_xticklabels(comparison_df.index, rotation=0, fontweight='bold')
ax.set_ylim([0, 1.15])
ax.legend(loc="upper right", ncol=3, frameon=True)
for p in ax.patches:
    height = p.get_height()
    if height > 0.05:
        ax.annotate(f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                    textcoords='offset points')
plt.tight_layout()
plt.savefig("visualizations/05_model_comparison_bar.png")
plt.close()
print("-> Saved: visualizations/05_model_comparison_bar.png")

# Visualization 6: K-Means Anomaly Cluster Projection
from sklearn.decomposition import PCA
sample_km = km_test_preds.sample(fraction=0.08, seed=42).toPandas()
sample_features = np.array([v.toArray() for v in sample_km["scaled_features"]])
pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(sample_features)
sample_km["pca1"] = coords_2d[:, 0]
sample_km["pca2"] = coords_2d[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Subplot 1: K-means Clusters
scatter1 = ax1.scatter(sample_km["pca1"], sample_km["pca2"], c=sample_km["cluster"], cmap='tab10', alpha=0.5, s=15)
ax1.set_title("K-Means Cluster Partitioning (PCA 2D)", fontsize=12, fontweight='bold')
ax1.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
fig.colorbar(scatter1, ax=ax1, label="Cluster Assignment")

# Subplot 2: Actual Fraud vs Distance Anomalies
norm_pts = sample_km[(sample_km["Class"] == 0) & (sample_km["anomaly_prediction"] == 0)]
anom_pts = sample_km[(sample_km["Class"] == 0) & (sample_km["anomaly_prediction"] == 1)]
fraud_pts = sample_km[sample_km["Class"] == 1]

ax2.scatter(norm_pts["pca1"], norm_pts["pca2"], c='#4a90e2', alpha=0.3, s=12, label="Normal Transaction")
ax2.scatter(anom_pts["pca1"], anom_pts["pca2"], c='#f39c12', alpha=0.7, s=25, marker='^', label="Unsupervised Anomaly (Dist > 98.5%)")
ax2.scatter(fraud_pts["pca1"], fraud_pts["pca2"], c='#e74c3c', alpha=1.0, s=45, marker='X', edgecolors='black', label="Ground Truth Fraud (Class 1)")
ax2.set_title("Unsupervised Anomaly Boundary vs Ground Truth Fraud", fontsize=12, fontweight='bold')
ax2.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax2.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax2.legend(loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig("visualizations/06_kmeans_anomaly_clusters.png")
plt.close()
print("-> Saved: visualizations/06_kmeans_anomaly_clusters.png")

# Visualization 7: Decision Tree Feature Importances & LR Coefficients
dt_importances = dt_model.featureImportances.toArray()
feat_imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": dt_importances}).sort_values(by="Importance", ascending=False).head(12)

lr_coeffs = np.abs(lr_model.coefficients.toArray())
feat_coef_df = pd.DataFrame({"Feature": feature_cols, "AbsCoefficient": lr_coeffs}).sort_values(by="AbsCoefficient", ascending=False).head(12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", palette="Blues_r", ax=ax1)
ax1.set_title("Top 12 Features - Decision Tree Gini Importance", fontsize=11, fontweight='bold')
ax1.set_xlabel("Relative Importance")

sns.barplot(data=feat_coef_df, x="AbsCoefficient", y="Feature", palette="Oranges_r", ax=ax2)
ax2.set_title("Top 12 Features - Logistic Regression |Beta Weight|", fontsize=11, fontweight='bold')
ax2.set_xlabel("Absolute Coefficient Magnitude")

plt.tight_layout()
plt.savefig("visualizations/07_feature_importance.png")
plt.close()
print("-> Saved: visualizations/07_feature_importance.png")

# 7. Batch Fraud Prediction Demonstration
print("\n[Step 8/8] Simulating Real-Time Transaction Screening Pipeline...")
# Select sample transactions: 5 frauds and 5 non-frauds
sample_frauds = test_prepared.filter("Class == 1").limit(5)
sample_legits = test_prepared.filter("Class == 0").limit(5)
demo_batch = sample_frauds.union(sample_legits)

# Predict using Logistic Regression & Decision Tree
demo_results = lr_model.transform(demo_batch).select(
    "Time", "Amount", "Class",
    vector_to_array("probability")[1].alias("LR_Fraud_Prob"),
    "prediction"
).toPandas()

print("\n--- SAMPLE BATCH TRANSACTION SCREENING RESULTS ---")
print(demo_results.to_string(index=False))

# Export models and summary results
os.makedirs("models", exist_ok=True)
summary_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "total_records": total_rows,
        "legitimate_records": legit_count,
        "fraudulent_records": fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "train_records": train_count,
        "test_records": test_count,
        "train_fraud": train_fraud,
        "test_fraud": test_fraud
    },
    "anomaly_threshold_kmeans": round(anomaly_threshold, 4),
    "benchmark_metrics": results
}

with open("models/metrics_summary.json", "w") as f:
    json.dump(summary_data, f, indent=4)
print("\n-> Saved metrics summary to models/metrics_summary.json")

# Export models
print("-> Saving PySpark MLlib Pipeline and Trained Models...")
try:
    feature_model.write().overwrite().save("models/feature_pipeline_model")
    lr_model.write().overwrite().save("models/logistic_regression_model")
    dt_model.write().overwrite().save("models/decision_tree_model")
    lsvc_model.write().overwrite().save("models/linearsvc_model")
    kmeans_model.write().overwrite().save("models/kmeans_model")
    print("-> Successfully serialized all PySpark models to disk!")
except Exception as e:
    print(f"-> Model serialization note (native winutils fallback): {e}")

spark.stop()
print("\n" + "=" * 80)
print("PYSPARK FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
