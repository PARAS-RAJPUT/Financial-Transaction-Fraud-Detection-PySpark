# Financial Transaction Fraud Detection using PySpark

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FPARAS-RAJPUT%2FFinancial-Transaction-Fraud-Detection-PySpark)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5.3-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PySpark MLlib](https://img.shields.io/badge/PySpark-MLlib-brightgreen)](https://spark.apache.org/mllib/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance distributed machine learning and anomaly detection system for credit card fraud screening using **Apache Spark 3.5.3** and **PySpark MLlib**, featuring a **live interactive web frontend deployable on Vercel in 1 click**. Designed to handle extreme class imbalance (**577 : 1** ratio) across **284,807 European cardholder transactions**.

---

## 🌐 Live Web Frontend (Vercel Deployable)

The project includes an interactive web dashboard (`index.html` + `vercel.json`) featuring:
1. **Real-Time Fraud Simulator:** Sliders for Amount, Time, and top PCA features ($V_{14}, V_{10}, V_{12}, V_{17}, V_4, V_7, V_{11}, V_3$).
2. **Instant Model Scoring:** Standardizes inputs using the exact Spark `StandardScaler` and computes fraud probabilities via cost-weighted `LogisticRegressionModel`.
3. **Interactive Decision Gauge:** Live risk meter displaying `APPROVE TRANSACTION` vs `WARNING: 2FA REQUIRED` vs `ALERT: FRAUD DETECTED - TRANSACTION BLOCKED`.
4. **Model Comparison Hub:** Interactive benchmark matrix, metric bar charts, and confusion matrix cards for all 4 models.
5. **Direct Capstone Report Download:** One-click download of the 23-page `CAPSTONE_PROJECT_REPORT.pdf`.

### 🚀 Deploy to Vercel in 60 Seconds:
1. Open [vercel.com/new](https://vercel.com/new) and log in with your GitHub account.
2. Select and import repository: `PARAS-RAJPUT/Financial-Transaction-Fraud-Detection-PySpark`.
3. Keep default settings (Root directory: `./`) and click **Deploy**!
4. Your live app will be published instantly with a free `.vercel.app` URL.

---

## 📊 Benchmark Results (Test Partition: 56,582 Records, 103 Frauds)

| Algorithm | Accuracy | Precision | Recall (TPR) | F1-Score | ROC-AUC | PR-AUC | TP | FP | TN | FN | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Weighted)** | 0.9897 | 0.1310 | **0.8252** | 0.2261 | 0.9750 | **0.6395** | **85** | 564 | 55,915 | **18** | 14.97 s |
| **Decision Tree Classifier** | **0.9992** | **0.8090** | 0.6990 | **0.7500** | 0.7472 | 0.5966 | 72 | **17** | **56,462** | 31 | **4.80 s** |
| **Linear Support Vector Machine** | 0.9884 | 0.1172 | **0.8252** | 0.2053 | **0.9759** | 0.6001 | **85** | 640 | 55,839 | **18** | 15.13 s |
| **K-Means Anomaly Detector ($k=4$)** | 0.9845 | 0.0756 | 0.6699 | 0.1358 | 0.9407 | 0.1810 | 69 | 844 | 55,635 | 34 | **8.88 s** |

> **Key Takeaway:** Logistic Regression and Linear SVM achieve **82.52% Recall**, capturing 85 of 103 fraudulent transactions. Decision Tree achieves the lowest false-alarm rate with **80.90% Precision** (only 17 false positives across 56,582 transactions). K-Means successfully identifies **66.99% of frauds completely unsupervised**.

---

## 📁 Repository Structure

```
.
├── index.html                              # Production Vercel interactive frontend
├── vercel.json                             # Vercel deployment & routing configuration
├── model_params.json                       # Serialized weights, intercepts, and scalers
├── sample_transactions.json                # Pre-extracted test cases (frauds and normal)
├── Financial_Transaction_Fraud_Detection_PySpark.ipynb  # Primary executed Jupyter Notebook
├── notebooks/
│   └── Financial_Transaction_Fraud_Detection_PySpark.ipynb
├── data/
│   └── creditcard.csv                      # (143.8 MB, auto-downloadable via script)
├── models/
│   ├── decision_tree_model/                # Serialized PySpark ML model
│   ├── feature_pipeline_model/             # Serialized VectorAssembler + StandardScaler
│   ├── kmeans_model/                       # Serialized K-Means clustering model
│   ├── linearsvc_model/                    # Serialized Linear Support Vector Machine
│   ├── logistic_regression_model/          # Serialized Logistic Regression model
│   └── metrics_summary.json                # Benchmark metrics summary in JSON
├── scripts/
│   ├── build_frontend.py                   # Generates the Vercel HTML frontend
│   ├── export_model_params.py              # Exports Spark weights to JSON
│   ├── extract_samples.py                  # Extracts realistic test cases
│   ├── train_and_evaluate.py               # Master distributed training & evaluation script
│   ├── download_data.py                    # Dataset downloader
│   └── export_pdf.py                       # Headless Edge PDF exporter
└── visualizations/
    ├── 01_class_imbalance.png              # Linear & log scale class distribution
    ├── 02_correlation_heatmap.png          # Correlation matrix of key PCA features
    ├── 03_confusion_matrices.png           # 2x2 confusion matrix grid
    ├── 04_roc_pr_curves.png                # ROC & Precision-Recall curves
    ├── 05_model_comparison_bar.png         # Grouped performance metrics bar chart
    ├── 06_kmeans_anomaly_clusters.png      # 2D PCA cluster space & anomaly boundary
    └── 07_feature_importance.png           # Gini importance & beta coefficients
```

---

## 🚀 Quickstart Guide

### 1. Run the Frontend Locally
Simply open `index.html` in your web browser or start a local server:
```bash
python -m http.server 3000
```
Then navigate to `http://localhost:3000`.

### 2. Run Distributed PySpark Training Pipeline
```bash
# 1. Install dependencies
pip install pyspark==3.5.3 findspark numpy pandas matplotlib seaborn scikit-learn

# 2. Download the Kaggle dataset
python scripts/download_data.py

# 3. Train models and generate benchmarks
python scripts/train_and_evaluate.py
```

### 3. Launch Interactive Jupyter Notebook
```bash
jupyter notebook Financial_Transaction_Fraud_Detection_PySpark.ipynb
```

---

## 📜 Academic Attribution
- **Institution:** School of Computer Science & Engineering, Lovely Professional University
- **Course Code:** INT315 — Capstone Project Report
