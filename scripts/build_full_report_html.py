import os
import base64
import json
import re

def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return ""

img_imbalance = get_base64_image("visualizations/01_class_imbalance.png")
img_heatmap = get_base64_image("visualizations/02_correlation_heatmap.png")
img_cm = get_base64_image("visualizations/03_confusion_matrices.png")
img_roc_pr = get_base64_image("visualizations/04_roc_pr_curves.png")
img_comparison = get_base64_image("visualizations/05_model_comparison_bar.png")
img_kmeans = get_base64_image("visualizations/06_kmeans_anomaly_clusters.png")
img_feat_imp = get_base64_image("visualizations/07_feature_importance.png")

with open("scripts/notebook_section.html", "r", encoding="utf-8") as f:
    notebook_html = f.read()

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CAPSTONE PROJECT REPORT - Financial Transaction Fraud Detection using PySpark</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    
    body {{
        font-family: "Segoe UI", Arial, Helvetica, sans-serif;
        color: #222222;
        line-height: 1.6;
        font-size: 10.5pt;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }}

    .page-header {{
        font-size: 8.5pt;
        color: #666666;
        border-bottom: 1px solid #d0d7de;
        padding-bottom: 5px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
    }}

    .page-break {{
        page-break-after: always;
        break-after: page;
    }}

    /* Title Page */
    .title-page {{
        text-align: center;
        padding-top: 35px;
        height: 90vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .uni-logo-box {{
        margin-bottom: 25px;
    }}
    .uni-title {{
        font-size: 26pt;
        font-weight: 800;
        color: #d35400;
        letter-spacing: 1px;
        margin: 0;
    }}
    .uni-subtitle {{
        font-size: 18pt;
        font-weight: 700;
        color: #2c3e50;
        letter-spacing: 2px;
        margin: 5px 0 0 0;
    }}
    .capstone-badge {{
        font-size: 13pt;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #333333;
        margin-top: 35px;
    }}
    .term-text {{
        font-size: 10.5pt;
        color: #666666;
        margin-top: 4px;
    }}
    .project-main-title {{
        font-size: 23pt;
        font-weight: 800;
        color: #111111;
        margin: 40px 0 10px 0;
        line-height: 1.25;
        text-transform: uppercase;
    }}
    .project-subtitle {{
        font-size: 12.5pt;
        font-weight: 600;
        color: #2980b9;
        margin: 0 0 35px 0;
    }}
    .divider-line {{
        width: 100%;
        height: 1px;
        background: #bdc3c7;
        margin: 25px auto;
    }}
    .submission-block {{
        font-size: 11pt;
        line-height: 1.8;
    }}
    .submission-name {{
        font-size: 12.5pt;
        font-weight: 700;
        color: #111111;
    }}
    .department-block {{
        font-size: 11pt;
        color: #2c3e50;
        margin-top: 25px;
        font-weight: 600;
        line-height: 1.5;
    }}

    /* Section Headings */
    h1 {{
        font-size: 17pt;
        color: #111111;
        text-transform: uppercase;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 5px;
        margin-top: 18px;
        margin-bottom: 16px;
    }}
    h2 {{
        font-size: 12.5pt;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
    }}
    h3 {{
        font-size: 11pt;
        color: #34495e;
        margin-top: 14px;
        margin-bottom: 8px;
    }}

    p {{
        text-align: justify;
        margin-bottom: 11px;
        line-height: 1.6;
    }}

    ul, ol {{
        margin-top: 5px;
        margin-bottom: 12px;
        padding-left: 22px;
    }}
    li {{
        margin-bottom: 5px;
        text-align: justify;
    }}

    /* Tables */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0 18px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #bdc3c7;
        padding: 7px 9px;
        text-align: left;
    }}
    th {{
        background-color: #f4f6f7;
        font-weight: 700;
        color: #2c3e50;
    }}
    tr:nth-child(even) {{
        background-color: #fafbfc;
    }}
    .table-caption {{
        font-size: 8.5pt;
        font-style: italic;
        text-align: center;
        color: #555555;
        margin-top: 5px;
        margin-bottom: 16px;
    }}

    /* Jupyter Notebook Styling */
    .nb-markdown {{
        background-color: #fdfdfe;
        border-left: 4px solid #3498db;
        padding: 8px 14px;
        margin: 16px 0 10px 0;
        font-size: 10pt;
    }}
    .nb-cell {{
        margin-top: 12px;
        margin-bottom: 4px;
        page-break-inside: avoid;
    }}
    .nb-input-label {{
        color: #2980b9;
        font-family: monospace;
        font-size: 9pt;
        font-weight: bold;
        margin-bottom: 2px;
    }}
    .nb-input-code pre {{
        background-color: #f8f9fa;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 10px 12px;
        font-family: "Consolas", monospace;
        font-size: 8.5pt;
        margin: 0;
        overflow-x: auto;
        line-height: 1.4;
    }}
    .nb-output {{
        margin-top: 4px;
        margin-bottom: 16px;
        page-break-inside: avoid;
    }}
    .nb-output-label {{
        color: #c0392b;
        font-family: monospace;
        font-size: 9pt;
        font-weight: bold;
        margin-bottom: 2px;
    }}
    .nb-output-stream {{
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #718096;
        padding: 9px 12px;
        font-family: "Consolas", monospace;
        font-size: 8.2pt;
        color: #2d3748;
        margin: 0;
        white-space: pre-wrap;
        line-height: 1.35;
    }}
    .nb-output-table table {{
        font-size: 8.2pt;
        margin: 6px 0;
    }}
    .nb-output-table th {{
        background-color: #edf2f7;
    }}

    /* Figures */
    .figure-container {{
        text-align: center;
        margin: 18px 0;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #e1e4e8;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .figure-caption {{
        font-size: 8.5pt;
        font-style: italic;
        color: #555555;
        margin-top: 6px;
    }}

    .signature-row {{
        display: flex;
        justify-content: space-between;
        margin-top: 50px;
    }}
    .signature-box {{
        width: 45%;
        line-height: 1.8;
    }}
</style>
</head>
<body>

<!-- ========================================== -->
<!-- 1. TITLE PAGE -->
<!-- ========================================== -->
<div class="title-page">
    <div class="uni-logo-box">
        <h1 class="uni-title" style="border:none; margin:0;">LOVELY</h1>
        <h2 class="uni-subtitle" style="margin:0;">PROFESSIONAL UNIVERSITY</h2>
    </div>

    <div>
        <div class="capstone-badge">CAPSTONE PROJECT REPORT</div>
        <div class="term-text">(Project Term: August – December)</div>
        
        <div class="project-main-title">Financial Transaction Fraud Detection using PySpark</div>
        <div class="project-subtitle">Distributed Machine Learning, Big Data Analytics, and Anomaly Detection on Large-Scale Financial Streams</div>
    </div>

    <div class="divider-line"></div>

    <div class="submission-block">
        <div>Submitted by</div>
        <div class="submission-name">Paras Rajput &nbsp;|&nbsp; Reg. No: 12304811</div>
        <div style="margin-top: 8px;">Course Code: <strong>INT315</strong></div>
        <div style="margin-top: 15px;">Under the Guidance of</div>
        <div style="font-weight: 700; color: #111;">Miss Zeenat Zehra, Assistant Professor</div>
    </div>

    <div class="department-block">
        School of Computer Science and Engineering<br>
        Lovely Professional University, Phagwara, Punjab — 144411
    </div>
</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 2. DECLARATION -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>2</span>
</div>

<h1>DECLARATION</h1>
<p>
I, the undersigned, confirm that the work presented in this capstone report titled <strong>‘Financial Transaction Fraud Detection using PySpark — Distributed Machine Learning and Anomaly Detection on Large-Scale Financial Streams’</strong> has been carried out independently by me during the project term, under the supervision of <strong>Miss Zeenat Zehra</strong>, Assistant Professor, in partial fulfilment of the requirements for the award of the B.Tech degree in Computer Science and Engineering from Lovely Professional University, Phagwara.
</p>
<p>
This work is original and has not been submitted, either in full or in part, to any other academic institution or degree programme. All external references, datasets, algorithmic frameworks, APIs, and published works that influenced this project are cited in the Bibliography. The source code and distributed pipelines were written and executed by me; the use of open-source libraries is acknowledged appropriately.
</p>
<p>
I declare that every claim, result, metric, and figure presented in this report accurately reflects the work performed and the empirical outcomes observed on the benchmark European Cardholders Credit Card Fraud dataset. No fabrication, falsification, or misrepresentation of data has occurred at any stage.
</p>

<div class="signature-row" style="margin-top: 60px;">
    <div class="signature-box">
        <strong>Name of Student:</strong> Paras Rajput<br>
        <strong>Registration Number:</strong> 12304811<br>
        <strong>Signature:</strong> ____________________________
    </div>
    <div class="signature-box" style="text-align: right;">
        <br><br>
        <strong>Date:</strong> 24-10-2026
    </div>
</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 3. CERTIFICATE -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>3</span>
</div>

<h1>CERTIFICATE</h1>
<p>
This is to certify that the project report titled <strong>‘Financial Transaction Fraud Detection using PySpark — Distributed Machine Learning and Anomaly Detection on Large-Scale Financial Streams’</strong> has been prepared and submitted by <strong>Paras Rajput (Reg. No: 12304811)</strong> under my direct supervision. The work represents a genuine, mathematically sound contribution in the application of distributed big data engineering, Apache Spark computing, PySpark MLlib pipelines, classification algorithms, and unsupervised anomaly detection to real-world financial fraud intelligence.
</p>
<p>
I am satisfied that the scope, methodology, distributed system design, validation protocols, and empirical outcomes documented here meet the standards expected for a Capstone Project submission at the School of Computer Science and Engineering, Lovely Professional University, Phagwara. To the best of my knowledge, this work has not been submitted elsewhere for any academic award.
</p>

<div class="signature-row" style="margin-top: 70px;">
    <div class="signature-box">
        <strong>Signature of the Mentor:</strong> ____________________________<br>
        <strong>Designation:</strong> Assistant Professor<br>
        School of Computer Science and Engineering,<br>
        Lovely Professional University,<br>
        Phagwara, Punjab — 144411.
    </div>
    <div class="signature-box" style="text-align: right;">
        <br><br>
        <strong>Date:</strong> ____________________________
    </div>
</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 4. ACKNOWLEDGEMENT -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>4</span>
</div>

<h1>ACKNOWLEDGEMENT</h1>
<p>
Building this distributed financial fraud detection platform required rigorous engineering, theoretical analysis, and supportive mentorship. I express my sincere gratitude to my project mentor, <strong>Miss Zeenat Zehra</strong>, Assistant Professor, School of Computer Science and Engineering, whose insightful critique, continuous encouragement, and high academic expectations shaped the problem formulation, challenged my algorithmic decisions, and refined successive iterations of this system and report.
</p>
<p>
I owe a profound debt to the researchers of the <strong>Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB)</strong> for publishing and curating the benchmark European cardholders fraud dataset, which forms the empirical baseline of this work.
</p>
<p>
I gratefully acknowledge the open-source community behind the distributed computing ecosystem: the <strong>Apache Software Foundation</strong> for Apache Spark, the maintainers of <strong>PySpark</strong>, <strong>Spark MLlib</strong>, <strong>Python</strong>, <strong>Pandas</strong>, <strong>NumPy</strong>, <strong>Scikit-Learn</strong>, <strong>Matplotlib</strong>, and <strong>Seaborn</strong>.
</p>
<p>
Finally, I express my deepest heartfelt thanks to my parents, family, and peers for their unwavering patience, encouragement, and understanding throughout this demanding capstone semester.
</p>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 5. TABLE OF CONTENTS -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>5</span>
</div>

<h1>TABLE OF CONTENTS</h1>
<table>
    <tr><th>Chapter / Section</th><th>Page</th></tr>
    <tr><td><strong>DECLARATION</strong></td><td>2</td></tr>
    <tr><td><strong>CERTIFICATE</strong></td><td>3</td></tr>
    <tr><td><strong>ACKNOWLEDGEMENT</strong></td><td>4</td></tr>
    <tr><td><strong>TABLE OF CONTENTS</strong></td><td>5</td></tr>
    <tr><td><strong>1. INTRODUCTION</strong></td><td>7</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;1.1 Background of the Study</td><td>7</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;1.2 Motivation</td><td>7</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;1.3 Problem Statement</td><td>8</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;1.4 Objectives</td><td>8</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;1.5 Overview of the Proposed System</td><td>8</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;1.6 Scope and Limitations</td><td>9</td></tr>
    <tr><td><strong>2. PROFILE OF THE PROBLEM</strong></td><td>10</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;2.1 Problem Statement</td><td>10</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;2.2 Rationale and Scope of the Study</td><td>10</td></tr>
    <tr><td><strong>3. EXISTING SYSTEM</strong></td><td>11</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;3.1 Introduction</td><td>11</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;3.2 Review of Existing Approaches</td><td>11</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;3.3 System Architecture of Existing Tools</td><td>11</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;3.4 What Is New in This System</td><td>12</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;3.5 Implications for System Design</td><td>12</td></tr>
    <tr><td><strong>4. PROBLEM ANALYSIS</strong></td><td>13</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;4.1 Product Definition</td><td>13</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;4.2 Feasibility Analysis</td><td>13</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;4.3 Risk Analysis</td><td>14</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;4.4 Project Plan</td><td>14</td></tr>
    <tr><td><strong>5. SOFTWARE REQUIREMENT ANALYSIS</strong></td><td>15</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;5.1 Functional Requirements</td><td>15</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;5.2 Non-Functional Requirements</td><td>15</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;5.3 General Description</td><td>16</td></tr>
    <tr><td><strong>6. DESIGN</strong></td><td>17</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;6.1 System Design</td><td>17</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;6.2 Pipeline and Data Design</td><td>17</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;6.3 Machine Learning Pipeline Specification</td><td>18</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;6.4 Detailed Module Design</td><td>19</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;6.5 Flowcharts & Sequence Architecture</td><td>20</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;6.6 Pseudocode Algorithms</td><td>21</td></tr>
    <tr><td><strong>7. TESTING</strong></td><td>23</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;7.1 Functional Testing</td><td>23</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;7.2 Structural Testing</td><td>24</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;7.3 Levels of Testing</td><td>24</td></tr>
    <tr><td><strong>8. IMPLEMENTATION & RESULT ANALYSIS</strong></td><td>25</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;8.1 Implementation of the Project</td><td>25</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;8.2 Conversion Plan</td><td>26</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;8.3 Post-Implementation and Maintenance</td><td>26</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;8.4 Result Analysis & Comparative Benchmarks</td><td>27</td></tr>
    <tr><td><strong>9. PROJECT LEGACY & ETHICS</strong></td><td>30</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;9.1 Current Status of the Project</td><td>30</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;9.2 Remaining Areas of Concern</td><td>30</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;9.3 Technical and Managerial Lessons Learnt</td><td>30</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;9.4 Ethics and Data Integrity</td><td>31</td></tr>
    <tr><td><strong>10. USER MANUAL (PYSPARK TERMINAL)</strong></td><td>32</td></tr>
    <tr><td><strong>11. COMPLETE JUPYTER NOTEBOOK IMPLEMENTATION (.ipynb CODE & OUTPUTS)</strong></td><td>34</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.1 Cell 1: Environment Setup & Spark Initialization</td><td>34</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.2 Cell 2: Distributed Data Ingestion & Schema Inspection</td><td>35</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.3 Cell 3: Exploratory Data Analysis & Imbalance Quantification</td><td>36</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.4 Cell 4: Transaction Amount Statistics by Class</td><td>37</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.5 Cell 5: Feature Engineering & Cost-Sensitive Loss Weighting</td><td>38</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.6 Cell 6: Logistic Regression Training & Evaluation Output</td><td>39</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.7 Cell 7: Decision Tree Classifier Training & Evaluation Output</td><td>40</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.8 Cell 8: Linear Support Vector Machine Training & Output</td><td>41</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.9 Cell 9: Unsupervised K-Means Anomaly Scoring & Threshold Output</td><td>42</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.10 Cell 10: Model Comparison Benchmark Summary Table</td><td>43</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;11.11 Cell 11: Real-Time Stream Scoring Simulation Output</td><td>44</td></tr>
    <tr><td><strong>12. BIBLIOGRAPHY</strong></td><td>45</td></tr>
</table>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 6. CHAPTER 1: INTRODUCTION -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>7</span>
</div>

<h1>1. INTRODUCTION</h1>
<p>
Electronic financial payments have become the cornerstone of global commerce, processing trillions of dollars across digital point-of-sale terminals, mobile wallets, and e-commerce payment gateways. Concurrently, criminal fraud syndicates have evolved sophisticated attack vectors—including automated credential stuffing, card-not-present (CNP) identity theft, and high-velocity micro-transaction testing. According to global banking security audits, credit card fraud causes tens of billions of dollars in losses annually, directly eroding consumer trust and imposing severe financial reconciliation overhead on banks, payment processors, and merchants.
</p>
<p>
Traditional fraud prevention architectures rely predominantly on static, rule-based screening systems (e.g., flagging transactions exceeding $5,000, or rejecting cards when multiple transactions originate across different countries within thirty minutes). While such static heuristic filters capture naive fraudulent attempts, they completely fail when faced with adaptive adversary evasion, non-linear fraud patterns, and micro-transaction card testing. Furthermore, as transactional velocity reaches tens of thousands of operations per second, monolithic relational database systems experience extreme computational bottlenecks, causing unacceptable payment processing latency.
</p>
<p>
This capstone project introduces <strong>Financial Transaction Fraud Detection using PySpark</strong>, a high-performance distributed machine learning architecture built on <strong>Apache Spark</strong> and <strong>PySpark MLlib</strong>. The system leverages distributed Resilient Distributed Datasets (RDDs) and in-memory Spark DataFrames to ingest, preprocess, assemble, scale, and evaluate millions of transactional records in parallel. By uniting cost-sensitive supervised classification models (<strong>Logistic Regression</strong>, <strong>Decision Trees</strong>, and <strong>Linear Support Vector Machines</strong>) with unsupervised distance-based anomaly clustering (<strong>K-Means</strong>), this platform provides a dual-layer defense against both known fraud signatures and zero-day anomalous attacks.
</p>

<h2>1.1 Background of the Study</h2>
<p>
The empirical foundation of modern fraud detection research centers on high-volume card transaction streams characterized by extreme class imbalance. In legitimate real-world operations, only a microscopic fraction of all card swipes represent criminal fraud—typically fewer than 0.2%. The benchmark <strong>Kaggle Credit Card Fraud Detection dataset</strong> (compiled by the Machine Learning Group at Université Libre de Bruxelles) contains 284,807 card transactions recorded over a two-day span, of which only 492 are fraudulent (an extreme imbalance ratio of 577 to 1).
</p>
<p>
Three fundamental characteristics define this problem space:
</p>
<ul>
    <li><strong>Extreme Class Imbalance:</strong> Naive machine learning models trained on this data achieve 99.83% accuracy simply by classifying every single transaction as non-fraudulent, yet fail to catch a single criminal dollar. Standard classification metrics like raw accuracy are entirely misleading; the system must optimize for Recall (Sensitivity), Precision, the Area Under the Precision-Recall Curve (PR-AUC), and the Area Under the ROC Curve (ROC-AUC).</li>
    <li><strong>Computational Scale & Latency:</strong> Real-world banking rails require real-time risk scoring within 50 to 100 milliseconds per transaction, combined with batch processing of daily historical ledgers containing billions of rows. Single-node tools (such as standard Scikit-Learn or Pandas) hit memory bounds and cannot scale across cluster nodes.</li>
    <li><strong>Information Privacy (PCA Transformed Features):</strong> Due to strict customer confidentiality laws (GDPR, PCI-DSS), raw account numbers, merchant names, and geographic coordinates are transformed via Principal Component Analysis (PCA) into latent orthogonal features (V1 through V28), alongside continuous Time and transaction Amount.</li>
</ul>

<h2>1.2 Motivation</h2>
<p>
The motivation behind this project is twofold:
</p>
<ul>
    <li><strong>Financial & Operational:</strong> Every false negative (a missed fraudulent charge) results in direct monetary theft, chargeback penalties, and potential regulatory liability. Conversely, every false positive (a legitimate customer purchase wrongly blocked at the checkout register) damages user loyalty and drives churn. An enterprise-grade fraud intelligence system must give credit card issuers fine-grained mathematical control over the precision-recall trade-off.</li>
    <li><strong>Architectural & Technological:</strong> Demonstrating that Apache Spark's distributed in-memory data processing paradigm can be cleanly integrated with PySpark MLlib pipelines to solve non-linear classification and unsupervised anomaly detection on massive datasets. The project explores distributed feature normalization (StandardScaler), vectorization (VectorAssembler), and custom cost-sensitive weighted loss functions to counteract severe imbalance without relying on synthetic data fabrication.</li>
</ul>

<h2>1.3 Problem Statement</h2>
<p>
Modern financial systems face four core challenges:
</p>
<ul>
    <li><strong>The Imbalance Trap:</strong> Standard statistical algorithms assume balanced class distributions. Under a 577:1 imbalance ratio, unweighted algorithms minimize cross-entropy loss by predicting the majority class, resulting in disastrous false-negative rates for fraud.</li>
    <li><strong>Single-Node Scalability Bottlenecks:</strong> Memory exhaustion and CPU saturation occur when single-machine frameworks attempt to ingest and train models on multi-gigabyte financial ledgers.</li>
    <li><strong>Zero-Day Novel Fraud Blindspots:</strong> Supervised models trained exclusively on historical fraud patterns cannot recognize novel, emergent attack vectors that have never been previously labeled.</li>
    <li><strong>Evaluation Metric Pitfalls:</strong> Relying on accuracy rather than ROC-AUC, PR-AUC, and cost-benefit confusion matrices conceals catastrophic model failure.</li>
</ul>

<h2>1.4 Objectives</h2>
<p>
Five measurable, verifiable objectives guide the development of this platform:
</p>
<ul>
    <li><strong>O1 — Distributed Ingestion & Hygiene:</strong> Build an Apache Spark distributed pipeline capable of ingesting high-volume financial transaction records, enforcing typed schemas, confirming zero null/corrupted fields, and maintaining microsecond distributed caching.</li>
    <li><strong>O2 — Imbalance-Compensating Feature Pipeline:</strong> Design a distributed preprocessing pipeline using PySpark's VectorAssembler and StandardScaler that dynamically calculates inverse-class-frequency cost weights (w0 ≈ 0.50, w1 ≈ 289.44), penalizing missed fraud 577x more heavily than false alarms.</li>
    <li><strong>O3 — Supervised Model Benchmarking:</strong> Train, tune, and evaluate three distinct PySpark MLlib classification models: Logistic Regression with elastic-net regularization and cost-sensitive loss, Decision Tree Classifier with entropy/Gini impurity optimization, and Linear Support Vector Machine (LinearSVC) with margin maximization.</li>
    <li><strong>O4 — Unsupervised Anomaly Exploration:</strong> Implement a distributed K-Means clustering model (k=4) that scores transactions by Euclidean distance to the nearest cluster centroid, establishing a 98.5th percentile anomaly threshold to detect zero-day fraud without supervision.</li>
    <li><strong>O5 — Comprehensive Empirical Evaluation & Visualization:</strong> Generate publication-grade comparative visualizations (Confusion Matrices, ROC Curves, Precision-Recall Curves, PCA Cluster Plots, and Feature Importances) and demonstrate simulated real-time stream scoring.</li>
</ul>

<h2>1.5 Overview of the Proposed System</h2>
<p>
The system is structured as a modular, four-tier distributed pipeline: Ingestion Tier, Feature Engineering Tier, Machine Learning Modeling Tier, and Scoring & Evaluation Tier.
</p>

<h2>1.6 Scope and Limitations</h2>
<p>
The scope covers distributed machine learning workflows on numerical financial transaction records, featuring 284,807 transactions. Evaluates three supervised classifiers and one unsupervised detector. The limitations are clearly stated: features V1 to V28 are pre-transformed via PCA, preventing semantic domain feature engineering (such as cardholder billing zip codes or merchant categories). The system operates on batch-distributed processing and simulated streaming.
</p>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 7. CHAPTER 2: PROFILE OF THE PROBLEM -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>10</span>
</div>

<h1>2. PROFILE OF THE PROBLEM</h1>

<h2>2.1 Problem Statement</h2>
<p>
In financial transaction processing, fraudulent transactions represent an adversarial needle in a haystack. While accounting for only 0.172% of total transaction volume, their financial impact is catastrophic. When a fraudulent transaction slips past security, the merchant suffers chargeback fines, the cardholder endures identity theft, and the issuing bank must reimburse the lost funds.
</p>
<p>
Existing single-node analytic scripts fail when datasets expand beyond system RAM. Furthermore, conventional machine learning models penalize false positives and false negatives equally. In fraud detection, however, a False Negative (allowing a $2,000 stolen-card transaction) is orders of magnitude more damaging than a False Positive (prompting a cardholder for an SMS two-factor verification code). A mathematically rigorous fraud detection architecture must directly incorporate this cost asymmetry into its training objective.
</p>

<h2>2.2 Rationale and Scope of the Study</h2>
<p>
The rationale for choosing Apache Spark and PySpark MLlib lies in Spark's memory-centric distributed architecture. By maintaining resilient distributed datasets in RAM across cluster partitions, PySpark delivers training and inference throughput up to 100 times faster than disk-bound MapReduce architectures and eliminates the memory limits of single-node Python libraries.
</p>

<div class="figure-container">
    <img src="{img_imbalance}" alt="Class Imbalance Distribution">
    <div class="figure-caption">Figure 2.1: Transaction Class Distribution showing severe 577:1 imbalance (Linear vs Logarithmic Scale)</div>
</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 8. CHAPTER 3: EXISTING SYSTEM -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>11</span>
</div>

<h1>3. EXISTING SYSTEM</h1>

<h2>3.1 Introduction</h2>
<p>
Before fixing the architecture of our PySpark engine, an exhaustive review of current industrial and academic fraud detection systems was conducted across three classes: manual heuristic rule engines, single-node Python data science scripts, and commercial enterprise suites.
</p>

<h2>3.2 Review of Existing Approaches</h2>
<p>
Table 3.1 summarizes the qualitative and quantitative comparison between existing legacy architectures and the proposed PySpark system.
</p>

<table>
    <tr><th>Capability / Metric</th><th>Legacy Rule Engines</th><th>Single-Node Python (Scikit-Learn)</th><th>Proprietary Commercial Suites</th><th>Proposed PySpark System</th></tr>
    <tr><td><strong>Data Scalability</strong></td><td>Low (Relational bottlenecks)</td><td>Low (Single-machine RAM limit)</td><td>High (Proprietary cluster)</td><td><strong>Very High (Horizontal Spark scaling)</strong></td></tr>
    <tr><td><strong>Adaptability to Novel Fraud</strong></td><td>Zero (Manual updates only)</td><td>Medium (Periodic retraining)</td><td>High (Continuous learning)</td><td><strong>High (Hybrid Supervised + K-Means)</strong></td></tr>
    <tr><td><strong>Class Imbalance Handling</strong></td><td>Arbitrary manual tuning</td><td>SMOTE (Risks synthetic noise)</td><td>Proprietary weighting</td><td><strong>Cost-Sensitive Loss Weighting (577:1)</strong></td></tr>
    <tr><td><strong>Inference Throughput</strong></td><td>High (Simple rules)</td><td>Low (Single-threaded)</td><td>High</td><td><strong>High (Parallelized distributed partitions)</strong></td></tr>
    <tr><td><strong>Openness & Extensibility</strong></td><td>Proprietary SQL rules</td><td>Open Source</td><td>Closed Commercial</td><td><strong>100% Open Source (Apache Spark)</strong></td></tr>
    <tr><td><strong>Reproducibility & Auditability</strong></td><td>Poor (Disjointed rule files)</td><td>Good</td><td>Poor (Proprietary black-box)</td><td><strong>Complete (MLlib Pipelines & JSON logs)</strong></td></tr>
</table>
<div class="table-caption">Table 3.1: Qualitative comparison of typical fraud detection approaches with the proposed PySpark system</div>

<h2>3.3 System Architecture of Existing Tools</h2>
<p>
Traditional bank fraud architectures route incoming transactions through a relational transactional database (OLTP) to an external rule engine. When a rule is tripped, an analyst queue ticket is generated. This architecture suffers from three major flaws: I/O disk bottlenecks, static vulnerability, and the equal loss assumption.
</p>

<h2>3.4 What Is New in This System</h2>
<ul>
    <li><strong>Native Distributed MLlib Pipeline:</strong> Combines VectorAssembler and StandardScaler into a single reproducible pipeline.</li>
    <li><strong>Cost-Sensitive Asymmetric Loss:</strong> Penalizes missed fraud by 289.44x, preventing models from degenerating into trivial zero-predictors.</li>
    <li><strong>Hybrid Zero-Day Anomaly Detection:</strong> Employs K-Means centroid distance to detect unlabelled fraud without supervision.</li>
</ul>

<div class="figure-container">
    <img src="{img_heatmap}" alt="Feature Correlation Heatmap">
    <div class="figure-caption">Figure 3.1: Correlation matrix of key PCA latent features showing predictive correlation with fraud Class</div>
</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 9. CHAPTER 4: PROBLEM ANALYSIS -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>13</span>
</div>

<h1>4. PROBLEM ANALYSIS</h1>

<h2>4.1 Product Definition</h2>
<p>
AquaLens-style intelligence platform: The system delivers automated distributed batch model training, evaluation, anomaly distance thresholding, and real-time transaction screening that outputs triage decisions (e.g., APPROVE vs BLOCK).
</p>

<h2>4.2 Feasibility Analysis</h2>
<ul>
    <li><strong>Technical Feasibility:</strong> Confirmed. The entire pipeline runs on Apache Spark 3.5.3, PySpark, Python 3.10, and Java 8 SE Runtime. Spark's in-memory execution engine easily accommodates the 284,807 transactions on local multi-core threads and can be seamlessly scaled to an AWS EMR or Databricks multi-node cluster without modifying code.</li>
    <li><strong>Operational Feasibility:</strong> Confirmed. The pipeline is fully automated and self-contained. Analysts can execute end-to-end training and evaluation through a single terminal command or explore interactively through a clean Jupyter Notebook.</li>
    <li><strong>Economic Feasibility:</strong> Outstanding. Zero software licensing fees by leveraging the open-source Apache Spark and Python ecosystem.</li>
</ul>

<h2>4.3 Risk Analysis</h2>
<table>
    <tr><th>Risk ID</th><th>Identified Risk</th><th>Severity</th><th>Probability</th><th>Mitigation Strategy</th></tr>
    <tr><td><strong>R1</strong></td><td>Severe class imbalance causing models to ignore fraud</td><td>Critical</td><td>High</td><td>Implement inverse-frequency cost-sensitive loss weighting (w_fraud = 289.44).</td></tr>
    <tr><td><strong>R2</strong></td><td>Data leakage between training and evaluation partitions</td><td>High</td><td>Medium</td><td>Enforce strict PySpark Pipeline architecture where scaler parameters are fit exclusively on the 80% train split.</td></tr>
    <tr><td><strong>R3</strong></td><td>Memory exhaustion during Spark worker execution</td><td>High</td><td>Low</td><td>Set spark.driver.memory=4g, spark.driver.maxResultSize=2g, and tune spark.sql.shuffle.partitions=10.</td></tr>
    <tr><td><strong>R4</strong></td><td>Python worker socket timeouts on Windows environments</td><td>High</td><td>Medium</td><td>Explicitly configure PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON to point to the active Python interpreter.</td></tr>
    <tr><td><strong>R5</strong></td><td>Overfitting on high-dimensional PCA features</td><td>Medium</td><td>Medium</td><td>Implement L2 regularization (regParam=0.01) on Logistic Regression / LinearSVM and constrain Decision Tree maxDepth=7.</td></tr>
</table>
<div class="table-caption">Table 4.1: Risk register and engineering mitigation strategies</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 10. CHAPTER 6: DESIGN -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>17</span>
</div>

<h1>6. DESIGN</h1>

<h2>6.1 System Design</h2>
<p>
The system follows a four-tier distributed pipeline: Ingestion, Feature Pipeline, Model Training, and Evaluation/Inference.
</p>

<h2>6.2 Data Pipeline Design</h2>
<p>
The schema comprises 31 attributes: Time, V1 through V28 (PCA orthogonal projections), Amount, and Class. All 30 continuous features are assembled into a dense vector via <code>VectorAssembler</code> and normalized to zero mean and unit variance via <code>StandardScaler</code>.
</p>

<h2>6.3 Algorithmic Model Specifications</h2>
<ol>
    <li><strong>Logistic Regression:</strong> Cost-sensitive weighted log-loss optimization with L2 regularization.</li>
    <li><strong>Decision Tree Classifier:</strong> Recursive partitioning based on Gini Impurity with max depth 7.</li>
    <li><strong>Linear Support Vector Machine (LinearSVC):</strong> Soft-margin hinge loss optimization with cost-sensitive class weights.</li>
    <li><strong>K-Means Anomaly Scorer:</strong> Unsupervised 4-cluster centroid partition scoring each point by Euclidean distance. Transactions exceeding the 98.5th percentile distance (τ = 12.1889) are flagged as anomalies.</li>
</ol>

<div class="figure-container">
    <img src="{img_kmeans}" alt="K-Means Anomaly Cluster Projection">
    <div class="figure-caption">Figure 6.1: 2D PCA projection of K-Means clusters and the unsupervised anomaly boundary separating normal vs fraud points</div>
</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 11. CHAPTER 7: TESTING -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>23</span>
</div>

<h1>7. TESTING</h1>

<h2>7.1 Functional Testing</h2>
<table>
    <tr><th>Test ID</th><th>Test Scenario</th><th>Expected Outcome</th><th>Actual Result Observed</th><th>Status</th></tr>
    <tr><td><strong>TC-01</strong></td><td>Distributed CSV Ingestion</td><td>Exact count = 284,807 rows across 31 cols</td><td>Loaded 284,807 rows in 13.61s</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-02</strong></td><td>Missing Values Audit</td><td>Zero null or NaN fields across all features</td><td>0 nulls detected across all 31 columns</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-03</strong></td><td>Imbalance Quantification</td><td>Exactly 284,315 Class 0 and 492 Class 1</td><td>Legit: 284,315 (99.83%), Fraud: 492 (0.17%)</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-04</strong></td><td>Loss Weight Generation</td><td>w0 ≈ 0.50, w1 ≈ 289.44</td><td>w0 = 0.5009, w1 = 289.4380</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-05</strong></td><td>Train-Test Partitioning</td><td>80% Train, 20% Test (Seed 42)</td><td>Train: 228,225 rows; Test: 56,582 rows</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-06</strong></td><td>VectorAssembler Transformation</td><td>Dense vector of length 30</td><td>Dense vector of length 30 confirmed</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-07</strong></td><td>StandardScaler Normalization</td><td>Mean ≈ 0.0, Std ≈ 1.0</td><td>Mean 0.0, Std 1.0 confirmed</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-08</strong></td><td>Logistic Regression Training</td><td>Convergence and valid probabilities</td><td>Converged in 14.97s; valid probabilities</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-09</strong></td><td>Decision Tree Fitting</td><td>Valid hierarchical decision tree</td><td>Fit in 4.80s; top split on V14</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-10</strong></td><td>LinearSVC Margin Fitting</td><td>Converged hyperplane parameters</td><td>Fit in 15.13s; valid margin outputs</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-11</strong></td><td>K-Means Anomaly Scoring</td><td>4 distinct centroids, distance threshold</td><td>Converged in 8.88s; cutoff τ = 12.1889</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-12</strong></td><td>Fraud Recall Benchmark</td><td>High fraud capture rate (Recall > 80%)</td><td>LR: 82.52% (85/103), SVM: 82.52% (85/103)</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-13</strong></td><td>ROC-AUC Evaluation</td><td>ROC-AUC > 0.90 for linear models</td><td>LR: 0.9750, Linear SVM: 0.9759</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-14</strong></td><td>Visualization Artifact Export</td><td>Seven 300 DPI PNG images generated</td><td>All 7 high-res PNG plots saved</td><td><strong>PASS</strong></td></tr>
    <tr><td><strong>TC-15</strong></td><td>Batch Prediction Simulation</td><td>Correct fraud probability and triage alert</td><td>Correct probabilities and block alerts</td><td><strong>PASS</strong></td></tr>
</table>
<div class="table-caption">Table 7.1: Functional test suite results on European Cardholders benchmark dataset</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 12. CHAPTER 8: IMPLEMENTATION & RESULTS -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>25</span>
</div>

<h1>8. IMPLEMENTATION & RESULT ANALYSIS</h1>

<h2>8.4 Result Analysis & Algorithmic Benchmarks</h2>
<p>
The empirical evaluation was conducted on a holdout test partition containing <strong>56,582 records</strong>, of which <strong>103 represent confirmed fraudulent transactions</strong>. Table 8.1 presents the complete performance breakdown.
</p>

<table>
    <tr><th>Model / Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall (TPR)</th><th>F1-Score</th><th>ROC-AUC</th><th>PR-AUC</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th><th>Train Time</th></tr>
    <tr><td><strong>Logistic Regression (Weighted)</strong></td><td>0.9897</td><td>0.1310</td><td><strong>0.8252</strong></td><td>0.2261</td><td>0.9750</td><td><strong>0.6395</strong></td><td><strong>85</strong></td><td>564</td><td>55,915</td><td><strong>18</strong></td><td>14.97 s</td></tr>
    <tr><td><strong>Decision Tree Classifier</strong></td><td><strong>0.9992</strong></td><td><strong>0.8090</strong></td><td>0.6990</td><td><strong>0.7500</strong></td><td>0.7472</td><td>0.5966</td><td>72</td><td><strong>17</strong></td><td><strong>56,462</strong></td><td>31</td><td><strong>4.80 s</strong></td></tr>
    <tr><td><strong>Linear Support Vector Machine</strong></td><td>0.9884</td><td>0.1172</td><td><strong>0.8252</strong></td><td>0.2053</td><td><strong>0.9759</strong></td><td>0.6001</td><td><strong>85</strong></td><td>640</td><td>55,839</td><td><strong>18</strong></td><td>15.13 s</td></tr>
    <tr><td><strong>K-Means Anomaly Detector</strong></td><td>0.9845</td><td>0.0756</td><td>0.6699</td><td>0.1358</td><td>0.9407</td><td>0.1810</td><td>69</td><td>844</td><td>55,635</td><td>34</td><td>8.88 s</td></tr>
</table>
<div class="table-caption">Table 8.1: Comparative benchmark metrics across PySpark MLlib classification and anomaly algorithms</div>

<div class="figure-container">
    <img src="{img_cm}" alt="Confusion Matrices">
    <div class="figure-caption">Figure 8.1: Confusion Matrix heatmaps across all four algorithms evaluated on 56,582 test transactions</div>
</div>

<div class="figure-container">
    <img src="{img_roc_pr}" alt="ROC and PR Curves">
    <div class="figure-caption">Figure 8.2: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) Curves</div>
</div>

<div class="figure-container">
    <img src="{img_comparison}" alt="Model Comparison Bar Chart">
    <div class="figure-caption">Figure 8.3: Grouped performance bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC</div>
</div>

<div class="figure-container">
    <img src="{img_feat_imp}" alt="Feature Importance">
    <div class="figure-caption">Figure 8.4: Top 12 predictive features from Decision Tree Gini importance and Logistic Regression coefficient weights</div>
</div>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 13. CHAPTER 9, 10: LEGACY & USER MANUAL -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>30</span>
</div>

<h1>9. PROJECT LEGACY & ETHICS</h1>
<p>
The project delivers a fully validated, production-grade PySpark distributed machine learning engine. Ethical safeguards ensure explainability in fraud triaging, non-discrimination by excluding demographic attributes, and stepped-up authentication protocols rather than immediate account closure.
</p>

<h1>10. USER MANUAL (PYSPARK TERMINAL)</h1>
<p>
The distributed fraud detection engine can be executed via terminal command line or Jupyter Notebook interface:
</p>
<pre>
# Navigate to directory
cd c:\\Users\\hp\\OneDrive\\Desktop\\Pyspa

# Run distributed training & evaluation pipeline via PySpark terminal
python scripts/train_and_evaluate.py

# Launch interactive Jupyter Notebook
jupyter notebook Financial_Transaction_Fraud_Detection_PySpark.ipynb
</pre>

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 14. CHAPTER 11: FULL NOTEBOOK IMPLEMENTATION (.ipynb CODE & OUTPUTS) -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>34</span>
</div>

<h1>11. COMPLETE JUPYTER NOTEBOOK IMPLEMENTATION (.ipynb CODE & OUTPUTS)</h1>
<p>
This chapter contains the complete, unabridged execution transcript of the primary Jupyter Notebook (<code>Financial_Transaction_Fraud_Detection_PySpark.ipynb</code>). Every code cell is displayed alongside its verbatim Apache Spark execution output, printed logs, distributed DataFrame tables, schema trees, evaluation metrics, and simulated transaction scoring decisions.
</p>

{notebook_html}

<div class="page-break"></div>

<!-- ========================================== -->
<!-- 15. CHAPTER 12: BIBLIOGRAPHY -->
<!-- ========================================== -->
<div class="page-header">
    <span>School of Computer Science & Engineering | Lovely Professional University</span>
    <span>45</span>
</div>

<h1>12. BIBLIOGRAPHY</h1>
<ol>
    <li>Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). <em>Calibrating Probability with Undersampling for Unbalanced Classification.</em> IEEE SSCI, pp. 159–166.</li>
    <li>Zaharia, M., et al. (2016). <em>Apache Spark: A Unified Engine for Big Data Processing.</em> Communications of the ACM, 59(11), 56–65.</li>
    <li>Meng, X., et al. (2016). <em>MLlib: Machine Learning in Apache Spark.</em> Journal of Machine Learning Research, 17(34), 1–7.</li>
    <li>Chawla, N. V., et al. (2002). <em>SMOTE: Synthetic Minority Over-sampling Technique.</em> JAIR, 16, 321–357.</li>
    <li>Breiman, L., et al. (1984). <em>Classification and Regression Trees.</em> CRC Press.</li>
    <li>Cortes, C., & Vapnik, V. (1995). <em>Support-Vector Networks.</em> Machine Learning, 20(3), 273–297.</li>
    <li>MacQueen, J. (1967). <em>Some Methods for Classification and Analysis of Multivariate Observations.</em> Proc. 5th Berkeley Symp. Math. Statist. Prob.</li>
    <li>Davis, J., & Goadrich, M. (2006). <em>The Relationship Between Precision-Recall and ROC Curves.</em> ICML, pp. 233–240.</li>
    <li>Apache Spark Documentation. <em>PySpark MLlib Guide.</em> Apache Software Foundation. https://spark.apache.org/docs/latest/ml-guide.html</li>
    <li>Kaggle MLG ULB. <em>Credit Card Fraud Detection Dataset.</em> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud</li>
</ol>

</body>
</html>
"""

with open("CAPSTONE_PROJECT_REPORT_PRINT.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Saved updated CAPSTONE_PROJECT_REPORT_PRINT.html with full .ipynb code and outputs!")
