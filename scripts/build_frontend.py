import json
import base64
import os

with open("model_params.json", "r") as f:
    model_params = json.load(f)

with open("sample_transactions.json", "r") as f:
    sample_transactions = json.load(f)

with open("models/metrics_summary.json", "r") as f:
    metrics_summary = json.load(f)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Transaction Fraud Detection — PySpark MLlib Platform</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220%22%20%22100%22><text y=%22.9em%22 font-size=%2290%22>🛡️</text></svg>">
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0b0f19;
            color: #f1f5f9;
        }}
        .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
        .glass-panel {{
            background: rgba(17, 24, 39, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .glass-card {{
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .gradient-text {{
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .risk-gauge {{
            transition: stroke-dashoffset 0.8s ease-in-out, stroke 0.5s ease;
        }}
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: #0b0f19;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #1e293b;
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #334155;
        }}
    </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-indigo-500 selection:text-white">

    <!-- Navbar -->
    <header class="sticky top-0 z-50 glass-panel border-b border-slate-800">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                    <i class="fa-solid fa-shield-halved text-white text-lg"></i>
                </div>
                <div>
                    <div class="font-extrabold text-lg tracking-tight flex items-center gap-2">
                        <span>PySpark FraudGuard</span>
                        <span class="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">v3.5.3</span>
                    </div>
                    <div class="text-xs text-slate-400">Distributed Financial Transaction Intelligence</div>
                </div>
            </div>

            <nav class="hidden md:flex items-center space-x-1">
                <button onclick="switchTab('scanner')" id="tab-btn-scanner" class="px-4 py-2 rounded-lg text-sm font-medium transition-all text-white bg-indigo-600 shadow-md shadow-indigo-600/30">
                    <i class="fa-solid fa-radar mr-2"></i>Live Scanner
                </button>
                <button onclick="switchTab('models')" id="tab-btn-models" class="px-4 py-2 rounded-lg text-sm font-medium transition-all text-slate-300 hover:text-white hover:bg-slate-800/60">
                    <i class="fa-solid fa-chart-line mr-2"></i>Model Benchmarks
                </button>
                <button onclick="switchTab('imbalance')" id="tab-btn-imbalance" class="px-4 py-2 rounded-lg text-sm font-medium transition-all text-slate-300 hover:text-white hover:bg-slate-800/60">
                    <i class="fa-solid fa-scale-unbalanced mr-2"></i>Data & Imbalance
                </button>
                <button onclick="switchTab('report')" id="tab-btn-report" class="px-4 py-2 rounded-lg text-sm font-medium transition-all text-slate-300 hover:text-white hover:bg-slate-800/60">
                    <i class="fa-solid fa-file-pdf mr-2"></i>Capstone Report
                </button>
            </nav>

            <div class="flex items-center space-x-3">
                <a href="CAPSTONE_PROJECT_REPORT.pdf" download="CAPSTONE_PROJECT_REPORT.pdf" class="inline-flex items-center px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-colors">
                    <i class="fa-solid fa-download mr-1.5"></i>PDF Report
                </a>
                <a href="https://github.com/PARAS-RAJPUT/Financial-Transaction-Fraud-Detection-PySpark" target="_blank" class="p-2 text-slate-400 hover:text-white transition-colors" title="View Source on GitHub">
                    <i class="fa-brands fa-github text-xl"></i>
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">

        <!-- ============================================================== -->
        <!-- TAB 1: LIVE TRANSACTION SCANNER & SIMULATOR -->
        <!-- ============================================================== -->
        <section id="tab-scanner" class="space-y-6">
            <!-- Hero Banner -->
            <div class="glass-panel p-6 rounded-2xl relative overflow-hidden border border-slate-800">
                <div class="absolute -right-10 -bottom-10 w-72 h-72 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <div class="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-3">
                            <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                            Real-Time PySpark MLlib Inference Engine
                        </div>
                        <h1 class="text-2xl sm:text-3xl font-extrabold tracking-tight">Financial Transaction Fraud Simulator</h1>
                        <p class="text-slate-400 text-sm mt-1 max-w-2xl">
                            Simulate credit card transactions and evaluate real-time fraud probability against the cost-weighted PySpark Logistic Regression model (<span class="text-indigo-400 font-mono">ROC-AUC: 0.9750</span>, <span class="text-emerald-400 font-mono">Recall: 82.52%</span>).
                        </p>
                    </div>
                    <!-- Quick Preset Dropdown -->
                    <div class="flex flex-col gap-2">
                        <label class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Quick Presets</label>
                        <div class="flex flex-wrap gap-2">
                            <button onclick="loadPreset(0)" class="px-3 py-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 rounded-lg text-xs font-medium transition-all">
                                <i class="fa-solid fa-triangle-exclamation mr-1"></i>Real Fraud #1
                            </button>
                            <button onclick="loadPreset(1)" class="px-3 py-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 rounded-lg text-xs font-medium transition-all">
                                <i class="fa-solid fa-skull mr-1"></i>Micro-Test Fraud
                            </button>
                            <button onclick="loadPreset(5)" class="px-3 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs font-medium transition-all">
                                <i class="fa-solid fa-cart-shopping mr-1"></i>Normal Purchase
                            </button>
                            <button onclick="randomizeInputs()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-xs font-medium transition-all">
                                <i class="fa-solid fa-shuffle mr-1"></i>Randomize
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Two-column scanner grid -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

                <!-- Left Column: Sliders & Controls (7 Cols) -->
                <div class="lg:col-span-7 space-y-6">
                    <div class="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
                        <div class="flex items-center justify-between border-b border-slate-800/80 pb-3">
                            <h2 class="text-base font-bold text-white flex items-center gap-2">
                                <i class="fa-solid fa-sliders text-indigo-400"></i> Transaction Attributes
                            </h2>
                            <span class="text-xs text-slate-400">Inputs auto-standardized with Spark StandardScaler</span>
                        </div>

                        <!-- Amount & Time Primary Inputs -->
                        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div class="glass-card p-4 rounded-xl border border-slate-800">
                                <div class="flex justify-between items-center mb-1">
                                    <label class="text-xs font-medium text-slate-300">Transaction Amount</label>
                                    <span id="label-amount" class="text-xs font-bold text-cyan-400 font-mono">€1809.68</span>
                                </div>
                                <input type="range" id="input-amount" min="0" max="3000" step="1" value="1809.68" oninput="updatePrediction()" class="w-full accent-cyan-400 h-2 bg-slate-700 rounded-lg cursor-pointer">
                                <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                                    <span>€0.00</span>
                                    <span>€1,500</span>
                                    <span>€3,000+</span>
                                </div>
                            </div>

                            <div class="glass-card p-4 rounded-xl border border-slate-800">
                                <div class="flex justify-between items-center mb-1">
                                    <label class="text-xs font-medium text-slate-300">Elapsed Time (s)</label>
                                    <span id="label-time" class="text-xs font-bold text-cyan-400 font-mono">9,064 s</span>
                                </div>
                                <input type="range" id="input-time" min="0" max="172800" step="100" value="9064" oninput="updatePrediction()" class="w-full accent-cyan-400 h-2 bg-slate-700 rounded-lg cursor-pointer">
                                <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                                    <span>0 h</span>
                                    <span>24 h</span>
                                    <span>48 h (Day 2)</span>
                                </div>
                            </div>
                        </div>

                        <!-- Key Predictive PCA Features -->
                        <div>
                            <div class="flex items-center justify-between mb-3">
                                <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                                    <i class="fa-solid fa-bolt text-amber-400"></i> Top Predictive Latent Features (PCA)
                                </h3>
                                <span class="text-[11px] text-slate-500">Ranked by Gini Importance & Beta Magnitudes</span>
                            </div>

                            <div class="space-y-3.5" id="top-pca-container">
                                <!-- Populated dynamically by JS -->
                            </div>
                        </div>

                        <!-- Collapsible for remaining PCA components -->
                        <div class="border-t border-slate-800/80 pt-3">
                            <button onclick="toggleRemainingFeatures()" class="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1.5 font-medium transition-colors">
                                <i id="remaining-toggle-icon" class="fa-solid fa-chevron-right text-[10px]"></i>
                                <span id="remaining-toggle-text">Show Remaining Latent Features (V1..V28)</span>
                            </button>
                            <div id="remaining-pca-container" class="hidden grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
                                <!-- Populated dynamically by JS -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right Column: Live Prediction Gauge & Decision Card (5 Cols) -->
                <div class="lg:col-span-5 space-y-6">
                    <!-- Decision Card -->
                    <div id="decision-card" class="glass-panel p-6 rounded-2xl border border-rose-500/40 relative overflow-hidden transition-all duration-500">
                        <div id="decision-glow" class="absolute -top-12 -right-12 w-48 h-48 bg-rose-500/20 rounded-full blur-3xl pointer-events-none transition-all duration-500"></div>

                        <div class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Transaction Assessment</div>

                        <!-- Gauge & Probability Display -->
                        <div class="flex items-center justify-center py-4">
                            <div class="relative w-44 h-44 flex items-center justify-center">
                                <svg class="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                                    <circle cx="50" cy="50" r="42" stroke="#1e293b" stroke-width="9" fill="transparent"/>
                                    <circle id="gauge-circle" cx="50" cy="50" r="42" stroke="#ef4444" stroke-width="9" stroke-linecap="round" fill="transparent"
                                            stroke-dasharray="263.89" stroke-dashoffset="10" class="risk-gauge"/>
                                </svg>
                                <div class="absolute flex flex-col items-center justify-center text-center">
                                    <span id="risk-pct" class="text-3xl font-extrabold tracking-tight font-mono text-white">99.8%</span>
                                    <span class="text-[10px] text-slate-400 uppercase tracking-widest font-semibold mt-0.5">Fraud Risk</span>
                                </div>
                            </div>
                        </div>

                        <!-- Status Badge -->
                        <div id="decision-badge" class="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-center justify-center gap-2.5 font-bold text-sm text-center">
                            <i id="decision-icon" class="fa-solid fa-ban text-rose-500 text-base"></i>
                            <span id="decision-text">ALERT: FRAUD DETECTED - TRANSACTION BLOCKED</span>
                        </div>

                        <!-- Sub-indicators -->
                        <div class="grid grid-cols-2 gap-3 mt-4">
                            <div class="glass-card p-3 rounded-lg border border-slate-800 text-center">
                                <div class="text-[10px] text-slate-400 font-medium">Logit Margin</div>
                                <div id="metric-logit" class="text-sm font-bold text-white font-mono mt-0.5">+4.184</div>
                            </div>
                            <div class="glass-card p-3 rounded-lg border border-slate-800 text-center">
                                <div class="text-[10px] text-slate-400 font-medium">K-Means Anomaly</div>
                                <div id="metric-anomaly" class="text-sm font-bold text-amber-400 font-mono mt-0.5">d = 16.42 (&gt; 12.18)</div>
                            </div>
                        </div>

                        <!-- Top Risk Drivers Breakdown -->
                        <div class="mt-5 border-t border-slate-800/80 pt-4">
                            <div class="text-xs font-semibold text-slate-300 mb-2 flex items-center justify-between">
                                <span>Key Factor Influence</span>
                                <span class="text-[10px] text-slate-500 font-mono">Weight * Z-Score</span>
                            </div>
                            <div id="factor-breakdown" class="space-y-2 text-xs">
                                <!-- Populated dynamically -->
                            </div>
                        </div>
                    </div>

                    <!-- Pipeline Architecture Micro-card -->
                    <div class="glass-panel p-5 rounded-2xl border border-slate-800 text-xs space-y-2.5">
                        <div class="font-bold text-slate-200 flex items-center gap-2">
                            <i class="fa-solid fa-server text-cyan-400"></i> PySpark MLlib Pipeline Specs
                        </div>
                        <div class="text-slate-400 leading-relaxed">
                            Feature vector passes through <span class="text-slate-200 font-mono">VectorAssembler(30)</span> followed by z-score standardization via <span class="text-slate-200 font-mono">StandardScaler(withMean=True, withStd=True)</span>. Cost-sensitive inverse loss weighting (<span class="text-indigo-400 font-mono">289.44x</span>) compensates for 577:1 imbalance.
                        </div>
                    </div>
                </div>

            </div>
        </section>

        <!-- ============================================================== -->
        <!-- TAB 2: MODEL BENCHMARKS -->
        <!-- ============================================================== -->
        <section id="tab-models" class="hidden space-y-6">
            <div class="glass-panel p-6 rounded-2xl border border-slate-800">
                <h2 class="text-2xl font-bold tracking-tight">PySpark MLlib Comparative Benchmarks</h2>
                <p class="text-slate-400 text-sm mt-1">
                    Evaluated on <strong>56,582 holdout transactions</strong> (including <strong>103 actual fraud cases</strong>). Models were trained concurrently in Apache Spark 3.5.3.
                </p>
            </div>

            <!-- Table Card -->
            <div class="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
                <div class="p-5 border-b border-slate-800 flex items-center justify-between">
                    <h3 class="text-base font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-table-list text-indigo-400"></i> Full Metric Matrix (Test Partition)
                    </h3>
                    <span class="text-xs text-slate-400">Seed = 42 | 80/20 Stratified Split</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="bg-slate-900/60 text-slate-400 uppercase tracking-wider font-semibold">
                            <tr>
                                <th class="p-4">Algorithm / Model</th>
                                <th class="p-4">Accuracy</th>
                                <th class="p-4">Precision</th>
                                <th class="p-4 text-emerald-400">Recall (TPR)</th>
                                <th class="p-4">F1-Score</th>
                                <th class="p-4 text-indigo-400">ROC-AUC</th>
                                <th class="p-4 text-cyan-400">PR-AUC</th>
                                <th class="p-4">TP / FP</th>
                                <th class="p-4">TN / FN</th>
                                <th class="p-4">Train Time</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800 font-mono">
                            <tr class="hover:bg-slate-800/40 transition-colors">
                                <td class="p-4 font-sans font-bold text-white flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-blue-500"></span> Logistic Regression (Weighted)
                                </td>
                                <td class="p-4">0.9897</td>
                                <td class="p-4">0.1310</td>
                                <td class="p-4 text-emerald-400 font-bold">0.8252 (85/103)</td>
                                <td class="p-4">0.2261</td>
                                <td class="p-4 text-indigo-400 font-bold">0.9750</td>
                                <td class="p-4 text-cyan-400 font-bold">0.6395</td>
                                <td class="p-4 text-slate-300">85 / 564</td>
                                <td class="p-4 text-slate-300">55,915 / 18</td>
                                <td class="p-4 text-slate-400">14.97 s</td>
                            </tr>
                            <tr class="hover:bg-slate-800/40 transition-colors">
                                <td class="p-4 font-sans font-bold text-white flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Decision Tree Classifier
                                </td>
                                <td class="p-4 font-bold text-slate-200">0.9992</td>
                                <td class="p-4 font-bold text-emerald-400">0.8090</td>
                                <td class="p-4">0.6990 (72/103)</td>
                                <td class="p-4 font-bold text-amber-400">0.7500</td>
                                <td class="p-4">0.7472</td>
                                <td class="p-4">0.5966</td>
                                <td class="p-4 text-slate-300">72 / 17</td>
                                <td class="p-4 text-slate-300">56,462 / 31</td>
                                <td class="p-4 text-slate-400 font-bold text-emerald-400">4.80 s</td>
                            </tr>
                            <tr class="hover:bg-slate-800/40 transition-colors">
                                <td class="p-4 font-sans font-bold text-white flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span> Linear SVM (Cost-Sensitive)
                                </td>
                                <td class="p-4">0.9884</td>
                                <td class="p-4">0.1172</td>
                                <td class="p-4 text-emerald-400 font-bold">0.8252 (85/103)</td>
                                <td class="p-4">0.2053</td>
                                <td class="p-4 text-indigo-400 font-bold">0.9759</td>
                                <td class="p-4 text-cyan-400">0.6001</td>
                                <td class="p-4 text-slate-300">85 / 640</td>
                                <td class="p-4 text-slate-300">55,839 / 18</td>
                                <td class="p-4 text-slate-400">15.13 s</td>
                            </tr>
                            <tr class="hover:bg-slate-800/40 transition-colors">
                                <td class="p-4 font-sans font-bold text-white flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span> K-Means Anomaly Scorer (k=4)
                                </td>
                                <td class="p-4">0.9845</td>
                                <td class="p-4">0.0756</td>
                                <td class="p-4 text-emerald-400">0.6699 (69/103)</td>
                                <td class="p-4">0.1358</td>
                                <td class="p-4 text-indigo-400 font-bold">0.9407</td>
                                <td class="p-4">0.1810</td>
                                <td class="p-4 text-slate-300">69 / 844</td>
                                <td class="p-4 text-slate-300">55,635 / 34</td>
                                <td class="p-4 text-slate-400">8.88 s</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Charts Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Bar Chart -->
                <div class="glass-panel p-5 rounded-2xl border border-slate-800">
                    <h3 class="text-sm font-bold text-white mb-4 flex items-center gap-2">
                        <i class="fa-solid fa-chart-column text-indigo-400"></i> Model Performance Metric Comparison
                    </h3>
                    <div class="relative h-64">
                        <canvas id="modelBarChart"></canvas>
                    </div>
                </div>

                <!-- Confusion Matrix Breakdown Cards -->
                <div class="glass-panel p-5 rounded-2xl border border-slate-800">
                    <h3 class="text-sm font-bold text-white mb-4 flex items-center gap-2">
                        <i class="fa-solid fa-table-cells text-emerald-400"></i> Confusion Matrix Comparison (Test Frauds = 103)
                    </h3>
                    <div class="grid grid-cols-2 gap-3 text-xs">
                        <div class="glass-card p-3.5 rounded-xl border border-blue-500/20">
                            <div class="font-bold text-blue-400 mb-1">Logistic Regression</div>
                            <div class="text-slate-300">TP: <strong class="text-emerald-400">85</strong> | FN: <strong class="text-rose-400">18</strong></div>
                            <div class="text-slate-400 text-[11px]">FP: 564 | TN: 55,915</div>
                            <div class="text-[10px] text-slate-500 mt-1">Recall: 82.52% | F1: 0.226</div>
                        </div>

                        <div class="glass-card p-3.5 rounded-xl border border-emerald-500/20">
                            <div class="font-bold text-emerald-400 mb-1">Decision Tree</div>
                            <div class="text-slate-300">TP: <strong class="text-emerald-400">72</strong> | FN: <strong class="text-rose-400">31</strong></div>
                            <div class="text-slate-400 text-[11px]">FP: <strong class="text-emerald-400">17</strong> | TN: 56,462</div>
                            <div class="text-[10px] text-slate-500 mt-1">Precision: 80.90% | F1: 0.750</div>
                        </div>

                        <div class="glass-card p-3.5 rounded-xl border border-amber-500/20">
                            <div class="font-bold text-amber-400 mb-1">Linear SVM</div>
                            <div class="text-slate-300">TP: <strong class="text-emerald-400">85</strong> | FN: <strong class="text-rose-400">18</strong></div>
                            <div class="text-slate-400 text-[11px]">FP: 640 | TN: 55,839</div>
                            <div class="text-[10px] text-slate-500 mt-1">ROC-AUC: 0.9759 | Recall: 82.52%</div>
                        </div>

                        <div class="glass-card p-3.5 rounded-xl border border-rose-500/20">
                            <div class="font-bold text-rose-400 mb-1">K-Means Anomaly</div>
                            <div class="text-slate-300">TP: <strong class="text-emerald-400">69</strong> | FN: <strong class="text-rose-400">34</strong></div>
                            <div class="text-slate-400 text-[11px]">FP: 844 | TN: 55,635</div>
                            <div class="text-[10px] text-slate-500 mt-1">Unsupervised Recall: 66.99%</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ============================================================== -->
        <!-- TAB 3: DATA & IMBALANCE -->
        <!-- ============================================================== -->
        <section id="tab-imbalance" class="hidden space-y-6">
            <div class="glass-panel p-6 rounded-2xl border border-slate-800">
                <h2 class="text-2xl font-bold tracking-tight">Kaggle European Cardholders Dataset & Imbalance Analysis</h2>
                <p class="text-slate-400 text-sm mt-1">
                    284,807 transactions recorded over 48 hours in September 2013 by European cardholders.
                </p>
            </div>

            <!-- Stats Grid -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div class="glass-panel p-4 rounded-xl border border-slate-800 text-center">
                    <div class="text-xs text-slate-400 font-medium">Total Records</div>
                    <div class="text-xl font-bold text-white font-mono mt-1">284,807</div>
                </div>
                <div class="glass-panel p-4 rounded-xl border border-slate-800 text-center">
                    <div class="text-xs text-slate-400 font-medium">Legitimate Transactions</div>
                    <div class="text-xl font-bold text-emerald-400 font-mono mt-1">284,315 <span class="text-xs font-normal">(99.83%)</span></div>
                </div>
                <div class="glass-panel p-4 rounded-xl border border-slate-800 text-center">
                    <div class="text-xs text-slate-400 font-medium">Fraudulent Transactions</div>
                    <div class="text-xl font-bold text-rose-400 font-mono mt-1">492 <span class="text-xs font-normal">(0.17%)</span></div>
                </div>
                <div class="glass-panel p-4 rounded-xl border border-slate-800 text-center">
                    <div class="text-xs text-slate-400 font-medium">Imbalance Ratio</div>
                    <div class="text-xl font-bold text-amber-400 font-mono mt-1">577.9 : 1</div>
                </div>
            </div>

            <!-- Imbalance Chart & Loss Weight Math -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="glass-panel p-5 rounded-2xl border border-slate-800">
                    <h3 class="text-sm font-bold text-white mb-3 flex items-center gap-2">
                        <i class="fa-solid fa-chart-pie text-cyan-400"></i> Linear vs Logarithmic Class Distribution
                    </h3>
                    <div class="relative h-64">
                        <canvas id="imbalanceChart"></canvas>
                    </div>
                </div>

                <div class="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                    <h3 class="text-base font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-calculator text-indigo-400"></i> Mathematical Cost-Sensitive Loss Weighting
                    </h3>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        Standard machine learning loss functions assume balanced class distribution ($50:50$). In credit card transactions, an unweighted algorithm achieves <span class="text-emerald-400 font-bold">99.83% accuracy</span> by predicting class 0 constantly, completely failing to catch any fraudulent activity.
                    </p>
                    <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 space-y-2">
                        <div class="text-slate-400">// Inverse Frequency Loss Formulation:</div>
                        <div>w_fraud = N / (2 * N_fraud) = 284807 / (2 * 492) = <span class="text-cyan-400 font-bold">289.4380</span></div>
                        <div>w_legit = N / (2 * N_legit) = 284807 / (2 * 284315) = <span class="text-emerald-400 font-bold">0.5009</span></div>
                    </div>
                    <p class="text-xs text-slate-400 leading-relaxed">
                        By integrating <span class="font-mono text-slate-300">weightCol="weight"</span> into PySpark's Logistic Regression and Linear Support Vector Machine optimizers, the model is penalized <strong>577 times more heavily</strong> for missing a fraud instance than for triggering a false alarm.
                    </p>
                </div>
            </div>
        </section>

        <!-- ============================================================== -->
        <!-- TAB 4: CAPSTONE REPORT & NOTEBOOK -->
        <!-- ============================================================== -->
        <section id="tab-report" class="hidden space-y-6">
            <div class="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div>
                    <h2 class="text-2xl font-bold tracking-tight">Academic Capstone Project Report</h2>
                    <p class="text-slate-400 text-sm mt-1">
                        Course Code: <strong class="text-slate-200">INT315</strong> • Lovely Professional University, Phagwara, Punjab
                    </p>
                </div>
                <div class="flex flex-wrap gap-2.5">
                    <a href="CAPSTONE_PROJECT_REPORT.pdf" download class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition-all shadow-lg shadow-emerald-600/30 flex items-center gap-2">
                        <i class="fa-solid fa-file-arrow-down"></i> Download PDF Report (23 Pages)
                    </a>
                    <a href="https://github.com/PARAS-RAJPUT/Financial-Transaction-Fraud-Detection-PySpark/blob/main/Financial_Transaction_Fraud_Detection_PySpark.ipynb" target="_blank" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-semibold transition-all border border-slate-700 flex items-center gap-2">
                        <i class="fa-solid fa-book-open"></i> View Jupyter Notebook
                    </a>
                </div>
            </div>

            <!-- Chapters Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
                <div class="glass-card p-4 rounded-xl border border-slate-800">
                    <div class="font-bold text-indigo-400 mb-1">Chapter 1: Introduction</div>
                    <p class="text-slate-400 leading-relaxed">Background of financial payments, CNP fraud risks, motivation, 5 measurable objectives (O1–O5), and distributed system scope.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-800">
                    <div class="font-bold text-indigo-400 mb-1">Chapter 3: Existing System</div>
                    <p class="text-slate-400 leading-relaxed">Comprehensive comparison matrix of legacy rule engines, single-node Scikit-Learn scripts, and commercial black-boxes.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-800">
                    <div class="font-bold text-indigo-400 mb-1">Chapter 6: System Design</div>
                    <p class="text-slate-400 leading-relaxed">Four-tier distributed architecture, PySpark MLlib Pipeline specs, sequence diagrams, and 3 pseudocode algorithms.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-800">
                    <div class="font-bold text-indigo-400 mb-1">Chapter 7: Testing</div>
                    <p class="text-slate-400 leading-relaxed">15 rigorous functional test cases (TC-01 to TC-15) covering schema, weights, PCA splits, and AUC metrics with 100% pass rate.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-800">
                    <div class="font-bold text-indigo-400 mb-1">Chapter 8: Results Analysis</div>
                    <p class="text-slate-400 leading-relaxed">Algorithmic benchmark matrix, confusion matrix heatmaps, ROC/PR curves, K-Means PCA clustering, and feature importance.</p>
                </div>
                <div class="glass-card p-4 rounded-xl border border-slate-800">
                    <div class="font-bold text-indigo-400 mb-1">Chapter 11: .ipynb Code &amp; Outputs</div>
                    <p class="text-slate-400 leading-relaxed">Full verbatim notebook transcript containing every code cell alongside its printed Apache Spark execution logs and DataFrame outputs.</p>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800/80 py-6 text-center text-xs text-slate-500">
        <div class="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div>
                <strong>Financial Transaction Fraud Detection using PySpark MLlib</strong> &bull; Paras Rajput (12304811)
            </div>
            <div class="flex items-center gap-4">
                <span>Supervisor: Miss Zeenat Zehra</span>
                <span>&bull;</span>
                <a href="https://github.com/PARAS-RAJPUT/Financial-Transaction-Fraud-Detection-PySpark" target="_blank" class="hover:text-slate-300">GitHub Repository</a>
            </div>
        </div>
    </footer>

    <!-- EMBEDDED MODEL PARAMETERS & SAMPLES -->
    <script>
        const MODEL_PARAMS = {json.dumps(model_params)};
        const SAMPLES = {json.dumps(sample_transactions)};
        const METRICS = {json.dumps(metrics_summary)};

        // Top 8 features sorted by absolute weight
        const TOP_FEATURES = ["V14", "V4", "V10", "V12", "V17", "V11", "V7", "V3"];
        
        let currentInputs = {{}};

        // Initialize default inputs
        MODEL_PARAMS.feature_names.forEach((f, i) => {{
            currentInputs[f] = MODEL_PARAMS.means[i];
        }});
        // Default to sample 0 (Real fraud)
        if (SAMPLES.length > 0) {{
            currentInputs = {{ ...SAMPLES[0].data }};
        }}

        function initUI() {{
            buildTopSliders();
            buildRemainingSliders();
            syncSlidersWithState();
            updatePrediction();
            initCharts();
        }}

        function buildTopSliders() {{
            const container = document.getElementById("top-pca-container");
            container.innerHTML = "";

            TOP_FEATURES.forEach(feat => {{
                const idx = MODEL_PARAMS.feature_names.indexOf(feat);
                const coef = MODEL_PARAMS.coefficients[idx];
                const val = currentInputs[feat] !== undefined ? currentInputs[feat] : 0.0;

                const div = document.createElement("div");
                div.className = "glass-card p-3 rounded-xl border border-slate-800/80";
                div.innerHTML = `
                    <div class="flex justify-between items-center mb-1">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-bold text-slate-200 font-mono">${{feat}}</span>
                            <span class="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 font-mono">beta: ${{coef.toFixed(3)}}</span>
                        </div>
                        <span id="label-${{feat}}" class="text-xs font-mono font-bold text-indigo-400">${{val.toFixed(2)}}</span>
                    </div>
                    <input type="range" id="input-${{feat}}" min="-15" max="15" step="0.1" value="${{val}}"
                           oninput="onSliderChange('${{feat}}', this.value)"
                           class="w-full accent-indigo-500 h-1.5 bg-slate-700 rounded-lg cursor-pointer">
                `;
                container.appendChild(div);
            }});
        }}

        function buildRemainingSliders() {{
            const container = document.getElementById("remaining-pca-container");
            container.innerHTML = "";

            MODEL_PARAMS.feature_names.forEach(feat => {{
                if (feat === "Amount" || feat === "Time" || TOP_FEATURES.includes(feat)) return;

                const val = currentInputs[feat] !== undefined ? currentInputs[feat] : 0.0;
                const div = document.createElement("div");
                div.className = "glass-card p-2.5 rounded-lg border border-slate-800/60";
                div.innerHTML = `
                    <div class="flex justify-between items-center mb-0.5">
                        <span class="text-[11px] font-mono text-slate-300">${{feat}}</span>
                        <span id="label-${{feat}}" class="text-[11px] font-mono font-bold text-slate-400">${{val.toFixed(2)}}</span>
                    </div>
                    <input type="range" id="input-${{feat}}" min="-10" max="10" step="0.2" value="${{val}}"
                           oninput="onSliderChange('${{feat}}', this.value)"
                           class="w-full accent-slate-400 h-1 bg-slate-700 rounded cursor-pointer">
                `;
                container.appendChild(div);
            }});
        }}

        function syncSlidersWithState() {{
            document.getElementById("input-amount").value = currentInputs["Amount"] || 0;
            document.getElementById("label-amount").textContent = "€" + (currentInputs["Amount"] || 0).toFixed(2);

            document.getElementById("input-time").value = currentInputs["Time"] || 0;
            document.getElementById("label-time").textContent = Math.round(currentInputs["Time"] || 0).toLocaleString() + " s";

            MODEL_PARAMS.feature_names.forEach(feat => {{
                if (feat === "Amount" || feat === "Time") return;
                const el = document.getElementById("input-" + feat);
                const lbl = document.getElementById("label-" + feat);
                if (el && lbl) {{
                    el.value = currentInputs[feat] || 0;
                    lbl.textContent = (currentInputs[feat] || 0).toFixed(2);
                }}
            }});
        }}

        function onSliderChange(feat, val) {{
            val = parseFloat(val);
            currentInputs[feat] = val;
            const lbl = document.getElementById("label-" + feat);
            if (lbl) lbl.textContent = val.toFixed(2);
            updatePrediction();
        }}

        function loadPreset(idx) {{
            if (idx >= SAMPLES.length) return;
            const sample = SAMPLES[idx];
            currentInputs = {{ ...sample.data }};
            syncSlidersWithState();
            updatePrediction();
        }}

        function randomizeInputs() {{
            MODEL_PARAMS.feature_names.forEach((feat, i) => {{
                if (feat === "Amount") {{
                    currentInputs[feat] = Math.random() < 0.2 ? Math.random() * 2000 : Math.random() * 80;
                }} else if (feat === "Time") {{
                    currentInputs[feat] = Math.random() * 172800;
                }} else {{
                    currentInputs[feat] = (Math.random() - 0.5) * 6;
                }}
            }});
            syncSlidersWithState();
            updatePrediction();
        }}

        function toggleRemainingFeatures() {{
            const c = document.getElementById("remaining-pca-container");
            const icon = document.getElementById("remaining-toggle-icon");
            const txt = document.getElementById("remaining-toggle-text");
            if (c.classList.contains("hidden")) {{
                c.classList.remove("hidden");
                icon.className = "fa-solid fa-chevron-down text-[10px]";
                txt.textContent = "Hide Remaining Latent Features";
            }} else {{
                c.classList.add("hidden");
                icon.className = "fa-solid fa-chevron-right text-[10px]";
                txt.textContent = "Show Remaining Latent Features (V1..V28)";
            }}
        }}

        function updatePrediction() {{
            currentInputs["Amount"] = parseFloat(document.getElementById("input-amount").value);
            document.getElementById("label-amount").textContent = "€" + currentInputs["Amount"].toFixed(2);

            currentInputs["Time"] = parseFloat(document.getElementById("input-time").value);
            document.getElementById("label-time").textContent = Math.round(currentInputs["Time"]).toLocaleString() + " s";

            // Standardize and calculate logistic regression logit
            let logit = MODEL_PARAMS.intercept;
            let contributions = [];

            MODEL_PARAMS.feature_names.forEach((feat, i) => {{
                const raw = currentInputs[feat] !== undefined ? currentInputs[feat] : 0.0;
                const mean = MODEL_PARAMS.means[i];
                const std = MODEL_PARAMS.stds[i] || 1.0;
                const zscore = (raw - mean) / std;
                const coef = MODEL_PARAMS.coefficients[i];
                const effect = zscore * coef;

                logit += effect;
                contributions.push({{ feat, effect, raw, zscore }});
            }});

            // Sigmoid probability
            const prob = 1.0 / (1.0 + Math.exp(-logit));
            const pct = (prob * 100).toFixed(1);

            // Update UI elements
            document.getElementById("risk-pct").textContent = pct + "%";
            document.getElementById("metric-logit").textContent = (logit >= 0 ? "+" : "") + logit.toFixed(3);

            // Distance proxy calculation for K-Means Anomaly
            let distProxy = Math.sqrt(contributions.reduce((acc, c) => acc + (c.zscore * c.zscore), 0));
            const isAnomaly = distProxy >= MODEL_PARAMS.anomaly_threshold_kmeans;
            document.getElementById("metric-anomaly").innerHTML = `d = ${{distProxy.toFixed(1)}} (${{isAnomaly ? '&gt;' : '&le;'}} 12.2)`;

            // SVG Gauge Circle
            // 263.89 is circumference of r=42
            const offset = 263.89 * (1 - prob);
            const circle = document.getElementById("gauge-circle");
            circle.style.strokeDashoffset = offset;

            const card = document.getElementById("decision-card");
            const glow = document.getElementById("decision-glow");
            const badge = document.getElementById("decision-badge");
            const icon = document.getElementById("decision-icon");
            const text = document.getElementById("decision-text");

            if (prob >= 0.5) {{
                // High Risk Fraud
                circle.style.stroke = "#ef4444";
                card.className = "glass-panel p-6 rounded-2xl border border-rose-500/50 relative overflow-hidden transition-all duration-500";
                glow.className = "absolute -top-12 -right-12 w-48 h-48 bg-rose-500/25 rounded-full blur-3xl pointer-events-none transition-all duration-500";
                badge.className = "p-3.5 rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-400 flex items-center justify-center gap-2.5 font-bold text-sm text-center";
                icon.className = "fa-solid fa-ban text-rose-500 text-base";
                text.textContent = "ALERT: FRAUD DETECTED - TRANSACTION BLOCKED";
            }} else if (prob >= 0.2) {{
                // Medium Risk / Review
                circle.style.stroke = "#f59e0b";
                card.className = "glass-panel p-6 rounded-2xl border border-amber-500/50 relative overflow-hidden transition-all duration-500";
                glow.className = "absolute -top-12 -right-12 w-48 h-48 bg-amber-500/20 rounded-full blur-3xl pointer-events-none transition-all duration-500";
                badge.className = "p-3.5 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-400 flex items-center justify-center gap-2.5 font-bold text-sm text-center";
                icon.className = "fa-solid fa-triangle-exclamation text-amber-400 text-base";
                text.textContent = "WARNING: ELEVATED RISK - 2FA REQUIRED";
            }} else {{
                // Low Risk Approved
                circle.style.stroke = "#10b981";
                card.className = "glass-panel p-6 rounded-2xl border border-emerald-500/40 relative overflow-hidden transition-all duration-500";
                glow.className = "absolute -top-12 -right-12 w-48 h-48 bg-emerald-500/15 rounded-full blur-3xl pointer-events-none transition-all duration-500";
                badge.className = "p-3.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 flex items-center justify-center gap-2.5 font-bold text-sm text-center";
                icon.className = "fa-solid fa-circle-check text-emerald-400 text-base";
                text.textContent = "TRANSACTION APPROVED - LOW RISK";
            }}

            // Render top 4 factor contributors
            contributions.sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));
            const top4 = contributions.slice(0, 4);
            const breakdownEl = document.getElementById("factor-breakdown");
            breakdownEl.innerHTML = top4.map(c => {{
                const isRisk = c.effect > 0;
                return `
                    <div class="flex items-center justify-between">
                        <span class="font-mono text-slate-300 font-bold">${{c.feat}} (${{c.raw.toFixed(1)}})</span>
                        <span class="${{isRisk ? 'text-rose-400 font-bold' : 'text-emerald-400'}} font-mono">
                            ${{c.effect >= 0 ? '+' : ''}}${{c.effect.toFixed(2)}} ${{isRisk ? '▲ (Risk)' : '▼ (Safe)'}}
                        </span>
                    </div>
                `;
            }}).join("");
        }}

        function switchTab(tabId) {{
            ['scanner', 'models', 'imbalance', 'report'].forEach(t => {{
                const sec = document.getElementById('tab-' + t);
                const btn = document.getElementById('tab-btn-' + t);
                if (t === tabId) {{
                    sec.classList.remove('hidden');
                    btn.className = 'px-4 py-2 rounded-lg text-sm font-medium transition-all text-white bg-indigo-600 shadow-md shadow-indigo-600/30';
                }} else {{
                    sec.classList.add('hidden');
                    btn.className = 'px-4 py-2 rounded-lg text-sm font-medium transition-all text-slate-300 hover:text-white hover:bg-slate-800/60';
                }}
            }});
        }}

        function initCharts() {{
            // Bar Chart in Models Tab
            const ctxBar = document.getElementById('modelBarChart').getContext('2d');
            new Chart(ctxBar, {{
                type: 'bar',
                data: {{
                    labels: ['Recall (TPR)', 'ROC-AUC', 'PR-AUC', 'Precision', 'F1-Score'],
                    datasets: [
                        {{ label: 'Logistic Reg (Weighted)', data: [0.825, 0.975, 0.639, 0.131, 0.226], backgroundColor: '#3b82f6' }},
                        {{ label: 'Decision Tree', data: [0.699, 0.747, 0.597, 0.809, 0.750], backgroundColor: '#10b981' }},
                        {{ label: 'Linear SVM', data: [0.825, 0.976, 0.600, 0.117, 0.205], backgroundColor: '#f59e0b' }},
                        {{ label: 'K-Means Anomaly', data: [0.670, 0.941, 0.181, 0.076, 0.136], backgroundColor: '#ef4444' }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 10 }} }} }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true, max: 1.05, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
                        x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#cbd5e1' }} }}
                    }}
                }}
            }});

            // Imbalance Doughnut Chart
            const ctxPie = document.getElementById('imbalanceChart').getContext('2d');
            new Chart(ctxPie, {{
                type: 'doughnut',
                data: {{
                    labels: ['Legitimate (284,315)', 'Fraud (492)'],
                    datasets: [{{
                        data: [284315, 492],
                        backgroundColor: ['#10b981', '#ef4444'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '70%',
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}
                    }}
                }}
            }});
        }}

        window.onload = initUI;
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Generated comprehensive Vercel frontend index.html successfully!")
