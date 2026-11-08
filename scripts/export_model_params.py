import pandas as pd
import numpy as np
import json

# Read LR model
lr_df = pd.read_parquet('models/logistic_regression_model/data')
lr_rec = lr_df.to_dict('records')[0]
intercept = float(lr_rec['interceptVector']['values'][0])
coefficients = [float(x) for x in lr_rec['coefficientMatrix']['values']]

# Read Scaler model
scaler_df = pd.read_parquet('models/feature_pipeline_model/stages/1_StandardScaler_713ec0f350ad/data')
scaler_rec = scaler_df.to_dict('records')[0]
means = [float(x) for x in scaler_rec['mean']['values']]
stds = [float(x) for x in scaler_rec['std']['values']]

feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

model_params = {
    "feature_names": feature_names,
    "intercept": intercept,
    "coefficients": coefficients,
    "means": means,
    "stds": stds,
    "anomaly_threshold_kmeans": 12.1889
}

with open("model_params.json", "w") as f:
    json.dump(model_params, f, indent=2)

print("Exported model_params.json successfully!")
