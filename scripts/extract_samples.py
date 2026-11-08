import pandas as pd
import json

df = pd.read_csv('data/creditcard.csv', nrows=40000)
frauds = df[df['Class'] == 1].head(5)
legits = df[df['Class'] == 0].head(5)

samples = []
for idx, r in frauds.iterrows():
    samples.append({
        'name': f'Fraud Incident #{len(samples)+1} (Amount: €{r["Amount"]:.2f})',
        'type': 'fraud',
        'data': {col: float(r[col]) for col in df.columns}
    })

for idx, r in legits.iterrows():
    samples.append({
        'name': f'Legitimate Purchase #{len(samples)-4} (Amount: €{r["Amount"]:.2f})',
        'type': 'legit',
        'data': {col: float(r[col]) for col in df.columns}
    })

with open('sample_transactions.json', 'w') as f:
    json.dump(samples, f, indent=2)

print(f"Extracted {len(samples)} sample transactions into sample_transactions.json")
