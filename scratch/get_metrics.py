import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import sys

sys.path.insert(0, str(Path('codes').absolute()))
from config import CLEAN_XL

df = pd.read_excel(CLEAN_XL).dropna(subset=['question','intent'])
X = df['question'].astype(str).values
y = df['intent'].astype(str).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': 'outputs/model_lr.pkl',
    'Linear SVM (dep.)': 'outputs/final_optimized_svm.pkl',
    'KNN (comparison baseline)': 'outputs/model_knn.pkl'
}

print('Model | Acc. (%) | Mac. F1 (%) | Wt. F1 (%)')
print('---|---|---|---')
for name, path in models.items():
    if Path(path).exists():
        with open(path, 'rb') as f:
            model = pickle.load(f)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        mac_f1 = f1_score(y_test, preds, average='macro') * 100
        wt_f1 = f1_score(y_test, preds, average='weighted') * 100
        print(f"{name} | {acc:.2f} | {mac_f1:.2f} | {wt_f1:.2f}")
