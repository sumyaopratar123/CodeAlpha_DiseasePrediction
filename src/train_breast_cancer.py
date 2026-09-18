import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from common import evaluate_models, save_metadata

data = load_breast_cancer(as_frame=True)
X = data.data.copy()
# sklearn target: 0=malignant, 1=benign. Convert to disease-positive malignant=1.
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = evaluate_models(X_train, X_test, y_train, y_test, "BreastCancer")
save_metadata("BreastCancer", list(X.columns))
print("\nBreast Cancer Results\n")
print(results.to_string(index=False))
