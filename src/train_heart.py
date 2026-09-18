import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from common import evaluate_models, save_metadata

heart = fetch_ucirepo(id=45)
X = heart.data.features.copy()
y = heart.data.targets.iloc[:, 0].copy()

# UCI Heart Disease target is 0 for no disease and 1-4 for disease severity.
y = (pd.to_numeric(y, errors="coerce") > 0).astype(int)

# Convert any categorical/object columns to numeric codes.
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.apply(pd.to_numeric, errors="coerce")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = evaluate_models(X_train, X_test, y_train, y_test, "Heart")
save_metadata("Heart", list(X.columns))
print("\nHeart Disease Results\n")
print(results.to_string(index=False))
