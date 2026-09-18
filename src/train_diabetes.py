import os
import pandas as pd
from sklearn.model_selection import train_test_split
from common import evaluate_models, save_metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "data", "diabetes.csv")

if not os.path.exists(path):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
            "BMI","DiabetesPedigreeFunction","Age","Outcome"]
    df = pd.read_csv(url, header=None, names=cols)
    df.to_csv(path, index=False)
else:
    df = pd.read_csv(path)

X = df.drop(columns=["Outcome"]).apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["Outcome"], errors="coerce").astype(int)

# Medical measurements coded as zero are commonly treated as missing in this dataset.
for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
    X.loc[X[col] == 0, col] = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = evaluate_models(X_train, X_test, y_train, y_test, "Diabetes")
save_metadata("Diabetes", list(X.columns))
print("\nDiabetes Results\n")
print(results.to_string(index=False))
