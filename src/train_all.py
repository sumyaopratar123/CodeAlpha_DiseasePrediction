import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
scripts = ["train_heart.py", "train_diabetes.py", "train_breast_cancer.py"]

for script in scripts:
    print("\n" + "="*70)
    print("Running:", script)
    print("="*70)
    subprocess.run([sys.executable, os.path.join(ROOT, script)], check=True)

print("\nAll disease models trained successfully.")
