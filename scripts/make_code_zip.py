"""Zip the code (no data/outputs) for upload as a Kaggle dataset `atml-pa3-code`."""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ("common", "task1", "task2", "task3", "manifests", "scripts", "notebooks")
FILES = ("requirements.txt", "README.md")

out = os.path.join(ROOT, "outputs", "atml-pa3-code.zip")
os.makedirs(os.path.dirname(out), exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d in INCLUDE:
        for root, _, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in root or ".ipynb_checkpoints" in root:
                continue
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, ROOT))
    for fn in FILES:
        if os.path.exists(os.path.join(ROOT, fn)):
            z.write(os.path.join(ROOT, fn), fn)
print(out, os.path.getsize(out) // 1024, "KB")
