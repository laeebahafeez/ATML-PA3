"""Tiny Task 2 manifests + random features for CPU smoke tests (outputs/smoke_task2)."""
import json
import os

import numpy as np

OUT = "outputs/smoke_task2"
M = "manifests/task2"
os.makedirs(f"{OUT}/manifests", exist_ok=True)
os.makedirs(f"{OUT}/features", exist_ok=True)
cap = json.load(open(f"{M}/captions.json"))
gr = json.load(open(f"{M}/grounding.json"))
vq = json.load(open(f"{M}/vqa.json"))
cap = {"train": cap["train"][:48], "dev": cap["dev"][:8]}
gr = {"train_pools": {k: v[:10] for k, v in gr["train_pools"].items()}, "dev": gr["dev"][:4] + gr["dev"][-4:],
      "test": gr["test"][:4] + gr["test"][-4:]}
vq = {"train": vq["train"][:10], "dev": vq["dev"][:6], "test": vq["test"][:6]}
imgs = sorted({c["image"] for s in cap.values() for c in s} |
              {i for p in gr["train_pools"]["presence"] + gr["train_pools"]["counting"] + gr["dev"] + gr["test"]
               for i in p["images"]} | {i for s in vq.values() for p in s for i in p["images"]})
for n, d in (("captions", cap), ("grounding", gr), ("vqa", vq)):
    json.dump(d, open(f"{OUT}/manifests/{n}.json", "w"))
rng = np.random.default_rng(0)
np.save(f"{OUT}/features/features.npy", rng.standard_normal((len(imgs), 196, 768)).astype(np.float16))
json.dump({str(i): r for r, i in enumerate(imgs)}, open(f"{OUT}/features/index.json", "w"))
print(len(imgs), "images")
