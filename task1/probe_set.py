"""Fixed activation-probe set: 32 GSM8K test questions + 32 HellaSwag validation contexts.

    python -m task1.probe_set        # -> manifests/task1_probe_set.json

Sampled with seed 6304 before any model results were inspected. Prompts use the
benchmark formats: GSM8K as lm-eval's gsm8k_cot zero-shot text ("Q: ...\\nA:") inside the
non-thinking chat template; HellaSwag as lm-eval's preprocessed completion context
(no chat template). The probe position is the last prompt token.
"""
import re

import numpy as np
from datasets import load_dataset

from common.utils import SEED, write_json

N = 32


def hellaswag_preprocess(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def main():
    rng = np.random.default_rng(SEED)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    gi = sorted(rng.choice(len(gsm), N, replace=False).tolist())
    hi = sorted(rng.choice(len(hs), N, replace=False).tolist())
    items = []
    for i in gi:
        items.append({"domain": "math", "dataset": "gsm8k/test", "index": i,
                      "user_text": f"Q: {gsm[i]['question']}\nA:", "chat": True})
    for i in hi:
        d = hs[i]
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        items.append({"domain": "nonmath", "dataset": "hellaswag/validation", "index": i,
                      "ind": d["ind"], "text": hellaswag_preprocess(d["activity_label"] + ": " + ctx),
                      "chat": False})
    write_json({"seed": SEED, "n_per_domain": N, "items": items}, "manifests/task1_probe_set.json")
    print("gsm8k", gi[:5], "hellaswag", hi[:5])


if __name__ == "__main__":
    main()
