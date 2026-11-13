"""Task 3 Step 4: parallel sampling vs one longer response vs sequential refinement.

    python -m task3.live --out outputs/task3_live        # GPU, resumable

Qwen2.5-3B-Instruct (pinned revision), FP16, batch size 1, T=0.7, top-p 0.8, top-k 20,
repetition penalty 1.0, on the first 50 MATH evaluation questions. Every call is
independently seeded and logged (prompt, response, tokens, time, cap hit) to calls.jsonl.
Scoring happens offline in task3.live_analysis.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import SEED, append_jsonl, read_json, write_json

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
INSTR = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its "
          "answer. Give your final answer inside \\boxed{}.")
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)


def last_boxed(text):
    """Content of the last literal \\boxed{...} with balanced braces; None if absent/empty/unclosed."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                s = text[start:j].strip()
                return s or None
        j += 1
    return None


class Runner:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                          torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def generate(self, messages, max_new, seed):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=max_new, **SAMPLING,
                                  pad_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0, enc["input_ids"].shape[1]:].tolist()
        stops = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        n = next((k for k, t in enumerate(gen) if t in stops), None)
        kept = gen[:n] if n is not None else gen
        resp = self.tok.decode(kept, skip_special_tokens=True)
        return {"prompt": text, "response": resp, "input_tokens": int(enc["input_ids"].shape[1]),
                "output_tokens": len(kept), "hit_cap": n is None and len(gen) >= max_new,
                "seconds": dt, "seed": seed, "max_new_tokens": max_new,
                "extracted": last_boxed(resp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="manifests/task3_live_questions.json")
    ap.add_argument("--out", default="outputs/task3_live")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "calls.jsonl")
    done = set()
    if os.path.exists(log):
        done = {(r["sample_id"], r["call"]) for r in map(json.loads, open(log, encoding="utf-8"))}
    qs = read_json(a.questions)
    t_load = time.time()
    R = Runner()
    load_s = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()
    for qi, q in enumerate(qs):
        base = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q["question"] + "\n\n" + INSTR}]
        seed = lambda c: SEED * 1000 + qi * 10 + c
        calls = {"single": (base, 3072, seed(0))}
        for k in range(3):
            calls[f"parallel_{k}"] = (base, 1024, seed(1 + k))
        for name, (msgs, cap, s) in calls.items():
            if (q["sample_id"], name) in done:
                continue
            r = R.generate(msgs, cap, s)
            append_jsonl({"sample_id": q["sample_id"], "call": name, **r}, log)
        if (q["sample_id"], "refine") not in done:
            # reuse the first parallel response as the initial answer
            init = next(r for r in map(json.loads, open(log, encoding="utf-8"))
                        if r["sample_id"] == q["sample_id"] and r["call"] == "parallel_0")
            msgs = base + [{"role": "assistant", "content": init["response"]},
                           {"role": "user", "content": REVIEW}]
            r = R.generate(msgs, 2048, seed(4))
            append_jsonl({"sample_id": q["sample_id"], "call": "refine", **r}, log)
        print(f"{qi + 1}/{len(qs)} {q['sample_id']}", flush=True)
    write_json({"model": MODEL, "revision": REVISION, "dtype": "float16", "batch_size": 1,
                "sampling": SAMPLING, "model_load_seconds": load_s,
                "peak_mem_bytes": torch.cuda.max_memory_allocated(),
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
               os.path.join(a.out, "run_info.json"))


if __name__ == "__main__":
    main()
