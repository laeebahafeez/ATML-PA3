"""Run lm-evaluation-harness + qualitative generations for Task 1 conditions.

    python -m task1.evaluate --models outputs/task1_eval_models --out results/task1/eval \
        --backend vllm --conditions baseline full

GSM8K: gsm8k_cot, 0-shot, chat template (thinking disabled in the model dir's
template), greedy, 1,024 new tokens, stock stop sequences and filters.
HellaSwag / BoolQ / ARC-Easy: stock completion prompts, no chat template.
"""
import argparse
import os
import subprocess
import sys
import time

import torch
import yaml

from common.utils import SEED, write_json

LIKELIHOOD_TASKS = "hellaswag,boolq,arc_easy"


def model_args(backend, path):
    if backend == "vllm":
        return (f"pretrained={path},dtype=float16,gpu_memory_utilization=0.85,"
                f"max_model_len=4096,seed={SEED}")
    return f"pretrained={path},dtype=float16"


def run_lmeval(backend, path, tasks, out, chat, gen_kwargs=None, batch="auto"):
    cmd = [sys.executable, "-m", "lm_eval", "--model", backend,
           "--model_args", model_args(backend, path), "--tasks", tasks,
           "--num_fewshot", "0", "--batch_size", str(batch), "--log_samples",
           "--output_path", out, "--seed", str(SEED)]
    if chat:
        cmd.append("--apply_chat_template")
    if gen_kwargs:
        cmd += ["--gen_kwargs", gen_kwargs]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def qualitative_prompts(yaml_path):
    """24 prompts in file order: 8 plain, 8 JSON (system prompt), 8 template."""
    items = []
    for q in yaml.safe_load(open(yaml_path, encoding="utf-8")):
        msgs = []
        if q.get("system"):
            msgs.append({"role": "system", "content": q["system"]})
        msgs.append({"role": "user", "content": q["paraphrases"][0]})
        fmt = "json" if q["id"].endswith("_json") else "template" if q["id"].endswith("_template") else "plain"
        base = q["id"].removesuffix("_json").removesuffix("_template")
        items.append({"id": q["id"], "question": base, "format": fmt, "messages": msgs})
    return items


@torch.no_grad()
def qualitative(path, yaml_path, out_path, max_new=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    items = qualitative_prompts(yaml_path)
    texts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for it in items]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    for it, g, prompt in zip(items, gen, texts):
        ids = g.tolist()
        n = next((i for i, t in enumerate(ids) if t in eos), None)
        it["prompt_text"] = prompt
        it["response"] = tok.decode(ids[:n] if n is not None else ids, skip_special_tokens=True)
        it["n_new_tokens"] = n if n is not None else len(ids)
        it["truncated"] = n is None
    write_json({"generation_seconds": time.time() - t0, "max_new_tokens": max_new,
                "items": items}, out_path)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="outputs/task1_eval_models")
    ap.add_argument("--out", default="results/task1/eval")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--yaml", default="manifests/first_plot_questions.yaml")
    ap.add_argument("--skip", nargs="*", default=[], choices=["gsm8k", "lik", "qual"])
    ap.add_argument("--gsm8k_batch", type=int, default=32)
    a = ap.parse_args()
    for cond in a.conditions:
        path = os.path.abspath(os.path.join(a.models, cond))
        out = os.path.join(a.out, cond)
        os.makedirs(out, exist_ok=True)
        timing = {}
        if "gsm8k" not in a.skip:
            timing["gsm8k_seconds"] = run_lmeval(
                a.backend, path, "gsm8k_cot", os.path.join(out, "gsm8k"), chat=True,
                gen_kwargs="max_gen_toks=1024,do_sample=False,temperature=0",
                # never "auto" here: for generation lm-eval probes batch sizes at the model's
                # max length (~40k tokens), OOMs at every size and falls back to batch size 1
                batch=a.gsm8k_batch)
        if "lik" not in a.skip:
            timing["likelihood_seconds"] = run_lmeval(
                a.backend, path, LIKELIHOOD_TASKS, os.path.join(out, "likelihood"), chat=False)
        if "qual" not in a.skip:
            t0 = time.time()
            qualitative(path, a.yaml, os.path.join(out, "qualitative.json"))
            timing["qualitative_seconds"] = time.time() - t0
        write_json(timing, os.path.join(out, f"timing_{int(time.time())}.json"))


if __name__ == "__main__":
    main()
