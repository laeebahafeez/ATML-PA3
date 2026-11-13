"""Build FP16 HF model directories for the seven Task 1 evaluation conditions.

    python -m task1.prepare_eval --runs outputs/task1 --out outputs/task1_eval_models

Conditions: baseline, full_u625, full (u1250), l2sp, lora4, lora16, oplora16.
Adapters are merged into a fresh FP16 backbone using the exported *effective* factors
(projected factors for OPLoRA), and merged-vs-unmerged logits are compared on a small batch.
Every directory gets a chat template that hard-codes enable_thinking=False so that any
evaluator (lm-eval hf/vllm, plain generate) uses the non-thinking template.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import write_json
from task1 import data as D
from task1.lora import LoRALinear, inject_adapters, merge_into

CONDITIONS = {
    "baseline": None,
    "full_u625": ("full", 625),
    "full": ("full", 1250),
    "l2sp": ("l2sp", 1250),
    "lora4": ("lora4", 1250),
    "lora16": ("lora16", 1250),
    "oplora16": ("oplora16", 1250),
}
CHECK_PROMPTS = [
    "What is 17 * 23?",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
    "Describe a sunny day at the beach.",
    "Is the following statement true: the sun rises in the west?",
]
NO_THINK = "{%- set enable_thinking = false %}"


def force_no_thinking(model_dir):
    path = os.path.join(model_dir, "tokenizer_config.json")
    cfg = json.load(open(path, encoding="utf-8"))
    tpl = cfg.get("chat_template")
    if tpl and not tpl.startswith(NO_THINK):
        cfg["chat_template"] = NO_THINK + tpl
        json.dump(cfg, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    # newer transformers also read chat_template.jinja
    jpath = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(jpath):
        t = open(jpath, encoding="utf-8").read()
        if not t.startswith(NO_THINK):
            open(jpath, "w", encoding="utf-8").write(NO_THINK + t)


def fresh_backbone(model_id, revision, device):
    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                torch_dtype=torch.float16).to(device).eval()


@torch.no_grad()
def logits_on(model, tok, device):
    texts = [D.prompt_text(tok, p) for p in CHECK_PROMPTS]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    out = model(**enc).logits.float()
    tok.padding_side = "right"
    return out[enc["attention_mask"].bool()]


def build_adapter_condition(run_dir, model_id, revision, device, update=1250):
    exp = os.path.join(run_dir, f"export_u{update}")
    acfg = json.load(open(os.path.join(exp, "adapter_config.json")))
    eff = load_file(os.path.join(exp, "adapter_effective.safetensors"))
    raw = load_file(os.path.join(exp, "adapter_raw.safetensors"))
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    # unmerged: FP16 backbone + FP32 adapters (raw factors, projection recomputed from W0)
    ref = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                               torch_dtype=torch.float32).to(device)
    mods, _ = inject_adapters(ref, acfg["r"], acfg["alpha"], 0.0, 0, protect_k=acfg["protect_k"])
    for n, m in mods.items():
        m.lora_A.data.copy_(raw[f"{n}.lora_A"])
        m.lora_B.data.copy_(raw[f"{n}.lora_B"])
    for n, p in ref.named_parameters():
        if "lora_" not in n:
            p.data = p.data.half()
    ref.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        l_unmerged = logits_on(ref, tok, device)
    # projected-factor export must equal the in-training effective factors
    max_factor_diff = 0.0
    for n, m in mods.items():
        A, B = m.effective_factors()
        max_factor_diff = max(max_factor_diff,
                              (A.cpu() - eff[f"{n}.lora_A"]).abs().max().item(),
                              (B.cpu() - eff[f"{n}.lora_B"]).abs().max().item())
    del ref, mods
    torch.cuda.empty_cache()

    merged = fresh_backbone(model_id, revision, device)
    merge_into(merged, eff, acfg["scale"])
    l_merged = logits_on(merged, tok, device)
    check = {
        "max_abs_logit_diff": (l_merged - l_unmerged).abs().max().item(),
        "mean_abs_logit_diff": (l_merged - l_unmerged).abs().mean().item(),
        "argmax_agreement": (l_merged.argmax(-1) == l_unmerged.argmax(-1)).float().mean().item(),
        "max_effective_factor_diff": max_factor_diff,
    }
    return merged, tok, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/task1")
    ap.add_argument("--out", default="outputs/task1_eval_models")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--model", default=D.MODEL_ID)
    ap.add_argument("--revision", default=D.MODEL_REV)
    ap.add_argument("--final_update", type=int, default=1250, help="override for smoke tests")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = {}
    for cond, src in CONDITIONS.items():
        if a.only and cond not in a.only:
            continue
        dst = os.path.join(a.out, cond)
        if os.path.exists(os.path.join(dst, "config.json")):
            print("exists", dst)
            continue
        if src is None:
            m = fresh_backbone(a.model, a.revision, device)
            tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
        elif src[0] in ("full", "l2sp"):
            upd = src[1] if src[1] != 1250 else a.final_update
            srcdir = os.path.join(a.runs, src[0], f"export_u{upd}", "model_fp16")
            shutil.copytree(srcdir, dst)
            force_no_thinking(dst)
            print("copied", srcdir)
            continue
        else:
            m, tok, checks[cond] = build_adapter_condition(os.path.join(a.runs, src[0]),
                                                           a.model, a.revision, device, a.final_update)
            print(cond, checks[cond])
        m.save_pretrained(dst, safe_serialization=True)
        tok.save_pretrained(dst)
        force_no_thinking(dst)
        del m
        torch.cuda.empty_cache()
    if checks:
        path = os.path.join(a.out, "merge_checks.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        write_json({**old, **checks}, path)


if __name__ == "__main__":
    main()
