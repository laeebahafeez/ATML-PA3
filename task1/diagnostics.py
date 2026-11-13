"""Task 1 weight-space and activation diagnostics on down_proj of layers 0, 13, 27 (FP32, CPU ok).

    python -m task1.diagnostics --runs outputs/task1 --out results/task1/diagnostics

Needs only the small exports: <run>/export_u*/diag_down_proj_fp32.safetensors (Full FT
variants) and <run>/export_u1250/adapter_effective.safetensors + adapter_config.json
(adapter variants; OPLoRA uses its projected factors).

Weight space:  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2,  eta = ||dW||_F / ||W0||_F
Activations:   R(z) = ||dW z|| / (||W0 z|| + 1e-8), z = starting model's down_proj input at
               the last prompt token of each probe prompt (math vs non-math).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import read_json, write_json
from task1 import data as D

LAYERS = (0, 13, 27)
K = 128
CONDITIONS = {"full_u625": ("full", 625), "full": ("full", 1250), "l2sp": ("l2sp", 1250),
              "lora4": ("lora4", 1250), "lora16": ("lora16", 1250), "oplora16": ("oplora16", 1250)}


def name(l):
    return f"model.layers.{l}.mlp.down_proj"


def delta_weights(run_dir, update, W0):
    """{layer: dW (FP32)} for one condition, or None if its export is missing."""
    exp = os.path.join(run_dir, f"export_u{update}")
    full = os.path.join(exp, "diag_down_proj_fp32.safetensors")
    if os.path.exists(full):
        sd = load_file(full)
        return {l: sd[name(l) + ".weight"].float() - W0[l] for l in LAYERS}
    eff = os.path.join(exp, "adapter_effective.safetensors")
    if os.path.exists(eff):
        cfg = read_json(os.path.join(exp, "adapter_config.json"))
        f = load_file(eff)
        return {l: cfg["scale"] * f[name(l) + ".lora_B"].float() @ f[name(l) + ".lora_A"].float() for l in LAYERS}
    return None


@torch.no_grad()
def cache_inputs(model, tok, probe):
    """down_proj inputs z (FP32) at the last prompt token: {layer: (n_probe, 3072)}."""
    store, hooks = {}, []
    for l in LAYERS:
        mod = model.model.layers[l].mlp.down_proj
        hooks.append(mod.register_forward_hook(lambda m, inp, out, l=l: store.__setitem__(l, inp[0][0, -1].float().clone())))
    Z = {l: [] for l in LAYERS}
    for it in probe:
        if it["chat"]:
            text = tok.apply_chat_template([{"role": "user", "content": it["user_text"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        else:
            text = it["text"]
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        model(input_ids=ids)          # single unpadded prompt: last position = last prompt token
        for l in LAYERS:
            Z[l].append(store[l])
    for h in hooks:
        h.remove()
    return {l: torch.stack(v) for l, v in Z.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/task1")
    ap.add_argument("--out", default="results/task1/diagnostics")
    ap.add_argument("--probe", default="manifests/task1_probe_set.json")
    ap.add_argument("--model", default=D.MODEL_ID)
    ap.add_argument("--revision", default=D.MODEL_REV)
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.float32).eval()
    W0 = {l: model.model.layers[l].mlp.down_proj.weight.detach().float().clone() for l in LAYERS}
    Uk = {}
    for l in LAYERS:
        U, S, Vh = torch.linalg.svd(W0[l], full_matrices=False)
        Uk[l] = U[:, :a.k]

    probe = read_json(a.probe)["items"]
    Z = cache_inputs(model, tok, probe)
    domain = np.array([it["domain"] for it in probe])
    torch.save({"Z": Z, "domain": domain.tolist()}, os.path.join(a.out, "probe_inputs.pt"))

    w_rows, a_rows = [], []
    for cond, (run, upd) in CONDITIONS.items():
        dW = delta_weights(os.path.join(a.runs, run), upd, W0)
        if dW is None:
            print("missing export for", cond)
            continue
        for l in LAYERS:
            d = dW[l]
            nd = d.norm().item()
            proj = (Uk[l].T @ d).norm().item() ** 2
            w_rows.append({"condition": cond, "layer": l, "rho_k": proj / max(nd ** 2, 1e-30),
                           "eta": nd / W0[l].norm().item(), "dW_fro": nd,
                           "right_rho_k": (d @ torch.linalg.svd(W0[l], full_matrices=False)[2][:a.k].T).norm().item() ** 2
                           / max(nd ** 2, 1e-30)})
            z = Z[l]
            R = (z @ d.T).norm(dim=1) / ((z @ W0[l].T).norm(dim=1) + 1e-8)
            for i, r in enumerate(R.tolist()):
                a_rows.append({"condition": cond, "layer": l, "probe": i, "domain": domain[i], "R": r})
    W = pd.DataFrame(w_rows)
    A = pd.DataFrame(a_rows)
    W.to_csv(os.path.join(a.out, "weight_space.csv"), index=False)
    A.to_csv(os.path.join(a.out, "activation_R.csv"), index=False)
    summ = A.groupby(["condition", "layer", "domain"]).R.agg(["mean", "median"]).unstack("domain")
    summ.columns = [f"{s}_{d}" for s, d in summ.columns]
    summ["math_over_nonmath_mean"] = summ["mean_math"] / summ["mean_nonmath"]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(a.out, "activation_summary.csv"), index=False)
    print(W.round(5).to_string())
    print(summ.round(5).to_string())
    write_json({"k": a.k, "layers": LAYERS, "n_probe": len(probe),
                "note": "dW for Full FT variants = W_trained - W0 (FP32 export); adapters = scale * B~ A~"},
               os.path.join(a.out, "info.json"))


if __name__ == "__main__":
    main()
