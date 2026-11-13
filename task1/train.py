"""Task 1 training: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16.

    python -m task1.train --variant full --out outputs/task1/full

Resumable: rerun the same command and it continues from <out>/resume. One epoch over
the 40k manifest examples in source order, 1,250 updates of 32 examples.
"""
import argparse
import os
import shutil
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from common.utils import (SEED, append_jsonl, dir_size_bytes, read_json, rng_state,
                          seed_all, set_rng_state, write_json)
from task1 import data as D
from task1.lora import adapter_state, inject_adapters

VARIANTS = {
    "full":     dict(kind="full", lr=2e-5, lam=0.0),
    "l2sp":     dict(kind="full", lr=2e-5, lam=1e-3),
    "lora4":    dict(kind="lora", lr=2e-4, r=4, alpha=8),
    "lora16":   dict(kind="lora", lr=2e-4, r=16, alpha=32),
    "oplora16": dict(kind="lora", lr=2e-4, r=16, alpha=32, k=128),
}
MICRO, ACCUM = 2, 16
EFF = MICRO * ACCUM
WARMUP = 38
SAVE_UPDATES = (312, 625, 937, 1250)
EXPORT_UPDATES = {"full": (625, 1250)}          # others export only at 1250
DIAG_LAYERS = (0, 13, 27)
LOSS_CHUNK = 1024


def build_model(cfg, model_id, revision, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    info = {}
    mods = None
    if cfg["kind"] == "lora":
        model.to(device)
        mods, info = inject_adapters(model, cfg["r"], cfg["alpha"], 0.05, SEED,
                                     protect_k=cfg.get("k"))
        # frozen FP16 backbone, FP32 adapters (and FP32 OPLoRA bases / rotary buffers);
        # CPU smoke tests keep FP32 because CPU half matmuls are extremely slow
        bb_dtype = torch.float16 if device == "cuda" else torch.float32
        for n, p in model.named_parameters():
            if "lora_" not in n:
                p.requires_grad_(False)
                p.data = p.data.to(bb_dtype)
    else:
        model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg["kind"] == "lora":
        model.enable_input_require_grads()
    return model, mods, info


def param_groups(model, wd=0.01):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def collate(batch, pad_id, device):
    T = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def _chunk_ce(h, w, y):
    return F.cross_entropy((h @ w.T).float(), y, reduction="sum")


def sum_ce(model, ids, labels, att):
    """Summed next-token CE over supervised positions; logits only where needed."""
    hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    h = hidden[:, :-1][mask]
    y = tgt[mask]
    w = model.lm_head.weight
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], LOSS_CHUNK):
        total = total + checkpoint(_chunk_ce, h[s:s + LOSS_CHUNK], w, y[s:s + LOSS_CHUNK],
                                   use_reentrant=False)
    return total, int(mask.sum())


def atomic_save_dir(tmp, final):
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="manifests/task1_train_manifest.json")
    ap.add_argument("--model", default=D.MODEL_ID)
    ap.add_argument("--revision", default=D.MODEL_REV)
    ap.add_argument("--max_updates", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--n_examples", type=int, default=None, help="limit data (smoke tests)")
    ap.add_argument("--extra_save_every", type=int, default=0,
                    help="additional resume saves every N updates (e.g. on Colab)")
    ap.add_argument("--time_limit_h", type=float, default=None,
                    help="save resume state and exit after this many hours")
    a = ap.parse_args()

    cfg = VARIANTS[a.variant]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    os.makedirs(a.out, exist_ok=True)
    seed_all(SEED)
    t_start = time.time()

    tok = D.load_tokenizer()
    examples = D.load_train_examples(a.manifest, tok)
    if a.n_examples:
        examples = examples[:a.n_examples]
    total_updates = len(examples) // EFF
    stop_at = min(total_updates, a.max_updates or total_updates)

    t0 = time.time()
    model, mods, prep = build_model(cfg, a.model, a.revision, device)
    prep["model_setup_seconds"] = time.time() - t0
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)

    # starting point theta_0: original bf16 weights stored exactly in bf16
    ref = None
    if cfg["kind"] == "full":
        # kept in (pinned) CPU memory and streamed in chunks: full FT already peaks at
        # ~12.9 GB of a 15 GB T4, so a resident 1.2 GB copy plus FP32 temporaries could OOM
        ref_dev = "cpu"
        ref = [p.detach().to(ref_dev, torch.bfloat16).clone() for p in model.parameters()]
        if device == "cuda":
            ref = [r.pin_memory() for r in ref]
        prep["reference_bytes"] = sum(r.numel() * r.element_size() for r in ref)
        prep["reference_device"] = ref_dev
        diag_names = {f"model.layers.{l}.mlp.down_proj.weight" for l in DIAG_LAYERS}

    # foreach=False avoids multi-tensor temporaries that would not fit a 16 GB T4 for full FT
    opt = torch.optim.AdamW(param_groups(model), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8,
                            foreach=cfg["kind"] != "full")
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ---------- resume ----------
    start_update, resume_dir = 0, os.path.join(a.out, "resume")
    log_path = os.path.join(a.out, "train_log.jsonl")
    prev_elapsed = 0.0
    if os.path.exists(os.path.join(resume_dir, "state.pt")):
        st = torch.load(os.path.join(resume_dir, "state.pt"), weights_only=False, map_location="cpu")
        weights = load_file(os.path.join(resume_dir, "trainable.safetensors"))
        named = dict(model.named_parameters())
        for n, t in weights.items():
            named[n].data.copy_(t.to(named[n].device, named[n].dtype))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start_update, prev_elapsed = st["update"], st["elapsed"]
        # drop log lines written after the checkpoint
        if os.path.exists(log_path):
            keep = [l for l in open(log_path, encoding="utf-8") if '"update"' in l
                    and int(l.split('"update": ')[1].split(",")[0]) <= start_update]
            open(log_path, "w", encoding="utf-8").writelines(keep)
        print(f"resumed at update {start_update}")
    elif os.path.exists(log_path):
        os.remove(log_path)

    def save_resume(update, elapsed):
        tmp = resume_dir + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        save_file({n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
                   if p.requires_grad}, os.path.join(tmp, "trainable.safetensors"))
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng": rng_state(), "update": update,
                    "examples_seen": update * EFF, "elapsed": elapsed},
                   os.path.join(tmp, "state.pt"))
        atomic_save_dir(tmp, resume_dir)
        return dir_size_bytes(resume_dir)

    def drop_old_resume():
        # full-FT resume states are ~7 GB; never keep two on Kaggle's 20 GB output disk
        if cfg["kind"] == "full" and os.path.exists(resume_dir):
            shutil.rmtree(resume_dir)

    REF_CHUNK = 1 << 23           # 8M elements: <= 32 MB FP32 temporaries per chunk

    @torch.no_grad()
    def ref_pass(lam=0.0):
        """sum_j (theta_j - theta0_j)^2, streaming theta0 from CPU in chunks; with lam > 0
        also adds lam * (theta - theta0) (the L2-SP gradient) to the unscaled grads."""
        sq = torch.zeros((), dtype=torch.float64, device=device)
        for p, r in zip(model.parameters(), ref):
            pf, rf = p.detach().view(-1), r.view(-1)
            gf = p.grad.view(-1) if lam > 0 else None
            for s in range(0, pf.numel(), REF_CHUNK):
                diff = pf[s:s + REF_CHUNK] - rf[s:s + REF_CHUNK].to(device, non_blocking=True).float()
                sq += diff.double().pow(2).sum()
                if gf is not None:
                    gf[s:s + REF_CHUNK].add_(diff, alpha=lam)
        return sq.item()

    def displacement():
        return ref_pass() ** 0.5

    def export(update):
        t_exp = time.time()
        d = os.path.join(a.out, f"export_u{update}")
        tmp = d + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        info = {"update": update, "examples_seen": update * EFF}
        if cfg["kind"] == "full":
            info["displacement_l2"] = displacement()
            sd = model.state_dict()
            save_file({n: sd[n].detach().float().cpu().contiguous() for n in diag_names},
                      os.path.join(tmp, "diag_down_proj_fp32.safetensors"))
            fp16 = {k: v.detach().half() for k, v in sd.items()
                    if not (k == "lm_head.weight" and model.config.tie_word_embeddings)}
            model.save_pretrained(os.path.join(tmp, "model_fp16"), safe_serialization=True,
                                  max_shard_size="5GB", state_dict=fp16)
            tok.save_pretrained(os.path.join(tmp, "model_fp16"))
        else:
            save_file(adapter_state(mods), os.path.join(tmp, "adapter_raw.safetensors"))
            save_file(adapter_state(mods, effective=True), os.path.join(tmp, "adapter_effective.safetensors"))
            write_json({"r": cfg["r"], "alpha": cfg["alpha"], "scale": cfg["alpha"] / cfg["r"],
                        "protect_k": cfg.get("k"), "base_model": a.model, "revision": a.revision},
                       os.path.join(tmp, "adapter_config.json"))
            if cfg.get("k"):
                info["oplora_constraint_check"] = oplora_check()
        info["export_seconds"] = time.time() - t_exp
        write_json(info, os.path.join(tmp, "export_info.json"))
        atomic_save_dir(tmp, d)
        info["export_bytes"] = dir_size_bytes(d)
        return info

    @torch.no_grad()
    def oplora_check():
        """Max |U_k^T dW| and |dW V_k| relative to |dW| over all modules."""
        worst_l, worst_r = 0.0, 0.0
        for m in mods.values():
            dW = m.delta_weight()
            nrm = dW.norm().item()
            if nrm < 1e-10:      # e.g. k >= d_out leaves no admissible update
                continue
            worst_l = max(worst_l, (m.U_k.T @ dW).norm().item() / nrm)
            worst_r = max(worst_r, (dW @ m.V_k).norm().item() / nrm)
        return {"max_rel_left_leak": worst_l, "max_rel_right_leak": worst_r}

    write_json({"variant": a.variant, **cfg, "micro_batch": MICRO, "grad_accum": ACCUM,
                "warmup": WARMUP, "total_updates": total_updates, "n_trainable": n_trainable,
                "seed": SEED, "model": a.model, "revision": a.revision, "device": device,
                "gpu": torch.cuda.get_device_name(0) if amp else None, **prep},
               os.path.join(a.out, "config.json"))
    print(f"[{a.variant}] trainable params {n_trainable:,}; updates {start_update}->{stop_at}")

    model.train()
    if amp:
        torch.cuda.reset_peak_memory_stats()
    t_loop = time.time()
    summary_path = os.path.join(a.out, "summary.json")
    summary = read_json(summary_path) if (start_update and os.path.exists(summary_path)) else \
        {"exports": {}, "checkpoint_bytes": {}, "skipped_updates": [], "peak_mem_bytes": 0}

    for u in range(start_update, stop_at):
        batch = examples[u * EFF:(u + 1) * EFF]
        n_tok = sum(sum(1 for t in y[1:] if t != -100) for _, y in batch)
        loss_sum = 0.0
        for m in range(ACCUM):
            ids, lab, att = collate(batch[m * MICRO:(m + 1) * MICRO], tok.pad_token_id, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                ce, _ = sum_ce(model, ids, lab, att)
            loss = ce / n_tok
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        rec = {"update": u + 1, "examples_seen": (u + 1) * EFF, "sft_loss": loss_sum,
               "lr": sched.get_last_lr()[0], "response_tokens": n_tok}
        if cfg["kind"] == "full" and cfg["lam"] > 0:
            t_pen = time.perf_counter()
            omega = 0.5 * ref_pass(cfg["lam"])                 # also adds d(lam*Omega)/dp
            if amp:
                torch.cuda.synchronize()
            rec.update(omega=omega, lam_omega=cfg["lam"] * omega,
                       total_loss=loss_sum + cfg["lam"] * omega,
                       penalty_seconds=time.perf_counter() - t_pen)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scale_before = scaler.get_scale() if amp else 1.0
        scaler.step(opt)
        scaler.update()
        skipped = amp and scaler.get_scale() < scale_before
        sched.step()
        opt.zero_grad(set_to_none=True)
        elapsed = prev_elapsed + (time.time() - t_loop)
        rec.update(grad_norm=float(gnorm), skipped=bool(skipped), loss_scale=scale_before,
                   elapsed_s=elapsed)
        if skipped:
            summary["skipped_updates"].append(u + 1)
        append_jsonl(rec, log_path)
        if (u + 1) % 10 == 0 or u == start_update:
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

        done = u + 1
        if amp:
            summary["peak_mem_bytes"] = max(summary["peak_mem_bytes"], torch.cuda.max_memory_allocated())
            summary["peak_mem_reserved_bytes"] = max(summary.get("peak_mem_reserved_bytes", 0),
                                                     torch.cuda.max_memory_reserved())
        if done in EXPORT_UPDATES.get(a.variant, ()) or done == total_updates:
            summary["exports"][str(done)] = export(done)
            print("exported", summary["exports"][str(done)], flush=True)
        is_save = done in SAVE_UPDATES or done == stop_at or \
            (a.extra_save_every and done % a.extra_save_every == 0)
        out_of_time = a.time_limit_h and (time.time() - t_start) / 3600 > a.time_limit_h
        if done == total_updates:
            # training finished: the export holds everything later steps need, so the
            # (same-sized) resume state from the last recovery checkpoint is dropped
            if os.path.exists(resume_dir):
                shutil.rmtree(resume_dir)
            summary.update(train_seconds=elapsed, last_update=done, n_trainable=n_trainable)
            write_json(summary, summary_path)
        elif is_save or out_of_time:
            drop_old_resume()
            summary["checkpoint_bytes"][str(done)] = save_resume(done, elapsed)
            summary.update(train_seconds=elapsed, last_update=done, n_trainable=n_trainable)
            write_json(summary, summary_path)
        if out_of_time:
            print(f"time limit reached at update {done}; rerun to resume")
            return

    summary.update(train_seconds=prev_elapsed + (time.time() - t_loop), last_update=stop_at,
                   n_trainable=n_trainable, wall_seconds_this_session=time.time() - t_start)
    write_json(summary, summary_path)
    print("done", summary)


if __name__ == "__main__":
    main()
