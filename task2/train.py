"""Task 2: three-stage curriculum for the SigLIP-Qwen VLM (resumable).

    python -m task2.train --features outputs/task2/features --out outputs/task2/run196

Stage 1 caption alignment (connector only, 2,250 updates), Stage 2 visual instruction
tuning (+LoRA r16 on q/k/v/o, 1,500 updates), Stage 3 natural VQA (750 updates).
Effective batch 16 = 2 microbatches of 8; each sampler draw contributes 2 examples
(a complementary question pair or two replayed captions). Loss: mean token CE within
each example, then mean over the effective batch.
"""
import argparse
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from common.utils import SEED, append_jsonl, read_json, rng_state, seed_all, set_rng_state, write_json
from task2 import vlm as V


STAGES = [
    {"name": "caption", "updates": 2250, "lr_conn": 1e-3, "lr_lora": None, "mix": None},
    {"name": "instruct", "updates": 1500, "lr_conn": 1e-4, "lr_lora": 2e-5,
     "mix": {"presence": 0.5, "counting": 0.25, "caption": 0.25}},
    {"name": "vqa", "updates": 750, "lr_conn": 5e-5, "lr_lora": 1e-5,
     "mix": {"vqa": 0.7, "presence": 0.1, "counting": 0.1, "caption": 0.1}},
]
MICRO, ACCUM = 8, 2
EFF = MICRO * ACCUM
DEV_EVERY = 500
CKPT_EVERY = 250


class Data:
    """Formatted examples + feature rows for every split."""

    def __init__(self, man, fidx, fmt):
        cap = read_json(f"{man}/captions.json")
        gr = read_json(f"{man}/grounding.json")
        vq = read_json(f"{man}/vqa.json")
        row = lambda i: fidx[str(i)]
        self.fmt, self.row = fmt, row
        self.captions = [fmt.example(None, c["caption"], row(c["image"]), kind="caption") for c in cap["train"]]
        self.dev_captions = [fmt.example(None, c["caption"], row(c["image"]), kind="caption") for c in cap["dev"]]
        self.pools = {k: gr["train_pools"][k] for k in ("presence", "counting")}
        self.vqa_train = vq["train"]
        self.suites = {"dev": {"grounding": gr["dev"], "vqa": vq["dev"]},
                       "test": {"grounding": gr["test"], "vqa": vq["test"]}}

    def pair_examples(self, p, image_for=None, with_image=True):
        """Two examples of a pair; image_for[k] overrides the image used for question k."""
        qs = [p["question"], p.get("question_2", p["question"])]
        out = []
        for k in range(2):
            img = p["images"][k] if image_for is None else image_for[k]
            out.append(self.fmt.example(qs[k], p["answers"][k], self.row(img) if with_image else None,
                                        kind="short", with_image=with_image))
        return out


class Sampler:
    """Seeded mixture sampler; state is checkpointed with the run."""

    def __init__(self, data, seed=SEED):
        self.d = data
        self.rng = np.random.default_rng(seed)
        self.counts = {}
        self.cap_order, self.cap_pos = None, 0

    def state(self):
        return {"rng": self.rng.bit_generator.state, "counts": self.counts,
                "cap_order": None if self.cap_order is None else self.cap_order.tolist(), "cap_pos": self.cap_pos}

    def load(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.counts = s["counts"]
        self.cap_order = None if s["cap_order"] is None else np.array(s["cap_order"])
        self.cap_pos = s["cap_pos"]

    def _tick(self, k, n=1):
        self.counts[k] = self.counts.get(k, 0) + n

    def stage1_batch(self):
        """Two passes over the 18,000 captions, reshuffled each pass."""
        out = []
        while len(out) < EFF:
            if self.cap_order is None or self.cap_pos >= len(self.cap_order):
                self.cap_order = self.rng.permutation(len(self.d.captions))
                self.cap_pos = 0
            out.append(self.d.captions[self.cap_order[self.cap_pos]])
            self.cap_pos += 1
        self._tick("caption", EFF)
        return out

    def mixture_batch(self, mix):
        kinds, probs = list(mix), np.array(list(mix.values()))
        out = []
        for _ in range(EFF // 2):
            k = kinds[self.rng.choice(len(kinds), p=probs / probs.sum())]
            self._tick(k)
            if k == "caption":
                out += [self.d.captions[j] for j in self.rng.choice(len(self.d.captions), 2, replace=False)]
            elif k == "vqa":
                out += self.d.pair_examples(self.d.vqa_train[self.rng.integers(len(self.d.vqa_train))])
            else:
                pool = self.d.pools[k]
                out += self.d.pair_examples(pool[self.rng.integers(len(pool))])
        return out


def batch_loss(model, fmt, examples, feats, device):
    nll = V.per_example_nll(model, fmt, examples, feats, fmt.tok.pad_token_id, device)
    return nll.sum() / EFF


@torch.no_grad()
def dev_eval(model, data, feats, device, bs=16):
    model.eval()
    fmt = data.fmt
    res = {}

    def mean_nll(exs, exclude_eot=True):
        vals = []
        for s in range(0, len(exs), bs):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                vals.append(V.per_example_nll(model, fmt, exs[s:s + bs], feats, fmt.tok.pad_token_id,
                                              device, exclude_eot=exclude_eot).float().cpu())
        return torch.cat(vals)

    res["caption_nll"] = mean_nll(data.dev_captions).mean().item()
    for suite, pairs in data.suites["dev"].items():
        correct = [e for p in pairs for e in data.pair_examples(p)]
        swap = [e for p in pairs for e in data.pair_examples(p, image_for=[p["images"][1], p["images"][0]])]
        noimg = [e for p in pairs for e in data.pair_examples(p, with_image=False)]
        nc, ns, nn_ = mean_nll(correct), mean_nll(swap), mean_nll(noimg)
        res[suite] = {"nll_correct": nc.mean().item(), "nll_swap": ns.mean().item(),
                      "nll_noimage": nn_.mean().item(), "swap_penalty": (ns - nc).mean().item()}
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/task2/features")
    ap.add_argument("--manifests", default="manifests/task2")
    ap.add_argument("--out", default="outputs/task2/run196")
    ap.add_argument("--n_vis", type=int, default=196, choices=[196, 49])
    ap.add_argument("--model", default=V.QWEN)
    ap.add_argument("--revision", default=V.QWEN_REV)
    ap.add_argument("--scale_updates", type=float, default=1.0, help="<1 for smoke tests")
    ap.add_argument("--time_limit_h", type=float, default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    seed_all(SEED)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    feats = torch.from_numpy(np.array(np.load(os.path.join(a.features, "features.npy"), mmap_mode="r")))
    if amp:
        feats = feats.pin_memory()
    fidx = read_json(os.path.join(a.features, "index.json"))
    assert feats.shape[1:] == (196, 768)
    tok = V.load_tokenizer(a.model, a.revision)
    fmt = V.Formatter(tok, a.n_vis)
    data = Data(a.manifests, fidx, fmt)
    model = V.VLM(device, a.n_vis, a.model, a.revision)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    stages = [dict(s, updates=max(2, int(s["updates"] * a.scale_updates))) for s in STAGES]
    bounds = np.cumsum([s["updates"] for s in stages]).tolist()
    total = bounds[-1]
    dev_points = sorted(set([0] + list(range(DEV_EVERY, total + 1, DEV_EVERY)) + bounds))

    sampler = Sampler(data)
    log_path, dev_path = os.path.join(a.out, "train_log.jsonl"), os.path.join(a.out, "dev_log.jsonl")
    ck = os.path.join(a.out, "ckpt")
    summary_path = os.path.join(a.out, "summary.json")

    # ---------- calibration (training data only) ----------
    cal_path = os.path.join(a.out, "calibration.json")
    rng = np.random.default_rng(SEED)
    cal_idx = rng.choice(len(data.captions), min(256, len(data.captions)), replace=False)
    if os.path.exists(cal_path):
        model.vis_scale.fill_(read_json(cal_path)["scale"])
    else:
        rows = [data.captions[i]["feat"] for i in cal_idx]
        txt = [t for i in cal_idx for t, l in zip(data.captions[i]["ids"], data.captions[i]["labels"]) if l != -100]
        cal = model.calibrate(feats[rows], torch.tensor(txt))
        cal["rule"] = ("scale = mean ||text input embedding|| over caption tokens of 256 training captions"
                       " / mean ||connector output|| at initialization over those images' patches")
        write_json(cal, cal_path)

    stage_i, update, start_elapsed = 0, 0, 0.0
    opt = sched = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    summary = {"stages": {}, "skipped_updates": []}

    def build_stage(i):
        nonlocal opt, sched
        s = stages[i]
        if s["lr_lora"] is not None and not model.lora:
            model.enable_lora()
            for n, p in model.lm.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()
        conn, lora = model.trainable_groups()
        decay = lambda ps: [p for p in ps if p.ndim >= 2]
        nodecay = lambda ps: [p for p in ps if p.ndim < 2]
        groups = [{"params": decay(conn), "lr": s["lr_conn"], "weight_decay": 0.01},
                  {"params": nodecay(conn), "lr": s["lr_conn"], "weight_decay": 0.0}]
        if s["lr_lora"] is not None:
            groups.append({"params": lora, "lr": s["lr_lora"], "weight_decay": 0.01})
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, V.lr_lambda(s["updates"]))
        n_tr = sum(p.numel() for g in groups for p in g["params"])
        summary["stages"].setdefault(s["name"], {})["trainable_params"] = n_tr
        summary["stages"][s["name"]]["connector_params"] = sum(p.numel() for p in conn)
        summary["stages"][s["name"]]["lora_params"] = sum(p.numel() for p in lora) if s["lr_lora"] else 0

    # ---------- resume ----------
    if os.path.exists(os.path.join(ck, "state.pt")):
        st = torch.load(os.path.join(ck, "state.pt"), weights_only=False, map_location="cpu")
        stage_i, update, start_elapsed = st["stage"], st["update"], st["elapsed"]
        summary = st["summary"]
        if stage_i >= 1:
            model.enable_lora()
            for n, p in model.lm.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()
        w = load_file(os.path.join(ck, "trainable.safetensors"))
        model.connector.load_state_dict({k[len("connector."):]: v for k, v in w.items() if k.startswith("connector.")})
        named = dict(model.lm.named_parameters())
        for k, v in w.items():
            if k.startswith("lm."):
                named[k[3:]].data.copy_(v)
        build_stage(stage_i)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        sampler.load(st["sampler"])
        set_rng_state(st["rng"])
        for path in (log_path, dev_path):
            if os.path.exists(path):
                import json
                keep = [l for l in open(path, encoding="utf-8") if json.loads(l)["update"] <= update]
                open(path, "w", encoding="utf-8").writelines(keep)
        print(f"resumed at update {update} (stage {stage_i})")
    else:
        for path in (log_path, dev_path):
            if os.path.exists(path):
                os.remove(path)
        build_stage(0)

    def trainable_state():
        sd = {f"connector.{k}": v.detach().cpu().contiguous() for k, v in model.connector.state_dict().items()}
        sd.update({f"lm.{n}": p.detach().cpu().contiguous() for n, p in model.lm.named_parameters() if "lora_" in n})
        return sd

    def save_ckpt(elapsed):
        tmp = ck + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        save_file(trainable_state(), os.path.join(tmp, "trainable.safetensors"))
        torch.save({"stage": stage_i, "update": update, "elapsed": elapsed, "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.state(),
                    "rng": rng_state(), "summary": summary}, os.path.join(tmp, "state.pt"))
        if os.path.exists(ck):
            import shutil
            shutil.rmtree(ck)
        os.replace(tmp, ck)

    if update == 0 and not os.path.exists(dev_path):
        r = dev_eval(model, data, feats, device)
        append_jsonl({"update": 0, "stage": "init", **r}, dev_path)
        print("dev@0", r, flush=True)

    checked = update > 0
    t_loop = time.time()
    stage_t0 = time.time()
    if amp:
        torch.cuda.reset_peak_memory_stats()
    while update < total:
        s = stages[stage_i]
        batch = sampler.stage1_batch() if s["mix"] is None else sampler.mixture_batch(s["mix"])
        loss_sum = 0.0
        for m in range(ACCUM):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = batch_loss(model, fmt, batch[m * MICRO:(m + 1) * MICRO], feats, device)
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        if not checked:   # gradients reach the connector, never the frozen backbone
            g = [p.grad for p in model.connector.parameters()]
            assert all(x is not None for x in g), "no gradient reaches the connector"
            assert all(p.grad is None for n, p in model.lm.named_parameters() if "lora_" not in n), \
                "the frozen backbone received gradients"
            # the first updates can overflow at the initial loss scale (the scaler skips them),
            # so the nonzero check waits for the first finite gradients
            if all(torch.isfinite(x).all() for x in g):
                assert all(x.abs().sum() > 0 for x in g), "connector gradients are all zero"
                checked = True
                print(f"gradient check passed at update {update + 1}", flush=True)
            else:
                print(f"update {update + 1}: non-finite gradients (loss-scale overflow), step skipped", flush=True)
                assert update < 50, "gradients stayed non-finite for 50 updates"
        params = [p for g in opt.param_groups for p in g["params"]]
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        lrs = [g["lr"] for g in opt.param_groups]       # the rates this update uses (before sched.step)
        sb = scaler.get_scale() if amp else 1.0
        scaler.step(opt)
        scaler.update()
        skipped = amp and scaler.get_scale() < sb
        sched.step()
        opt.zero_grad(set_to_none=True)
        update += 1
        elapsed = start_elapsed + time.time() - t_loop
        rec = {"update": update, "stage": s["name"], "loss": loss_sum, "grad_norm": float(gnorm),
               "lr_conn": lrs[0], "lr_lora": lrs[2] if len(lrs) > 2 else None,
               "skipped": bool(skipped), "mixture_counts": dict(sampler.counts), "elapsed_s": elapsed}
        if skipped:
            summary["skipped_updates"].append(update)
        append_jsonl(rec, log_path)
        if update % 25 == 0:
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

        stage_end = update == bounds[stage_i]
        if update in dev_points:
            r = dev_eval(model, data, feats, device)
            append_jsonl({"update": update, "stage": s["name"], **r}, dev_path)
            print("dev", update, r, flush=True)
        if stage_end:
            st = summary["stages"][s["name"]]
            st["seconds"] = st.get("seconds", 0.0) + (time.time() - stage_t0)
            if amp:
                st["peak_mem_bytes"] = max(st.get("peak_mem_bytes", 0), torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            st["mixture_counts_end"] = dict(sampler.counts)
            save_file(trainable_state(), os.path.join(a.out, f"stage{stage_i + 1}_{s['name']}.safetensors"))
            stage_t0 = time.time()
            if stage_i + 1 < len(stages):
                stage_i += 1
                build_stage(stage_i)
        out_of_time = a.time_limit_h and (time.time() - t_start) / 3600 > a.time_limit_h
        if update % CKPT_EVERY == 0 or stage_end or out_of_time:
            if not stage_end:
                st = summary["stages"][s["name"]]
                st["seconds"] = st.get("seconds", 0.0) + (time.time() - stage_t0)
                if amp:
                    st["peak_mem_bytes"] = max(st.get("peak_mem_bytes", 0), torch.cuda.max_memory_allocated())
                stage_t0 = time.time()
            save_ckpt(elapsed)
            write_json(summary, summary_path)
        if out_of_time:
            print("time limit reached; rerun to resume")
            return
    summary["train_seconds"] = start_elapsed + time.time() - t_loop
    summary["calibration"] = read_json(cal_path)
    write_json(summary, summary_path)
    print("done", summary)


if __name__ == "__main__":
    main()
