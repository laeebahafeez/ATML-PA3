"""Generate the Kaggle/Colab notebooks in notebooks/ from the cell sources below."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nb(cells):
    return {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, s in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


SETUP = r'''
# --- locate code (Kaggle dataset "atml-pa3-code" containing atml-pa3-code.zip, or Colab Drive) ---
import glob, os, shutil, subprocess, sys, zipfile
ON_KAGGLE = os.path.exists("/kaggle")
WORK = "/kaggle/working" if ON_KAGGLE else "/content/drive/MyDrive/atml-pa3"
if not ON_KAGGLE:
    from google.colab import drive; drive.mount("/content/drive")
os.makedirs(WORK, exist_ok=True)
REPO = "/tmp/ATML-PA3"           # code lives outside the saved output
zips = glob.glob("/kaggle/input/**/atml-pa3-code.zip", recursive=True) + glob.glob(f"{WORK}/atml-pa3-code.zip")
# Kaggle usually unpacks an uploaded zip, so also accept an already-extracted copy
dirs = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/task1/train.py", recursive=True)]
shutil.rmtree(REPO, ignore_errors=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(REPO)
else:
    assert dirs, "upload outputs/atml-pa3-code.zip as a Kaggle dataset (or to Drive/atml-pa3/)"
    shutil.copytree(dirs[0], REPO)
print("code from", zips[:1] or dirs[:1])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "hf_xet"], check=True)
os.chdir(REPO)
os.makedirs(f"{WORK}/outputs", exist_ok=True)
if not os.path.exists("outputs"):
    os.symlink(f"{WORK}/outputs", "outputs")
import torch; print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
'''

TASK1_TRAIN = [
    ("markdown", r'''
# ATML PA3 — Task 1 training

**Kaggle:** Accelerator = *GPU T4 x2*, Internet = on. Add the `atml-pa3-code` dataset.
Then **Save Version → Save & Run All (Commit)** so it runs in the background (no disconnects).
Each GPU trains one variant. Suggested sessions (keeps the 20 GB output disk safe):

| Session | GPU 0 | GPU 1 |
|---|---|---|
| 1 | `full` | `lora4` |
| 2 | `l2sp` | `lora16` |
| 3 | `oplora16` | (eval of finished runs) |

**Resuming:** if a run was cut off, add the previous version's output as an input (Add Input →
Your Work → this notebook) — the cell below copies any `resume/` state back, and training continues.

**Colab:** runs one variant on the single T4; outputs go to Drive `MyDrive/atml-pa3/outputs`.
Set `EXTRA_SAVE` to e.g. 100 so a disconnect loses little work, then just rerun all cells.
'''),
    ("code", SETUP),
    ("code", r'''
VARIANTS = ["l2sp", "lora16"]     # <- edit per session (one per GPU); session 3: ["oplora16"]
EXTRA_SAVE = 0 if ON_KAGGLE else 100
TIME_LIMIT_H = 11.3 if ON_KAGGLE else 11.0   # save + exit cleanly before the platform limit
'''),
    ("code", r'''
# restore earlier runs (previous notebook versions attached as inputs)
for prev in glob.glob("/kaggle/input/*/outputs/task1/*") if ON_KAGGLE else []:
    v = os.path.basename(prev)
    dst = f"outputs/task1/{v}"
    if not os.path.exists(dst):
        print("restoring", prev, "->", dst)
        shutil.copytree(prev, dst)
'''),
    ("code", r'''
import time
n_gpu = max(torch.cuda.device_count(), 1)
procs = {}
def launch(v, gpu):
    os.makedirs(f"outputs/task1/{v}", exist_ok=True)
    log = open(f"outputs/task1/{v}/stdout.log", "a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TOKENIZERS_PARALLELISM="false")
    cmd = [sys.executable, "-m", "task1.train", "--variant", v, "--out", f"outputs/task1/{v}",
           "--extra_save_every", str(EXTRA_SAVE), "--time_limit_h", str(TIME_LIMIT_H)]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

queue = list(VARIANTS)
free = list(range(n_gpu))
while queue or procs:
    while queue and free:
        g = free.pop(0); v = queue.pop(0); procs[v] = (launch(v, g), g); print("started", v, "on gpu", g)
    time.sleep(120)
    for v, (p, g) in list(procs.items()):
        tail = open(f"outputs/task1/{v}/stdout.log").read().splitlines()[-1:]
        print(time.strftime("%H:%M"), v, tail, flush=True)
        if p.poll() is not None:
            print(v, "exited with", p.returncode); procs.pop(v); free.append(g)
'''),
    ("code", r'''
for v in VARIANTS:
    print(v, open(f"outputs/task1/{v}/summary.json").read() if os.path.exists(f"outputs/task1/{v}/summary.json") else "no summary")
!du -sh outputs/task1/*
'''),
]

TASK1_EVAL = [
    ("markdown", r'''
# ATML PA3 — Task 1 evaluation

Attach `atml-pa3-code` **and** the output(s) of the training notebook versions (Add Input →
Your Work). GPU T4 x2, Internet on, then *Save & Run All*. Conditions whose training export
is missing are skipped, so this can run as soon as some runs finish; rerunning skips
conditions that already have results.
'''),
    ("code", SETUP),
    ("code", r'''
CONDITIONS = ["baseline", "full_u625", "full", "l2sp", "lora4", "lora16", "oplora16"]
# restore training exports + any earlier eval results from attached notebook outputs
for pat in ["/kaggle/input/*/outputs/task1/*", "/kaggle/input/*/outputs/task1_eval_models/*",
            "/kaggle/input/*/outputs/results_task1/*"]:
    for prev in glob.glob(pat) if ON_KAGGLE else []:
        dst = os.path.join("outputs", os.path.relpath(prev, os.path.dirname(os.path.dirname(prev))))
        if not os.path.exists(dst) and not prev.endswith("/resume"):
            print("restoring", prev); shutil.copytree(prev, dst, ignore=shutil.ignore_patterns("resume"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm_eval==0.4.13", "langdetect", "immutabledict"], check=True)
'''),
    ("code", r'''
# try vLLM (much faster GSM8K generation); fall back to HF if it does not work on this GPU
BACKEND = "hf"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True)
test = subprocess.run([sys.executable, "-c",
    "from vllm import LLM, SamplingParams; m=LLM('Qwen/Qwen3-0.6B', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.5);"
    "print(m.generate(['1+1='], SamplingParams(max_tokens=5))[0].outputs[0].text)"], capture_output=True, text=True)
print(test.stdout[-500:], test.stderr[-1500:])
if test.returncode == 0:
    BACKEND = "vllm"
print("BACKEND =", BACKEND)
'''),
    ("code", r'''
def have(cond):
    src = {"full_u625": "full/export_u625", "full": "full/export_u1250", "l2sp": "l2sp/export_u1250",
           "lora4": "lora4/export_u1250", "lora16": "lora16/export_u1250", "oplora16": "oplora16/export_u1250"}
    return cond == "baseline" or os.path.exists(f"outputs/task1/{src[cond]}")
ready = [c for c in CONDITIONS if have(c)]
print("ready:", ready)
subprocess.run([sys.executable, "-m", "task1.prepare_eval", "--only", *ready], check=True)
'''),
    ("code", r'''
RES = "outputs/results_task1"
todo = [c for c in ready if not os.path.exists(f"{RES}/{c}/qualitative.json")]
print("evaluating", todo)
for c in todo:   # one condition at a time so partial progress is kept
    subprocess.run([sys.executable, "-m", "task1.evaluate", "--backend", BACKEND, "--out", RES,
                    "--conditions", c], check=True)
'''),
    ("code", r'''
# eval models are rebuilt cheaply; drop them so the saved output stays small
shutil.rmtree("outputs/task1_eval_models", ignore_errors=True)
!du -sh outputs/*
'''),
]

TASK2 = [
    ("markdown", r'''
# ATML PA3 — Task 2 VLM (features, 3-stage curriculum, final evaluation)

GPU T4 x2, Internet on, attach `atml-pa3-code`, then *Save & Run All (Commit)*.
GPU 0 trains the main 196-token model; GPU 1 trains the optional 49-token variant at no
extra Kaggle-hour cost (set `RUN_49 = False` to skip). If cut off, attach this notebook's
previous output and rerun: features and checkpoints are restored and training resumes.
'''),
    ("code", SETUP),
    ("code", r'''
RUN_49 = True
for prev in glob.glob("/kaggle/input/*/outputs/task2") if ON_KAGGLE else []:
    for item in os.listdir(prev):
        dst = f"outputs/task2/{item}"
        if not os.path.exists(dst):
            print("restoring", item); shutil.copytree(os.path.join(prev, item), dst)
if not os.path.exists("outputs/task2/features/info.json"):
    subprocess.run([sys.executable, "-m", "task2.features"], check=True)
print(open("outputs/task2/features/info.json").read())
'''),
    ("code", r'''
import time
runs = {"run196": ["--n_vis", "196"]}
if RUN_49 and torch.cuda.device_count() > 1:
    runs["run49"] = ["--n_vis", "49"]
procs = {}
for gpu, (name, extra) in enumerate(runs.items()):
    os.makedirs(f"outputs/task2/{name}", exist_ok=True)
    log = open(f"outputs/task2/{name}/stdout.log", "a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    procs[name] = subprocess.Popen([sys.executable, "-m", "task2.train", "--out", f"outputs/task2/{name}",
                                    "--time_limit_h", "10.5", *extra], stdout=log, stderr=subprocess.STDOUT, env=env)
while any(p.poll() is None for p in procs.values()):
    time.sleep(180)
    for name in procs:
        tail = open(f"outputs/task2/{name}/stdout.log").read().strip().splitlines()[-1:]
        print(time.strftime("%H:%M"), name, tail[0][:300] if tail else "(starting)", flush=True)
print({n: p.returncode for n, p in procs.items()})
'''),
    ("code", r'''
for name, extra in runs.items():
    if os.path.exists(f"outputs/task2/{name}/stage3_vqa.safetensors"):
        subprocess.run([sys.executable, "-m", "task2.evaluate", "--run", f"outputs/task2/{name}",
                        "--out", f"outputs/task2/results/{name}", *extra], check=True)
!du -sh outputs/task2/*
'''),
]

TASK3_LIVE = [
    ("markdown", r'''
# ATML PA3 — Task 3 live generation (Step 4)

Single T4 is enough (Kaggle: Accelerator *GPU T4 x2* or *P100*, Internet on). Attach `atml-pa3-code`, then
*Save & Run All (Commit)*. Resumable: if cut off, attach this notebook's previous output and rerun; finished
calls are restored and skipped. Download `task3_live.zip` from the Output tab.
'''),
    ("code", SETUP),
    ("code", r'''
# restore finished calls from an attached earlier version of this notebook
prev = glob.glob("/kaggle/input/**/outputs/task3_live/calls.jsonl", recursive=True) if ON_KAGGLE else []
os.makedirs("outputs/task3_live", exist_ok=True)
if prev and not os.path.exists("outputs/task3_live/calls.jsonl"):
    shutil.copy(prev[0], "outputs/task3_live/calls.jsonl")
    print("restored", sum(1 for _ in open(prev[0])), "calls from", prev[0])
subprocess.run([sys.executable, "-m", "task3.live", "--out", "outputs/task3_live"], check=True)
'''),
    ("code", r'''
n = sum(1 for _ in open("outputs/task3_live/calls.jsonl"))
print(n, "calls (expected 250 = 50 questions x 5 calls)")
shutil.make_archive(f"{WORK}/task3_live", "zip", "outputs/task3_live")
print(f"{WORK}/task3_live.zip", os.path.getsize(f"{WORK}/task3_live.zip") // 1024, "KB")
'''),
]

TASK1_FINISH = [
    ("markdown", r'''
# ATML PA3 — Task 1 finish: OPLoRA-16 training (GPU 1) + evaluation of all 7 conditions (GPU 0)

GPU T4 x2, Internet on. Attach `atml-pa3-code` **and** the outputs of training sessions 1 and 2
(Add Input → Your Work → the training notebook → pick each version). Then *Save & Run All (Commit)*.
GPU 0 evaluates every condition whose export exists (lm-eval HF backend + 24 qualitative prompts);
OPLoRA is evaluated as soon as its training on GPU 1 finishes. Rerunning with this notebook's own
output attached skips anything already finished.
'''),
    ("code", SETUP),
    ("code", r'''
# restore training exports/logs (not resume states) and earlier eval results from attached outputs
for prev in sorted(glob.glob("/kaggle/input/**/outputs/task1/*", recursive=True)):
    dst = f"outputs/task1/{os.path.basename(prev)}"
    if os.path.isdir(prev) and not os.path.exists(dst):
        print("restoring", prev); shutil.copytree(prev, dst, ignore=shutil.ignore_patterns("resume.tmp"))
for prev in sorted(glob.glob("/kaggle/input/**/outputs/results_task1/*", recursive=True)):
    dst = f"outputs/results_task1/{os.path.basename(prev)}"
    if os.path.isdir(prev) and not os.path.exists(dst):
        print("restoring", prev); shutil.copytree(prev, dst)
print("attached inputs:", os.listdir("/kaggle/input") if ON_KAGGLE else "-")
need = {"full": "export_u1250", "lora4": "export_u1250", "l2sp": "export_u1250", "lora16": "export_u1250"}
missing = [v for v, e in need.items() if not os.path.exists(f"outputs/task1/{v}/{e}/export_info.json")]
assert not missing, (f"training exports not found for {missing}: attach the session-1 and session-2 versions "
                     "of the training notebook (Add Input -> Your Work -> Notebooks) and rerun")
print("found all training exports")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm_eval==0.4.13", "langdetect", "immutabledict"], check=True)
!ls outputs/task1/*
'''),
    ("code", r'''
import json, time
T_START, BUDGET_H = time.time(), 10.5
EXPORT = {"full_u625": "full/export_u625", "full": "full/export_u1250", "l2sp": "l2sp/export_u1250",
          "lora4": "lora4/export_u1250", "lora16": "lora16/export_u1250", "oplora16": "oplora16/export_u1250"}
CONDITIONS = ["baseline", "full_u625", "full", "l2sp", "lora4", "lora16", "oplora16"]
RES, MODELS = "outputs/results_task1", "/kaggle/temp/eval_models" if ON_KAGGLE else "/content/eval_models"
ready = lambda c: c == "baseline" or os.path.exists(f"outputs/task1/{EXPORT[c]}/export_info.json")
evaluated = lambda c: os.path.exists(f"{RES}/{c}/qualitative.json")

train = None
if not ready("oplora16"):
    os.makedirs("outputs/task1/oplora16", exist_ok=True)
    env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="1" if torch.cuda.device_count() > 1 else "0",
                PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    train = subprocess.Popen([sys.executable, "-m", "task1.train", "--variant", "oplora16",
                              "--out", "outputs/task1/oplora16", "--time_limit_h", "10.3"],
                             stdout=open("outputs/task1/oplora16/stdout.log", "a"), stderr=subprocess.STDOUT, env=env1)
    print("started oplora16 training")

env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
while True:
    todo = [c for c in CONDITIONS if ready(c) and not evaluated(c)]
    if not todo:
        if train is not None and train.poll() is None:
            print(time.strftime("%H:%M"), "waiting for oplora16:",
                  open("outputs/task1/oplora16/stdout.log").read().splitlines()[-1:], flush=True)
            time.sleep(300); continue
        break
    c = todo[0]
    t0 = time.time()
    left = BUDGET_H * 3600 - (time.time() - T_START)
    if left < 600:
        print("time budget reached; attach this output and rerun to continue", flush=True)
        break
    subprocess.run([sys.executable, "-m", "task1.prepare_eval", "--out", MODELS, "--only", c], check=True, env=env0)
    try:   # stop cleanly before Kaggle's 12 h limit so everything finished so far is saved
        subprocess.run([sys.executable, "-m", "task1.evaluate", "--backend", "hf", "--models", MODELS,
                        "--out", RES, "--conditions", c], check=True, env=env0, timeout=left)
    except subprocess.TimeoutExpired:
        print("time budget reached during", c, flush=True)
        break
    shutil.rmtree(f"{MODELS}/{c}", ignore_errors=True)
    print(time.strftime("%H:%M"), "evaluated", c, f"in {(time.time() - t0) / 60:.0f} min", flush=True)
# earlier conditions ran GSM8K at batch size 1 (lm-eval "auto" fallback), later ones at 32:
# rerun the baseline at 32 once to measure whether batching changes the scores
CHECK = f"{RES}/_batchcheck"
left = BUDGET_H * 3600 - (time.time() - T_START)
if all(evaluated(c) for c in CONDITIONS) and left > 1800 and not glob.glob(f"{CHECK}/baseline/gsm8k/**/results_*.json", recursive=True):
    subprocess.run([sys.executable, "-m", "task1.prepare_eval", "--out", MODELS, "--only", "baseline"], check=True, env=env0)
    try:
        subprocess.run([sys.executable, "-m", "task1.evaluate", "--backend", "hf", "--models", MODELS, "--out", CHECK,
                        "--conditions", "baseline", "--skip", "lik", "qual"], check=True, env=env0, timeout=left)
    except subprocess.TimeoutExpired:
        print("time budget reached during batch check", flush=True)
if os.path.exists(f"{MODELS}/merge_checks.json"):
    os.makedirs(RES, exist_ok=True)
    old = json.load(open(f"{RES}/merge_checks.json")) if os.path.exists(f"{RES}/merge_checks.json") else {}
    json.dump({**old, **json.load(open(f"{MODELS}/merge_checks.json"))}, open(f"{RES}/merge_checks.json", "w"), indent=2)
print("missing:", [c for c in CONDITIONS if not evaluated(c)])
'''),
    ("code", r'''
import json
for c in CONDITIONS:
    for f in glob.glob(f"{RES}/{c}/**/results_*.json", recursive=True):
        r = json.load(open(f))["results"]
        print(c, {t: {k: round(v, 4) for k, v in m.items() if isinstance(v, float) and "stderr" not in k} for t, m in r.items()})
!du -sh outputs/*
'''),
]

BATCH_CELL = r'''
# GSM8K batch size per condition, and what batch size 1 vs 32 changed where both runs exist
def latest(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    return files[-1] if files else None

def flex_samples(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        if x.get("filter") == "flexible-extract":
            out[x["doc_id"]] = (x["resps"][0][0] if isinstance(x["resps"][0], list) else x["resps"][0],
                                float(x["exact_match"]))
    return out

batch = {"batch_size": {}, "bs1_vs_bs32": {}}
for c in CONDS:
    r = json.load(open(latest(f"{RES}/{c}/gsm8k/**/results_*.json")))
    batch["batch_size"][c] = r.get("config", {}).get("batch_size")
    cur = latest(f"{RES}/{c}/gsm8k/**/samples_gsm8k_cot_*.jsonl")
    bs1 = latest(f"{RES}/{c}/gsm8k_bs1/**/samples_gsm8k_cot_*.jsonl")
    chk = latest(f"{RES}/_batchcheck/{c}/gsm8k/**/samples_gsm8k_cot_*.jsonl")
    old, new = (bs1, cur) if bs1 else (cur, chk) if chk else (None, None)
    if old:
        a, b = flex_samples(old), flex_samples(new)
        batch["bs1_vs_bs32"][c] = {
            "n": len(b), "bs1_%": 100 * sum(v[1] for v in a.values()) / len(a),
            "bs32_%": 100 * sum(v[1] for v in b.values()) / len(b),
            "identical_responses": sum(a[d][0] == b[d][0] for d in b),
            "wrong_to_right": sum(a[d][1] == 0 and b[d][1] == 1 for d in b),
            "right_to_wrong": sum(a[d][1] == 1 and b[d][1] == 0 for d in b)}
json.dump(batch, open(f"{OUT}/gsm8k_batch_settings.json", "w"), indent=1, default=int)
print(json.dumps(batch, indent=1, default=int))
'''

TASK1_ANALYZE = [
    ("markdown", r'''
# ATML PA3 — Task 1 analysis: tables, plots, diagnostics, annotation sheet, extraction check

No GPU needed. Internet on. Attach `atml-pa3-code` **and** the finished output of the Task 1 finish
notebook (Add Input → Your Work → Notebooks). Then *Save & Run All (Commit)*, or just run all cells
interactively. Everything is collected in `task1_analysis.zip` in the Output tab.
'''),
    ("code", SETUP),
    ("code", r'''
import json, random, re
import pandas as pd
CONDS = ["baseline", "full_u625", "full", "l2sp", "lora4", "lora16", "oplora16"]
roots = glob.glob("/kaggle/input/**/outputs/results_task1", recursive=True)
assert roots, "attach the finish notebook's output: Add Input -> Your Work -> Notebooks"
RES = max(roots, key=lambda r: sum(os.path.exists(f"{r}/{c}/qualitative.json") for c in CONDS))
TRAIN = os.path.join(os.path.dirname(RES), "task1")
OUT = "outputs/analysis_task1"
os.makedirs(OUT, exist_ok=True)
print("results:", RES, "\ntraining:", TRAIN)
'''),
    ("code", r'''
# benchmark table, per-example GSM8K files, learning curves, resources, qualitative sheets
subprocess.run([sys.executable, "-m", "task1.analyze", "--runs", TRAIN, "--eval", RES, "--out", OUT], check=True)
'''),
    ("code", r'''
# weight-space (rho_k, eta) and activation (R) diagnostics on down_proj of layers 0, 13, 27
subprocess.run([sys.executable, "-m", "task1.diagnostics", "--runs", TRAIN, "--out", f"{OUT}/diagnostics"], check=True)
'''),
    ("code", r'''
# Does GSM8K's flexible extraction favour one answer style? Stock scores stay the primary metric.
# Heuristic rescoring: use the response's last \boxed{} (else a bold number near its end) when it
# has one and was not truncated, otherwise the stock extraction; compare numbers, not strings.
def num(s):
    s = re.sub(r"\\text\{[^}]*\}|\\[,!]|[$,%*\\]", "", str(s))
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None

def boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 1, i + 7
    while j < len(t) and depth:
        depth += {"{": 1, "}": -1}.get(t[j], 0)
        j += 1
    return t[i + 7:j - 1] if depth == 0 else None

def final_answer(t):
    b = boxed(t)
    if b is not None and num(b) is not None:
        return num(b)
    for b in reversed(re.findall(r"\*\*(.+?)\*\*", t[-300:], re.S)):
        if num(b) is not None:
            return num(b)
    return None

def same(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6

rows, frames = [], {}
for c in CONDS:
    g = pd.read_csv(f"{OUT}/gsm8k_per_example_{c}.csv", dtype={"extracted": str, "target": str}, keep_default_na=False)
    final = [None if tr else final_answer(t) for t, tr in zip(g.response, g.truncated)]
    g["final"] = final
    g["alt"] = [f if f is not None else num(e) for f, e in zip(final, g.extracted)]
    g["alt_correct"] = [float(same(a, num(t))) for a, t in zip(g.alt, g.target)]
    frames[c] = g
    rows.append({"condition": c, "stock_%": 100 * g.correct.mean(), "rescored_%": 100 * g.alt_correct.mean(),
                 "wrong_to_right": int(((g.correct == 0) & (g.alt_correct == 1)).sum()),
                 "right_to_wrong": int(((g.correct == 1) & (g.alt_correct == 0)).sum()),
                 "has_final_marker": int(g.final.notna().sum())})
E = pd.DataFrame(rows)
for k in ("stock_%", "rescored_%"):
    E[k + "_change"] = E[k] - E[k].iloc[0]
E.to_csv(f"{OUT}/gsm8k_extraction_check.csv", index=False)
print(E.round(2).to_string())

# sheet for reading by hand: is the heuristic right, and what did it miss on the baseline?
b = frames["baseline"]
flips = b[(b.correct == 0) & (b.alt_correct == 1)]
wrong = b[(b.correct == 0) & (b.alt_correct == 0)]
picked = wrong.sample(min(20, len(wrong)), random_state=6304)
md = []
for title, part in (("Baseline: stock wrong, rescored right", flips), ("Baseline: wrong under both (random 20)", picked)):
    md.append(f"\n## {title} ({len(part)})\n")
    for _, r in part.iterrows():
        md.append(f"**doc {r.doc_id}** gold {r.target}, stock {r.extracted}, rescored {r.alt}\n\n```\n...{r.response[-250:]}\n```\n")
open(f"{OUT}/gsm8k_extraction_review_baseline.md", "w", encoding="utf-8").write("\n".join(md))
for _, r in flips.head(5).iterrows():
    print(f"\ndoc {r.doc_id}: gold {r.target}, stock {r.extracted}, rescored {r.alt}\n...{r.response[-150:]}")
'''),
    ("code", BATCH_CELL),
    ("code", r'''
# benchmark changes from baseline: one panel per benchmark on a common scale
import matplotlib.pyplot as plt
T = pd.read_csv(f"{OUT}/benchmarks.csv").set_index("condition")
order = ["full_u625", "full", "l2sp", "lora4", "lora16", "oplora16"]
panels = ["GSM8K (flex EM)", "HellaSwag acc_norm", "BoolQ acc", "ARC-E acc_norm"]
fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True, sharex=True)
for ax, p in zip(axes, panels):
    v = T.loc[order, p + " Δpp"]
    ax.barh(range(len(order)), v, height=0.55, color="#2a78d6")
    for i, x in enumerate(v):
        ax.text(x + (0.6 if x >= 0 else -0.6), i, f"{x:+.1f}", va="center", ha="left" if x >= 0 else "right", fontsize=8, color="#333333")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlim(-12, 30)
    ax.set_title(p, fontsize=10, loc="left")
    ax.set_xlabel("change from baseline (pp)", fontsize=8)
    ax.tick_params(length=0, labelsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
axes[0].set_yticks(range(len(order)), [T.loc[c, "label"] for c in order])
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig(f"{OUT}/benchmark_changes.png", dpi=200)
plt.show()
'''),
    ("code", r'''
# one file to download
shutil.make_archive(f"{WORK}/task1_analysis", "zip", OUT)
print(f"{WORK}/task1_analysis.zip", os.path.getsize(f"{WORK}/task1_analysis.zip") // 1024, "KB")
for f in sorted(glob.glob(f"{OUT}/**", recursive=True)):
    if os.path.isfile(f):
        print(os.path.getsize(f) // 1024, "KB", os.path.relpath(f, OUT))
print(open(f"{OUT}/benchmarks.md", encoding="utf-8").read())
'''),
]

TASK1_RERUN = [
    ("markdown", r'''
# ATML PA3 — Task 1: GSM8K at batch size 32 for Full FT (mid), Full FT and L2-SP, then full analysis

The first evaluations ran GSM8K at batch size 1 for baseline, Full FT (mid), Full FT and L2-SP, and at 32
for the LoRA runs. This reruns the three Full FT conditions at 32 (one per GPU), takes the baseline at 32
from the earlier batch check, and keeps every batch-size-1 run as `<condition>/gsm8k_bs1`. Then it
redoes the whole Task 1 analysis on the consistent results.

GPU T4 x2, Internet on. Attach `atml-pa3-code` (latest version) **and** the finished output of the
Task 1 finish notebook (Add Input → Your Work → Notebooks). *Save & Run All (Commit)*: about 1.5 h.
Download `task1_analysis.zip` from the Output tab.
'''),
    ("code", SETUP),
    ("code", r'''
import json, random, re, threading, time
import pandas as pd
CONDS = ["baseline", "full_u625", "full", "l2sp", "lora4", "lora16", "oplora16"]
RERUN = ["full_u625", "full", "l2sp"]
roots = glob.glob("/kaggle/input/**/outputs/results_task1", recursive=True)
assert roots, "attach the finish notebook's output: Add Input -> Your Work -> Notebooks"
SRC = max(roots, key=lambda r: sum(os.path.exists(f"{r}/{c}/qualitative.json") for c in CONDS))
TRAIN = os.path.join(os.path.dirname(SRC), "task1")
RES, TMP, OUT = "outputs/results_task1", "outputs/gsm8k_bs32", "outputs/analysis_task1"
MODELS = "/kaggle/temp/eval_models" if ON_KAGGLE else "/content/eval_models"
if not os.path.exists(RES):
    shutil.copytree(SRC, RES)       # writable copy; the reruns are swapped into it below
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
print("results from:", SRC, "\ntraining:", TRAIN)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm_eval==0.4.13", "langdetect", "immutabledict"], check=True)
'''),
    ("code", r'''
todo = [c for c in RERUN if not os.path.exists(f"{RES}/{c}/gsm8k_bs1")]
status = {}
if todo:
    subprocess.run([sys.executable, "-m", "task1.prepare_eval", "--runs", TRAIN, "--out", MODELS, "--only", *todo], check=True)
    n_gpu = max(torch.cuda.device_count(), 1)

    def worker(gpu, conds):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        for c in conds:
            with open(f"{TMP}/{c}.log", "w") as log:
                status[c] = subprocess.run([sys.executable, "-m", "task1.evaluate", "--backend", "hf", "--models", MODELS,
                                            "--out", TMP, "--conditions", c, "--skip", "lik", "qual"],
                                           env=env, stdout=log, stderr=subprocess.STDOUT).returncode

    threads = [threading.Thread(target=worker, args=(g, todo[g::n_gpu])) for g in range(n_gpu)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(300)
        for c in todo:
            if os.path.exists(f"{TMP}/{c}.log"):
                tail = open(f"{TMP}/{c}.log").read().replace("\r", "\n").strip().splitlines()[-1:]
                print(time.strftime("%H:%M"), c, tail, flush=True)
    print("exit codes:", status)
    shutil.rmtree(MODELS, ignore_errors=True)
'''),
    ("code", r'''
# swap the batch-32 runs in; batch-size-1 runs are kept beside them as gsm8k_bs1
for c in RERUN:
    new = f"{TMP}/{c}/gsm8k"
    if os.path.exists(f"{RES}/{c}/gsm8k_bs1"):
        continue
    assert status.get(c) == 0 and glob.glob(f"{new}/**/results_*.json", recursive=True), f"rerun of {c} failed: see {TMP}/{c}.log"
    shutil.move(f"{RES}/{c}/gsm8k", f"{RES}/{c}/gsm8k_bs1")
    shutil.move(new, f"{RES}/{c}/gsm8k")
if not os.path.exists(f"{RES}/baseline/gsm8k_bs1"):
    shutil.move(f"{RES}/baseline/gsm8k", f"{RES}/baseline/gsm8k_bs1")
    shutil.copytree(f"{RES}/_batchcheck/baseline/gsm8k", f"{RES}/baseline/gsm8k")
for c in CONDS:
    f = sorted(glob.glob(f"{RES}/{c}/gsm8k/**/results_*.json", recursive=True))[-1]
    r = json.load(open(f))
    print(c, "batch", r["config"]["batch_size"], "flex", round(r["results"]["gsm8k_cot"]["exact_match,flexible-extract"], 4))
    assert str(r["config"]["batch_size"]) == "32", c
'''),
    *TASK1_ANALYZE[3:],
]

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for name, cells in {"task1_train.ipynb": TASK1_TRAIN, "task1_eval.ipynb": TASK1_EVAL,
                        "task3_live.ipynb": TASK3_LIVE, "task2_vlm.ipynb": TASK2,
                        "task1_finish.ipynb": TASK1_FINISH, "task1_analyze.ipynb": TASK1_ANALYZE,
                        "task1_gsm8k_bs32.ipynb": TASK1_RERUN}.items():
        with open(os.path.join(ROOT, "notebooks", name), "w", encoding="utf-8") as f:
            json.dump(nb(cells), f, indent=1)
        print("wrote", name)
