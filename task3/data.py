"""Task 3: load supplied rollouts, verify alignment, build answer groups and labels.

    python -m task3.data            # writes outputs/task3/{math,mathvista}.pkl + manifests

Answer groups are formed per question WITHOUT gold answers (string normalization plus
math-verify equivalence for MATH; MathVista answer-type/choice/precision rules). Gold is
then graded against each group's representative with the same rules. Missing
extractions form one NO_ANSWER group that is always incorrect.
"""
import os
import pickle
import re
from importlib.metadata import version

import numpy as np
import pandas as pd

from common.utils import SEED, read_json, write_json

DATA = "data/task3"
OUT = "outputs/task3"
NO_ANSWER = "NO_ANSWER"
N_ROLLOUTS = 32
N_ORDERS = 10
DEV_SIZE = 200
LIVE_SIZE = 50
DATASETS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
SCORER_VERSION = {"math_verify": version("math-verify"), "mathvista_rules": "pa3-v1"}


# ---------------------------------------------------------------- MATH (math-verify)
_mv_parse_cache, _mv_eq_cache = {}, {}


def _mv_parse(ans):
    from math_verify import parse
    if ans not in _mv_parse_cache:
        _mv_parse_cache[ans] = parse("$\\boxed{" + ans + "}$", parsing_timeout=None)
    return _mv_parse_cache[ans]


def math_equiv(a, b):
    """Symmetric math-verify equivalence of two boxed-answer strings."""
    if a == b:
        return True
    key = (a, b) if a < b else (b, a)
    if key not in _mv_eq_cache:
        from math_verify import verify
        pa, pb = _mv_parse(a), _mv_parse(b)
        ok = False
        if pa and pb:
            try:
                ok = verify(pa, pb, timeout_seconds=None) or verify(pb, pa, timeout_seconds=None)
            except Exception:
                ok = False
        if not ok:   # fall back to exact match of whitespace-stripped strings
            ok = re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)
        _mv_eq_cache[key] = ok
    return _mv_eq_cache[key]


# ---------------------------------------------------------------- MathVista rules
_LATEX_WRAP = re.compile(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\{([^{}]*)\}")


def _clean(s):
    s = str(s).strip()
    s = s.replace("\\,", " ").replace("\\!", "").replace("\\;", " ").replace("\\ ", " ")
    s = s.replace("\\%", "%").replace("\\$", "$").replace("$", "")
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
    s = _LATEX_WRAP.sub(r"\1", s)
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("°", "").replace("\\circ", "")
    # scientific notation: 2.79 \times 10^{-1} -> 2.79e-1 ; bare 10^8 -> 1e8
    s = re.sub(r"(-?\d+\.?\d*)\s*(?:\\times|\\cdot)\s*10\^\{?(-?\d+)\}?", r"\1e\2", s)
    s = re.sub(r"^10\^\{?(-?\d+)\}?", r"1e\1", s)
    return re.sub(r"\s+", " ", s).strip()


_NUM = re.compile(r"(?<![\^\d.])(?:-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+)")  # skip unit exponents (s^2)


def _to_float(s):
    s = _clean(s)
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?\s*/\s*\(?(-?[\d.]+)\)?", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    nums = _NUM.findall(s)
    if len(nums) != 1:          # ambiguous or none
        return None
    rest = _NUM.sub("", s, count=1).strip(" .%")
    # allow trailing units such as "cm", "\mathrm{~m} / \mathrm{s}^2" (wrappers already removed)
    if rest and not re.fullmatch(r"[A-Za-z ~/^0-9²³{}\\]*", rest):
        return None
    try:
        return float(nums[0].replace(",", ""))
    except ValueError:
        return None


def _letters(meta):
    return [chr(ord("A") + i) for i in range(len(meta["choices"] or []))]


def mathvista_normalize(ans, meta):
    """Canonical key for a MathVista answer (option letter, number, list or text)."""
    if ans is None or (isinstance(ans, float) and np.isnan(ans)) or str(ans).strip() == "":
        return NO_ANSWER
    s = _clean(ans)
    if meta["question_type"] == "multi_choice":
        letters = _letters(meta)
        m = re.fullmatch(r"\(?([A-Za-z])\)?[.:)]?", s)
        if m and m.group(1).upper() in letters:
            return m.group(1).upper()
        m = re.match(r"^\(?([A-Z])\)[\s.:]", s) or re.match(r"^([A-Z])[.:]\s", s)
        if m and m.group(1) in letters:
            return m.group(1)
        norm = lambda x: re.sub(r"[\s$]", "", _clean(x)).lower().rstrip(".")
        for L, c in zip(letters, meta["choices"]):
            if norm(c) == norm(s):
                return L
        f = _to_float(s)
        if f is not None:
            hits = [L for L, c in zip(letters, meta["choices"])
                    if _to_float(c) is not None and abs(_to_float(c) - f) < 1e-9]
            if len(hits) == 1:
                return hits[0]
        return "OTHER:" + s.lower()
    at = meta["answer_type"]
    if at == "integer":
        f = _to_float(s)
        return str(int(f)) if f is not None else "TEXT:" + s.lower()
    if at == "float":
        f = _to_float(s)
        prec = int(meta["precision"] or 0)
        return f"{round(f, prec):.{prec}f}" if f is not None else "TEXT:" + s.lower()
    if at == "list":
        parts = [_to_float(p) for p in re.split(r"[,;]|\band\b", s.strip("[]() "))]
        if parts and all(p is not None for p in parts):
            return "LIST:" + ",".join(f"{p:g}" for p in parts)
        return "TEXT:" + s.lower()
    return "TEXT:" + re.sub(r"[\s.]", "", s.lower())


def mathvista_gold(gold, meta):
    if meta["question_type"] == "multi_choice":
        return str(gold).strip().upper()
    return mathvista_normalize(gold, meta)


# ---------------------------------------------------------------- grouping
def group_math(extracted):
    """extracted: list of 32 strings/None in rollout-id order -> (group ids, reps)."""
    reps, gid = [], []
    for e in extracted:
        if e is None or str(e).strip() == "":
            key = NO_ANSWER
        else:
            key = None
            for j, r in enumerate(reps):
                if r != NO_ANSWER and math_equiv(r, e):
                    key = j
                    break
            if key is None:
                reps.append(e)
                key = len(reps) - 1
            gid.append(key)
            continue
        if NO_ANSWER not in reps:
            reps.append(NO_ANSWER)
        gid.append(reps.index(NO_ANSWER))
    return np.array(gid), reps


def group_keys(keys):
    reps, gid = [], []
    for k in keys:
        if k not in reps:
            reps.append(k)
        gid.append(reps.index(k))
    return np.array(gid), reps


# ---------------------------------------------------------------- splits / orders
def make_split(ids):
    ids = sorted(ids)
    perm = np.random.default_rng(SEED).permutation(len(ids))
    ordered = [ids[i] for i in perm]
    return {"dev": ordered[:DEV_SIZE], "eval": ordered[DEV_SIZE:], "live_math": ordered[DEV_SIZE:DEV_SIZE + LIVE_SIZE]}


def make_orders(qids):
    rng = np.random.default_rng(SEED)
    return {q: np.stack([rng.permutation(N_ROLLOUTS) for _ in range(N_ORDERS)]) for q in sorted(qids)}


# ---------------------------------------------------------------- load
def load(name):
    folder = os.path.join(DATA, DATASETS[name])
    keys = ["dataset", "sample_id", "rollout_id"]
    g = pd.read_parquet(os.path.join(folder, "generations.parquet"))
    p = pd.read_parquet(os.path.join(folder, "probability_metrics.parquet"))
    df = g.merge(p, on=keys, how="inner", validate="one_to_one")
    if name == "mathvista":
        a = pd.read_parquet(os.path.join(folder, "attention_metrics.parquet"))
        a = a.rename(columns={"n_generated_text_tokens": "attn_prev_counts"})
        df = df.merge(a, on=keys, how="inner", validate="one_to_one")
    assert len(df) == len(g) == 32_000, len(df)
    # token-list alignment
    lists = ["generated_token_ids", "logp", "kl_u_p"] + (
        ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total", "attn_prev_counts"]
        if name == "mathvista" else [])
    bad = 0
    for col in lists:
        lens = df[col].map(len).to_numpy()
        bad += int((lens != df["n_generated_text_tokens"].to_numpy()).sum())
    assert bad == 0, f"{bad} misaligned token lists"
    return df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)


def build(name):
    df = load(name)
    meta = None
    if name == "math":
        math_cache = read_json(os.path.join(OUT, "math_groups.json"))
    if name == "mathvista":
        meta = {m["pid"]: m for m in read_json("manifests/mathvista_testmini_metadata.json")}
    questions = {}
    for qid, d in df.groupby("sample_id", sort=True):
        d = d.sort_values("rollout_id")
        assert list(d.rollout_id) == list(range(N_ROLLOUTS))
        ext = [None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
               for x in d.extracted_answer]
        gold = d.gold_answer.iloc[0]
        if name == "math":
            mg = math_cache[qid]            # task3.math_groups (parallel, timeout-guarded)
            gid, reps, rep_ok = np.array(mg["gid"]), mg["reps"], mg["rep_ok"]
        else:
            m = meta[qid]
            gid, reps = group_keys([mathvista_normalize(e, m) for e in ext])
            gk = mathvista_gold(gold, m)
            rep_ok = [r != NO_ANSWER and r == gk for r in reps]
        T = d.n_generated_text_tokens.to_numpy()
        q = {
            "gid": gid, "reps": reps, "group_correct": np.array(rep_ok, bool),
            "correct": np.array(rep_ok, bool)[gid], "no_answer": np.array([r == NO_ANSWER for r in reps])[gid],
            "T": T, "capped": d.hit_max_new_tokens.to_numpy(),
            "L": np.array([np.mean(x) if len(x) else -np.inf for x in d.logp]),
            "K": np.array([np.mean(x) if len(x) else -np.inf for x in d.kl_u_p]),
            "kl_tokens": [np.asarray(x, np.float64) for x in d.kl_u_p],
            "gold": gold, "extracted": ext,
        }
        if name == "mathvista":
            v = []
            for im, pr, ge in zip(d.img_attn_total, d.prompt_text_attn_total, d.generated_text_attn_total):
                im, pr, ge = (np.asarray(x, np.float64) for x in (im, pr, ge))
                v.append(im / np.maximum(im + pr + ge, 1e-12))
            q["v_tokens"] = v
            q["V"] = np.array([x.mean() if len(x) else 0.0 for x in v])
            q["meta"] = {k: meta[qid][k] for k in ("question_type", "answer_type", "precision", "choices")}
        questions[qid] = q
    return questions


def main():
    import sys
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:] or DATASETS:
        qs = build(name)
        split = make_split(list(qs))
        orders = make_orders(list(qs))
        if name != "math":
            split.pop("live_math")
        write_json(split, f"manifests/task3_split_{name}.json")
        np.savez_compressed(f"manifests/task3_orders_{name}.npz", **{k: v for k, v in orders.items()})
        with open(f"{OUT}/{name}.pkl", "wb") as f:
            pickle.dump({"questions": qs, "split": split, "orders": orders,
                         "scorer_version": SCORER_VERSION}, f)
        corr = np.concatenate([q["correct"] for q in qs.values()])
        na = np.concatenate([q["no_answer"] for q in qs.values()])
        print(name, "rollout acc %.3f, NO_ANSWER %.3f, mean groups %.2f" % (
            corr.mean(), na.mean(), np.mean([len(q["reps"]) for q in qs.values()])))
    write_json(SCORER_VERSION, "results/task3/scorer_version.json")


if __name__ == "__main__":
    main()
