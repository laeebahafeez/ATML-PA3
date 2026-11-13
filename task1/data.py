"""MetaMathQA selection and tokenization for Task 1.

Selection rule (handout): walk MetaMathQA in source order and keep the first
40,000 examples that (a) have a non-empty query and response, (b) are at most
2,048 tokens once formatted, and (c) whose query / original_question does not
match a GSM8K test question after text normalization. No truncation, no shuffle.

    python -m task1.data --out manifests/task1_train_manifest.json
"""
import argparse
import re
import unicodedata

from datasets import load_dataset
from transformers import AutoTokenizer

from common.utils import read_json, write_json

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_ID = "meta-math/MetaMathQA"
N_TRAIN = 40_000
MAX_LEN = 2048
END_OF_TURN = "<|im_end|>"


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", s).strip()


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    tok.padding_side = "right"
    return tok


def prompt_text(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def encode_example(tok, question: str, response: str):
    """Return (input_ids, labels); loss only on response tokens + <|im_end|>."""
    p_ids = tok(prompt_text(tok, question), add_special_tokens=False)["input_ids"]
    r_ids = tok(response + END_OF_TURN, add_special_tokens=False)["input_ids"]
    return p_ids + r_ids, [-100] * len(p_ids) + r_ids


def build_manifest(out_path: str, n_train: int = N_TRAIN):
    tok = load_tokenizer()
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_test = {normalize_text(q) for q in gsm["question"]}
    ds = load_dataset(DATA_ID, split="train")

    kept, excluded = [], {"empty": 0, "too_long": 0, "gsm8k_test_match": 0}
    for i, ex in enumerate(ds):
        q, r = ex["query"], ex["response"]
        if not (q and q.strip()) or not (r and r.strip()):
            excluded["empty"] += 1
            continue
        if normalize_text(q) in gsm_test or normalize_text(ex["original_question"]) in gsm_test:
            excluded["gsm8k_test_match"] += 1
            continue
        ids, _ = encode_example(tok, q, r)
        if len(ids) > MAX_LEN:
            excluded["too_long"] += 1
            continue
        kept.append(i)
        if len(kept) == n_train:
            last_scanned = i
            break
    else:
        raise RuntimeError(f"only {len(kept)} eligible examples")

    manifest = {
        "dataset": DATA_ID,
        "dataset_fingerprint": ds._fingerprint,
        "model": MODEL_ID, "model_revision": MODEL_REV,
        "n_train": len(kept), "max_len": MAX_LEN,
        "last_source_index_scanned": last_scanned,
        "excluded_before_cutoff": excluded,
        "source_indices": kept,
    }
    write_json(manifest, out_path, indent=None)
    print({k: v for k, v in manifest.items() if k != "source_indices"})
    return manifest


def load_train_examples(manifest_path: str, tok=None):
    """Tokenized training examples in manifest (= source) order."""
    tok = tok or load_tokenizer()
    man = read_json(manifest_path)
    ds = load_dataset(DATA_ID, split="train").select(man["source_indices"])
    out = []
    for ex in ds:
        ids, labels = encode_example(tok, ex["query"], ex["response"])
        assert len(ids) <= MAX_LEN
        out.append((ids, labels))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifests/task1_train_manifest.json")
    ap.add_argument("--n", type=int, default=N_TRAIN)
    a = ap.parse_args()
    build_manifest(a.out, a.n)
