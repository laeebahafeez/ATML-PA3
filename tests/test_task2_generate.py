"""Batched left-padded greedy decoding == one-at-a-time decoding (tiny model, CPU)."""
import numpy as np
import torch

from common.utils import read_json
from task2 import vlm as V
from task2.train import Data

S = "outputs/smoke_task2"
torch.manual_seed(0)
tok = V.load_tokenizer("outputs/tiny_qwen3", "main")
fmt = V.Formatter(tok)
feats = torch.from_numpy(np.load(f"{S}/features/features.npy")).float()
data = Data(f"{S}/manifests", read_json(f"{S}/features/index.json"), fmt)
model = V.VLM("cpu", 196, "outputs/tiny_qwen3", "main").eval()
model.vis_scale.fill_(0.1)
p = data.suites["test"]["vqa"]
exs = data.pair_examples(p[0]) + data.pair_examples(p[1], with_image=False) + data.pair_examples(p[2])
batched = V.greedy_generate(model, fmt, exs, feats, "cpu", max_new=6)
single = [V.greedy_generate(model, fmt, [e], feats, "cpu", max_new=6)[0] for e in exs]
print(batched)
print(single)
assert batched == single
print("ok")
