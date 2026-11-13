"""Download the selected COCO images and cache frozen SigLIP patch features (FP16).

    python -m task2.features --images manifests/task2/images.json --out outputs/task2/features

Output: features.npy (N x 196 x 768, float16, rows in sorted image-id order) and
index.json {image_id: row}. Uses the 196 final patch features in 14x14 order, i.e.
SiglipVisionModel.last_hidden_state, which has the final post-layernorm applied once.
The vision encoder is deleted from GPU memory afterwards.
"""
import argparse
import io
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from common.utils import read_json, write_json

SIGLIP = "google/siglip-base-patch16-224"


def fetch(url, path, tries=5):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    for t in range(tries):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            Image.open(io.BytesIO(data)).verify()
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return True
        except Exception:
            time.sleep(2 * (t + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="manifests/task2/images.json")
    ap.add_argument("--img_dir", default="/kaggle/temp/coco" if os.path.exists("/kaggle") else "data/task2/images")
    ap.add_argument("--out", default="outputs/task2/features")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests")
    a = ap.parse_args()
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    os.makedirs(a.img_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    images = read_json(a.images)
    ids = sorted(int(k) for k in images)[: a.limit]
    paths = {i: os.path.join(a.img_dir, f"{i}.jpg") for i in ids}

    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(lambda i: fetch(images[str(i)]["url"], paths[i]), ids))
    missing = [i for i, o in zip(ids, ok) if not o]
    assert not missing, f"failed downloads: {missing[:10]} ({len(missing)})"
    t_download = time.time() - t0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = SiglipImageProcessor.from_pretrained(SIGLIP)
    vcfg = SiglipVisionConfig.from_pretrained(SIGLIP)      # vision tower of the dual-encoder checkpoint
    model = SiglipVisionModel.from_pretrained(SIGLIP, config=vcfg, torch_dtype=dtype).to(device).eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    feats = np.lib.format.open_memmap(os.path.join(a.out, "features.npy"), mode="w+",
                                      dtype=np.float16, shape=(len(ids), 196, 768))
    t1 = time.time()
    checked = False
    with torch.no_grad():
        for s in range(0, len(ids), a.batch):
            batch = [Image.open(paths[i]).convert("RGB") for i in ids[s:s + a.batch]]
            px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device, dtype)
            h = model(pixel_values=px).last_hidden_state
            assert h.shape[1:] == (196, 768), h.shape
            if not checked:   # last_hidden_state == post_layernorm(encoder output): applied exactly once
                vm = model.vision_model
                manual = vm.post_layernorm(vm.encoder(vm.embeddings(px)).last_hidden_state)
                assert torch.allclose(manual.float(), h.float(), atol=1e-2), "unexpected layer-norm placement"
                checked = True
            feats[s:s + len(batch)] = h.float().cpu().numpy().astype(np.float16)
    feats.flush()
    t_encode = time.time() - t1
    info = {"model": SIGLIP, "n_images": len(ids), "shape": [len(ids), 196, 768], "dtype": "float16",
            "download_seconds": t_download, "encode_seconds": t_encode,
            "peak_mem_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "bytes": os.path.getsize(os.path.join(a.out, "features.npy")),
            "preprocessing": "SiglipImageProcessor defaults (224x224 bicubic resize, rescale, mean/std 0.5)"}
    write_json({str(i): r for r, i in enumerate(ids)}, os.path.join(a.out, "index.json"), indent=None)
    write_json(info, os.path.join(a.out, "info.json"))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(info)


if __name__ == "__main__":
    main()
