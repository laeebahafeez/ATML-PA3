"""SigLIP-Qwen VLM: MLP connector + visual-token insertion into Qwen's chat format.

User turn: <|vision_start|> 196 x <|image_pad|> <|vision_end|> \n question  (thinking disabled).
The <|image_pad|> embeddings are replaced by scale * g_phi(F(I)). For no-image inputs the
whole vision span is removed, so masks and positions are rebuilt from the shorter sequence.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN = "Qwen/Qwen3-0.6B"
QWEN_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
N_VIS = 196
CAPS = {"question": 80, "caption": 64, "answer": 16}
CAPTION_PROMPT = "Describe this image in one sentence."
SHORT_SUFFIX = "\nAnswer the question using a single word or number."


class Connector(nn.Module):
    def __init__(self, d_in=768, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def pool_2x2(feats):
    """Optional 49-token variant: average each 2x2 block of the 14x14 grid."""
    B = feats.shape[0]
    g = feats.view(B, 14, 14, -1).permute(0, 3, 1, 2)
    return F.avg_pool2d(g, 2).permute(0, 2, 3, 1).reshape(B, 49, -1)


class Formatter:
    def __init__(self, tok, n_vis=N_VIS):
        self.tok, self.n_vis = tok, n_vis
        self.pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
        self.vs = "<|vision_start|>" + "<|image_pad|>" * n_vis + "<|vision_end|>\n"
        self.eot = tok.convert_tokens_to_ids("<|im_end|>")

    def _cap(self, text, n):
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:n]
        return self.tok.decode(ids)

    def prompt_ids(self, question, with_image=True, kind="short"):
        q = CAPTION_PROMPT if kind == "caption" else self._cap(question, CAPS["question"]) + SHORT_SUFFIX
        content = (self.vs if with_image else "") + q
        text = self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def target_ids(self, target, kind="short"):
        cap = CAPS["caption"] if kind == "caption" else CAPS["answer"]
        return self.tok(target.strip(), add_special_tokens=False)["input_ids"][:cap] + [self.eot]

    def example(self, question, target, feat_row, kind="short", with_image=True):
        p = self.prompt_ids(question, with_image, kind)
        t = self.target_ids(target, kind)
        return {"ids": p + t, "labels": [-100] * len(p) + t, "feat": feat_row if with_image else None,
                "n_prompt": len(p)}


class VLM(nn.Module):
    def __init__(self, device, n_vis=N_VIS, model_id=QWEN, revision=QWEN_REV):
        super().__init__()
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype,
                                                       attn_implementation="sdpa").to(device)
        self.lm.config.use_cache = False
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.connector = Connector(768, self.lm.config.hidden_size).to(device).float()
        self.register_buffer("vis_scale", torch.ones((), device=device))
        self.n_vis = n_vis
        self.device_ = device
        self.lora = {}

    def enable_lora(self, r=16, alpha=32, dropout=0.05, seed=6304):
        from task1.lora import inject_adapters
        mods, _ = inject_adapters(self.lm, r, alpha, dropout, seed,
                                  targets=("q_proj", "k_proj", "v_proj", "o_proj"))
        self.lora = mods
        return mods

    def trainable_groups(self):
        conn = [p for p in self.connector.parameters()]
        lora = [p for n, p in self.lm.named_parameters() if "lora_" in n]
        return conn, lora

    def visual_tokens(self, feats):
        if self.n_vis == 49:
            feats = pool_2x2(feats)
        return self.connector(feats.float()) * self.vis_scale

    @torch.no_grad()
    def calibrate(self, feats, text_ids):
        """Rule: scale so the mean L2 norm of connector outputs (at init) equals the mean L2
        norm of Qwen input embeddings of training-caption tokens."""
        self.vis_scale.fill_(1.0)
        v = self.visual_tokens(feats.to(self.device_)).norm(dim=-1).mean()
        e = self.lm.get_input_embeddings()(text_ids.to(self.device_)).float().norm(dim=-1).mean()
        self.vis_scale.fill_((e / v).item())
        return {"text_embed_mean_norm": e.item(), "connector_init_mean_norm": v.item(),
                "scale": (e / v).item()}

    def embed(self, ids, feats, fmt):
        """ids: (B, T); feats: (n_img, 196, 768) for rows that contain an image, in order."""
        emb = self.lm.get_input_embeddings()(ids)
        mask = ids == fmt.pad_id
        if feats is not None and feats.shape[0]:
            vis = self.visual_tokens(feats).to(emb.dtype)
            emb = emb.clone()
            emb[mask] = vis.reshape(-1, vis.shape[-1])
        return emb

    def hidden(self, ids, att, feats, fmt):
        emb = self.embed(ids, feats, fmt)
        return self.lm.model(inputs_embeds=emb, attention_mask=att).last_hidden_state


def collate(examples, feat_store, pad_id, device):
    T = max(len(e["ids"]) for e in examples)
    B = len(examples)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    rows = []
    for i, e in enumerate(examples):
        n = len(e["ids"])
        ids[i, :n] = torch.tensor(e["ids"])
        lab[i, :n] = torch.tensor(e["labels"])
        att[i, :n] = 1
        if e["feat"] is not None:
            rows.append(e["feat"])
    feats = feat_store[rows].to(device, non_blocking=True) if rows else None
    return ids.to(device), lab.to(device), att.to(device), feats


def per_example_nll(model, fmt, examples, feat_store, pad_id, device, exclude_eot=False):
    """Mean token NLL over each example's supervised tokens -> tensor (B,)."""
    ids, lab, att, feats = collate(examples, feat_store, pad_id, device)
    h = model.hidden(ids, att, feats, fmt)
    tgt = lab[:, 1:].clone()
    if exclude_eot:
        tgt[tgt == fmt.eot] = -100
    mask = tgt != -100
    hs = h[:, :-1][mask]
    logits = model.lm.lm_head(hs).float()
    tok_nll = F.cross_entropy(logits, tgt[mask], reduction="none")
    row = mask.nonzero()[:, 0]
    sums = torch.zeros(len(examples), device=device).index_add_(0, row, tok_nll)
    cnt = mask.sum(1).clamp(min=1)
    return sums / cnt


@torch.no_grad()
def greedy_generate(model, fmt, examples, feat_store, device, max_new=16):
    """Batched greedy decoding without KV cache reuse issues: left-pad, recompute each step."""
    tok = fmt.tok
    prompts = [e["ids"][: e["n_prompt"]] for e in examples]
    rows = [e["feat"] for e in examples if e["feat"] is not None]
    feats = feat_store[rows].to(device) if rows else None
    L = max(len(p) for p in prompts)
    pad = tok.pad_token_id
    ids = torch.full((len(prompts), L), pad, dtype=torch.long)
    att = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p)
        att[i, L - len(p):] = 1
    ids, att = ids.to(device), att.to(device)
    emb = model.embed(ids, feats, fmt)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    out = model.lm(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1)
    gen = [nxt]
    done = nxt == fmt.eot
    for _ in range(max_new - 1):
        att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
        pos = pos[:, -1:] + 1
        e = model.lm.get_input_embeddings()(nxt[:, None])
        out = model.lm(inputs_embeds=e, attention_mask=att, position_ids=pos, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, fmt.eot), nxt)
        gen.append(nxt)
        done = done | (nxt == fmt.eot)
        if done.all():
            break
    gen = torch.stack(gen, 1).tolist()
    texts = []
    for g in gen:
        g = g[: g.index(fmt.eot)] if fmt.eot in g else g
        texts.append(tok.decode(g, skip_special_tokens=True).strip())
    return texts


def lr_lambda(total, warm_frac=0.05, floor=0.1):
    warm = max(1, round(total * warm_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def load_tokenizer(model_id=QWEN, revision=QWEN_REV):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return tok
