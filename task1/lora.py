"""LoRA and OPLoRA adapters (own implementation so both share init and code path).

LoRA:   W = W0 + (alpha/r) B A,            A: r x d_in (random), B: d_out x r (zero)
OPLoRA: W = W0 + (alpha/r) B~ A~,          B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T
where U_k, V_k are the top-k left/right singular vectors of W0 (computed once, FP32,
frozen). The projections are applied to the factors, never as dense d x d matrices.
"""
import math
import time

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float,
                 generator: torch.Generator, U_k=None, V_k=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.r, self.scale = r, alpha / r
        dev = base.weight.device
        A = torch.empty(r, d_in, dtype=torch.float32)
        # same init as peft / nn.Linear default (kaiming-uniform, a=sqrt(5)), drawn from
        # a dedicated generator so LoRA-16 and OPLoRA-16 get identical initial factors
        bound = 1.0 / math.sqrt(d_in)
        A.uniform_(-bound, bound, generator=generator)
        self.lora_A = nn.Parameter(A.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=dev))
        self.dropout = nn.Dropout(dropout)
        self.protected = U_k is not None
        if self.protected:
            self.register_buffer("U_k", U_k.float().contiguous(), persistent=False)
            self.register_buffer("V_k", V_k.float().contiguous(), persistent=False)

    def effective_factors(self):
        """(A~, B~) in FP32; equal to (A, B) for plain LoRA."""
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            A, B = self.lora_A.float(), self.lora_B.float()
            if self.protected:
                B = B - self.U_k @ (self.U_k.T @ B)
                A = A - (A @ self.V_k) @ self.V_k.T
        return A, B

    def delta_weight(self):
        A, B = self.effective_factors()
        return self.scale * (B @ A)

    def forward(self, x):
        y = self.base(x)
        A, B = self.effective_factors()
        h = self.dropout(x)
        upd = (h @ A.to(h.dtype).T) @ B.to(h.dtype).T
        return y + self.scale * upd.to(y.dtype)


def top_k_bases(W: torch.Tensor, k: int):
    """Leading k left/right singular vectors of W (computed in FP32)."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return U[:, :k].contiguous(), Vh[:k].T.contiguous()


def inject_adapters(model, r, alpha, dropout, seed, protect_k=None, targets=TARGETS):
    """Replace target linears in every decoder layer, in a fixed order.

    Returns (modules dict name->LoRALinear, info dict with preprocessing time and
    basis storage bytes)."""
    gen = torch.Generator().manual_seed(seed)
    mods, basis_bytes, t_svd = {}, 0, 0.0
    for li, layer in enumerate(model.model.layers):
        for name in targets:
            parent = layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            base = getattr(parent, name)
            U_k = V_k = None
            if protect_k:
                t0 = time.perf_counter()
                U_k, V_k = top_k_bases(base.weight.data, protect_k)
                if U_k.is_cuda:
                    torch.cuda.synchronize()
                t_svd += time.perf_counter() - t0
                basis_bytes += (U_k.numel() + V_k.numel()) * 4
            m = LoRALinear(base, r, alpha, dropout, gen, U_k, V_k)
            setattr(parent, name, m)
            mods[f"model.layers.{li}.{'self_attn' if parent is layer.self_attn else 'mlp'}.{name}"] = m
    return mods, {"svd_seconds": t_svd, "basis_bytes": basis_bytes}


def adapter_state(mods, effective=False):
    """Raw (A, B) or effective (A~, B~) factors keyed by module name."""
    out = {}
    for n, m in mods.items():
        if effective:
            A, B = m.effective_factors()
        else:
            A, B = m.lora_A, m.lora_B
        out[f"{n}.lora_A"] = A.detach().float().cpu().contiguous()
        out[f"{n}.lora_B"] = B.detach().float().cpu().contiguous()
    return out


@torch.no_grad()
def merge_into(model, factors: dict, scale: float):
    """Add scale * B A into a plain (un-injected) model's target weights in place.

    `factors` holds the exported effective factors (projected ones for OPLoRA)."""
    sd = dict(model.named_modules())
    names = {k.rsplit(".lora_", 1)[0] for k in factors}
    for n in names:
        lin = sd[n]
        A = factors[f"{n}.lora_A"].to(lin.weight.device, torch.float32)
        B = factors[f"{n}.lora_B"].to(lin.weight.device, torch.float32)
        lin.weight.copy_((lin.weight.float() + scale * (B @ A)).to(lin.weight.dtype))
    return model
