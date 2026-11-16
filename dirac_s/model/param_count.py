"""Exact parameter ledger — dirac_formalization.md Section 9 / spec Section 18. Buckets every parameter
into the spec's listed categories and reports the exact count, never inferred from nominal architecture
names. Enforces the complex-parameter-counting convention by construction: this codebase represents
complex quantities as (re, im) pairs of REAL nn.Linear layers (never a torch.complex-dtype parameter), so
numel() on each already sums to the correct 2*d_in*d_out real-scalar count with no special-case logic
needed — `assert_no_complex_dtype_params` exists purely as a guard against that assumption silently
breaking if someone later introduces a true complex-dtype layer without updating this file.
"""
from __future__ import annotations
import torch
import torch.nn as nn

CATEGORY_RULES = [
    ("conv", lambda name, mod: isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d))),
    ("norm", lambda name, mod: isinstance(mod, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm3d))),
    ("attention", lambda name, mod: isinstance(mod, nn.TransformerEncoderLayer) or "self_attn" in name),
    ("graph_layers", lambda name, mod: "gcn" in name.lower() or "graph_head" in name.lower()),
    ("topology_embeddings", lambda name, mod: "topology_head" in name.lower()),
    ("complex_projections", lambda name, mod: "density_head" in name.lower() or "head_re" in name
                                              or "head_im" in name),
    ("decoders_adapters", lambda name, mod: "decoder" in name.lower() or "fusion" in name.lower()
                                            or "classical_pool" in name.lower()),
    ("mlp", lambda name, mod: isinstance(mod, nn.Linear)),
]


def assert_no_complex_dtype_params(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        assert not p.is_complex(), (
            f"parameter {name!r} has complex dtype — param_count's counting convention assumes complex "
            f"quantities are always (re, im) real-Linear pairs (formalization Section 9). A true complex "
            f"parameter here would need explicit 2x-real-scalar accounting, not a plain numel() sum."
        )


def _categorize(full_name: str, module: nn.Module) -> str:
    for category, rule in CATEGORY_RULES:
        if rule(full_name, module):
            return category
    return "other"


def parameter_ledger(model: nn.Module) -> dict:
    assert_no_complex_dtype_params(model)
    module_of = {}
    for mod_name, mod in model.named_modules():
        for p_name, _ in mod.named_parameters(recurse=False):
            module_of[f"{mod_name}.{p_name}" if mod_name else p_name] = (mod_name, mod)

    buckets: dict[str, int] = {}
    biases = 0
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if name.endswith(".bias") or name == "bias":
            biases += n
            continue
        mod_name, mod = module_of.get(name, ("", model))
        category = _categorize(mod_name, mod)
        buckets[category] = buckets.get(category, 0) + n
    buckets["biases"] = biases
    buckets["total"] = total
    return buckets


def exact_param_count(model: nn.Module) -> int:
    assert_no_complex_dtype_params(model)
    return sum(p.numel() for p in model.parameters())
