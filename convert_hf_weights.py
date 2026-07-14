#!/usr/bin/env python3
"""
Convert HuggingFace nvidia/mit-b3 weights to mix_transformer format.
Generates mit_b3.pth compatible with the GF-Transformer codebase.

HF state dict has 628 keys (separate key/value/query).
Custom mix_transformer has 572 keys (kv is merged from key+value).
"""

import sys
import os
import torch
from collections import OrderedDict

# ── Step 1: Load HF model ──────────────────────────────────────────────
print("Loading HuggingFace nvidia/mit-b3 ...")
from transformers import SegformerModel

hf_model = SegformerModel.from_pretrained("nvidia/mit-b3")
hf_sd = hf_model.state_dict()
print(f"  HF keys: {len(hf_sd)}")

# ── Step 2: Load custom model to get target keys ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from mix_transformer import mit_b3

custom_model = mit_b3()
custom_sd = custom_model.state_dict()
print(f"  Custom keys: {len(custom_sd)}")

# ── Step 3: Build mapping ──────────────────────────────────────────────
new_sd = OrderedDict()
unmapped = []
skipped = []

# Architecture of mit_b3:
#   stage 0 (block1): layers 0,1,2     — 3 layers, has SR + attn.norm
#   stage 1 (block2): layers 0,1,2,3   — 4 layers, has SR + attn.norm
#   stage 2 (block3): layers 0..17     — 18 layers, has SR + attn.norm
#   stage 3 (block4): layers 0,1,2     — 3 layers, NO SR, NO attn.norm

STAGE_CONFIG = [
    (1, 3, True),   # block1: 3 layers, has SR + attn.norm
    (2, 4, True),   # block2: 4 layers, has SR + attn.norm
    (3, 18, True),  # block3: 18 layers, has SR + attn.norm
    (4, 3, False),  # block4: 3 layers, no SR, no attn.norm
]

for block_id, num_layers, has_sr_norm in STAGE_CONFIG:
    hf_stage = block_id - 1  # HF uses 0-indexed stages
    for layer in range(num_layers):
        prefix_hf = f"encoder.block.{hf_stage}.{layer}."
        prefix_cu = f"block{block_id}.{layer}."

        # ── Attention query ──
        new_sd[prefix_cu + "attn.q.weight"] = hf_sd.pop(prefix_hf + "attention.self.query.weight")
        new_sd[prefix_cu + "attn.q.bias"] = hf_sd.pop(prefix_hf + "attention.self.query.bias")

        # ── Attention kv (merge key + value) ──
        k_w = hf_sd.pop(prefix_hf + "attention.self.key.weight")
        v_w = hf_sd.pop(prefix_hf + "attention.self.value.weight")
        new_sd[prefix_cu + "attn.kv.weight"] = torch.cat([k_w, v_w], dim=0)

        k_b = hf_sd.pop(prefix_hf + "attention.self.key.bias")
        v_b = hf_sd.pop(prefix_hf + "attention.self.value.bias")
        new_sd[prefix_cu + "attn.kv.bias"] = torch.cat([k_b, v_b], dim=0)

        # ── Attention SR (only stages 0,1,2) ──
        if has_sr_norm:
            new_sd[prefix_cu + "attn.sr.weight"] = hf_sd.pop(prefix_hf + "attention.self.sr.weight")
            new_sd[prefix_cu + "attn.sr.bias"] = hf_sd.pop(prefix_hf + "attention.self.sr.bias")
            new_sd[prefix_cu + "attn.norm.weight"] = hf_sd.pop(prefix_hf + "attention.self.layer_norm.weight")
            new_sd[prefix_cu + "attn.norm.bias"] = hf_sd.pop(prefix_hf + "attention.self.layer_norm.bias")

        # ── Attention proj (output.dense) ──
        new_sd[prefix_cu + "attn.proj.weight"] = hf_sd.pop(prefix_hf + "attention.output.dense.weight")
        new_sd[prefix_cu + "attn.proj.bias"] = hf_sd.pop(prefix_hf + "attention.output.dense.bias")

        # ── Layer norms ──
        new_sd[prefix_cu + "norm1.weight"] = hf_sd.pop(prefix_hf + "layer_norm_1.weight")
        new_sd[prefix_cu + "norm1.bias"] = hf_sd.pop(prefix_hf + "layer_norm_1.bias")
        new_sd[prefix_cu + "norm2.weight"] = hf_sd.pop(prefix_hf + "layer_norm_2.weight")
        new_sd[prefix_cu + "norm2.bias"] = hf_sd.pop(prefix_hf + "layer_norm_2.bias")

        # ── MLP ──
        new_sd[prefix_cu + "mlp.fc1.weight"] = hf_sd.pop(prefix_hf + "mlp.dense1.weight")
        new_sd[prefix_cu + "mlp.fc1.bias"] = hf_sd.pop(prefix_hf + "mlp.dense1.bias")
        new_sd[prefix_cu + "mlp.dwconv.dwconv.weight"] = hf_sd.pop(prefix_hf + "mlp.dwconv.dwconv.weight")
        new_sd[prefix_cu + "mlp.dwconv.dwconv.bias"] = hf_sd.pop(prefix_hf + "mlp.dwconv.dwconv.bias")
        new_sd[prefix_cu + "mlp.fc2.weight"] = hf_sd.pop(prefix_hf + "mlp.dense2.weight")
        new_sd[prefix_cu + "mlp.fc2.bias"] = hf_sd.pop(prefix_hf + "mlp.dense2.bias")

# ── Patch embeddings ──
for i in range(4):
    src = f"encoder.patch_embeddings.{i}."
    dst = f"patch_embed{i + 1}."
    new_sd[dst + "proj.weight"] = hf_sd.pop(src + "proj.weight")
    new_sd[dst + "proj.bias"] = hf_sd.pop(src + "proj.bias")
    new_sd[dst + "norm.weight"] = hf_sd.pop(src + "layer_norm.weight")
    new_sd[dst + "norm.bias"] = hf_sd.pop(src + "layer_norm.bias")

# ── Final layer norms ──
for i in range(4):
    src = f"encoder.layer_norm.{i}."
    dst = f"norm{i + 1}."
    new_sd[dst + "weight"] = hf_sd.pop(src + "weight")
    new_sd[dst + "bias"] = hf_sd.pop(src + "bias")

# ── Step 4: Report remaining keys ───────────────────────────────────────
print(f"\n  Mapped: {len(new_sd)} keys")
print(f"  Remaining in HF: {len(hf_sd)}")
for k in sorted(hf_sd.keys()):
    print(f"    [UNMAPPED HF] {k}")

# Check for missing custom keys
missing = [k for k in custom_sd.keys() if k not in new_sd]
if missing:
    print(f"\n  Missing custom keys ({len(missing)}):")
    for k in missing[:15]:
        print(f"    [MISSING] {k}")
    if len(missing) > 15:
        print(f"    ... +{len(missing)-15} more")

# ── Step 5: Validate shapes ─────────────────────────────────────────────
print("\n─ Validating shapes...")
mismatched = []
for k in new_sd:
    if k in custom_sd:
        if new_sd[k].shape != custom_sd[k].shape:
            mismatched.append((k, list(new_sd[k].shape), list(custom_sd[k].shape)))
    else:
        mismatched.append((k, list(new_sd[k].shape), "NOT IN CUSTOM"))

if mismatched:
    print(f"  Shape mismatches ({len(mismatched)}):")
    for k, hf_s, cu_s in mismatched[:10]:
        print(f"    {k}: HF={hf_s} vs Custom={cu_s}")
    if len(mismatched) > 10:
        print(f"    ... +{len(mismatched)-10} more")
else:
    print("  All shapes match!")

# ── Step 6: Save ────────────────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(__file__), "mit_b3.pth")
# Add head weights (random init — will be popped by GFformer)
if "head.weight" not in new_sd and "head.weight" in custom_sd:
    new_sd["head.weight"] = custom_sd["head.weight"].clone()
if "head.bias" not in new_sd and "head.bias" in custom_sd:
    new_sd["head.bias"] = custom_sd["head.bias"].clone()

torch.save(new_sd, output_path)
print(f"\n{'='*60}")
print(f"Saved to: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")
print(f"Total keys: {len(new_sd)}")
