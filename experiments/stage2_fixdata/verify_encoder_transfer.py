#!/usr/bin/env python3
"""
Gate 2: verify the Stage-1 -> Stage-2 backbone parameter transfer.

Uses the SAME function that train_segformer_cls.py calls at startup
(ckpt_transfer.transfer_stage1_weights), then checks:

  * Stage-1 encoder tensor count vs Stage-2 backbone tensor count
  * matched count and coverage (requirement: > 95%)
  * missing / shape-mismatch lists
  * fixed spot-checks: max_abs_diff < 1e-7 on >=3 corresponding parameters

Exit code 0 = PASS, 1 = FAIL.  Stage 2 must NOT be trained on a FAIL.
"""

import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'model'))

from paths import STAGE1_LOC_CKPT
from ckpt_transfer import transfer_stage1_weights, build_transfer_map
from gfmodel import GFformer_two

# Fixed spot-check pairs (stage2 name -> stage1 name).
SPOT_CHECKS = [
    ('rgb_net.patch_embed1.proj.weight', 'encoder.patch_embed1.proj.weight'),
    ('rgb_net.block1.0.norm1.weight', 'encoder.block1.0.norm1.weight'),
    ('rgb_net.block3.0.attn.q.weight', 'encoder.block3.0.attn.q.weight'),
    ('rgb_net.block4.2.mlp.fc2.weight', 'encoder.block4.2.mlp.fc2.weight'),
    ('rgb_net.norm4.weight', 'encoder.norm4.weight'),
]

MAX_DIFF = 1e-7


def main():
    print('=' * 76)
    print(' GATE 2: Stage-1 encoder -> Stage-2 backbone transfer')
    print('=' * 76)
    print(f"  checkpoint: {STAGE1_LOC_CKPT}")

    ck = torch.load(STAGE1_LOC_CKPT, map_location='cpu')
    stage1_sd = ck['state_dict']

    # Production path: GFformer_two() first loads ImageNet mit_b3.pth, then
    # we overwrite the backbone from the frozen Stage-1 checkpoint.
    model = GFformer_two().cuda()
    sd = model.state_dict()

    # ── What same-name matching alone would transfer (the OLD logic) ──
    same_name_matched_backbone = sum(
        1 for k in sd
        if (k.startswith('rgb_net.') or k.startswith('post_net.'))
        and k in stage1_sd and sd[k].size() == stage1_sd[k].size())

    # ── Explicit transfer (same call train_segformer_cls.py makes) ──
    report = transfer_stage1_weights(model, STAGE1_LOC_CKPT, verbose=False)

    print('-' * 76)
    print(' Transfer report')
    print('-' * 76)
    print(f"  Stage-1 encoder tensors:   {report['n_stage1_encoder']}")
    print(f"  Stage-2 backbone tensors:  {report['n_stage2_backbone']}")
    print(f"  matched (total):           {report['matched']}")
    print(f"  matched (backbone):        {report['matched_backbone']}")
    print(f"  coverage (backbone):       {report['coverage_backbone']:.4f}  "
          f"(requirement > 0.95)")
    print(f"  shape mismatches:          {len(report['shape_mismatch'])}")
    for s in report['shape_mismatch'][:5]:
        print(f"      {s}")
    print(f"  missing (unmatched k2->k1):{len(report['missing'])}")
    for m in report['missing'][:5]:
        print(f"      {m}")
    print(f"  [diagnostic] same-name-only backbone matches (old logic): "
          f"{same_name_matched_backbone}")

    # ── Spot checks: max_abs_diff on fixed corresponding params ──
    print('-' * 76)
    print(' Spot checks (max_abs_diff < 1e-7 required)')
    print('-' * 76)
    all_ok = True
    for k2, k1 in SPOT_CHECKS:
        if k2 not in sd or k1 not in stage1_sd:
            print(f"  {k2}: MISSING KEY -> FAIL")
            all_ok = False
            continue
        d = (sd[k2] - stage1_sd[k1].to(sd[k2].device)).abs().max().item()
        ok = d < MAX_DIFF
        all_ok &= ok
        print(f"  {k2} vs {k1}: max_abs_diff = {d:.3e}  "
              f"{'OK' if ok else 'FAIL'}")

    # post_net alias shares storage with rgb_net (same module instance)
    shared = (model.rgb_net.patch_embed1.proj.weight.data_ptr()
              == model.post_net.patch_embed1.proj.weight.data_ptr())
    print(f"  rgb_net/post_net share the same backbone storage: {shared}")
    all_ok &= shared

    coverage_ok = report['coverage_backbone'] > 0.95
    all_ok &= coverage_ok

    print('=' * 76)
    print(f" GATE 2: {'PASS' if all_ok else 'FAIL'}")
    print('=' * 76)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
