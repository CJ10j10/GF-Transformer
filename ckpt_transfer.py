"""Explicit Stage-1 -> Stage-2 parameter transfer (Gate 2).

Key fact: GFformer_one saves its backbone under 'encoder.*' (mix_transformer
mit_b3), while GFformer_two exposes the SAME backbone under 'rgb_net.*' (and
shares it with 'post_net.*' — one module registered twice).  Same-name
matching therefore transfers NOTHING from the backbone and silently falls
back to the ImageNet-pretrained mit_b3 — which is what the old Stage-2 run
did.  Never rely on same-name matching: use build_transfer_map() below.

train_segformer_cls.py and experiments/stage2_fixdata/verify_encoder_transfer.py
both call these functions, so the verified logic IS the production logic.
"""

import torch

# Names under which GFformer_two exposes the Stage-1 backbone.
BACKBONE_ALIASES = ('rgb_net.', 'post_net.')

# Minimum acceptable backbone coverage (Gate 2 requirement: > 95%).
MIN_COVERAGE = 0.95


def build_transfer_map(stage1_sd, stage2_sd):
    """Build the explicit {stage2_key: stage1_tensor} transfer map.

    Backbone: 'encoder.<rest>' -> 'rgb_net.<rest>' / 'post_net.<rest>'.
    All other keys fall back to same-name matching (Stage-1 decoder conv
    blocks re-used as Stage-2 decoder init), same-shape only.

    Returns (mapped, report).  report contains:
      n_stage1_encoder, n_stage2_backbone, matched, matched_backbone,
      coverage_backbone, missing [(k2, k1)], shape_mismatch [(k2, k1, s2, s1)]
    """
    mapped = {}
    report = {
        'n_stage1_encoder': 0, 'n_stage2_backbone': 0,
        'matched': 0, 'matched_backbone': 0, 'coverage_backbone': 0.0,
        'missing': [], 'shape_mismatch': [],
    }

    report['n_stage1_encoder'] = sum(
        1 for k in stage1_sd if k.startswith('encoder.'))
    report['n_stage2_backbone'] = sum(
        1 for k in stage2_sd if k.startswith(BACKBONE_ALIASES))

    for k2 in stage2_sd:
        k1 = None
        for prefix in BACKBONE_ALIASES:
            if k2.startswith(prefix):
                k1 = 'encoder.' + k2[len(prefix):]
                break
        if k1 is None:
            k1 = k2  # non-backbone: same-name fallback

        if k1 in stage1_sd:
            if tuple(stage2_sd[k2].size()) == tuple(stage1_sd[k1].size()):
                mapped[k2] = stage1_sd[k1]
                report['matched'] += 1
                if k2.startswith(BACKBONE_ALIASES):
                    report['matched_backbone'] += 1
            else:
                report['shape_mismatch'].append(
                    (k2, k1, tuple(stage2_sd[k2].size()),
                     tuple(stage1_sd[k1].size())))
        else:
            report['missing'].append((k2, k1))

    report['coverage_backbone'] = (
        report['matched_backbone'] / report['n_stage2_backbone']
        if report['n_stage2_backbone'] else 0.0)
    return mapped, report


def transfer_stage1_weights(model, ckpt_path, verbose=True):
    """Load the frozen Stage-1 checkpoint into `model` (GFformer_two) using
    the explicit mapping.  Raises RuntimeError if backbone coverage < 95%.
    Returns the transfer report."""
    ck = torch.load(ckpt_path, map_location='cpu')
    stage1_sd = ck['state_dict']
    sd = model.state_dict()

    mapped, report = build_transfer_map(stage1_sd, sd)
    for k2, t in mapped.items():
        sd[k2] = t
    model.load_state_dict(sd)

    report['stage1_epoch'] = ck.get('epoch')
    report['stage1_best_score'] = ck.get('best_score')
    report['ckpt_path'] = ckpt_path

    if report['coverage_backbone'] < MIN_COVERAGE:
        raise RuntimeError(
            f"Stage-1 backbone transfer coverage "
            f"{report['coverage_backbone']:.4f} < {MIN_COVERAGE} (Gate 2) — "
            f"refusing to continue. See "
            f"experiments/stage2_fixdata/verify_encoder_transfer.py")

    if verbose:
        print(f"  transfer: {report['matched']} tensors mapped "
              f"(backbone {report['matched_backbone']}/"
              f"{report['n_stage2_backbone']}, coverage "
              f"{report['coverage_backbone']:.4f}), "
              f"stage1 epoch {report['stage1_epoch']}, "
              f"best_score {report['stage1_best_score']:.4f}")
        if report['shape_mismatch']:
            print(f"  [shape mismatch (skipped)] {report['shape_mismatch'][:5]}")
    return report
