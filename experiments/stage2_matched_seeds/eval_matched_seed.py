#!/usr/bin/env python3
"""Independently re-evaluate one completed matched-seed best checkpoint."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_segformer_cls as T
from experiments.stage2_matched_seeds import train_matched_seed as M
from experiments.stage2_fixdata.eval_repo_bs4 import METRIC_PATTERN, sha256_file

METRICS = ('F1b', 'F1d', 'F1s', 'F1_0', 'F1_1', 'F1_2', 'F1_3')


def main():
    args = M.parse_args()
    _, freeze = M.check_plan(args)
    out = M.run_dir(args.variant, args.seed)
    ckpt_path = out / 'ckpt' / f'GFformer_cls_3_{args.variant}_ddp2_seed{args.seed}_best14'
    history_path = out / 'results' / 'validation_history.csv'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for production validation')
    torch.cuda.set_device(0)
    T.cudnn.benchmark = True
    _, val_idxs = T.train_test_split(
        np.arange(len(T.all_files)), test_size=0.1, random_state=3)
    split_sha = M.audit_fixed_validation(val_idxs, freeze)
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    checkpoint_sha = sha256_file(ckpt_path)
    model = T.GFformer_two(
        use_kalman=args.variant in ('k2', 'k2b'),
        kalman_mode='post_gf' if args.variant == 'k2b' else 'change'
    ).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    loader = DataLoader(T.ValData(val_idxs), batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)
    captured = []
    original_dprint = T.dprint
    def capture_print(*items, **kwargs):
        line = ' '.join(str(item) for item in items)
        if line.startswith('Val Score:'):
            captured.append(line)
        original_dprint(*items, **kwargs)
    T.dprint = capture_print
    try:
        score = T.validate(model, loader)
    finally:
        T.dprint = original_dprint
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one validation metric line')
    match = METRIC_PATTERN.fullmatch(captured[0])
    if match is None:
        raise RuntimeError(f'Unexpected metric line: {captured[0]}')
    metrics = {key: float(value) for key, value in match.groupdict().items()}
    if abs(score - float(checkpoint['best_score'])) > 0.00005:
        raise RuntimeError('Independent F1s disagrees with checkpoint')
    with history_path.open(newline='', encoding='utf-8') as handle:
        history = list(csv.DictReader(handle))
    if len(history) != 25:
        raise RuntimeError(f'Expected 25 validation rows, got {len(history)}')
    best = max(history, key=lambda row: float(row['F1s']))
    if int(best['checkpoint_epoch_1based']) != int(checkpoint['epoch']):
        raise RuntimeError('Best epoch disagrees with history')
    for key in METRICS:
        if abs(metrics[key] - float(best[key])) > 0.00005:
            raise RuntimeError(f'{key} differs from best history row')
    result = {
        'variant': args.variant,
        'train_seed': args.seed,
        'mode': 'single',
        'views': ['original'],
        'checkpoint': str(ckpt_path.relative_to(ROOT)),
        'checkpoint_sha256': checkpoint_sha,
        'checkpoint_epoch_1based': int(checkpoint['epoch']),
        'checkpoint_best_score': float(checkpoint['best_score']),
        'F1s_unrounded': float(score),
        'metrics': metrics,
        'validation_split_seed': 3,
        'validation_images': len(val_idxs),
        'validation_split_sha256': split_sha,
        'localization_masks': freeze['localization_masks'],
        'validation_mask_manifest_sha256': freeze['validation_mask_manifest_sha256'],
        'metric_code': 'train_segformer_cls.validate',
        'metric_code_sha256': freeze['metric_code_sha256'],
        'eval_git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
    }
    destination = out / 'results' / 'single_view.json'
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Result saved: {destination}')
    print(f'Seed {args.seed} {args.variant}: {metrics}')


if __name__ == '__main__':
    main()
