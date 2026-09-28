#!/usr/bin/env python3
"""Paired image bootstrap of the unchanged global TP/FP/FN validation metric."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_segformer_cls as T
from experiments.stage2_fixdata.eval_repo_bs4 import sha256_file

B0_CKPT = ROOT / 'experiments/stage2_b0_ddp2/ckpt/GFformer_cls_3_b0_ddp2_best14'
K2_CKPT = ROOT / 'experiments/stage2_k2_kalman_refine/ckpt_ddp2/GFformer_cls_3_k2_ddp2_best14'
B0_RESULT = ROOT / 'experiments/stage2_b0_ddp2/results/ddp2_single_view.json'
K2_RESULT = ROOT / 'experiments/stage2_k2_kalman_refine/results/ddp2_single_view.json'
OUT_DIR = ROOT / 'experiments/stage2_b0_ddp2/results'
N_BOOTSTRAP = 10000
SEED = 20260928
METRICS = ('F1b', 'F1d', 'F1s', 'F1_0', 'F1_1', 'F1_2', 'F1_3')


def metrics_from_counts(counts):
    """Exact arithmetic of train_segformer_cls.validate, after count summation."""
    tp = counts[..., 0].astype(np.float64)
    fp = counts[..., 1].astype(np.float64)
    fn = counts[..., 2].astype(np.float64)
    f1b = 2 * tp[..., 4] / (2 * tp[..., 4] + fp[..., 4] + fn[..., 4] + 1e-8)
    class_f1 = 2 * tp[..., :4] / (2 * tp[..., :4] + fp[..., :4] + fn[..., :4] + 1e-8)
    f1d = 4 / np.sum(1.0 / (class_f1 + 1e-6), axis=-1)
    result = {'F1b': f1b, 'F1d': f1d, 'F1s': 0.3 * f1b + 0.7 * f1d}
    result.update({f'F1_{c}': class_f1[..., c] for c in range(4)})
    return result


def count_one(sample, model):
    """The same masks, softmax/argmax, gating, and class count rules as validate."""
    msks = sample['msk'].numpy()[0]
    lbl_msk = sample['lbl_msk'].numpy()[0]
    msk_loc = sample['msk_loc'].numpy()[0].astype(bool)
    imgs = sample['img'].cuda(non_blocking=True)
    with torch.no_grad():
        out = model(imgs)
        damage_probs = torch.softmax(out, dim=1).cpu().numpy()[0, 1:, ...]
    counts = np.zeros((5, 3), dtype=np.int64)  # class, [TP, FP, FN]
    gt_bld = msks[0] > 0
    counts[4, 0] = np.count_nonzero(gt_bld & msk_loc)
    counts[4, 1] = np.count_nonzero(gt_bld & ~msk_loc)
    counts[4, 2] = np.count_nonzero(~gt_bld & msk_loc)
    targ = lbl_msk[gt_bld]
    pred = damage_probs.argmax(axis=0)
    pred = pred * (msk_loc > 0.4)
    pred = pred[gt_bld]
    for c in range(4):
        counts[c, 0] = np.count_nonzero((pred == c) & (targ == c))
        counts[c, 1] = np.count_nonzero((pred == c) & (targ != c))
        counts[c, 2] = np.count_nonzero((pred != c) & (targ == c))
    return counts


def collect(ckpt_path, use_kalman, val_idxs, val_files, expected):
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    model = T.GFformer_two(use_kalman=use_kalman).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    loader = DataLoader(T.ValData(val_idxs), batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)
    rows = []
    for i, sample in enumerate(T.tqdm(loader, desc='K2' if use_kalman else 'B0')):
        if sample['fn'][0] != val_files[i]:
            raise RuntimeError('Validation image order changed')
        rows.append(count_one(sample, model))
    counts = np.stack(rows)
    observed = metrics_from_counts(counts.sum(axis=0))
    for key in METRICS:
        if abs(float(observed[key]) - expected['metrics'][key]) > 0.00005:
            raise RuntimeError(f'{ckpt_path.name} {key}: counts disagree with exact validate')
    if abs(float(observed['F1s']) - float(checkpoint['best_score'])) > 0.00005:
        raise RuntimeError('Counts disagree with checkpoint score')
    del loader, model, checkpoint
    torch.cuda.empty_cache()
    return counts, observed


def main():
    torch.cuda.set_device(0)
    T.cudnn.benchmark = True
    b0_result = json.loads(B0_RESULT.read_text())
    k2_result = json.loads(K2_RESULT.read_text())
    _, val_idxs = T.train_test_split(
        np.arange(len(T.all_files)), test_size=0.1, random_state=3)
    val_files = [T.all_files[i] for i in val_idxs]
    split_sha = hashlib.sha256('\n'.join(val_files).encode()).hexdigest()
    assert len(val_files) == 917
    assert split_sha == b0_result['validation_split_sha256'] == k2_result['validation_split_sha256']
    assert sha256_file(ROOT / 'train_segformer_cls.py') == b0_result['metric_code_sha256'] == k2_result['metric_code_sha256']
    assert sha256_file(B0_CKPT) == b0_result['checkpoint_sha256']
    assert sha256_file(K2_CKPT) == k2_result['checkpoint_sha256']
    assert b0_result['localization_masks'] == k2_result['localization_masks']
    b0_counts, b0_observed = collect(B0_CKPT, False, val_idxs, val_files, b0_result)
    k2_counts, k2_observed = collect(K2_CKPT, True, val_idxs, val_files, k2_result)
    assert np.array_equal(b0_counts[:, 4], k2_counts[:, 4])

    # A replicate samples identical image indices for both models, sums its
    # sufficient statistics, then computes global metrics. Per-image F1 is
    # never averaged.
    rng = np.random.default_rng(SEED)
    draws = {key: np.empty(N_BOOTSTRAP, dtype=np.float64) for key in METRICS}
    chunk = 200
    for start in range(0, N_BOOTSTRAP, chunk):
        end = min(start + chunk, N_BOOTSTRAP)
        index = rng.integers(0, len(val_files), size=(end - start, len(val_files)))
        b0_metric = metrics_from_counts(b0_counts[index].sum(axis=1))
        k2_metric = metrics_from_counts(k2_counts[index].sum(axis=1))
        for key in METRICS:
            draws[key][start:end] = b0_metric[key] - k2_metric[key]
    summary = {}
    for key in METRICS:
        values = draws[key]
        lo, hi = np.quantile(values, [0.025, 0.975])
        summary[key] = {
            'observed_delta_b0_minus_k2': float(b0_observed[key] - k2_observed[key]),
            'bootstrap_percentile_95_ci': [float(lo), float(hi)],
            'bootstrap_standard_error': float(np.std(values, ddof=1)),
            'fraction_delta_positive': float(np.mean(values > 0)),
        }
    result = {
        'method': 'paired image resampling with replacement; sum TP/FP/FN then compute global F1b, harmonic F1d, and F1s',
        'resamples': N_BOOTSTRAP,
        'seed': SEED,
        'validation_images': len(val_files),
        'validation_split_sha256': split_sha,
        'metric_code_sha256': b0_result['metric_code_sha256'],
        'b0_checkpoint_sha256': b0_result['checkpoint_sha256'],
        'k2_checkpoint_sha256': k2_result['checkpoint_sha256'],
        'damage_classes': ['no_damage', 'minor_damage', 'major_damage', 'destroyed'],
        'results': summary,
        'scope': 'image sampling uncertainty on this fixed validation set and selected checkpoints; does not include training-seed or checkpoint-selection uncertainty',
    }
    counts_path = OUT_DIR / 'paired_per_image_sufficient_statistics.npz'
    np.savez_compressed(counts_path, b0_tp_fp_fn=b0_counts, k2_tp_fp_fn=k2_counts,
                        validation_files=np.asarray(val_files))
    summary_path = OUT_DIR / 'paired_bootstrap.json'
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Counts: {counts_path} ({counts_path.stat().st_size} bytes)')
    print(f'Summary: {summary_path}')
    for key in ('F1s', 'F1d', 'F1_1'):
        print(key, summary[key])


if __name__ == '__main__':
    main()
