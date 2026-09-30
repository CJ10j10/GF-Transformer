#!/usr/bin/env python3
"""Frozen-protocol 2x2 seed training; launch only after matching preflight."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import timeit
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_segformer_cls as T
from experiments.stage2_b0_ddp2 import train_b0_ddp2 as P
from experiments.stage2_fixdata.eval_repo_bs4 import METRIC_PATTERN, sha256_file

EXP = Path(__file__).resolve().parent
PLAN = EXP / 'seed_plan.json'
FROZEN = ROOT / 'experiments/stage2_b0_ddp2/results/freeze_manifest.json'
PER_GPU_BATCH = 2
WORLD_SIZE = 2
GRAD_ACCUM = 1
HISTORY_FIELDS = P.HISTORY_FIELDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('b0', 'k2', 'k2b'), required=True)
    parser.add_argument('--seed', type=int, choices=(3, 11, 23), required=True)
    return parser.parse_args()


def run_dir(variant, seed):
    if variant == 'k2b':
        return ROOT / 'experiments' / 'stage2_k2b_post_gf' / 'runs' / f'seed{seed}'
    return EXP / 'runs' / f'{variant}_seed{seed}'


def check_plan(args):
    P.assert_protocol()
    plan = json.loads(PLAN.read_text())
    freeze = json.loads(FROZEN.read_text())
    assert freeze['status'] == 'FROZEN'
    assert plan['train_seeds_for_all_variants'] == [3, 11, 23]
    if args.variant == 'k2b':
        assert args.variant in plan['future_variants_using_same_seed_set']
        assert args.seed in plan['train_seeds_for_all_variants']
    else:
        assert args.seed in plan['new_matched_seed_screening']
        assert args.variant in plan['variants_in_screening']
    assert plan['validation_split_seed'] == 3
    assert plan['validation_split_sha256'] == freeze['image_ids_sha256']
    assert sha256_file(ROOT / 'train_segformer_cls.py') == freeze['metric_code_sha256']
    assert T.LOC_FOLDER == str(ROOT / freeze['localization_masks'])
    return plan, freeze


def audit_fixed_validation(val_idxs, freeze):
    val_files = [T.all_files[i] for i in val_idxs]
    split_sha = hashlib.sha256('\n'.join(val_files).encode()).hexdigest()
    assert len(val_files) == freeze['image_count'] == 917
    assert split_sha == freeze['image_ids_sha256']
    masks = hashlib.sha256()
    for image in val_files:
        mask = Path(T.LOC_FOLDER) / (Path(image).stem + '_part1.png')
        if not mask.is_file():
            raise FileNotFoundError(mask)
        masks.update(str(mask.relative_to(ROOT)).encode() + b'\0')
        masks.update(sha256_file(mask).encode() + b'\n')
    assert masks.hexdigest() == freeze['validation_mask_manifest_sha256']
    return split_sha


def make_train_loader(train_idxs, rank, seed):
    data = T.TrainData(train_idxs)
    sampler = T.DistributedSampler(data, num_replicas=WORLD_SIZE, rank=rank,
                                    shuffle=True, seed=seed, drop_last=True)
    loader = T.DataLoader(data, batch_size=PER_GPU_BATCH, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)
    assert len(loader) == 3351
    return loader, sampler


def main():
    args = parse_args()
    plan, freeze = check_plan(args)
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank not in (0, 1) or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError('Matched-seed training requires exactly two visible GPUs')
    torch.cuda.set_device(local_rank)
    T.dist.init_process_group(backend='nccl')
    rank = T.dist.get_rank()
    out = run_dir(args.variant, args.seed)
    ckpt_dir = out / 'ckpt'
    history_csv = out / 'results' / 'validation_history.csv'
    try:
        assert T.dist.get_world_size() == WORLD_SIZE
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history_csv.parent.mkdir(parents=True, exist_ok=True)
        if any(ckpt_dir.iterdir()) or history_csv.exists():
            raise RuntimeError(f'Refusing resume or overwrite in {out}')
        preflight = json.loads((out / 'audit' / 'preflight.json').read_text())
        if not (preflight.get('ready_for_training') and
                preflight.get('variant') == args.variant and
                preflight.get('train_seed') == args.seed and
                preflight.get('world_size') == WORLD_SIZE and
                preflight.get('validation_split_sha256') == freeze['image_ids_sha256'] and
                preflight.get('validation_mask_manifest_sha256') == freeze['validation_mask_manifest_sha256']):
            raise RuntimeError('Matched-seed preflight does not match this run')
        t0 = timeit.default_timer()
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        T.cudnn.benchmark = True

        train_idxs, val_idxs = P.build_split()  # fixed split seed 3
        split_sha = audit_fixed_validation(val_idxs, freeze)
        train_loader, sampler = make_train_loader(train_idxs, rank, args.seed)
        val_loader = P.make_val_loader(val_idxs) if rank == 0 else None
        model = T.GFformer_two(
            use_kalman=args.variant in ('k2', 'k2b'),
            kalman_mode='post_gf' if args.variant == 'k2b' else 'change'
        ).cuda(local_rank)
        transfer = T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT,
                                              verbose=T.is_main())
        assert transfer['coverage_backbone'] == 1.0
        backbone = [p for name, p in model.named_parameters()
                    if name.startswith(('rgb_net.', 'post_net.'))]
        assert backbone and all(p.requires_grad for p in backbone)
        optimizer = T.AdamW(model.parameters(), lr=T.LR,
                             weight_decay=T.WEIGHT_DECAY)
        assert len(optimizer.param_groups) == 1
        assert {id(p) for p in model.parameters()} == {
            id(p) for p in optimizer.param_groups[0]['params']}
        model = T.DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        scheduler = T.lr_scheduler.MultiStepLR(
            optimizer, milestones=T.MILESTONES, gamma=T.GAMMA)
        seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0},
                                per_image=False).cuda()
        ce_loss = torch.nn.CrossEntropyLoss().cuda()
        T.PHYSICAL_BATCH = PER_GPU_BATCH
        T.dprint(f'[Matched-seed protocol] variant={args.variant} run_seed={args.seed} '
                 f'split_seed={plan["validation_split_seed"]} split_sha256={split_sha} '
                 f'2xRTX4090 batch_per_gpu=2 accum=1 global_batch=4 '
                 f'epochs={T.TOTAL_EPOCHS} lr={T.LR} wd={T.WEIGHT_DECAY} '
                 f'milestones={T.MILESTONES} gamma={T.GAMMA} '
                 f'crop={T.INPUT_SHAPE} FP32 val_batch={T.VAL_BATCH} '
                 f'updates_per_epoch={len(train_loader)} checkpoint_dir={ckpt_dir}')
        best_score = 0.0
        if rank == 0:
            with history_csv.open('w', newline='', encoding='utf-8') as handle:
                csv.DictWriter(handle, fieldnames=HISTORY_FIELDS,
                               lineterminator='\n').writeheader()
        for epoch in range(T.TOTAL_EPOCHS):
            loss, cce, updates = T.train_epoch(
                epoch, seg_loss, ce_loss, model, optimizer, scheduler,
                train_loader, sampler, grad_accum=GRAD_ACCUM,
                world_size=WORLD_SIZE)
            assert updates == len(train_loader) == 3351
            if epoch % 2 == 0 and rank == 0:
                torch.cuda.empty_cache()
                model.eval()
                captured = []
                original_dprint = T.dprint
                def capture_print(*items, **kwargs):
                    line = ' '.join(str(item) for item in items)
                    if line.startswith('Val Score:'):
                        captured.append(line)
                    original_dprint(*items, **kwargs)
                T.dprint = capture_print
                try:
                    score = T.validate(model.module, val_loader)
                finally:
                    T.dprint = original_dprint
                if len(captured) != 1:
                    raise RuntimeError('Expected exactly one validation metric line')
                match = METRIC_PATTERN.fullmatch(captured[0])
                if match is None or abs(float(match['F1s']) - score) > 0.000051:
                    raise RuntimeError(f'Invalid validation metric line: {captured[0]}')
                improved = score > best_score
                if improved:
                    torch.save({'epoch': epoch + 1,
                                'state_dict': model.module.state_dict(),
                                'best_score': score},
                               ckpt_dir / f'GFformer_cls_3_{args.variant}_ddp2_seed{args.seed}_best14')
                    best_score = score
                row = {
                    'epoch_0based': epoch,
                    'checkpoint_epoch_1based': epoch + 1,
                    **{key: float(value) for key, value in match.groupdict().items()},
                    'best_score': round(best_score, 4),
                    'improved': int(improved),
                }
                with history_csv.open('a', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS,
                                            lineterminator='\n')
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                T.dprint(f'score: {score:.4f}\tscore_best: {best_score:.4f}')
            T.barrier()
        T.dprint(f'[Matched-seed done] variant={args.variant} seed={args.seed} '
                 f'time_hours={(timeit.default_timer() - t0) / 3600:.2f}')
    finally:
        T.dist.destroy_process_group()


if __name__ == '__main__':
    main()
