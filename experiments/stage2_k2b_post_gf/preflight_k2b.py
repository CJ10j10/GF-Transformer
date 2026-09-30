#!/usr/bin/env python3
"""K2b seed-3 DDP2 preflight: formula, gain, gradients, and 20 real updates."""

import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_segformer_cls as T
from experiments.stage2_b0_ddp2 import train_b0_ddp2 as P
from experiments.stage2_matched_seeds import train_matched_seed as M

SEED = 3
N_UPDATES = 20
B0_CKPT = ROOT / 'experiments/stage2_b0_ddp2/ckpt/GFformer_cls_3_b0_ddp2_best14'


def finite(items):
    return all(bool(torch.isfinite(value).all()) for value in items)


def initialization_check(local_rank):
    checkpoint = torch.load(B0_CKPT, map_location='cpu')
    b0 = T.GFformer_two().cuda(local_rank).eval()
    b0.load_state_dict(checkpoint['state_dict'], strict=True)
    k2b = T.GFformer_two(use_kalman=True, kalman_mode='post_gf').cuda(local_rank).eval()
    result = k2b.load_state_dict(checkpoint['state_dict'], strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(k.startswith('kalman') for k in result.missing_keys)
    assert all(torch.equal(value, k2b.state_dict()[name])
               for name, value in b0.state_dict().items())
    for i in (1, 2, 3):
        getattr(k2b, f'kalman{i}').record_k_stats = True
    torch.manual_seed(753 + local_rank)
    image = torch.randn(1, 6, 512, 512, device=f"cuda:{local_rank}")
    with torch.no_grad():
        baseline = b0(image)
        refined = k2b(image)
    diff = float((baseline - refined).abs().max())
    assert math.isfinite(diff) and diff > 0
    stats = {}
    for i in (1, 2, 3):
        entry = getattr(k2b, f'kalman{i}').last_k_stats
        assert entry is not None and all(math.isfinite(v) for v in entry.values())
        assert 0 <= entry['min'] <= entry['max'] <= 1
        stats[f'K{i}'] = entry
    counts = {'b0': sum(p.numel() for p in b0.parameters()),
              'k2b': sum(p.numel() for p in k2b.parameters())}
    del b0, k2b, image, baseline, refined, checkpoint
    torch.cuda.empty_cache()
    return diff, stats, counts


def main():
    args = M.parse_args()
    assert args.variant == 'k2b' and args.seed == SEED
    _, freeze = M.check_plan(args)
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank not in (0, 1) or torch.cuda.device_count() != 2:
        raise RuntimeError('K2b preflight requires exactly two visible GPUs')
    torch.cuda.set_device(local_rank)
    T.dist.init_process_group('nccl')
    rank = T.dist.get_rank()
    out = M.run_dir('k2b', SEED)
    audit_path = out / 'audit/preflight.json'
    ckpt_dir = out / 'ckpt'
    history = out / 'results/validation_history.csv'
    try:
        assert T.dist.get_world_size() == 2
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        history.parent.mkdir(parents=True, exist_ok=True)
        assert not any(ckpt_dir.iterdir()) and not history.exists()
        if rank == 0:
            audit_path.write_text(json.dumps({'ready_for_training': False}) + '\n')
        T.dist.barrier()
        diff, stats, counts = initialization_check(local_rank)
        np.random.seed(SEED + rank)
        random.seed(SEED + rank)
        torch.manual_seed(SEED + rank)
        T.cudnn.benchmark = True
        train_idxs, val_idxs = P.build_split()
        split_sha = M.audit_fixed_validation(val_idxs, freeze)
        data = T.TrainData(np.asarray(train_idxs[:N_UPDATES * 4]))
        sampler = T.DistributedSampler(data, num_replicas=2, rank=rank,
                                       shuffle=True, seed=SEED, drop_last=True)
        loader = T.DataLoader(data, batch_size=2, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
        assert len(loader) == N_UPDATES
        model = T.GFformer_two(use_kalman=True, kalman_mode='post_gf').cuda(local_rank)
        transfer = T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT, verbose=False)
        assert transfer['coverage_backbone'] == 1.0
        backbone = [p for name, p in model.named_parameters()
                    if name.startswith(('rgb_net.', 'post_net.'))]
        assert backbone and all(p.requires_grad for p in backbone)
        checked_steps = [0]
        gradients = []
        step_times = []
        step_start = [time.perf_counter()]

        class CheckedAdamW(T.AdamW):
            def step(self, *items, **kwargs):
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and finite(grads), f'rank {rank}: nonfinite gradient'
                if checked_steps[0] < 3:
                    entry = {'step': checked_steps[0] + 1, 'branches': {}}
                    for i in (1, 2, 3):
                        layer = getattr(model, f'kalman{i}')
                        branch = {}
                        for name, param in (('obs', layer.obs_proj.weight),
                                            ('P', layer.p_net.weight),
                                            ('R', layer.r_net.weight)):
                            assert param.grad is not None and bool(torch.isfinite(param.grad).all())
                            branch[name] = float(param.grad.detach().abs().max())
                        entry['branches'][f'K{i}'] = branch
                    gradients.append(entry)
                result = super().step(*items, **kwargs)
                checked_steps[0] += 1
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - step_start[0])
                step_start[0] = time.perf_counter()
                return result

        optimizer = CheckedAdamW(model.parameters(), lr=T.LR,
                                  weight_decay=T.WEIGHT_DECAY)
        assert len(optimizer.param_groups) == 1
        assert {id(p) for p in model.parameters()} == {
            id(p) for p in optimizer.param_groups[0]['params']}
        ddp_model = T.DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        scheduler = T.lr_scheduler.MultiStepLR(
            optimizer, milestones=T.MILESTONES, gamma=T.GAMMA)
        seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
        ce_loss = torch.nn.CrossEntropyLoss().cuda()
        T.PHYSICAL_BATCH = 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss, cce, updates = T.train_epoch(
            0, seg_loss, ce_loss, ddp_model, optimizer, scheduler,
            loader, sampler, grad_accum=1, world_size=2)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert updates == checked_steps[0] == N_UPDATES
        assert scheduler.last_epoch == 1
        assert math.isfinite(loss) and math.isfinite(cce) and finite(model.parameters())
        for entry in gradients:
            for branch in entry['branches'].values():
                assert all(v > 0 for v in branch.values()), branch
        local = {
            'rank': rank,
            'gpu': torch.cuda.get_device_name(local_rank),
            'initial_logits_max_abs_diff_from_b0': diff,
            'kalman_gain': stats,
            'parameter_counts': counts,
            'updates': updates,
            'loss': loss,
            'cross_entropy': cce,
            'elapsed_seconds': elapsed,
            'median_steady_step_seconds': statistics.median(step_times[3:]),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'first_three_step_gradients': gradients,
            'backbone_trainable_tensors': len(backbone),
            'stage1_encoder_coverage': transfer['coverage_backbone'],
        }
        ranks = [None, None]
        T.dist.all_gather_object(ranks, local)
        validation_score = None
        if rank == 0:
            val_loader = T.DataLoader(T.ValData(np.asarray([val_idxs[0]])),
                                      batch_size=1, shuffle=False,
                                      num_workers=0, pin_memory=True)
            ddp_model.eval()
            validation_score = T.validate(ddp_model.module, val_loader)
            assert math.isfinite(validation_score)
        T.dist.barrier()
        assert not any(ckpt_dir.iterdir()) and not history.exists()
        if rank == 0:
            report = {
                'ready_for_training': True,
                'variant': 'k2b',
                'train_seed': SEED,
                'validation_split_seed': 3,
                'validation_split_sha256': split_sha,
                'validation_mask_manifest_sha256': freeze['validation_mask_manifest_sha256'],
                'metric_code_sha256': freeze['metric_code_sha256'],
                'validation_images': len(val_idxs),
                'validation_sample_score': validation_score,
                'world_size': 2,
                'per_gpu_batch': 2,
                'grad_accum': 1,
                'global_batch': 4,
                'updates_per_full_epoch': 3351,
                'formal_epochs': T.TOTAL_EPOCHS,
                'lr': T.LR,
                'weight_decay': T.WEIGHT_DECAY,
                'scheduler_milestones': T.MILESTONES,
                'scheduler_gamma': T.GAMMA,
                'crop': list(T.INPUT_SHAPE),
                'amp': T.AMP_ENABLED,
                'stage1_checkpoint': T.STAGE1_LOC_CKPT,
                'checkpoint_dir': str(ckpt_dir),
                'code_sha256': {
                    str(path.relative_to(ROOT)): M.sha256_file(path)
                    for path in (
                        ROOT / 'model/gfmodel.py',
                        ROOT / 'model/kalman_refine.py',
                        ROOT / 'experiments/stage2_matched_seeds/train_matched_seed.py',
                    )
                },
                'ranks': ranks,
            }
            audit_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'K2B PREFLIGHT PASS seed={SEED}; peak allocated '
                  f'{ranks[0]["peak_allocated_gib"]:.2f}/'
                  f'{ranks[1]["peak_allocated_gib"]:.2f} GiB', flush=True)
    finally:
        T.dist.destroy_process_group()


if __name__ == '__main__':
    main()
