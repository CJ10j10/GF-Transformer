#!/usr/bin/env python3
"""24 real DDP2 K2 updates plus one rank-0 validation batch; no checkpoint."""

import json
import math
import os
import random
import statistics
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import train_segformer_cls as T
from experiments.stage2_k2_kalman_refine import train_k2_ddp2 as P

N_UPDATES = 24
AUDIT_JSON = os.path.join(P.EXP_DIR, 'audit', 'ddp2_preflight.json')
SINGLE_AUDIT = os.path.join(P.EXP_DIR, 'audit', 'preflight.json')


def finite(tensors):
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def first_global_indices(train_idxs, count):
    """The first `count` entries of Baseline-B's existing oversampled order."""
    assert len(train_idxs) >= count
    return np.asarray(train_idxs[:count])


def main():
    P.assert_protocol()
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    assert local_rank in (0, 1) and torch.cuda.device_count() == 2
    torch.cuda.set_device(local_rank)
    T.dist.init_process_group(backend='nccl')
    rank = T.dist.get_rank()
    try:
        assert T.dist.get_world_size() == 2
        assert not os.listdir(P.CKPT_DIR), 'DDP2 checkpoint directory must be empty'
        with open(SINGLE_AUDIT, encoding='utf-8') as f:
            previous = json.load(f)
        assert previous['ready_for_training'] and previous['max_abs_diff'] < 1e-6
        assert previous['base_initialization_equal']
        if rank == 0:
            with open(AUDIT_JSON, 'w', encoding='utf-8') as f:
                json.dump({'ready_for_training': False, 'world_size': 2}, f, indent=2)
                f.write('\n')
        T.dist.barrier()

        np.random.seed(P.SEED + rank)
        random.seed(P.SEED + rank)
        torch.manual_seed(P.SEED + rank)
        T.cudnn.benchmark = True
        train_idxs, val_idxs0 = P.build_split()
        assert len(val_idxs0) == 917
        for i in val_idxs0:
            basename = os.path.basename(T.all_files[i]).replace('.png', '_part1.png')
            assert os.path.isfile(os.path.join(T.LOC_FOLDER, basename))
        train_data = T.TrainData(first_global_indices(
            train_idxs, N_UPDATES * P.PER_GPU_BATCH * P.WORLD_SIZE))
        sampler = T.DistributedSampler(
            train_data, num_replicas=2, rank=rank, shuffle=True,
            seed=P.SEED, drop_last=True)
        sampler.set_epoch(0)
        local_indices = list(iter(sampler))
        gathered_indices = [None, None]
        T.dist.all_gather_object(gathered_indices, local_indices)
        assert len(local_indices) == N_UPDATES * P.PER_GPU_BATCH
        assert not (set(gathered_indices[0]) & set(gathered_indices[1]))
        assert len(set(gathered_indices[0] + gathered_indices[1])) == 96
        loader = T.DataLoader(train_data, batch_size=P.PER_GPU_BATCH,
                              sampler=sampler, num_workers=4,
                              pin_memory=True, drop_last=True)
        assert len(loader) == N_UPDATES

        model = T.GFformer_two(use_kalman=True).cuda(local_rank)
        transfer = T.transfer_stage1_weights(
            model, T.STAGE1_LOC_CKPT, verbose=False)
        assert transfer['coverage_backbone'] == 1.0
        backbone = [p for n, p in model.named_parameters()
                    if n.startswith(('rgb_net.', 'post_net.'))]
        assert backbone and all(p.requires_grad for p in backbone)
        assert all(float(getattr(model, f'kalman{i}').gamma) == 0
                   for i in (1, 2, 3))

        checked_steps = [0]
        step_times = []
        gradients = []
        step_start = [time.perf_counter()]

        class CheckedAdamW(T.AdamW):
            def step(self, *args, **kwargs):
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and finite(grads), f'rank {rank}: nonfinite gradient'
                if checked_steps[0] < 3:
                    entry = {'step': checked_steps[0] + 1, 'branches': {}}
                    for i in (1, 2, 3):
                        layer = getattr(model, f'kalman{i}')
                        branch = {}
                        for name, p in (('gamma', layer.gamma),
                                        ('obs', layer.obs_proj.weight),
                                        ('P', layer.p_net.weight),
                                        ('R', layer.r_net.weight)):
                            assert p.grad is not None and torch.isfinite(p.grad).all()
                            branch[name] = p.grad.detach().abs().max().item()
                        entry['branches'][f'K{i}'] = branch
                    gradients.append(entry)
                result = super().step(*args, **kwargs)
                checked_steps[0] += 1
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - step_start[0])
                step_start[0] = time.perf_counter()
                return result

        optimizer = CheckedAdamW(model.parameters(), lr=T.LR,
                                  weight_decay=T.WEIGHT_DECAY)
        assert len(optimizer.param_groups) == 1
        opt_ids = {id(p) for group in optimizer.param_groups
                   for p in group['params']}
        assert all(id(p) in opt_ids for p in model.parameters())
        ddp_model = T.DDP(model, device_ids=[local_rank],
                          find_unused_parameters=True)
        scheduler = T.lr_scheduler.MultiStepLR(
            optimizer, milestones=T.MILESTONES, gamma=T.GAMMA)
        seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0},
                                per_image=False).cuda()
        ce_loss = torch.nn.CrossEntropyLoss().cuda()
        T.PHYSICAL_BATCH = P.PER_GPU_BATCH
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss, cce, updates = T.train_epoch(
            0, seg_loss, ce_loss, ddp_model, optimizer, scheduler,
            loader, sampler, grad_accum=P.GRAD_ACCUM, world_size=2)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        train_allocated = torch.cuda.max_memory_allocated() / 2**30
        train_reserved = torch.cuda.max_memory_reserved() / 2**30
        assert updates == checked_steps[0] == N_UPDATES
        assert scheduler.last_epoch == 1
        assert math.isfinite(loss) and math.isfinite(cce)
        assert finite(model.parameters()), f'rank {rank}: nonfinite parameter'
        gamma_after = {f'K{i}': float(getattr(model, f'kalman{i}').gamma.detach())
                       for i in (1, 2, 3)}
        assert all(v != 0 for v in gamma_after.values())
        for i in (1, 2, 3):
            for branch in ('obs', 'P', 'R'):
                assert any(gradients[j]['branches'][f'K{i}'][branch] > 0
                           for j in (1, 2)), (rank, i, branch)
        local_report = {
            'rank': rank, 'gpu': torch.cuda.get_device_name(local_rank),
            'updates': updates, 'loss': loss, 'cross_entropy': cce,
            'train_elapsed_seconds': elapsed,
            'train_peak_allocated_gib': train_allocated,
            'train_peak_reserved_gib': train_reserved,
            'iteration_time_mean_seconds': statistics.mean(step_times),
            'iteration_time_median_steady_seconds': statistics.median(step_times[3:]),
            'gamma_after_smoke': gamma_after,
            'first_three_gradient_max_abs': gradients,
            'backbone_trainable_tensors': len(backbone),
            'stage1_encoder_coverage': transfer['coverage_backbone'],
        }
        ranks = [None, None]
        T.dist.all_gather_object(ranks, local_report)

        # Rank 1 waits while rank 0 runs a real 1024x1024 ValData batch via
        # model.module. No DDP forward or write occurs during validation.
        validation_score = None
        validation_seconds = None
        if rank == 0:
            val_loader = T.DataLoader(
                T.ValData(np.asarray([val_idxs0[0]])), batch_size=T.VAL_BATCH,
                shuffle=False, num_workers=0, pin_memory=True)
            ddp_model.eval()
            v0 = time.perf_counter()
            validation_score = T.validate(ddp_model.module, val_loader)
            torch.cuda.synchronize()
            validation_seconds = time.perf_counter() - v0
            assert math.isfinite(validation_score)
        T.dist.barrier()
        assert not os.listdir(P.CKPT_DIR), 'DDP2 smoke wrote a formal checkpoint'
        if rank == 0:
            report = {
                'ready_for_training': True,
                'world_size': 2,
                'per_gpu_batch': P.PER_GPU_BATCH,
                'grad_accum': P.GRAD_ACCUM,
                'global_batch': P.PER_GPU_BATCH * P.WORLD_SIZE * P.GRAD_ACCUM,
                'train_images_oversampled': len(train_idxs),
                'full_epoch_updates_per_rank': 3351,
                'fixed_validation_images': len(val_idxs0),
                'validation_protocol': 'single-view, rank0 only, batch1',
                'validation_sample_score': validation_score,
                'validation_sample_seconds': validation_seconds,
                'baseline_equivalence_max_abs_diff': previous['max_abs_diff'],
                'base_initialization_equal': previous['base_initialization_equal'],
                'single_gpu_reference_audit': SINGLE_AUDIT,
                'stage1_checkpoint': T.STAGE1_LOC_CKPT,
                'localization_masks': T.LOC_FOLDER,
                'checkpoint_dir': P.CKPT_DIR,
                'formal_epochs': T.TOTAL_EPOCHS,
                'lr': T.LR,
                'weight_decay': T.WEIGHT_DECAY,
                'scheduler_milestones': T.MILESTONES,
                'scheduler_gamma': T.GAMMA,
                'crop': list(T.INPUT_SHAPE),
                'amp': T.AMP_ENABLED,
                'ranks': ranks,
            }
            with open(AUDIT_JSON, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
                f.write('\n')
            print(f'K2 DDP2 PREFLIGHT PASS: 2x{N_UPDATES} local steps, '
                  f'global_batch=4; peaks allocated='
                  f'{ranks[0]["train_peak_allocated_gib"]:.2f}/'
                  f'{ranks[1]["train_peak_allocated_gib"]:.2f} GiB; '
                  f'elapsed={max(r["train_elapsed_seconds"] for r in ranks):.1f}s; '
                  f'rank0 validation={validation_seconds:.1f}s', flush=True)
    finally:
        T.dist.destroy_process_group()


if __name__ == '__main__':
    main()
