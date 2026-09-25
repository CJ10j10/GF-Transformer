#!/usr/bin/env python3
"""24 real matched B0 DDP2 updates and one rank-0 validation batch; no checkpoint."""

import hashlib
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
from experiments.stage2_b0_ddp2 import train_b0_ddp2 as P

N_UPDATES = 24
AUDIT_JSON = os.path.join(P.EXP_DIR, 'audit', 'preflight.json')
BASELINE_CKPT = os.path.join(ROOT, 'experiments', 'stage2_fixdata',
                             'ckpt_repo_bs4', 'GFformer_cls_3_fixdata_best14')
BASELINE_RESULT = os.path.join(ROOT, 'experiments', 'stage2_fixdata',
                               'results_repo_bs4', 'single_view.json')


def finite(tensors):
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def check_baseline_compatibility(rank):
    """B0 has exactly Baseline-B's parameters and K2's base initialization."""
    torch.manual_seed(P.SEED + rank)
    model = T.GFformer_two(use_kalman=False)
    assert not any(name.startswith('kalman') for name in model.state_dict())
    assert not any(hasattr(model, f'kalman{i}') for i in (1, 2, 3))
    assert sum(p.numel() for p in model.parameters()) == 56517168
    base_state = model.state_dict()
    torch.manual_seed(P.SEED + rank)
    k2 = T.GFformer_two(use_kalman=True)
    k2_state = k2.state_dict()
    assert all(torch.equal(value, k2_state[name])
               for name, value in base_state.items())
    del k2, k2_state, base_state
    if rank == 0:
        # Strictly load the frozen single-card baseline checkpoint into the
        # baseline architecture. This is a compatibility check only; training
        # below begins from the same Stage1 transfer as K2-D2.
        checkpoint = torch.load(BASELINE_CKPT, map_location='cpu')
        probe = T.GFformer_two(use_kalman=False)
        probe.load_state_dict(checkpoint['state_dict'], strict=True)
        assert checkpoint['best_score'] > 0.74
        del probe, checkpoint
    return model


def main():
    P.assert_protocol()
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    assert local_rank in (0, 1) and torch.cuda.device_count() == 2
    torch.cuda.set_device(local_rank)
    T.dist.init_process_group(backend='nccl')
    rank = T.dist.get_rank()
    try:
        assert T.dist.get_world_size() == 2
        os.makedirs(P.CKPT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(AUDIT_JSON), exist_ok=True)
        assert not os.listdir(P.CKPT_DIR), 'B0-D2 checkpoint directory must be empty'
        assert not os.path.exists(P.HISTORY_CSV), 'Formal validation history already exists'
        if rank == 0:
            with open(AUDIT_JSON, 'w', encoding='utf-8') as handle:
                json.dump({'ready_for_training': False, 'world_size': 2}, handle,
                          indent=2)
                handle.write('\n')
        T.dist.barrier()

        np.random.seed(P.SEED + rank)
        random.seed(P.SEED + rank)
        torch.manual_seed(P.SEED + rank)
        T.cudnn.benchmark = True
        train_idxs, val_idxs0 = P.build_split()
        with open(BASELINE_RESULT, encoding='utf-8') as handle:
            frozen = json.load(handle)
        val_files = [T.all_files[i] for i in val_idxs0]
        split_sha = hashlib.sha256('\n'.join(val_files).encode()).hexdigest()
        assert split_sha == frozen['validation_split_sha256']
        assert len(val_idxs0) == frozen['validation_images'] == 917
        for i in val_idxs0:
            basename = os.path.basename(T.all_files[i]).replace('.png', '_part1.png')
            assert os.path.isfile(os.path.join(T.LOC_FOLDER, basename))
        assert os.path.isfile(T.STAGE1_LOC_CKPT)

        train_data = T.TrainData(np.asarray(
            train_idxs[:N_UPDATES * P.PER_GPU_BATCH * P.WORLD_SIZE]))
        sampler = T.DistributedSampler(train_data, num_replicas=2,
                                        rank=rank, shuffle=True,
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

        model = check_baseline_compatibility(rank).cuda(local_rank)
        transfer = T.transfer_stage1_weights(
            model, T.STAGE1_LOC_CKPT, verbose=False)
        assert transfer['coverage_backbone'] == 1.0
        backbone = [p for name, p in model.named_parameters()
                    if name.startswith(('rgb_net.', 'post_net.'))]
        assert backbone and all(p.requires_grad for p in backbone)

        checked_steps = [0]
        step_times = []
        step_start = [time.perf_counter()]

        class CheckedAdamW(T.AdamW):
            def step(self, *args, **kwargs):
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and finite(grads), f'rank {rank}: nonfinite gradient'
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
        local_report = {
            'rank': rank, 'gpu': torch.cuda.get_device_name(local_rank),
            'updates': updates, 'loss': loss, 'cross_entropy': cce,
            'train_elapsed_seconds': elapsed,
            'train_peak_allocated_gib': train_allocated,
            'train_peak_reserved_gib': train_reserved,
            'iteration_time_mean_seconds': statistics.mean(step_times),
            'iteration_time_median_steady_seconds': statistics.median(step_times[3:]),
            'backbone_trainable_tensors': len(backbone),
            'stage1_encoder_coverage': transfer['coverage_backbone'],
        }
        ranks = [None, None]
        T.dist.all_gather_object(ranks, local_report)

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
        assert not os.listdir(P.CKPT_DIR), 'Preflight wrote a formal checkpoint'
        assert not os.path.exists(P.HISTORY_CSV), 'Preflight wrote validation history'
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
                'validation_split_sha256': split_sha,
                'validation_protocol': 'single-view, rank0 only, batch1',
                'validation_sample_score': validation_score,
                'validation_sample_seconds': validation_seconds,
                'base_initialization_equal_to_k2': True,
                'frozen_baseline_checkpoint_strict_load': True,
                'baseline_parameter_count': sum(p.numel() for p in model.parameters()),
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
            with open(AUDIT_JSON, 'w', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2)
                handle.write('\n')
            print(f'B0 DDP2 PREFLIGHT PASS: 2x{N_UPDATES} local steps, '
                  f'global_batch=4; peaks allocated='
                  f'{ranks[0]["train_peak_allocated_gib"]:.2f}/'
                  f'{ranks[1]["train_peak_allocated_gib"]:.2f} GiB; '
                  f'elapsed={max(r["train_elapsed_seconds"] for r in ranks):.1f}s; '
                  f'rank0 validation={validation_seconds:.1f}s', flush=True)
    finally:
        T.dist.destroy_process_group()


if __name__ == '__main__':
    main()
