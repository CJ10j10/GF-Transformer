#!/usr/bin/env python3
"""20 real two-GPU updates for each preregistered B0/K2 seed; no checkpoint."""

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
from experiments.stage2_matched_seeds import train_matched_seed as M

N_UPDATES = 20


def finite(tensors):
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def check_shared_initialization(seed, rank):
    torch.manual_seed(seed + rank)
    b0 = T.GFformer_two(use_kalman=False)
    torch.manual_seed(seed + rank)
    k2 = T.GFformer_two(use_kalman=True)
    k2_state = k2.state_dict()
    assert all(torch.equal(value, k2_state[name])
               for name, value in b0.state_dict().items())
    assert not any(name.startswith('kalman') for name in b0.state_dict())
    assert sum(p.numel() for p in b0.parameters()) == 56517168
    assert sum(p.numel() for p in k2.parameters()) == 57326963
    del b0, k2, k2_state


def main():
    args = M.parse_args()
    _, freeze = M.check_plan(args)
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank not in (0, 1) or torch.cuda.device_count() != 2:
        raise RuntimeError('Matched-seed preflight requires two visible GPUs')
    torch.cuda.set_device(local_rank)
    T.dist.init_process_group(backend='nccl')
    rank = T.dist.get_rank()
    out = M.run_dir(args.variant, args.seed)
    audit = out / 'audit' / 'preflight.json'
    ckpt_dir = out / 'ckpt'
    history = out / 'results' / 'validation_history.csv'
    try:
        assert T.dist.get_world_size() == 2
        audit.parent.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        assert not list(ckpt_dir.iterdir()), 'Formal checkpoint already exists'
        assert not history.exists(), 'Formal validation history already exists'
        if rank == 0:
            audit.write_text(json.dumps({'ready_for_training': False,
                                          'variant': args.variant,
                                          'train_seed': args.seed}, indent=2) + '\n')
        T.dist.barrier()
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        T.cudnn.benchmark = True
        train_idxs, val_idxs = P.build_split()  # fixed split seed 3
        split_sha = M.audit_fixed_validation(val_idxs, freeze)
        sample_count = N_UPDATES * M.PER_GPU_BATCH * M.WORLD_SIZE
        data = T.TrainData(np.asarray(train_idxs[:sample_count]))
        sampler = T.DistributedSampler(data, num_replicas=2, rank=rank,
                                        shuffle=True, seed=args.seed,
                                        drop_last=True)
        sampler.set_epoch(0)
        local_indices = list(iter(sampler))
        gathered = [None, None]
        T.dist.all_gather_object(gathered, local_indices)
        assert len(local_indices) == N_UPDATES * M.PER_GPU_BATCH
        assert not (set(gathered[0]) & set(gathered[1]))
        loader = T.DataLoader(data, batch_size=M.PER_GPU_BATCH,
                              sampler=sampler, num_workers=4,
                              pin_memory=True, drop_last=True)
        assert len(loader) == N_UPDATES
        check_shared_initialization(args.seed, rank)
        torch.manual_seed(args.seed + rank)
        model = T.GFformer_two(use_kalman=args.variant == 'k2').cuda(local_rank)
        transfer = T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT,
                                              verbose=False)
        assert transfer['coverage_backbone'] == 1.0
        backbone = [p for name, p in model.named_parameters()
                    if name.startswith(('rgb_net.', 'post_net.'))]
        assert backbone and all(p.requires_grad for p in backbone)
        if args.variant == 'b0':
            assert not any(name.startswith('kalman') for name in model.state_dict())
        else:
            assert all(float(getattr(model, f'kalman{i}').gamma) == 0
                       for i in (1, 2, 3))

        checked_steps = [0]
        step_times = []
        step_start = [time.perf_counter()]
        class CheckedAdamW(T.AdamW):
            def step(self, *items, **kwargs):
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and finite(grads), f'rank {rank}: nonfinite gradients'
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
        ddp_model = T.DDP(model, device_ids=[local_rank],
                          find_unused_parameters=True)
        scheduler = T.lr_scheduler.MultiStepLR(
            optimizer, milestones=T.MILESTONES, gamma=T.GAMMA)
        seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0},
                                per_image=False).cuda()
        ce_loss = torch.nn.CrossEntropyLoss().cuda()
        T.PHYSICAL_BATCH = M.PER_GPU_BATCH
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss, cce, updates = T.train_epoch(
            0, seg_loss, ce_loss, ddp_model, optimizer, scheduler,
            loader, sampler, grad_accum=M.GRAD_ACCUM, world_size=2)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert updates == checked_steps[0] == N_UPDATES
        assert scheduler.last_epoch == 1
        assert math.isfinite(loss) and math.isfinite(cce)
        assert finite(model.parameters()), f'rank {rank}: nonfinite parameters'
        gamma = None
        if args.variant == 'k2':
            gamma = {f'K{i}': float(getattr(model, f'kalman{i}').gamma.detach())
                     for i in (1, 2, 3)}
            assert all(math.isfinite(value) and value != 0
                       for value in gamma.values())
        local = {
            'rank': rank,
            'gpu': torch.cuda.get_device_name(local_rank),
            'updates': updates,
            'loss': loss,
            'cross_entropy': cce,
            'train_elapsed_seconds': elapsed,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'median_steady_step_seconds': statistics.median(step_times[3:]),
            'backbone_trainable_tensors': len(backbone),
            'stage1_encoder_coverage': transfer['coverage_backbone'],
            'gamma_after_smoke': gamma,
        }
        ranks = [None, None]
        T.dist.all_gather_object(ranks, local)
        validation_score = None
        if rank == 0:
            val_loader = T.DataLoader(
                T.ValData(np.asarray([val_idxs[0]])), batch_size=1,
                shuffle=False, num_workers=0, pin_memory=True)
            ddp_model.eval()
            validation_score = T.validate(ddp_model.module, val_loader)
            assert math.isfinite(validation_score)
        T.dist.barrier()
        assert not list(ckpt_dir.iterdir()) and not history.exists()
        if rank == 0:
            report = {
                'ready_for_training': True,
                'variant': args.variant,
                'train_seed': args.seed,
                'validation_split_seed': 3,
                'validation_split_sha256': split_sha,
                'validation_mask_manifest_sha256': freeze['validation_mask_manifest_sha256'],
                'metric_code_sha256': freeze['metric_code_sha256'],
                'world_size': 2,
                'per_gpu_batch': M.PER_GPU_BATCH,
                'grad_accum': M.GRAD_ACCUM,
                'global_batch': 4,
                'updates_per_full_epoch': 3351,
                'smoke_updates_per_rank': N_UPDATES,
                'validation_images': len(val_idxs),
                'validation_sample_score': validation_score,
                'shared_b0_k2_base_initialization_equal': True,
                'formal_epochs': T.TOTAL_EPOCHS,
                'lr': T.LR,
                'weight_decay': T.WEIGHT_DECAY,
                'scheduler_milestones': T.MILESTONES,
                'scheduler_gamma': T.GAMMA,
                'crop': list(T.INPUT_SHAPE),
                'amp': T.AMP_ENABLED,
                'stage1_checkpoint': T.STAGE1_LOC_CKPT,
                'checkpoint_dir': str(ckpt_dir),
                'ranks': ranks,
            }
            audit.write_text(json.dumps(report, indent=2) + '\n')
            print(f'MATCHED-SEED PREFLIGHT PASS: {args.variant} seed={args.seed} '
                  f'2x{N_UPDATES} local steps; peak allocated '
                  f'{ranks[0]["peak_allocated_gib"]:.2f}/'
                  f'{ranks[1]["peak_allocated_gib"]:.2f} GiB', flush=True)
    finally:
        T.dist.destroy_process_group()


if __name__ == '__main__':
    main()
