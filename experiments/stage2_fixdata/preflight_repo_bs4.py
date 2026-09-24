#!/usr/bin/env python3
"""Baseline-B smoke: 24 real Stage2 training batches, no validation or checkpoint."""
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
import train_segformer_cls as T

N_BATCHES = 24


def all_finite(tensors):
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(3)
    np.random.seed(3)
    assert T.PHYSICAL_BATCH == 4 and T.GRAD_ACCUM_STEPS == 1
    assert T.PHYSICAL_BATCH * T.GRAD_ACCUM_STEPS == 4
    assert T.TOTAL_EPOCHS == 50 and T.VAL_BATCH == 1
    assert T.LR == 2e-4 and T.WEIGHT_DECAY == 1e-6
    assert T.MILESTONES == [3, 9] and T.GAMMA == 0.5 and not T.AMP_ENABLED
    assert T.INPUT_SHAPE == (512, 512)
    assert not os.listdir(T.MODELS_FOLDER), 'Baseline-B checkpoint directory must be empty'

    model = T.GFformer_two().cuda()
    T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT, verbose=False)
    backbone = [p for name, p in model.named_parameters()
                if name.startswith(('rgb_net.', 'post_net.'))]
    assert backbone and all(p.requires_grad for p in backbone), 'backbone frozen'
    checked_steps = [0]
    class CheckedAdamW(T.AdamW):
        def step(self, *args, **kwargs):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all_finite(grads), 'non-finite gradient'
            result = super().step(*args, **kwargs)
            assert all_finite(model.parameters()), 'non-finite parameter'
            checked_steps[0] += 1
            return result

    optimizer = CheckedAdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    assert len(optimizer.param_groups) == 1, 'differential LR'
    assert optimizer.param_groups[0]['lr'] == T.LR
    opt_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in opt_ids for p in backbone), 'backbone missing from optimizer'

    # Take the first 96 entries from the production split and oversampling order.
    train_idxs0, val_idxs0 = T.train_test_split(
        np.arange(len(T.all_files)), test_size=0.1, random_state=3)
    assert len(T.all_files) == 9168 and len(val_idxs0) == 917
    assert T.LOC_FOLDER.endswith('experiments/stage1_fixdata_eval/loc_masks')
    indices = []
    for i in train_idxs0:
        indices.append(i)
        mask_path = T.all_files[i].replace('/images/', '/masks/').replace(
            '_pre_disaster', '_post_disaster')
        damage_mask = T.cv2.imread(mask_path, T.cv2.IMREAD_UNCHANGED)
        classes = [(c in damage_mask) for c in range(1, 5)]
        if any(classes[1:]):
            indices.append(i)
        if any(classes[1:3]):
            indices.append(i)
        if len(indices) >= N_BATCHES * T.PHYSICAL_BATCH:
            break
    dataset = T.TrainData(np.asarray(indices[:N_BATCHES * T.PHYSICAL_BATCH]))
    sampler = T.DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=3)
    loader = DataLoader(dataset, batch_size=T.PHYSICAL_BATCH, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)
    assert len(loader) == N_BATCHES
    seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    # The optimizer subclass checks gradients and parameters at each step.
    scheduler = T.lr_scheduler.MultiStepLR(optimizer, milestones=T.MILESTONES,
                                            gamma=T.GAMMA)

    # Production also wraps the model in one-rank NCCL DDP.
    rendezvous = f'/tmp/gf_stage2_repo_bs4_smoke_{os.getpid()}'
    T.dist.init_process_group(backend='nccl', init_method=f'file://{rendezvous}',
                              rank=0, world_size=1)
    ddp_model = T.DDP(model, device_ids=[0], find_unused_parameters=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        loss, cce, updates = T.train_epoch(
            0, seg_loss, ce_loss, ddp_model, optimizer, scheduler, loader, sampler,
            grad_accum=T.GRAD_ACCUM_STEPS, world_size=1)
    finally:
        T.dist.destroy_process_group()
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    assert math.isfinite(loss) and math.isfinite(cce), 'non-finite loss'
    assert updates == checked_steps[0] == N_BATCHES, 'optimizer update mismatch'
    assert all_finite(model.parameters()), 'non-finite parameter'
    assert scheduler.last_epoch == 1, 'scheduler did not step once after subset epoch'
    assert not os.listdir(T.MODELS_FOLDER), 'smoke wrote a formal checkpoint'
    print(f'BASELINE-B SMOKE PASS: batches={N_BATCHES}, updates={updates}, '
          f'loss={loss:.6f}, cce={cce:.6f}, '
          f'peak_allocated={peak_allocated:.2f} GiB, '
          f'peak_reserved={peak_reserved:.2f} GiB, elapsed={elapsed:.1f}s')
    print(f'backbone: {len(backbone)} trainable tensors, all in one AdamW group; '
          'gradients and parameters finite at every optimizer step; no checkpoint')


if __name__ == '__main__':
    main()
