#!/usr/bin/env python3
"""
Final single-GPU preflight for the official Stage-2 30-epoch baseline.

Runs the REAL train_segformer_cls.train_epoch with gradient accumulation
(physical batch 4, accum 8 -> effective batch 32) on a small dataset, and
checks:

  1. backbone (rgb_net) is fully trainable and present in the optimizer
  2. no OOM, no NaN/Inf in loss / gradients / parameters
  3. peak allocated + reserved memory
  4. gradient-accumulation accounting (micro batches vs optimizer updates)

NO checkpoint is written.  Importing train_segformer_cls reuses the exact
production Dataset / model / losses / transfer path / accumulation loop.
"""

import os
import sys
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

import train_segformer_cls as T  # noqa: E402

N_SAMPLES = 128  # -> 32 micro-batches at physical batch 4


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    np.random.seed(0)

    print('=' * 72)
    print(' FINAL PREFLIGHT — real train_epoch, accum=%d (eff batch 32)' % T.GRAD_ACCUM_STEPS)
    print('=' * 72)

    # ── Model + frozen Stage-1 backbone (production path) ──
    model = T.GFformer_two().cuda()
    report = T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT, verbose=False)
    print(f"  backbone transfer coverage: {report['coverage_backbone']:.4f}")

    # ── 1. Backbone trainability ────────────────────────────────────────
    backbone_params = [p for k, p in model.named_parameters()
                       if k.startswith(('rgb_net.', 'post_net.'))]
    backbone_total = sum(p.numel() for p in backbone_params)
    backbone_trainable = [p for p in backbone_params if p.requires_grad]
    all_req_grad = all(p.requires_grad for p in backbone_params)

    optimizer = T.AdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    opt_ids = set()
    opt_total = 0
    for g in optimizer.param_groups:
        for p in g['params']:
            opt_ids.add(id(p))
            opt_total += p.numel()
    backbone_ids = set(id(p) for p in backbone_params)
    backbone_in_optimizer = sum(1 for p in backbone_params if id(p) in opt_ids)
    backbone_in_optimizer_numel = sum(p.numel() for p in backbone_params
                                     if id(p) in opt_ids)

    print('-' * 72)
    print(' 1. BACKBONE TRAINABILITY')
    print('-' * 72)
    print(f"  backbone params (unique):   {len(backbone_params)}")
    print(f"  backbone total numel:       {backbone_total:,}")
    print(f"  backbone trainable numel:   {sum(p.numel() for p in backbone_trainable):,}")
    print(f"  all requires_grad:          {all_req_grad}")
    print(f"  backbone params in optimizer: {backbone_in_optimizer} "
          f"({backbone_in_optimizer_numel:,} params)")
    print(f"  optimizer total numel:      {opt_total:,}")
    assert backbone_in_optimizer > 0, 'backbone NOT in optimizer — abort'
    assert all_req_grad, 'some backbone params frozen — abort'

    # ── Small dataset driving the REAL train_epoch ──────────────────────
    data_train = T.TrainData(np.arange(N_SAMPLES))
    sampler = T.DistributedSampler(data_train, num_replicas=1, rank=0, shuffle=True)
    loader = DataLoader(data_train, batch_size=T.PHYSICAL_BATCH, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)
    n_batches = len(loader)
    print(f"\n  micro-batches for smoke: {n_batches} "
          f"(physical {T.PHYSICAL_BATCH}, accum {T.GRAD_ACCUM_STEPS})")

    scheduler = T.lr_scheduler.MultiStepLR(optimizer, milestones=T.MILESTONES,
                                           gamma=T.GAMMA)
    seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()

    # ── 2/3. Real accumulation loop (forward+backward+step) ─────────────
    ls, lc, n_updates = T.train_epoch(
        0, seg_loss, ce_loss, model, optimizer, scheduler, loader, sampler,
        grad_accum=T.GRAD_ACCUM_STEPS, world_size=1)

    dt = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    loss_finite = math.isfinite(ls)
    params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())

    # ── gradient finiteness (one extra backward, then zero) ─────────────
    sample = next(iter(loader))
    imgs = sample["img"].cuda(non_blocking=True)
    msks = sample["msk"].cuda(non_blocking=True)
    lbl = sample["lbl_msk"].cuda(non_blocking=True)
    out = model(imgs)
    l = (0.1 * seg_loss(out[:, 0], msks[:, 0].float()) +
         0.1 * seg_loss(out[:, 1], msks[:, 1].float()) +
         0.1 * seg_loss(out[:, 2], msks[:, 2].float()) +
         0.6 * seg_loss(out[:, 3], msks[:, 3].float()) +
         0.1 * seg_loss(out[:, 4], msks[:, 4].float()) +
         11 * ce_loss(out, lbl))
    l.backward()
    grads_finite = all(torch.isfinite(p.grad).all().item()
                       for p in model.parameters() if p.grad is not None)
    optimizer.zero_grad()

    expected_updates = (n_batches + T.GRAD_ACCUM_STEPS - 1) // T.GRAD_ACCUM_STEPS
    print('-' * 72)
    print(' 2/3. MEMORY + ACCUMULATION SMOKE')
    print('-' * 72)
    print(f"  micro batches:      {n_batches}")
    print(f"  optimizer updates:  {n_updates}  (expected {expected_updates})")
    print(f"  accum steps:        {T.GRAD_ACCUM_STEPS}")
    print(f"  effective batch:    {T.PHYSICAL_BATCH} * 1 * {T.GRAD_ACCUM_STEPS} = "
          f"{T.PHYSICAL_BATCH * T.GRAD_ACCUM_STEPS}")
    print(f"  peak allocated:     {peak_alloc:.2f} GiB")
    print(f"  peak reserved:      {peak_reserved:.2f} GiB")
    print(f"  loss finite:        {loss_finite}  (avg loss {ls:.4f})")
    print(f"  params finite:      {params_finite}")
    print(f"  grads finite:       {grads_finite}")
    print(f"  OOM:                {'NO' if peak_alloc < 23.5 else 'YES'}")
    print(f"  time:               {dt:.1f} s for {n_batches} micro-batches "
          f"({dt/n_batches:.2f} s/micro-batch)")

    ok = (loss_finite and params_finite and grads_finite
          and peak_alloc < 23.5 and n_updates == expected_updates
          and backbone_in_optimizer > 0 and all_req_grad)
    print('-' * 72)
    print(f" FINAL PREFLIGHT: {'PASS' if ok else 'FAIL'}")
    print('  (no checkpoint was written; ckpt_baseline/ untouched)')
    print('=' * 72)


if __name__ == '__main__':
    main()
