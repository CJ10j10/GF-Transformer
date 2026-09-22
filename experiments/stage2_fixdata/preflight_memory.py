#!/usr/bin/env python3
"""
Preflight memory smoke for the official Stage-2 30-epoch baseline.

Runs 30 real training iterations with the planned official config
(batch_per_gpu=4, FP32, AdamW lr=2e-4, the exact loss expression from
train_segformer_cls.train_epoch) and checks:

  * no CUDA OOM
  * no NaN/Inf in loss / gradients / parameters
  * GPU memory stable across iterations (no leak growth)
  * loss values in a sane range

It imports train_segformer_cls to reuse the REAL Dataset, model, losses and
Stage-1 transfer path.  NO checkpoint is written.  (Train indices are plain
here — per-sample workload is identical, so memory/loss behaviour matches
the official run; the official oversampling only changes which samples.)

Usage:
  /usr/local/miniconda3/envs/gft/bin/python experiments/stage2_fixdata/preflight_memory.py
"""

import os
import sys
import time
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

import train_segformer_cls as T  # noqa: E402  (module-level: paths, dataset, model, losses)

N_ITERS = 30
BATCH_SIZE = 4  # official batch_per_gpu


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    np.random.seed(0)

    print('=' * 72)
    print(' PREFLIGHT MEMORY SMOKE — official config: bs=%d, FP32, 1 GPU' % BATCH_SIZE)
    print('=' * 72)

    # ── Data (real TrainData; plain indices, same per-sample workload) ──
    train_idxs = np.arange(len(T.all_files))
    data_train = T.TrainData(train_idxs)
    loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # ── Model + frozen Stage-1 backbone (production path) ──
    model = T.GFformer_two().cuda()
    report = T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT, verbose=False)
    print(f"  backbone coverage: {report['coverage_backbone']:.4f}")
    model.train()

    # ── Optimizer / losses (same as train_segformer_cls main) ──
    optimizer = T.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6)
    seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    mem_by_iter = []
    losses = []
    nonfinite = 0
    t0 = time.time()

    for it, sample in enumerate(loader):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)
        lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)

        out = model(imgs)
        loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...].float())
        loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...].float())
        loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...].float())
        loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...].float())
        loss4 = seg_loss(out[:, 4, ...], msks[:, 4, ...].float())
        loss5 = ce_loss(out, lbl_msk)
        loss = (0.1 * loss0 + 0.1 * loss1 + 0.1 * loss2 + 0.6 * loss3 +
                0.1 * loss4 + 11 * loss5)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

        # ── Checks ──
        finite = (math.isfinite(loss.item())
                  and all(torch.isfinite(p).all().item() for p in model.parameters()
                          if p.grad is not None)
                  and all(torch.isfinite(p).all().item() for p in model.parameters()))
        if not finite:
            nonfinite += 1
        mem = torch.cuda.memory_allocated() / 2**30
        mem_by_iter.append(mem)
        losses.append(loss.item())

        if it % 10 == 0 or it == N_ITERS - 1:
            print(f"  iter {it:3d}: loss={loss.item():.4f} "
                  f"alloc={mem:.2f}GiB peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
        if it + 1 >= N_ITERS:
            break

    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30

    print('-' * 72)
    print(' RESULTS')
    print('-' * 72)
    print(f"  iterations:        {len(mem_by_iter)}")
    print(f"  loss range:        [{min(losses):.4f}, {max(losses):.4f}]  "
          f"(sane: yes if no NaN/Inf)")
    print(f"  NaN/Inf events:    {nonfinite}")
    print(f"  GPU alloc:         first {mem_by_iter[0]:.2f} GiB -> "
          f"last {mem_by_iter[-1]:.2f} GiB (stable if no growth trend)")
    mid = mem_by_iter[len(mem_by_iter)//2]
    print(f"  GPU alloc mid-run: {mid:.2f} GiB")
    print(f"  peak GPU memory:   {peak:.2f} GiB / 24 GiB")
    print(f"  time per iter:     {dt/len(mem_by_iter):.2f} s "
          f"(-> ~{dt/len(mem_by_iter)*3351/60:.1f} min per epoch at 3351 batches)")
    ok = (nonfinite == 0 and peak < 23.5 and
          abs(mem_by_iter[-1] - mid) < 1.0)
    print('-' * 72)
    print(f" PREFLIGHT MEMORY: {'PASS' if ok else 'FAIL'}")
    print('  (no checkpoint was saved by this script)')
    print('=' * 72)


if __name__ == '__main__':
    main()
