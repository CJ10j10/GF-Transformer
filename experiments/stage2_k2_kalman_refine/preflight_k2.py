#!/usr/bin/env python3
"""K2 gates: baseline equality, Kalman values, and 24 real FP32 train updates."""

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
from experiments.stage2_k2_kalman_refine.train_k2 import assert_protocol

EXP_DIR = os.path.abspath(os.path.dirname(__file__))
CKPT_DIR = os.path.join(EXP_DIR, 'ckpt')
AUDIT_JSON = os.path.join(EXP_DIR, 'audit', 'preflight.json')
BASELINE_CKPT = os.path.join(ROOT, 'experiments', 'stage2_fixdata',
                             'ckpt_repo_bs4', 'GFformer_cls_3_fixdata_best14')
N_BATCHES = 24


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def finite(tensors):
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def main():
    assert_protocol()
    assert torch.cuda.is_available()
    assert os.path.isfile(BASELINE_CKPT)
    os.makedirs(CKPT_DIR, exist_ok=True)
    assert not os.listdir(CKPT_DIR), 'K2 checkpoint directory must stay empty'
    torch.cuda.set_device(0)
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    T.cudnn.benchmark = False
    report = {'ready_for_training': False, 'baseline_checkpoint': BASELINE_CKPT,
              'stage1_checkpoint': T.STAGE1_LOC_CKPT,
              'localization_masks': T.LOC_FOLDER,
              'iterations': N_BATCHES,
              'protocol': {'batch': T.PHYSICAL_BATCH,
                           'grad_accum': T.GRAD_ACCUM_STEPS,
                           'epochs': T.TOTAL_EPOCHS, 'optimizer': 'AdamW',
                           'lr': T.LR, 'weight_decay': T.WEIGHT_DECAY,
                           'scheduler_milestones': T.MILESTONES,
                           'scheduler_gamma': T.GAMMA,
                           'crop': list(T.INPUT_SHAPE), 'amp': T.AMP_ENABLED,
                           'val_batch': T.VAL_BATCH,
                           'eval': 'single-view'}}
    with open(AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')

    # Equality gate: same checkpoint and exact same real augmented input.
    baseline = T.GFformer_two().cuda().eval()
    k2 = T.GFformer_two(use_kalman=True).cuda().eval()
    checkpoint = torch.load(BASELINE_CKPT, map_location='cpu')
    weights = checkpoint['state_dict']
    baseline.load_state_dict(weights, strict=True)
    missing, unexpected = k2.load_state_dict(weights, strict=False)
    assert not unexpected and missing and all(n.startswith('kalman') for n in missing), (missing, unexpected)
    report['baseline_parameters'] = count_params(baseline)
    report['k2_parameters'] = count_params(k2)
    report['added_parameters'] = report['k2_parameters'] - report['baseline_parameters']
    assert all(float(getattr(k2, f'kalman{i}').gamma) == 0 for i in (1, 2, 3))
    assert not hasattr(k2, 'kalman4')

    observed_shapes = {}
    handles = []
    for i in (1, 2, 3):
        def gf_hook(module, inp, out, i=i):
            observed_shapes[f'GF{i}'] = {'pre': list(inp[0].shape),
                                        'post': list(inp[1].shape),
                                        'global': list(out.shape)}
        def csgf_hook(module, inp, out, i=i):
            observed_shapes[f'CSGF{i}'] = {'pre': list(inp[0].shape),
                                          'post': list(inp[1].shape),
                                          'global': list(inp[2].shape)}
        handles.append(getattr(k2, f'gfm{i}').register_forward_hook(gf_hook))
        handles.append(getattr(k2, f'CSGF{i}').register_forward_hook(csgf_hook))
        getattr(k2, f'kalman{i}').record_k_stats = True
    train_idxs0, val_idxs0 = T.train_test_split(np.arange(len(T.all_files)),
                                                 test_size=0.1, random_state=3)
    assert len(val_idxs0) == 917
    report['validation_images'] = len(val_idxs0)
    for i in val_idxs0:
        mask_name = os.path.basename(T.all_files[i]).replace('.png', '_part1.png')
        assert os.path.isfile(os.path.join(T.LOC_FOLDER, mask_name)), mask_name
    sample = T.TrainData(np.asarray([train_idxs0[0]]))[0]
    same_input = sample['img'].unsqueeze(0).cuda()
    assert list(same_input.shape) == [1, 6, 512, 512]
    with torch.no_grad():
        baseline_logits = baseline(same_input)
        k2_logits = k2(same_input)
        diff = (baseline_logits - k2_logits).abs().max().item()
    for handle in handles:
        handle.remove()
    report['max_abs_diff'] = diff
    report['insertion_shapes'] = observed_shapes
    report['kalman_k'] = {f'K{i}': getattr(k2, f'kalman{i}').last_k_stats
                          for i in (1, 2, 3)}
    assert diff < 1e-6, f'Baseline equivalence failed: {diff}'
    for i in (1, 2, 3):
        s = report['kalman_k'][f'K{i}']
        assert s is not None and all(math.isfinite(v) for v in s.values())
        assert 0 <= s['min'] <= s['max'] <= 1, s
        gf = observed_shapes[f'GF{i}']
        csgf = observed_shapes[f'CSGF{i}']
        assert gf['pre'] == csgf['pre'] and gf['post'] == csgf['post']
        assert gf['global'] == csgf['global']
    print(f'BASELINE EQUIVALENCE PASS: max_abs_diff={diff:.9g}', flush=True)
    print('K stats:', report['kalman_k'], flush=True)
    print('Shapes:', observed_shapes, flush=True)
    del baseline, k2, checkpoint, weights, same_input, baseline_logits, k2_logits
    torch.cuda.empty_cache()

    # With identical seed, every pre-existing tensor must initialize identically.
    torch.manual_seed(3)
    baseline_initial = T.GFformer_two().cuda()
    torch.manual_seed(3)
    model = T.GFformer_two(use_kalman=True).cuda()
    baseline_state = baseline_initial.state_dict()
    k2_state = model.state_dict()
    assert all(torch.equal(v, k2_state[n]) for n, v in baseline_state.items()), \
        'K2 changed baseline parameter initialization'
    report['base_initialization_equal'] = True
    del baseline_initial, baseline_state, k2_state
    torch.cuda.empty_cache()

    # Production initialization: Stage1 encoder transfer, random K2 head.
    transfer = T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT, verbose=False)
    assert transfer['coverage_backbone'] >= 0.95
    backbone = [p for n, p in model.named_parameters()
                if n.startswith(('rgb_net.', 'post_net.'))]
    assert backbone and all(p.requires_grad for p in backbone)
    report['stage1_encoder_coverage'] = transfer['coverage_backbone']
    report['backbone_trainable_tensors'] = len(backbone)
    report['gamma_initial'] = {f'K{i}': float(getattr(model, f'kalman{i}').gamma.detach())
                               for i in (1, 2, 3)}

    gradient_audit = []
    step_times = []
    checked_steps = [0]
    step_start = [time.perf_counter()]

    class CheckedAdamW(T.AdamW):
        def step(self, *args, **kwargs):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and finite(grads), 'nonfinite gradient'
            entry = {'step': checked_steps[0] + 1, 'branches': {}}
            for i in (1, 2, 3):
                layer = getattr(model, f'kalman{i}')
                sub = {}
                for name, p in [('gamma', layer.gamma),
                                ('obs', layer.obs_proj.weight),
                                ('P', layer.p_net.weight),
                                ('R', layer.r_net.weight)]:
                    assert p.grad is not None and torch.isfinite(p.grad).all(), (i, name)
                    sub[name] = p.grad.detach().abs().max().item()
                entry['branches'][f'K{i}'] = sub
            if checked_steps[0] < 3:
                gradient_audit.append(entry)
            result = super().step(*args, **kwargs)
            checked_steps[0] += 1
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start[0])
            step_start[0] = time.perf_counter()
            return result

    optimizer = CheckedAdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    assert len(optimizer.param_groups) == 1
    param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in param_ids for p in backbone)
    assert all(id(p) in param_ids for p in model.parameters())
    scheduler = T.lr_scheduler.MultiStepLR(optimizer, milestones=T.MILESTONES,
                                            gamma=T.GAMMA)

    # First 96 entries in the exact production split and oversampling order.
    indices = []
    for i in train_idxs0:
        indices.append(i)
        mask_path = T.all_files[i].replace('/images/', '/masks/').replace(
            '_pre_disaster', '_post_disaster')
        damage_mask = T.cv2.imread(mask_path, T.cv2.IMREAD_UNCHANGED)
        classes = [c in damage_mask for c in range(1, 5)]
        if any(classes[1:]):
            indices.append(i)
        if any(classes[1:3]):
            indices.append(i)
        if len(indices) >= N_BATCHES * T.PHYSICAL_BATCH:
            break
    dataset = T.TrainData(np.asarray(indices[:N_BATCHES * T.PHYSICAL_BATCH]))
    sampler = T.DistributedSampler(dataset, num_replicas=1, rank=0,
                                    shuffle=True, seed=3)
    loader = T.DataLoader(dataset, batch_size=T.PHYSICAL_BATCH,
                          sampler=sampler, num_workers=4, pin_memory=True,
                          drop_last=True)
    assert len(loader) == N_BATCHES
    seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    rendezvous = f'/tmp/gf_stage2_k2_smoke_{os.getpid()}'
    T.dist.init_process_group(backend='nccl', init_method=f'file://{rendezvous}',
                              rank=0, world_size=1)
    ddp_model = T.DDP(model, device_ids=[0], find_unused_parameters=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        loss, cce, updates = T.train_epoch(
            0, seg_loss, ce_loss, ddp_model, optimizer, scheduler, loader, sampler,
            grad_accum=1, world_size=1)
    finally:
        T.dist.destroy_process_group()
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report['optimizer_updates'] = updates
    report['loss'] = loss
    report['cross_entropy'] = cce
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    report['smoke_elapsed_seconds'] = elapsed
    report['iteration_time_mean_seconds'] = statistics.mean(step_times)
    report['iteration_time_median_seconds'] = statistics.median(step_times)
    report['first_three_gradient_max_abs'] = gradient_audit
    report['gamma_after_smoke'] = {f'K{i}': float(getattr(model, f'kalman{i}').gamma.detach())
                                   for i in (1, 2, 3)}
    assert updates == checked_steps[0] == N_BATCHES
    assert math.isfinite(loss) and math.isfinite(cce)
    assert finite(model.parameters()), 'nonfinite parameter'
    assert scheduler.last_epoch == 1
    assert not os.listdir(CKPT_DIR), 'smoke wrote checkpoint'
    assert all(v != 0 for v in report['gamma_after_smoke'].values()), 'gamma stayed zero'
    for i in (1, 2, 3):
        for branch in ('obs', 'P', 'R'):
            assert any(gradient_audit[j]['branches'][f'K{i}'][branch] > 0
                       for j in (1, 2)), (i, branch, gradient_audit)
    report['ready_for_training'] = True
    with open(AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(f'K2 SMOKE PASS: batches={N_BATCHES}, updates={updates}, '
          f'peak_allocated={report["peak_allocated_gib"]:.2f} GiB, '
          f'peak_reserved={report["peak_reserved_gib"]:.2f} GiB, '
          f'mean_step={report["iteration_time_mean_seconds"]:.2f}s, '
          f'elapsed={elapsed:.1f}s', flush=True)
    print('gamma:', report['gamma_after_smoke'], flush=True)
    print('gradient audit:', gradient_audit, flush=True)


if __name__ == '__main__':
    main()
