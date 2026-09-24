#!/usr/bin/env python3
"""K2 formal Stage2 entry point. Run manually through run_k2.sh only."""

import os
import random
import sys
import timeit

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import train_segformer_cls as T

EXP_DIR = os.path.abspath(os.path.dirname(__file__))
CKPT_DIR = os.path.join(EXP_DIR, 'ckpt')


def assert_protocol():
    assert T.PHYSICAL_BATCH == 4 and T.GRAD_ACCUM_STEPS == 1
    assert T.TOTAL_EPOCHS == 50 and T.VAL_BATCH == 1
    assert T.LR == 2e-4 and T.WEIGHT_DECAY == 1e-6
    assert T.MILESTONES == [3, 9] and T.GAMMA == 0.5
    assert T.INPUT_SHAPE == (512, 512) and not T.AMP_ENABLED
    assert len(T.all_files) == 9168
    assert T.LOC_FOLDER == os.path.join(ROOT, 'experiments', 'stage1_fixdata_eval', 'loc_masks')


def main():
    assert_protocol()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    T.dist.init_process_group(backend='nccl')
    rank, world_size = T.dist.get_rank(), T.dist.get_world_size()
    try:
        if world_size != 1:
            raise RuntimeError(f'K2 requires one GPU, got world_size={world_size}')
        os.makedirs(CKPT_DIR, exist_ok=True)
        if os.listdir(CKPT_DIR):
            raise RuntimeError(f'K2 checkpoint directory must be empty: {CKPT_DIR}')
        if not os.path.isfile(T.STAGE1_LOC_CKPT):
            raise FileNotFoundError(T.STAGE1_LOC_CKPT)

        t0 = timeit.default_timer()
        seed = 3
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        T.cudnn.benchmark = True

        # Same class scan, split, and oversampling order as Baseline-B.
        file_classes = []
        for fn in T.tqdm(T.all_files, disable=not T.is_main(), desc='Scan classes'):
            mask_path = fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')
            damage_mask = T.cv2.imread(mask_path, T.cv2.IMREAD_UNCHANGED)
            file_classes.append([c in damage_mask for c in range(1, 5)])
        file_classes = np.asarray(file_classes)
        train_idxs0, val_idxs0 = T.train_test_split(
            np.arange(len(T.all_files)), test_size=0.1, random_state=seed)
        assert len(val_idxs0) == 917
        train_idxs = []
        for i in train_idxs0:
            train_idxs.append(i)
            if file_classes[i, 1:].max():
                train_idxs.append(i)
            if file_classes[i, 1:3].max():
                train_idxs.append(i)
        train_idxs = np.asarray(train_idxs)

        # ValData uses the verified localization masks; fail if any is missing.
        for i in val_idxs0:
            basename = os.path.basename(T.all_files[i]).replace('.png', '_part1.png')
            path = os.path.join(T.LOC_FOLDER, basename)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        train_data = T.TrainData(train_idxs)
        val_data = T.ValData(val_idxs0)
        sampler = T.DistributedSampler(train_data, num_replicas=world_size,
                                       rank=rank, shuffle=True, seed=seed)
        train_loader = T.DataLoader(train_data, batch_size=T.PHYSICAL_BATCH,
                                    sampler=sampler, num_workers=4,
                                    pin_memory=True, drop_last=True)
        val_loader = T.DataLoader(val_data, batch_size=T.VAL_BATCH,
                                  shuffle=False, num_workers=4, pin_memory=True)

        model = T.GFformer_two(use_kalman=True).cuda(local_rank)
        report = T.transfer_stage1_weights(model, T.STAGE1_LOC_CKPT,
                                           verbose=T.is_main())
        if report['coverage_backbone'] < 0.95:
            raise RuntimeError(f"Stage1 encoder coverage too low: {report['coverage_backbone']}")
        optimizer = T.AdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
        assert len(optimizer.param_groups) == 1
        model = T.DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        scheduler = T.lr_scheduler.MultiStepLR(optimizer, milestones=T.MILESTONES,
                                                gamma=T.GAMMA)
        seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
        ce_loss = torch.nn.CrossEntropyLoss().cuda()

        T.dprint(f'[K2 protocol] GPU=1xRTX4090 batch={T.PHYSICAL_BATCH} '
                 f'accum={T.GRAD_ACCUM_STEPS} epochs={T.TOTAL_EPOCHS} '
                 f'AdamW lr={T.LR} wd={T.WEIGHT_DECAY} '
                 f'MultiStepLR{T.MILESTONES} gamma={T.GAMMA} '
                 f'crop={T.INPUT_SHAPE} FP32 val_batch={T.VAL_BATCH} '
                 f'train={len(train_idxs)} val={len(val_idxs0)} '
                 f'Stage1={T.STAGE1_LOC_CKPT} loc={T.LOC_FOLDER}')
        best_score = 0.0
        for epoch in range(T.TOTAL_EPOCHS):
            _, _, updates = T.train_epoch(epoch, seg_loss, ce_loss, model,
                                           optimizer, scheduler, train_loader,
                                           sampler, grad_accum=T.GRAD_ACCUM_STEPS,
                                           world_size=world_size)
            assert updates == len(train_loader)
            if epoch % 2 == 0:
                torch.cuda.empty_cache()
                model.eval()
                score = T.validate(model, val_loader)  # single-view, unchanged metric
                if score > best_score:
                    torch.save({'epoch': epoch + 1,
                                'state_dict': model.module.state_dict(),
                                'best_score': score},
                               os.path.join(CKPT_DIR, 'GFformer_cls_3_k2_best14'))
                    best_score = score
                T.dprint(f'score: {score:.4f}\tscore_best: {best_score:.4f}')
            T.barrier()
        T.dprint(f'K2 done. Time: {(timeit.default_timer()-t0)/3600:.2f} h')
    finally:
        T.dist.destroy_process_group()


if __name__ == '__main__':
    main()
