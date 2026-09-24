#!/usr/bin/env python3
"""Independent 2xRTX4090 K2 protocol. Start manually through run_k2_ddp2.sh."""

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
CKPT_DIR = os.path.join(EXP_DIR, 'ckpt_ddp2')
PER_GPU_BATCH = 2
WORLD_SIZE = 2
GRAD_ACCUM = 1
SEED = 3


def assert_protocol():
    assert T.PHYSICAL_BATCH == 4 and T.GRAD_ACCUM_STEPS == 1
    assert PER_GPU_BATCH * WORLD_SIZE * GRAD_ACCUM == 4
    assert T.TOTAL_EPOCHS == 50 and T.VAL_BATCH == 1
    assert T.LR == 2e-4 and T.WEIGHT_DECAY == 1e-6
    assert T.MILESTONES == [3, 9] and T.GAMMA == 0.5
    assert T.INPUT_SHAPE == (512, 512) and not T.AMP_ENABLED
    assert len(T.all_files) == 9168
    assert T.LOC_FOLDER == os.path.join(ROOT, 'experiments', 'stage1_fixdata_eval', 'loc_masks')


def build_split():
    """Copy Baseline-B class scan and oversampling exactly; only sampling is sharded."""
    file_classes = []
    for fn in T.tqdm(T.all_files, disable=not T.is_main(), desc='Scan classes'):
        mask_path = fn.replace('/images/', '/masks/').replace(
            '_pre_disaster', '_post_disaster')
        damage_mask = T.cv2.imread(mask_path, T.cv2.IMREAD_UNCHANGED)
        file_classes.append([c in damage_mask for c in range(1, 5)])
    file_classes = np.asarray(file_classes)
    train_idxs0, val_idxs0 = T.train_test_split(
        np.arange(len(T.all_files)), test_size=0.1, random_state=SEED)
    assert len(val_idxs0) == 917
    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if file_classes[i, 1:].max():
            train_idxs.append(i)
        if file_classes[i, 1:3].max():
            train_idxs.append(i)
    train_idxs = np.asarray(train_idxs)
    assert len(train_idxs) == 13407
    return train_idxs, val_idxs0


def make_train_loader(train_idxs, rank):
    data = T.TrainData(train_idxs)
    # Drop one global tail index, then each rank drops its one-image tail batch.
    # 3351 updates/rank/epoch x 2 images/rank x 2 ranks = 13404 images,
    # matching Baseline-B's 3351 updates x 4 images.
    sampler = T.DistributedSampler(data, num_replicas=WORLD_SIZE, rank=rank,
                                    shuffle=True, seed=SEED, drop_last=True)
    loader = T.DataLoader(data, batch_size=PER_GPU_BATCH, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)
    assert len(loader) == 3351
    return loader, sampler


def make_val_loader(val_idxs0):
    for i in val_idxs0:
        basename = os.path.basename(T.all_files[i]).replace('.png', '_part1.png')
        path = os.path.join(T.LOC_FOLDER, basename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return T.DataLoader(T.ValData(val_idxs0), batch_size=T.VAL_BATCH,
                        shuffle=False, num_workers=4, pin_memory=True)


def main():
    assert_protocol()
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank not in (0, 1) or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError('K2 DDP2 needs exactly two visible CUDA devices')
    torch.cuda.set_device(local_rank)
    T.dist.init_process_group(backend='nccl')
    rank, world_size = T.dist.get_rank(), T.dist.get_world_size()
    try:
        if world_size != WORLD_SIZE:
            raise RuntimeError(f'K2 DDP2 requires world_size=2, got {world_size}')
        os.makedirs(CKPT_DIR, exist_ok=True)
        if os.listdir(CKPT_DIR):
            raise RuntimeError(f'K2 DDP2 checkpoint directory must be empty: {CKPT_DIR}')
        if not os.path.isfile(T.STAGE1_LOC_CKPT):
            raise FileNotFoundError(T.STAGE1_LOC_CKPT)
        t0 = timeit.default_timer()
        np.random.seed(SEED + rank)
        random.seed(SEED + rank)
        torch.manual_seed(SEED + rank)
        T.cudnn.benchmark = True

        train_idxs, val_idxs0 = build_split()
        train_loader, sampler = make_train_loader(train_idxs, rank)
        val_loader = make_val_loader(val_idxs0) if rank == 0 else None

        model = T.GFformer_two(use_kalman=True).cuda(local_rank)
        transfer = T.transfer_stage1_weights(
            model, T.STAGE1_LOC_CKPT, verbose=T.is_main())
        if transfer['coverage_backbone'] < 0.95:
            raise RuntimeError(f"Stage1 encoder coverage too low: {transfer['coverage_backbone']}")
        optimizer = T.AdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
        assert len(optimizer.param_groups) == 1
        model = T.DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        scheduler = T.lr_scheduler.MultiStepLR(optimizer, milestones=T.MILESTONES,
                                                gamma=T.GAMMA)
        seg_loss = T.ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
        ce_loss = torch.nn.CrossEntropyLoss().cuda()
        # train_epoch prints its effective batch using this imported constant.
        T.PHYSICAL_BATCH = PER_GPU_BATCH

        T.dprint(f'[K2 DDP2 protocol] GPUs=2xRTX4090 per_gpu_batch={PER_GPU_BATCH} '
                 f'accum={GRAD_ACCUM} global_batch={PER_GPU_BATCH * WORLD_SIZE} '
                 f'epochs={T.TOTAL_EPOCHS} AdamW lr={T.LR} wd={T.WEIGHT_DECAY} '
                 f'MultiStepLR{T.MILESTONES} gamma={T.GAMMA} '
                 f'crop={T.INPUT_SHAPE} FP32 val_batch={T.VAL_BATCH} '
                 f'train={len(train_idxs)} updates_per_epoch={len(train_loader)} '
                 f'val={len(val_idxs0)} Stage1={T.STAGE1_LOC_CKPT} loc={T.LOC_FOLDER}')
        best_score = 0.0
        for epoch in range(T.TOTAL_EPOCHS):
            _, _, updates = T.train_epoch(
                epoch, seg_loss, ce_loss, model, optimizer, scheduler,
                train_loader, sampler, grad_accum=GRAD_ACCUM, world_size=world_size)
            assert updates == len(train_loader) == 3351
            if epoch % 2 == 0 and rank == 0:
                torch.cuda.empty_cache()
                model.eval()
                # Use the raw module: rank 1 waits at barrier and does no DDP
                # forward, so validation cannot trigger a buffer-sync collective.
                score = T.validate(model.module, val_loader)
                if score > best_score:
                    torch.save({'epoch': epoch + 1,
                                'state_dict': model.module.state_dict(),
                                'best_score': score},
                               os.path.join(CKPT_DIR, 'GFformer_cls_3_k2_ddp2_best14'))
                    best_score = score
                T.dprint(f'score: {score:.4f}\tscore_best: {best_score:.4f}')
            T.barrier()
        T.dprint(f'K2 DDP2 done. Time: {(timeit.default_timer()-t0)/3600:.2f} h')
    finally:
        T.dist.destroy_process_group()


if __name__ == '__main__':
    main()
