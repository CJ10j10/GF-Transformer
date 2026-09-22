"""Stage 1: Building Localization with GFformer_one — 4-GPU DDP version."""

import os
import sys
import gc
import random
import timeit

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from apex import amp
from adamw import AdamW
from losses import dice_round, ComboLoss
from utils import *
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'model'))
from gfmodel import GFformer_one

DATA_BASE = os.path.join(BASE_DIR, 'data', 'xBD')
TRAIN_DIRS = [os.path.join(DATA_BASE, 'train'), os.path.join(DATA_BASE, 'tier3')]
MODELS_FOLDER = os.path.join(BASE_DIR, 'tune_weight')
INPUT_SHAPE = (512, 512)
EXP_NAME = os.environ.get('GF_EXP_NAME', 'fixdata')
RESUME_FROM_CHECKPOINT = os.environ.get('GF_RESUME', '0') == '1'
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ── DDP helpers ────────────────────────────────────────────────────────
def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()

def dprint(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)

# ── Data ───────────────────────────────────────────────────────────────
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

all_files = []
for d in TRAIN_DIRS:
    for f in sorted(os.listdir(os.path.join(d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(os.path.join(d, 'images', f))


class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]
        fn = all_files[_idx]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        if random.random() > 0.985:
            img = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)

        if random.random() > 0.5:
            img = img[::-1, ...]
            msk0 = msk0[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                msk0 = np.rot90(msk0, k=rot)

        if random.random() > 0.9:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)

        if random.random() > 0.9:
            rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320),
                       img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)

        crop_size = INPUT_SHAPE[0]
        if random.random() > 0.3:
            crop_size = random.randint(int(INPUT_SHAPE[0] / 1.1), int(INPUT_SHAPE[0] / 0.9))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        for _ in range(random.randint(1, 5)):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk0[y0:y0 + crop_size, x0:x0 + crop_size].sum()
            if _sc > bst_sc:
                bst_sc, bst_x0, bst_y0 = _sc, x0, y0
        x0, y0 = bst_x0, bst_y0
        img = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk0 = msk0[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != INPUT_SHAPE[0]:
            img = cv2.resize(img, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.99:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        if random.random() > 0.99:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.99:
            if random.random() > 0.99:
                img = clahe(img)
            elif random.random() > 0.99:
                img = gauss_noise(img)
            elif random.random() > 0.99:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.99:
            if random.random() > 0.99:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.999:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        msk = (msk0[..., np.newaxis] > 127).astype(np.float32)
        img = preprocess_inputs(img)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).float()

        return {'img': img, 'msk': msk, 'fn': fn}


class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]
        fn = all_files[_idx]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        msk = (msk0[..., np.newaxis] > 127).astype(np.float32)
        img = preprocess_inputs(img)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).float()
        return {'img': img, 'msk': msk, 'fn': fn}


# ── Validation ─────────────────────────────────────────────────────────
def validate(net, data_loader):
    dices = []
    with torch.no_grad():
        for sample in tqdm(data_loader, disable=not is_main(), desc='Val'):
            imgs = sample["img"].cuda(non_blocking=True)
            msks = sample["msk"].numpy()
            out = net(imgs)
            msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()
            for j in range(msks.shape[0]):
                dices.append(dice(msks[j, 0], msk_pred[j] > 0.5))
    return np.mean(dices)


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    if not is_main():
        return best_score
    model.eval()
    d = validate(model, data_val)
    if d > best_score:
        torch.save(
            {'epoch': current_epoch + 1, 'state_dict': model.module.state_dict(),
             'best_score': d},
            os.path.join(MODELS_FOLDER, snapshot_name + '_best2'))
        best_score = d
    dprint(f"score: {d:.4f}\tscore_best: {best_score:.4f}")
    return best_score


# ── Training epoch ─────────────────────────────────────────────────────
def train_epoch(current_epoch, seg_loss, model, optimizer, scheduler, train_loader, sampler):
    losses_m = AverageMeter()
    dices_m = AverageMeter()
    model.train()
    sampler.set_epoch(current_epoch)

    iterator = train_loader
    if is_main():
        iterator = tqdm(train_loader, desc=f'Epoch {current_epoch}')

    for sample in iterator:
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)

        out = model(imgs)
        loss = seg_loss(out, msks)

        with torch.no_grad():
            probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(probs, msks[:, 0, ...])

        losses_m.update(loss.item(), imgs.size(0))
        dices_m.update(dice_sc, imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.1)
        optimizer.step()

        if is_main() and hasattr(iterator, 'set_description'):
            iterator.set_description(
                f"epoch: {current_epoch}; lr {scheduler.get_last_lr()[-1]:.7f}; "
                f"Loss {losses_m.val:.4f} ({losses_m.avg:.4f}); Dice {dices_m.val:.4f} ({dices_m.avg:.4f})")

    scheduler.step()
    dprint(f"epoch: {current_epoch}; lr {scheduler.get_last_lr()[-1]:.7f}; "
           f"Loss {losses_m.avg:.4f}; Dice {dices_m.avg:.4f}")


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # DDP init
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dprint(f'DDP: rank={rank}/{world_size}, GPU={torch.cuda.get_device_name(local_rank)}')

    t0 = timeit.default_timer()
    seed = 3
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    cudnn.benchmark = True

    # ── Data split ──
    all_idxs = np.arange(len(all_files))
    train_idxs, val_idxs = train_test_split(all_idxs, test_size=0.1, random_state=seed,
                                             shuffle=True)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_sampler = DistributedSampler(data_train, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=seed)
    batch_size = int(os.environ.get('GF_BATCH_SIZE', 4))
    val_batch_size = int(os.environ.get('GF_VAL_BATCH_SIZE', batch_size))

    train_loader = DataLoader(data_train, batch_size=batch_size, sampler=train_sampler,
                               num_workers=4, pin_memory=True, drop_last=True)
    # Validation is run only on rank0, over the full validation split.
    val_loader = DataLoader(val_train, batch_size=val_batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    dprint(f'Train: {len(train_idxs)} imgs, {len(train_loader)} batches/epoch')
    dprint(f'Val:   {len(val_idxs)} imgs, {len(val_loader)} batches')

    # ── Model ──
    model = GFformer_one().cuda(local_rank)
    optimizer = AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6)
    # FP32 full precision (no AMP — more stable, RTX 4090 fits easily)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    start_epoch = 0
    ckpt_path = os.path.join(MODELS_FOLDER, f'GFformer_loc_{seed}_{EXP_NAME}_best2')
    if RESUME_FROM_CHECKPOINT and os.path.exists(ckpt_path):
        dprint(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.module.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_score = ckpt.get('best_score', 0.0)
        dprint(f"  Resume epoch: {start_epoch}, best_score: {best_score:.4f}")
        del ckpt
        gc.collect()
    else:
        best_score = 0.0

    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180, 190],
        gamma=0.5)

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 6.0}, per_image=False).cuda()

    # ── Training loop ──
    total_epochs = int(os.environ.get('GF_TOTAL_EPOCHS', 200))
    snapshot_name = f'GFformer_loc_{seed}_{EXP_NAME}'

    dprint(f'Starting Stage 1 training: epochs {start_epoch}-{total_epochs-1}, '
           f'eff_BS={batch_size * world_size}, FP32 (no AMP)')
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, total_epochs):
        train_epoch(epoch, seg_loss, model, optimizer, scheduler, train_loader, train_sampler)
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            best_score = evaluate_val(val_loader, best_score, model, snapshot_name, epoch)
        barrier()

    dist.destroy_process_group()
    elapsed = timeit.default_timer() - t0
    dprint(f'Stage 1 done. Time: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)')
