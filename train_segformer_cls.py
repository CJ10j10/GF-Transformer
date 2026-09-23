"""Stage 2: Damage Classification with GFformer_two — 4-GPU DDP version."""

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
from TripletLoss import TripletMarginLoss
from utils import *
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from skimage.morphology import square, dilation
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'model'))
from gfmodel import GFformer_two
from paths import STAGE1_LOC_CKPT  # the ONLY allowed Stage-1 checkpoint
from ckpt_transfer import transfer_stage1_weights  # explicit encoder mapping

DATA_BASE = os.path.join(BASE_DIR, 'data', 'xBD')
TRAIN_DIRS = [os.path.join(DATA_BASE, 'train'), os.path.join(DATA_BASE, 'tier3')]
# Stage-2 artifacts live in their own experiment directory — never mix with
# the legacy tune_weight/ files, the smoke checkpoints (ckpt_smoke/), or
# any previous Stage-2 run.
EXP_DIR = os.path.join(BASE_DIR, 'experiments', 'stage2_fixdata')
MODELS_FOLDER = os.path.join(EXP_DIR, 'ckpt_baseline')
# Localization masks generated from the verified Stage-1 checkpoint
# (inference_loc.py writes here).
LOC_FOLDER = os.path.join(BASE_DIR, 'experiments', 'stage1_fixdata_eval', 'loc_masks')
INPUT_SHAPE = (512, 512)
EXP_NAME = 'fixdata'
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(LOC_FOLDER, exist_ok=True)

# ── Training protocol (repo-aligned baseline, single-GPU effective batch 32) ──
PHYSICAL_BATCH = 4                    # batch per GPU
VAL_BATCH = 1                         # FP32 full-res 1024x1024 validation
GRAD_ACCUM_STEPS = int(os.environ.get('GF_GRAD_ACCUM', '8'))  # eff batch = 4 * 1 * 8 = 32
LR = 2e-4
WEIGHT_DECAY = 1e-6                   # repo-aligned AdamW (≈ Adam)
MILESTONES = [3, 9]
GAMMA = 0.5
TOTAL_EPOCHS = 30
AMP_ENABLED = False                   # FP32 full precision

# ── DDP helpers ───────────────────────────────────────────────────────
def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()

def dprint(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)

# ── Data ──────────────────────────────────────────────────────────────
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
        img2 = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(
            fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'),
            cv2.IMREAD_UNCHANGED)

        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255
        msk1[lbl_msk1 == 1] = 255

        # ── Augmentation (applied to pre + post jointly) ──
        if random.random() > 0.5:
            img, img2 = img[::-1, ...], img2[::-1, ...]
            msk0, msk1, msk2, msk3, msk4 = [m[::-1, ...] for m in (msk0, msk1, msk2, msk3, msk4)]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img, img2 = np.rot90(img, k=rot), np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)

        if random.random() > 0.8:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            for m in (msk0, msk1, msk2, msk3, msk4):
                _ = shift_image(m, shift_pnt)  # applied in-place via return
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)

        if random.random() > 0.2:
            rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320),
                       img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        # ── Crop ──
        crop_size = INPUT_SHAPE[0]
        if random.random() > 0.1:
            crop_size = random.randint(int(INPUT_SHAPE[0] / 1.15), int(INPUT_SHAPE[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        for _ in range(random.randint(1, 10)):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = (msk2[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 5 +
                   msk3[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 5 +
                   msk4[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 2 +
                   msk1[y0:y0 + crop_size, x0:x0 + crop_size].sum())
            if _sc > bst_sc:
                bst_sc, bst_x0, bst_y0 = _sc, x0, y0
        x0, y0 = bst_x0, bst_y0

        img = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
        img2 = img2[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk0 = msk0[y0:y0 + crop_size, x0:x0 + crop_size]
        msk1 = msk1[y0:y0 + crop_size, x0:x0 + crop_size]
        msk2 = msk2[y0:y0 + crop_size, x0:x0 + crop_size]
        msk3 = msk3[y0:y0 + crop_size, x0:x0 + crop_size]
        msk4 = msk4[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != INPUT_SHAPE[0]:
            img = cv2.resize(img, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)

        # ── Color augmentation ──
        if random.random() > 0.96:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        if random.random() > 0.96:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = change_hsv(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        for im, cond in [(img, 0.9), (img2, 0.9)]:
            if random.random() > cond:
                if random.random() > 0.96:
                    im[...] = clahe(im)
                elif random.random() > 0.96:
                    im[...] = gauss_noise(im)
                elif random.random() > 0.96:
                    im[...] = cv2.blur(im, (3, 3))
            elif random.random() > cond:
                if random.random() > 0.96:
                    im[...] = saturation(im, 0.9 + random.random() * 0.2)
                elif random.random() > 0.96:
                    im[...] = brightness(im, 0.9 + random.random() * 0.2)
                elif random.random() > 0.96:
                    im[...] = contrast(im, 0.9 + random.random() * 0.2)

        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)
        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img2 = el_det.augment_image(img2)

        # ── Build multi-class mask ──
        msks = np.stack([msk0, msk1, msk2, msk3, msk4], axis=2)
        msks = (msks > 127)
        # damage hierarchy logic (original paper)
        msks[..., 0] = True
        msks[..., 1] = dilation(msks[..., 1], square(5))
        msks[..., 2] = dilation(msks[..., 2], square(5))
        msks[..., 3] = dilation(msks[..., 3], square(5))
        msks[..., 4] = dilation(msks[..., 4], square(5))
        msks[..., 1][msks[..., 2:].max(axis=2)] = False
        msks[..., 3][msks[..., 2]] = False
        msks[..., 4][msks[..., 2]] = False
        msks[..., 4][msks[..., 3]] = False
        msks[..., 0][msks[..., 1:].max(axis=2)] = False
        msks = msks.astype(np.float32)

        lbl_msk = msks.argmax(axis=2)

        img_cat = np.concatenate([img, img2], axis=2)
        img_cat = preprocess_inputs(img_cat)

        img_t = torch.from_numpy(img_cat.transpose((2, 0, 1))).float()
        msk_t = torch.from_numpy(msks.transpose((2, 0, 1))).long()

        return {'img': img_t, 'msk': msk_t, 'lbl_msk': lbl_msk, 'fn': fn}


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
        img2 = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        # Load stage-1 predicted localization mask (if available)
        loc_path = os.path.join(LOC_FOLDER,
            f"{fn.split('/')[-1].replace('.png', '_part1.png')}")
        msk_loc = np.zeros((1024, 1024), dtype=bool)
        if os.path.exists(loc_path):
            msk_loc = cv2.imread(loc_path, cv2.IMREAD_UNCHANGED) > int(0.4 * 255)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(
            fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'),
            cv2.IMREAD_UNCHANGED)

        msk1 = (lbl_msk1 == 1).astype(np.uint8) * 255
        msk2 = (lbl_msk1 == 2).astype(np.uint8) * 255
        msk3 = (lbl_msk1 == 3).astype(np.uint8) * 255
        msk4 = (lbl_msk1 == 4).astype(np.uint8) * 255

        msks = np.stack([msk0, msk1, msk2, msk3, msk4], axis=2)
        msks = (msks > 127).astype(np.float32)
        lbl_msk = msks[..., 1:].argmax(axis=2)

        img_cat = np.concatenate([img, img2], axis=2)
        img_cat = preprocess_inputs(img_cat)

        img_t = torch.from_numpy(img_cat.transpose((2, 0, 1))).float()
        msk_t = torch.from_numpy(msks.transpose((2, 0, 1))).long()

        return {'img': img_t, 'msk': msk_t, 'lbl_msk': lbl_msk, 'fn': fn,
                'msk_loc': msk_loc}


# ── Validation ────────────────────────────────────────────────────────
def validate(net, data_loader):
    tp = np.zeros((5,))
    fp = np.zeros((5,))
    fn = np.zeros((5,))
    thr = 0.4

    with torch.no_grad():
        for sample in tqdm(data_loader, disable=not is_main(), desc='Val'):
            msks = sample["msk"].numpy()
            lbl_msk = sample["lbl_msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            msk_loc = sample["msk_loc"].numpy().astype(bool)

            out = net(imgs)
            msk_pred = msk_loc
            msk_damage_pred = torch.softmax(out, dim=1).cpu().numpy()[:, 1:, ...]

            for j in range(msks.shape[0]):
                gt_bld = msks[j, 0] > 0
                tp[4] += (gt_bld & msk_pred[j]).sum()
                fn[4] += (~gt_bld & msk_pred[j]).sum()
                fp[4] += (gt_bld & ~msk_pred[j]).sum()

                targ = lbl_msk[j][gt_bld]
                pred = msk_damage_pred[j].argmax(axis=0)
                pred = pred * (msk_pred[j] > thr)
                pred = pred[gt_bld]
                for c in range(4):
                    tp[c] += ((pred == c) & (targ == c)).sum()
                    fn[c] += ((pred != c) & (targ == c)).sum()
                    fp[c] += ((pred == c) & (targ != c)).sum()

    d0 = 2 * tp[4] / (2 * tp[4] + fp[4] + fn[4] + 1e-8)
    f1_sc = np.zeros(4)
    for c in range(4):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-8)
    f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))
    sc = 0.3 * d0 + 0.7 * f1

    dprint(f"Val Score: {sc:.4f}, Dice: {d0:.4f}, F1: {f1:.4f}, "
           f"F1_0:{f1_sc[0]:.4f} F1_1:{f1_sc[1]:.4f} F1_2:{f1_sc[2]:.4f} F1_3:{f1_sc[3]:.4f}")
    return sc


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    if not is_main():
        return best_score
    model.eval()
    d = validate(model, data_val)
    if d > best_score:
        torch.save(
            {'epoch': current_epoch + 1, 'state_dict': model.module.state_dict(),
             'best_score': d},
            os.path.join(MODELS_FOLDER, snapshot_name + '_best14'))
        best_score = d
    dprint(f"score: {d:.4f}\tscore_best: {best_score:.4f}")
    return best_score


# ── Training epoch ────────────────────────────────────────────────────
def train_epoch(current_epoch, seg_loss, ce_loss, model, optimizer, scheduler,
                train_loader, sampler, grad_accum=1, world_size=1):
    """One training epoch with gradient accumulation.

    Accumulation contract:
      * per micro-batch loss is scaled by 1/grad_accum before backward
      * optimizer.step() + clip only after a full accumulation cycle
        (or on the trailing partial batch at the end of the epoch)
      * optimizer.zero_grad() only after an optimizer update
      * scheduler.step() once per epoch, never per micro-batch
    Returns (avg_loss, avg_cce, optimizer_updates).
    """
    losses_seg = AverageMeter()
    losses_cce = AverageMeter()
    dices_m = AverageMeter()
    model.train()
    sampler.set_epoch(current_epoch)

    iterator = train_loader
    if is_main():
        iterator = tqdm(train_loader, desc=f'Epoch {current_epoch}')

    n_batches = len(train_loader)
    optimizer_updates = 0
    accum_count = 0
    optimizer.zero_grad()  # start the first accumulation cycle with zero grads

    for bi, sample in enumerate(iterator):
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

        # scale so the accumulated gradient == effective batch semantics
        (loss / grad_accum).backward()
        accum_count += 1

        with torch.no_grad():
            probs = 1 - torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(probs, 1 - msks[:, 0, ...])

        losses_seg.update(loss.item(), imgs.size(0))
        losses_cce.update(loss5.item(), imgs.size(0))
        dices_m.update(dice_sc, imgs.size(0))

        is_last = (bi == n_batches - 1)
        if accum_count % grad_accum == 0 or is_last:
            # clip + step only when the cycle is complete (or on the tail)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
            optimizer.step()
            optimizer.zero_grad()   # zero only after an optimizer update
            optimizer_updates += 1
            accum_count = 0

        if is_main() and hasattr(iterator, 'set_description'):
            iterator.set_description(
                f"epoch: {current_epoch}; lr {scheduler.get_last_lr()[-1]:.7f}; "
                f"Loss {losses_seg.val:.4f} ({losses_seg.avg:.4f}); "
                f"CCE {losses_cce.val:.4f} ({losses_cce.avg:.4f}); "
                f"Dice {dices_m.val:.4f} ({dices_m.avg:.4f}); "
                f"upd {optimizer_updates}")

    scheduler.step()  # once per epoch — never per micro-batch
    dprint(f"epoch: {current_epoch}; lr {scheduler.get_last_lr()[-1]:.7f}; "
           f"Loss {losses_seg.avg:.4f}; CCE {losses_cce.avg:.4f}; "
           f"Dice {dices_m.avg:.4f}; micro_batches {n_batches}; "
           f"optimizer_updates {optimizer_updates}; "
           f"accum {grad_accum}; "
           f"eff_bs {PHYSICAL_BATCH * world_size * grad_accum}")
    return losses_seg.avg, losses_cce.avg, optimizer_updates


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dprint(f'DDP: rank={rank}/{world_size}, GPU={torch.cuda.get_device_name(local_rank)}')

    if is_main() and not os.listdir(LOC_FOLDER):
        print(f'WARNING: {LOC_FOLDER} is empty — run inference_loc.py first, '
              f'otherwise Stage-2 validation uses empty localization masks.')

    t0 = timeit.default_timer()
    seed = 3
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    cudnn.benchmark = True

    # ── Build class-balanced training indices ──
    file_classes = []
    for fn in tqdm(all_files, disable=not is_main(), desc='Scan classes'):
        fl = np.zeros(4, dtype=bool)
        msk_path = fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')
        msk1 = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        for c in range(1, 5):
            fl[c - 1] = c in msk1
        file_classes.append(fl)
    file_classes = np.asarray(file_classes)

    train_idxs0, val_idxs0 = train_test_split(np.arange(len(all_files)),
                                               test_size=0.1, random_state=seed)

    # Over-sample damaged images for class balance
    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if file_classes[i, 1:].max():
            train_idxs.append(i)
        if file_classes[i, 1:3].max():
            train_idxs.append(i)
    train_idxs = np.asarray(train_idxs)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs0)

    train_sampler = DistributedSampler(data_train, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=seed)
    batch_size = PHYSICAL_BATCH
    # Validation runs FP32 at full 1024x1024: the GFM bmm tensors
    # ((B, h*w, h*w)) need ~4 GiB per batch element at 128x128 feature maps,
    # so bs=4 OOMs on 24 GB (the legacy run used AMP, which halved this).
    # Measured peak with bs=1: 4.4 GiB. Batch size does not change the
    # metric values (per-image accumulation), only memory.
    val_batch_size = VAL_BATCH

    train_loader = DataLoader(data_train, batch_size=batch_size, sampler=train_sampler,
                               num_workers=4, pin_memory=True, drop_last=True)
    # Validation is run only on rank0, over the full validation split.
    val_loader = DataLoader(val_train, batch_size=val_batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    dprint(f'Train: {len(train_idxs)} imgs (oversampled), {len(train_loader)} batches/epoch')
    dprint(f'Val:   {len(val_idxs0)} imgs, {len(val_loader)} batches')

    # ── Model ──
    model = GFformer_two().cuda(local_rank)

    # Load stage-1 weights — ONLY the frozen, independently re-evaluated
    # checkpoint defined in paths.STAGE1_LOC_CKPT
    # (see experiments/stage1_fixdata_eval/README.md). Never tune_weight/.
    # Backbone keys differ between the two models ('encoder.*' vs
    # 'rgb_net.*'/'post_net.*'), so same-name matching would transfer
    # nothing — use the explicit mapping in ckpt_transfer.py (Gate 2).
    ckpt_path = STAGE1_LOC_CKPT
    dprint(f"Loading stage-1 checkpoint '{ckpt_path}'...")
    if os.path.exists(ckpt_path):
        report = transfer_stage1_weights(model, ckpt_path, verbose=is_main())
        dprint(f"  backbone transfer coverage {report['coverage_backbone']:.4f} "
               f"({report['matched_backbone']}/{report['n_stage2_backbone']}) — "
               f"Gate 2 requires > {0.95}")
    else:
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {ckpt_path}. Restore it to "
            f"experiments/stage1_fixdata_eval/ckpt/ (sha256 "
            f"cd9409890d146fcc020c5448fd533cccec4a9a2ea157542b43890cabbeed01ed).")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # FP32 full precision (no AMP — stable for DDP training)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    seg_loss = ComboLoss({'dice': 0.5, 'focal': 8.0}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()

    # ── Training loop ──
    best_score = 0.0
    total_epochs = TOTAL_EPOCHS
    snapshot_name = f'GFformer_cls_{seed}_{EXP_NAME}'

    dprint(f'[protocol] GPU=1xRTX4090 crop={INPUT_SHAPE} epochs={total_epochs} '
           f'lr={LR} physical_batch={batch_size} grad_accum={GRAD_ACCUM_STEPS} '
           f'eff_batch={batch_size * world_size * GRAD_ACCUM_STEPS} '
           f'optimizer=AdamW(wd={WEIGHT_DECAY}) '
           f'scheduler=MultiStepLR{MILESTONES},gamma={GAMMA} '
           f'AMP={AMP_ENABLED} val_batch={val_batch_size}')
    dprint(f'Starting Stage 2 training: {total_epochs} epochs, '
           f'physical_BS={batch_size}, accum={GRAD_ACCUM_STEPS}, '
           f'eff_BS={batch_size * world_size * GRAD_ACCUM_STEPS}')
    torch.cuda.empty_cache()

    for epoch in range(total_epochs):
        ls, lc, _ = train_epoch(epoch, seg_loss, ce_loss, model, optimizer, scheduler,
                                train_loader, train_sampler,
                                grad_accum=GRAD_ACCUM_STEPS, world_size=world_size)
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            best_score = evaluate_val(val_loader, best_score, model, snapshot_name, epoch)
        barrier()

    dist.destroy_process_group()
    elapsed = timeit.default_timer() - t0
    dprint(f'Stage 2 done. Time: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)')
