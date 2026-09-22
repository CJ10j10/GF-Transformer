"""Generate building localization masks from Stage 1 checkpoint for Stage 2 training."""
import os, sys, gc, cv2
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'model'))
from gfmodel import GFformer_one
from utils import preprocess_inputs

DATA_BASE = os.path.join(BASE_DIR, 'data', 'xBD')
TRAIN_DIRS = [os.path.join(DATA_BASE, 'train'), os.path.join(DATA_BASE, 'tier3')]
LOC_FOLDER = os.path.join(BASE_DIR, 'loc_segformer')
EXP_NAME = 'fixdata'
CKPT_PATH = os.path.join(BASE_DIR, 'tune_weight', f'GFformer_loc_3_{EXP_NAME}_best2')
os.makedirs(LOC_FOLDER, exist_ok=True)

# Collect all pre-disaster images
all_files = []
for d in TRAIN_DIRS:
    for f in sorted(os.listdir(os.path.join(d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(os.path.join(d, 'images', f))

print(f'Total images to infer: {len(all_files)}')
print(f'Loading checkpoint: {CKPT_PATH}')

model = GFformer_one().cuda()
ckpt = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f'  Loaded epoch {ckpt["epoch"]}, best_score {ckpt["best_score"]:.4f}')

batch_size = 8
bsz = batch_size
total = len(all_files)
gen = 0

for start in tqdm(range(0, total, bsz), desc='Inference'):
    end = min(start + bsz, total)
    batch_imgs = []
    batch_fns = []
    for i in range(start, end):
        fn = all_files[i]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = preprocess_inputs(img)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        batch_imgs.append(img)
        batch_fns.append(fn.split('/')[-1].replace('.png', '_part1.png'))

    x = torch.stack(batch_imgs).cuda()
    with torch.no_grad():
        out = model(x)
        preds = torch.sigmoid(out[:, 0, ...]).cpu().numpy()

    for j, fname in enumerate(batch_fns):
        mask = (preds[j] > 0.4).astype(np.uint8) * 255
        out_path = os.path.join(LOC_FOLDER, fname)
        cv2.imwrite(out_path, mask)
        gen += 1

print(f'\nGenerated {gen} masks in {LOC_FOLDER}')
print('Inference complete.')
