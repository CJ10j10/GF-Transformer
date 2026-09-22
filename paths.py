"""Single source of truth for model checkpoint paths.

Stage 2 (train_segformer_cls.py) and inference_loc.py MUST reference
STAGE1_LOC_CKPT below — never tune_weight/GFformer_loc_*.  tune_weight/
is legacy and no longer used by Stage 2.

Provenance of STAGE1_LOC_CKPT:
    experiments/stage1_fixdata_eval/README.md  (independent re-evaluation)
    global F1 0.8628, mean per-image Dice 0.8830 (gate PASS, F1 >= 0.85)
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stage-1 (building localization) checkpoint — frozen, permanently named,
# sha256-verified (see experiments/stage1_fixdata_eval/README.md).
STAGE1_LOC_CKPT = os.path.join(
    BASE_DIR, 'experiments', 'stage1_fixdata_eval', 'ckpt',
    'GFformer_loc_fixdata_ep57_valdice0.8830_sha_cd940989.pt')
