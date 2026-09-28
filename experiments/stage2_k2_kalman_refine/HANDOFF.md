# GF-Transformer / K2 交接记录

更新：2026-09-28。项目目录：`/workspace/GF-Transformer`。本文件记录已完成的实验、封板证据和新种子实验准备；不记录进行中训练的临时进度或指标。

## 状态与分支

- 稳定 Baseline-B 分支：`fix/stage1-ckpt-isolation`，最后相关提交 `c566247`。
- K2 单卡准备：`exp/k2-gf-kalman-refine`，提交 `60403ca`。
- **双卡 K2 分支**：`exp/k2-gf-kalman-refine-ddp2`；训练代码提交 `2031786`。用户在 tmux 会话 `k2_ddp2` 中手动启动，50 epochs 已完整结束。
- **双卡 B0 对照分支**：`exp/b0-ddp2-control`，封板提交 `e5bcf7c`、标签 `b0-ddp2-frozen`。B0 和 K2 的 seed 3 配对结果已固定。
- **新种子筛选分支**：`exp/matched-seed-screening`；种子计划与训练入口提交 `d81f2f2`，四组预检提交 `1b1b701`。正式新种子实验的实时状态以各运行目录日志为准。
- 正式日志路径记录在 `current_k2_ddp2_log.txt`，本次为 `logs/stage2_k2_ddp2_20260924_133257.log`；torchrun 于 2026-09-25 04:00 UTC 正常退出，耗时 `14.45 h`。
- 最佳 checkpoint：`ckpt_ddp2/GFformer_cls_3_k2_ddp2_best14`（记录 epoch `27`，对应完成第 0-based epoch `26` 后的验证），SHA256 `8238addd6d0845515e6a893c5ae73c0bed93f7ffb3569a67b6e2488eb47f360a`。不要覆盖或从这个 best 权重继续启动新的正式实验。
- [独立 917 张 single-view 结果](results/ddp2_single_view.json)精确复现 checkpoint best score：`0.74992773` 对 `0.74992861`；split、localization 目录和 metric 代码哈希均与 Baseline-B 一致。

## 可信的前置复现

| 环节 | 固定结果 / 路径 |
| --- | --- |
| Stage1 checkpoint | `experiments/stage1_fixdata_eval/ckpt/GFformer_loc_fixdata_ep57_valdice0.8830_sha_cd940989.pt`；SHA256 `cd9409890d146fcc020c5448fd533cccec4a9a2ea157542b43890cabbeed01ed` |
| Stage1 独立重评估 | 固定 917 张验证，mean per-image Dice `0.882985`，与记录 best score `0.882980` 一致；sigmoid 阈值 0.5 下 global F1 `0.862771`。见 [Stage1 README](../stage1_fixdata_eval/README.md) |
| localization masks | 同一 Stage1 checkpoint 生成 9,168 张，missing/broken `0/0`；按 Stage2 使用的 0.4 阈值审计，global F1 `0.872038`。见 [mask 审计](../stage1_fixdata_eval/results/loc_mask_audit.json) |
| encoder transfer | Stage1 → Stage2 显式映射，backbone coverage `100%`，参数 spot check max abs diff `0`；此前已完成 Gate，无须重做 |
| Stage2 早期 smoke | building Dice `0.8720`；FP32 验证 batch 固定为 `1` |

Stage1 的 global F1 `0.862771` 与 localization-mask 审计 F1 `0.872038` **采用不同阈值**，不能当作同一评估结果比较。`tune_weight/` 是旧实验目录，正式 Stage2 不从中加载权重。

### Baseline-A 到 Baseline-B

Baseline-A 为单卡 physical batch `4`、accum `8`、effective batch `32`、30 epochs，F1b/F1d/F1s 为 `0.8720 / 0.4612 / 0.5845`。由于 optimizer 更新次数下降，但 scheduler 仍按 epoch `[3,9]` 衰减，随后运行了更贴近仓库协议的 Baseline-B。

Baseline-B 为单卡 batch `4`、accum `1`、50 epochs、AdamW `lr=2e-4`/`wd=1e-6`、MultiStepLR `[3,9]`/`gamma=0.5`、crop `512×512`、FP32、validation batch `1`。冻结 best checkpoint：`experiments/stage2_fixdata/ckpt_repo_bs4/GFformer_cls_3_fixdata_best14`（epoch 7，SHA256 `19523418fe2e149d44ab83a3c5e1fcfc95a6b554bc04d130e350d111a44d650d`）。固定 917 张的 single-view 结果：

| F1b | F1d | F1s | F1_0 | F1_1（minor） | F1_2 | F1_3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8720 | 0.6949 | 0.7480 | 0.9379 | 0.4917 | 0.7015 | 0.8122 |

4-way TTA（原图、水平/垂直翻转、180° 旋转；逆变换后平均 logits）在该 checkpoint 上 F1s `0.7450`、minor F1 `0.4834`，低于 single-view，因此 K2 正式比较继续使用 single-view。完整记录见 [Baseline-B 报告](../stage2_fixdata/results_repo_bs4/baseline_report.json)。Baseline-B 50 epochs 实际耗时 `20.46` 小时。

## K2 seed 3：damage-change guided refinement

**保留 `post−pre` 创新项，不是后续拟做的 `post−GF` K2b。** [KalmanRefine 实现](../../model/kalman_refine.py) 位于原 GF 与原 CSGF 之间，不改 GF、CSGF、loss、sampling、augmentation 或 backbone：

```text
innovation = post_feat - pre_feat
z = obs_proj(innovation)
P = softplus(p_net(global_feat)) + 1e-6
R = softplus(r_net(abs(innovation))) + 1e-6
K = P / (P + R)
refined = global_feat + gamma * K * z
```

`gamma` 可学习且初始为 `0`。仅在实际 forward 的 GF1/GF2/GF3 后插入 KF1/KF2/KF3，再送入 CSGF1/2/3；`gfm4` 未使用。

| 层 | pre/post 特征（512 crop） | GF 输出 / KF 输出 |
| --- | --- | --- |
| 1 | `B×64×128×128` | `B×128×128×128` |
| 2 | `B×128×64×64` | `B×320×64×64` |
| 3 | `B×320×32×32` | `B×512×32×32` |

单卡 [K2 预检](audit/preflight.json)：同一 Baseline-B 权重及同一输入、eval 模式下 logits `max_abs_diff=0`；原有参数在同一随机种子下逐项相同。K1/K2/K3 的 K mean/std 为 `0.5089/0.0787`、`0.4982/0.0861`、`0.5010/0.0575`，均有限且在 `[0,1]`。三个 gamma 在真实训练步后离开 0；P/R/obs 在第 2 步起出现有限非零梯度。24 次更新通过，单卡峰值 allocated/reserved `14.01/14.43 GiB`。Baseline/K2 参数量 `56,517,168 / 57,326,963`（增加 `809,795`）。

## 双卡训练协议与预检

[双卡训练入口](train_k2_ddp2.py)和[手动启动脚本](run_k2_ddp2.sh)与单卡 `ckpt/` 分开，正式输出为 `ckpt_ddp2/`，无自动 resume，也不读取 Baseline-B 或 smoke checkpoint。两张 RTX4090 每卡 batch `2`、accum `1`，全局 batch `4`；50 epochs，优化器、LR、scheduler、crop、FP32、Stage1 checkpoint、split、masks、loss 和增强保持 Baseline-B 协议。过采样后训练索引 `13,407`；`DistributedSampler(drop_last=True)` 加 DataLoader `drop_last=True` 后每卡每 epoch `3,351` 次更新、合计使用 `13,404` 张，更新数与单卡 Baseline-B 一致。

仅 rank 0 对固定 917 张运行原 `train_segformer_cls.validate`，使用 `model.module` 避开另一 rank 等待时的 DDP forward 同步，并仅由 rank 0 保存 best checkpoint。每个偶数 epoch（0、2、4、…）验证一次，best 依据原 score `0.3×F1b + 0.7×F1d`，不按 minor F1 挑 checkpoint。

[双卡预检](audit/ddp2_preflight.json)在两进程上各完成 `24` 次真实 FP32 optimizer 更新：loss/梯度/参数有限，gamma 更新、P/R/obs 梯度正常，无 OOM；每卡峰值 allocated `11.21 GiB`、reserved `16.79–16.81 GiB`。rank 0 还跑过**单张**真实 1024×1024 验证图，只用于验证链路，不能代替 917 张正式指标。预检未写 checkpoint。

双卡短测稳态约 `0.319 秒/更新`；据此作出的训练前耗时估计约 `15.7` 小时，实际 K2 完整训练耗时 `14.45` 小时。短测速度不能保证长期速度。虽然全局 batch 与更新数一致，每卡 BatchNorm 统计量不同，双卡 K2 与单卡 Baseline-B 的训练轨迹不会逐项相同。

## 双卡 K2 正式结果与 Baseline-B 比较

训练日志包含完整 50 epochs、每 epoch 3,351 次 optimizer 更新及 25 次完整 validation；无 Traceback、OOM 或非有限值。训练平均 loss 从 epoch 0 的 `1.3986` 降至 epoch 49 的 `0.4813`。best 在 epoch 26 后出现，最后一次验证（epoch 48 后）F1s 为 `0.7403`，低于 best `0.7499`；正式比较使用预先固定的 best checkpoint 规则。

独立重评估由 [eval_k2_ddp2.py](eval_k2_ddp2.py) 调用原 `train_segformer_cls.validate` 完成，结果见 [ddp2_single_view.json](results/ddp2_single_view.json)：

逐次验证曲线已从原始 tqdm 日志提取为小型 [K2 CSV](results/ddp2_validation_history.csv) 和 [Baseline-B CSV](results/baseline_b_validation_history.csv)，各 25 行；提取脚本为 [summarize_validation_history.py](summarize_validation_history.py)。原始大日志不纳入本轮提交。

| 指标 | Baseline-B（1 GPU） | K2 best（2 GPU） | K2 − Baseline-B |
| --- | ---: | ---: | ---: |
| F1b | 0.8720 | 0.8720 | 0.0000 |
| F1d | 0.6949 | 0.6976 | +0.0027 |
| F1s | 0.7480 | 0.7499 | +0.0019 |
| F1_0 | 0.9379 | 0.9406 | +0.0027 |
| F1_1（minor） | 0.4917 | 0.5024 | +0.0107 |
| F1_2（moderate） | 0.7015 | 0.6840 | −0.0175 |
| F1_3 | 0.8122 | 0.8206 | +0.0084 |

这只是相对于**单卡** Baseline-B 的历史比较：minor 增加 1.07 个百分点，F1s 增加 0.19 个百分点，moderate 下降 1.75 个百分点。单卡 B0 与双卡 K2 的 BatchNorm 本地统计量不同，不能把这些差值归因于 KalmanRefine。下节的 B0-D2 是真正同协议的双卡对照。

## B0-D2 同协议对照与正式封板

[B0-D2 实验说明](../stage2_b0_ddp2/README.md)记录了独立训练入口、预检、日志和 checkpoint 路径。它从双卡 K2 代码派生，仅将 `GFformer_two(use_kalman=True)` 改为 `False`，即原 GF + CSGF 结构。B0-D2 仍从固定 Stage1 encoder 初始化，不从训练好的 Baseline-B/K2 checkpoint resume。两卡每卡 batch 2、accum 1、全局 batch 4、每 epoch 3,351 次更新；其余优化器、scheduler、crop、FP32、loss、增强、过采样、固定 split、localization masks 及 single-view metric 与 K2-D2 一致。B0 预检确认原模型参数集与 Baseline-B checkpoint 严格兼容、共享参数的同种子初始化与 K2 一致；Stage1 encoder transfer 为 100%。

正式 B0-D2 完整运行 50 epochs、25 次验证，耗时 `14.60 h`，无 Traceback、OOM 或非有限值。冻结 best checkpoint：`experiments/stage2_b0_ddp2/ckpt/GFformer_cls_3_b0_ddp2_best14`（epoch 13，SHA256 `3c10b6f1ea807ee77dddc11743bbf73e471ab6ad8a2473fc424116721f8a1193`）。[独立重评结果](../stage2_b0_ddp2/results/ddp2_single_view.json)按原 `train_segformer_cls.validate` 在同一 917 张图像上复现 checkpoint 分数 `0.7596456`；[小型验证曲线 CSV](../stage2_b0_ddp2/results/validation_history.csv)有 25 行，原始 54 MB tqdm 日志和 216 MB checkpoint 不提交仓库。

| 指标 | 单卡 Baseline-B | 双卡 B0-D2 | 双卡 K2-D2 | B0-D2 − K2-D2 |
| --- | ---: | ---: | ---: | ---: |
| F1b | 0.8720 | 0.8720 | 0.8720 | 0.0000 |
| F1d | 0.6949 | **0.7115** | 0.6976 | +0.0139 |
| F1s | 0.7480 | **0.7596** | 0.7499 | +0.0097 |
| F1_0 | 0.9379 | 0.9398 | 0.9406 | −0.0008 |
| F1_1（minor） | 0.4917 | **0.5276** | 0.5024 | +0.0252 |
| F1_2 | 0.7015 | 0.6970 | 0.6840 | +0.0130 |
| F1_3 | 0.8122 | 0.8142 | 0.8206 | −0.0064 |

因此，K2 相对单卡 Baseline-B 的微小提升在同双卡协议对照下**不成立为 Kalman 增益**：seed 3 的最佳点估计反而是 B0-D2 较高。这不是 K2 稳定有害的证明，仍需看配对不确定性和训练种子差异。

[paired_bootstrap.json](../stage2_b0_ddp2/results/paired_bootstrap.json)明确写明 `delta_definition = B0-D2 minus K2-D2`、`bootstrap_rng_seed = 20260928`、`n_boot = 10000`、`image_count = 917`、image ID/split SHA256 `7e3c7e8cf1762512697055f6a7bfd5d51050a6d03fc4d93e0d86e1ffaa2c480e`。逐图 TP/FP/FN 保存在 [38 KB 压缩计数](../stage2_b0_ddp2/results/paired_per_image_sufficient_statistics.npz)；每次对两个模型重采样**同一组**图像索引，先汇总各类 TP/FP/FN，再算原全局 F1b、调和 F1d 与 F1s，未平均逐图 F1。

| B0-D2 − K2-D2 | 点估计 | 配对 bootstrap 95% percentile 区间 |
| --- | ---: | ---: |
| F1s | +0.0097 | [−0.0031, +0.0259] |
| F1d | +0.0139 | [−0.0044, +0.0370] |
| minor F1 | +0.0252 | [−0.0057, +0.0620] |

区间均跨 0；这项 bootstrap 只覆盖**固定最佳 checkpoint 条件下的验证图像抽样**，不覆盖训练种子或在同一 validation split 上挑选 best checkpoint 的不确定性。[封板清单](../stage2_b0_ddp2/results/freeze_manifest.json)核对两份独立重评估的 917 张 image IDs、同一 localization 目录、metric code SHA256 `906d6f68e9a7919c4a97944cace628678160888ffa53c835a587d84ecfba92d9`；917 张 mask 内容清单 SHA256 为 `37e841011b556f48e6158f6feb5df9efccbe854bdc0843c5be49e69d6e3b6f55`。两模型逐图 building 计数相同；mask 文件时间早于两次独立重评估。B0-D2 seed 3 阶段以标签 `b0-ddp2-frozen` 正式封板。

## 新种子 K2 screening：已准备的固定方案

[种子计划](../stage2_matched_seeds/seed_plan.json)在新增正式训练前固定 `[3, 11, 23]`，未来 **B0、K1、K2、K2b、K3** 均使用这组训练种子。seed 3 的 B0/K2 已完成；本轮新增 seed 11、23 的 B0/K2 各一轮。**训练种子与数据划分种子分离**：Python/NumPy/PyTorch 每 rank 使用 `train_seed + rank`，DDP sampler 使用 `train_seed`；917 张验证集始终由 split seed 3 决定，split SHA 与上节完全相同，mask 内容清单和 metric code SHA 也被训练入口检查。切换训练 seed 不改变 loss、sampling 规则、augmentation、GF/CSGF、backbone、优化器或验证协议。

筛选的预定主比较为**同 seed 的 K2 − B0 最佳 single-view F1s**；minor `F1_1` 是次要指标。每轮只按 25 次验证中的最高未舍入 F1s 选 checkpoint，并汇报全部三个 seed 的配对差值，不剔除不利结果。[训练入口](../stage2_matched_seeds/train_matched_seed.py)、[手动启动脚本](../stage2_matched_seeds/run_matched_seed.sh)、[独立重评脚本](../stage2_matched_seeds/eval_matched_seed.py)和[三种子汇总脚本](../stage2_matched_seeds/summarize_screening.py)已准备，每个 variant/seed 有独立 `ckpt/`、`audit/`、`results/`、时间戳日志路径和 compact validation CSV。无自动 resume，也不访问旧 checkpoint 输出目录。

四组 [预检报告目录](../stage2_matched_seeds/runs/)均为 `ready_for_training=true`：每卡 `20` 次真实 FP32 更新、loss/梯度/参数有限、Stage1 encoder coverage `100%`、固定 split/mask hash 一致，单张 rank 0 validation 链路正常，预检不写正式 checkpoint。B0 两种 seed 的峰值 allocated 均约 `10.88 GiB/卡`，K2 均约 `11.21 GiB/卡`；K2 的三个 gamma 在 seed 11、23 下均离开 0。[新种子 README](../stage2_matched_seeds/README.md)包含逐个 tmux 启动、日志查询和后续独立重评命令。K2b 的 `post−GF` canonical observation 更新仍可作为独立 hypothesis 继续，不依赖本次 K2 screening 是否成功。

## 遇到的问题及处理

| 问题 | 处理 / 结论 |
| --- | --- |
| Stage1 旧 checkpoint 与 Stage2 旧输出混用 | 冻结并校验正确 Stage1 checkpoint；Baseline-B、K2、B0-D2 和新种子运行各用独立 checkpoint 目录，legacy `tune_weight/` 不再用于正式训练 |
| 旧 localization mask 与 encoder 键名不匹配 | 重新生成并审计全部 9,168 张 mask；使用显式 encoder 权重映射，coverage `100%` |
| FP32 全分辨率验证 batch=4 OOM | 改为 val batch=1，固定 917 张和原 metric；不改变评估数值 |
| Baseline-A 的 accum=8 与按 epoch 衰减组合 | 运行 batch=4、accum=1 的 Baseline-B，并冻结其 single-view 结果 |
| Baseline-B 的 4-way TTA 未提升 | 报告 TTA 结果，K2 继续采用 single-view 作公平比较 |
| K2 新模块插入位置改变原 decoder 随机初始化 | 将 Kalman 参数构造放在原模型参数构造之后；相同种子下核对所有原有参数逐项相同 |
| 单卡 K2 入口拒绝双卡；多 rank 验证可能重复执行/写 checkpoint | 独立实现 `train_k2_ddp2.py`；只由 rank 0 验证和保存，其他 rank 等待 barrier；双卡预检覆盖此链路 |
| 双卡默认 sampler 可能补齐样本，使更新数变化 | 显式 `drop_last=True`，固定 3,351 更新/epoch |
| B0 预检中的 `torch.load(weights_only=False)` 与当前旧版 PyTorch 不兼容 | 移除该参数，保持独立 checkpoint 兼容性探针；重跑双卡短预检通过 |
| 根目录 `.gitignore` 的 `runs/` 规则遮蔽新种子预检报告 | 仅解除 `experiments/stage2_matched_seeds/runs/` 的忽略，继续忽略大型 checkpoint 和原始日志；四份预检 JSON 已提交 |
| agent 环境缺少 GitHub HTTPS 凭据 | 用户在已认证终端手动推送；本地标签和分支提交仍可用于精确复现 |
| VS Code/Codex 在非 Git 大目录搜索导致 CPU 高 | 只打开 `/workspace` 后 CPU 恢复；与训练计算无关 |
| 终端缺少 `rg` | 查询命令使用系统自带 `grep`；空 checkpoint 在 epoch 0 首次验证结束前属正常现象 |

## 查看已完成的训练日志

在新终端执行（每个新 shell 都要重新读取 `LOG`）：

```bash
cd /workspace/GF-Transformer
LOG=$(cat experiments/stage2_k2_kalman_refine/current_k2_ddp2_log.txt)
tail -n 50 "$LOG"
```

查看已经完成的验证和 best score：

```bash
tr '\r' '\n' < "$LOG" | grep -E 'Val Score:|score_best:|K2 DDP2 done' | tail -n 30
ls -lh experiments/stage2_k2_kalman_refine/ckpt_ddp2/
```

若 tmux 会话仍保留，可用 `tmux ls` 查看、`tmux attach -t k2_ddp2` 进入。日志文件中进度条使用回车符，故查询历史时先用 `tr '\r' '\n'` 展开。

## 下一步与交接约束

1. seed 11、23 的 B0/K2 四个任务按 [新种子 README](../stage2_matched_seeds/README.md) 逐个运行；每个任务独占双卡。运行中的 epoch、临时验证分数与 ETA **不写入本静态交接文档**。
2. 四轮全部完成后，对每个 best checkpoint 运行独立 single-view 重评，再用 `summarize_screening.py` 生成三种子 compact CSV/JSON；注意 seed 3 bootstrap 的 delta 是 **B0−K2**，三种子汇总预定方向是 **K2−B0**。
3. K2 的模型效应应根据同协议、多 seed 的配对结果判断；K2b 作为独立 `post−GF` 假设另建分支/目录，仍用 `[3,11,23]`。不覆盖 B0-D2 seed 3 封板结果，也不把当前 K2 换成 K2b。
