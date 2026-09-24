# GF-Transformer / K2 交接记录

更新：2026-09-24 13:54 UTC。项目目录：`/workspace/GF-Transformer`。以下“当前进度”是写文档时的快照，后续以实时日志为准。

## 当前状态与分支

- 稳定 Baseline-B 分支：`fix/stage1-ckpt-isolation`，最后相关提交 `c566247`。
- K2 单卡准备：`exp/k2-gf-kalman-refine`，提交 `60403ca`。
- **正在使用的双卡 K2 分支**：`exp/k2-gf-kalman-refine-ddp2`；训练代码提交 `2031786`。训练由用户在 tmux 会话 `k2_ddp2` 中手动启动，切断 SSH 不会终止 tmux 内进程。
- 当前日志路径记录在 `current_k2_ddp2_log.txt`，本次为 `logs/stage2_k2_ddp2_20260924_133257.log`。
- 写文档时 epoch 0 已完成 3,351 次更新及全部 917 张验证，epoch 1 已开始。首次验证：F1b/建筑 Dice `0.8720`、F1d `0.3835`、F1s `0.5300`，minor F1 `0.1573`。这是**第一个 epoch 的中间结果，不是最终 K2 指标**。
- 首个 checkpoint 已生成于 `ckpt_ddp2/GFformer_cls_3_k2_ddp2_best14`。训练入口拒绝非空 checkpoint 目录，**不要在当前训练未结束时再次执行启动脚本**。

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

## 当前 K2：damage-change guided refinement

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

双卡短测稳态约 `0.319 秒/更新`；按每 epoch `3,351` 次更新与 Baseline-B 约 `51` 分钟的总验证耗时推算，中心估计约 `15.7` 小时，实际可按 `15–18` 小时规划。双卡通信、数据读取和 24 步样本的局限使这个值不是保证值。虽然全局 batch 与更新数一致，每卡 BatchNorm 统计量不同，双卡 K2 与单卡 Baseline-B 的训练轨迹不会逐项相同。

## 遇到的问题及处理

| 问题 | 处理 / 结论 |
| --- | --- |
| Stage1 旧 checkpoint 与 Stage2 旧输出混用 | 冻结并校验正确 Stage1 checkpoint；Stage2/Baseline-B/K2 各用独立 checkpoint 目录，legacy `tune_weight/` 不再用于正式训练 |
| 旧 localization mask 与 encoder 键名不匹配 | 重新生成并审计全部 9,168 张 mask；使用显式 encoder 权重映射，coverage `100%` |
| FP32 全分辨率验证 batch=4 OOM | 改为 val batch=1，固定 917 张和原 metric；不改变评估数值 |
| Baseline-A 的 accum=8 与按 epoch 衰减组合 | 运行 batch=4、accum=1 的 Baseline-B，并冻结其 single-view 结果 |
| Baseline-B 的 4-way TTA 未提升 | 报告 TTA 结果，K2 继续采用 single-view 作公平比较 |
| K2 新模块插入位置改变原 decoder 随机初始化 | 将 Kalman 参数构造放在原模型参数构造之后；相同种子下核对所有原有参数逐项相同 |
| 单卡 K2 入口拒绝双卡；多 rank 验证可能重复执行/写 checkpoint | 独立实现 `train_k2_ddp2.py`；只由 rank 0 验证和保存，其他 rank 等待 barrier；双卡预检覆盖此链路 |
| 双卡默认 sampler 可能补齐样本，使更新数变化 | 显式 `drop_last=True`，固定 3,351 更新/epoch |
| VS Code/Codex 在非 Git 大目录搜索导致 CPU 高 | 只打开 `/workspace` 后 CPU 恢复；与训练计算无关 |
| 终端缺少 `rg` | 查询命令使用系统自带 `grep`；空 checkpoint 在 epoch 0 首次验证结束前属正常现象 |

## 监控正在运行的训练

在新终端执行（每个新 shell 都要重新读取 `LOG`）：

```bash
cd /workspace/GF-Transformer
LOG=$(cat experiments/stage2_k2_kalman_refine/current_k2_ddp2_log.txt)
tail -f "$LOG"
```

查看已经完成的验证和 best score：

```bash
tr '\r' '\n' < "$LOG" | grep -E 'Val Score:|score_best:|K2 DDP2 done' | tail -n 30
ls -lh experiments/stage2_k2_kalman_refine/ckpt_ddp2/
```

查看 tmux：`tmux ls`；重新进入：`tmux attach -t k2_ddp2`；暂时离开：按 `Ctrl-b` 后按 `d`。日志文件中进度条使用回车符，故查询历史时先用 `tr '\r' '\n'` 展开。

## 训练结束后的待办

1. 确认 `K2 DDP2 done`、50 个 epoch 完成、best checkpoint 存在；记录 checkpoint SHA256、最佳 epoch、日志时间与最终耗时。
2. 用**同一固定 917 validation split、同一 localization masks、原 `train_segformer_cls.validate`、single-view** 对 best checkpoint 独立重评估，冻结 F1b/F1d/F1s 和 F1_0–F1_3，再与 Baseline-B 比较。当前没有为 K2 best checkpoint 编写独立的完整重评估脚本；不要把一张验证图的 smoke 分数当正式结果。
3. 重点观察 minor F1 `0.4917` 基线，同时报告整体 F1s。若后续研究 `post−GF`，将其作为独立 **K2b** 实验，不能覆盖本次 K2 权重、日志或结果。
