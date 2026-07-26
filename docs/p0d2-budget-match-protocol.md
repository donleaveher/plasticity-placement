# P0-D2 参数预算匹配实验协议

- **目标：** 区分 P0-D1 局部 band 劣势中的参数容量因素与跨层覆盖因素
- **状态：** implementation complete / GPU execution pending
- **结果边界：** 本文只冻结设计与代码验收；所有真实 P0-D2 数值均为 TBD

## 1. 来源与不变量

P0-D2 只接受完整 verified 的 P0-C Confirmatory manifest，并冻结：

- 24 个 selected lesson IDs：12 `fact_mapping` + 12 `procedure_recovery`；
- 三个训练 seeds；
- model revision、4-bit/precision regime；
- P0-C calibration 选出的 base rank、alpha、learning rate 和 max steps；
- compiler/probe hashes、training JSONL 和 26-probe panel；
- `q_proj/v_proj` target modules。

P0-D2 不读取 P0-D1 target results 来选择 band、rank 或 modules，也不修改 P0-C/P0-D1
manifest、adapter、raw rows 或 aggregate。

## 2. 冻结七条件矩阵

令 P0-C calibration 配置为 base rank \(r_0\) 和 alpha \(\alpha_0\)：

| Condition | Layers | Rank | Alpha | Budget regime | Primary role |
|---|---|---:|---:|---|---|
| `full-base` | full | \(r_0\) | \(\alpha_0\) | reference full | 主参考 |
| `early-base` | early third | \(r_0\) | \(\alpha_0\) | fixed local | 复现 P0-D1 |
| `middle-base` | middle third | \(r_0\) | \(\alpha_0\) | fixed local | 复现 P0-D1 |
| `late-base` | late third | \(r_0\) | \(\alpha_0\) | fixed local | 复现 P0-D1 |
| `early-matched` | early third | \(3r_0\) | \(3\alpha_0\) | matched to Full | 容量匹配 |
| `middle-matched` | middle third | \(3r_0\) | \(3\alpha_0\) | matched to Full | 容量匹配 |
| `late-matched` | late third | \(3r_0\) | \(3\alpha_0\) | matched to Full | 容量匹配 |

三倍 alpha 保持 alpha/rank 不变。正式 runner 根据实际 `num_hidden_layers` 解析层索引，
并在 GPU 工作前验证 `selected layer count × rank` 的名义预算。对当前 24-layer 来源，
三个 8-layer matched conditions 的名义预算应与 Full 完全相等。

## 3. 规模与运行身份

- 7 LoRA conditions × 24 lessons × 3 seeds = **504 adapter units**；
- No-write/External 每 lesson 各运行一次；
- Parametric/rollback 每 adapter unit 各运行一次；
- 在 26-probe panel 下，完整 aggregate 应含 **27,456 probe rows**。

Run ID 覆盖：

- P0-D2 code SHA 和 `p0d2-config-v1`；
- P0-C source manifest SHA/run ID/calibration SHA；
- selected lesson IDs、seeds、model revision 和 compiler hashes；
- 每个 condition 的 resolved layers、rank、alpha、target modules 和 budget regime；
- retention margin 与实际预算容差。

任一字段改变必须生成新 run/stage attempt。

## 4. 实际参数预算门

名义 layer × rank 相等不是最终验收。每个 matched unit 在训练完成后必须相对相同
lesson/seed 的 `full-base` unit 检查：

\[
\epsilon =
\frac{|P_{\text{matched}}-P_{\text{full}}|}{P_{\text{full}}}
\le 0.01,
\]

其中 \(P\) 是 PEFT 实际报告的 trainable parameter count。Manifest 记录 reference
condition、两侧实际参数量、relative error 和 `within_tolerance`。任一 matched unit
越过容差即进入不可变 `failed` 状态，当前 attempt 不得聚合或覆盖。

## 5. Manifest、恢复与原始产物

P0-D2 使用独立：

- `p0d2-config-v1`
- `p0d2-manifest-v1`
- `p0d2-budget_match-<hash>` run ID

Unit identity 仍为 `condition::lesson::seed-N`，状态为：

```text
pending → training → trained → evaluated → verified
                                  └──────────────→ failed
```

`verified` 和 `failed` 均不可逆。普通 Colab 断线在代码、来源、矩阵、容差和环境一致时
恢复同一 attempt；已有 verified units 零工作跳过。出现 `failed` unit 时保留原目录并
递增 `BUDGET_MATCH_ATTEMPT`。

输出布局：

```text
STAGE_DIR/
├── manifest.json
├── notebook_context.json
├── source_sessions.json
├── environment.json
├── environment_sessions.json
├── compiled/
├── adapters/<condition>/<lesson>/seed-*/
└── results/
    ├── raw/base/<lesson>/base_arms.jsonl
    ├── raw/conditions/<condition>/<lesson>/seed-*/adapter_arms.jsonl
    └── aggregate/
        ├── summary.json
        ├── main_table.md
        └── next_stage_candidates.json
```

## 6. 聚合与 estimands

Aggregate 首先保留 condition × lesson × seed metrics，再在 lesson 内等权汇总 seeds，
最后以 24 lessons 为独立 cluster 做 paired bootstrap。

冻结 contrast families：

1. **replication：** `early/middle/late-base − full-base`；
2. **matched-vs-full：** 三个 matched bands 分别减 `full-base`；
3. **capacity-rescue：** `band-matched − band-base`；
4. **write-gain：** 每个训练 condition 减 No-write；
5. **external anchor：** External 减 No-write。

同时报告 TG、IR、exact、conflict、invalid、seed range、lesson type、
`fact_mapping × act_k2`、trainable parameters、adapter bytes、write time、peak memory、
latency 和 input tokens。

Aggregate 拒绝：

- 缺失/额外的 504-unit matrix；
- 非 verified unit、rollback 不为 1 或 manifest errors；
- condition-specific rank/alpha/modules/selected layers 与 metadata 不一致；
- adapter、training data、training config 或 compiler hashes 改变；
- mixed precision 或同 condition 中混合 trainable parameter count；
- matched unit 的实际 budget validation 缺失、改变或越过容差；
- 缺失、重复或 provenance 不一致的 probe rows。

## 7. 决策门

对 matched band \(b\)，主要 retention estimand 为：

\[
\Delta_b^{TG}=TG(b_{\text{matched}})-TG(\text{full-base}).
\]

- 95% CI 下界 ≥ `−0.05`：该 matched band 保留 Full 收益；
- 95% CI 上界 < `−0.05`：该 matched band 在预算匹配后仍实质落后；
- `band-matched − band-base` CI 下界 > 0：容量增加带来正向 rescue；
- retaining band 还需相对另一个 matched band 的 paired CI 完整高于 0，并通过
  2/3 seed、两个 lesson types 和 leave-largest-out safeguards，才进入人工冻结的
  Narrow Scan。

机器可读状态：

- `manual_freeze_required`
- `sparse_nonunique`
- `capacity_matched_bands_inferior`
- `inconclusive`

Notebook 和 aggregate 都不会自动启动 Narrow Scan。

## 8. Colab

入口：

```text
notebooks/p0d2_budget_match/p0d2_budget_match_colab.ipynb
```

Drive namespace：

```text
/content/drive/MyDrive/plasticity-p0d/budget-match/v1/pipelines/
```

首次保持 `RUN_BUDGET_MATCH = False`，核对 source manifest、code/model revision、
7 conditions、resolved rank/alpha/layers、504 units、27,456 rows、预算容差和 stage
目录；确认后再显式打开运行开关。

## 9. 实现验收

最低验收包括：

1. config 测试冻结七条件矩阵、condition-specific hyperparameters 和 nominal budget；
2. manifest 测试覆盖恢复、不可变终态与 hyperparameter change rejection；
3. fake backend 完成 504 adapters、base arms、rollback 和 27,456 rows；
4. 第二次运行不修改 verified adapter mtime；
5. aggregate 覆盖 replication、matched-vs-full、capacity-rescue、budget gate 和 locus gate；
6. 实际参数 mismatch、partial matrix、adapter/metadata tamper 被拒绝；
7. notebook generator/JSON/AST、独立 Drive namespace 和 run switch 测试；
8. 完整 pytest、Ruff、compileall、CLI help、notebook JSON 和 `git diff --check` 通过。

当前代码实现和 fake backend 验证完成后，状态可更新为
`implementation complete / GPU execution pending`；在真实 Colab aggregate 产生前，
所有 P0-D2 结果保持 TBD。
