# P0-D2 参数预算匹配 Colab

本目录提供独立的 P0-D2 入口：

- `p0d2_budget_match_colab.ipynb`
- `build_p0d2_notebook.py`

它只读取完整 verified 的 P0-C Confirmatory manifest，不修改 P0-C 或 P0-D1 产物，并写入：

```text
/content/drive/MyDrive/plasticity-p0d/budget-match/v1/pipelines/
```

## 冻结矩阵

正式矩阵包含：

1. `full-base`
2. `early-base`
3. `middle-base`
4. `late-base`
5. `early-matched`
6. `middle-matched`
7. `late-matched`

Base 条件使用 P0-C calibration 的 rank/alpha；Matched 条件使用三倍 rank/alpha，
保持 alpha/rank 不变。运行时同时检查名义 layer × rank 预算和相对 `full-base`
实际 trainable parameter count，默认容差为 1%。

总规模为 504 adapter units；在 26-probe panel 下正式聚合必须包含 27,456 rows。

## 执行顺序

1. 选择 GPU runtime；
2. 设置真实的 `P0C_PIPELINE_ATTEMPT` 与 `P0C_CONFIRMATORY_ATTEMPT`；
3. 建议把 `REQUESTED_CODE_REVISION` 固定为本次实现提交的完整 SHA；
4. 保持 `RUN_BUDGET_MATCH = False` 运行 preflight；
5. 核对 7 个条件、504 units、27,456 rows、resolved layers/rank/alpha 和 Drive 路径；
6. 设置 `RUN_BUDGET_MATCH = True`，从顶部重新运行；
7. 审阅 `summary.json`、`main_table.md`、budget validation、gates 和
   `next_stage_candidates.json`。

Notebook 不会自动启动 Narrow Scan。任何 `failed` unit 都应保留原 attempt，并把
`BUDGET_MATCH_ATTEMPT` 从 `a1` 增加到 `a2`。

## 生成规则

修改编排代码后运行：

```bash
uv run python notebooks/p0d2_budget_match/build_p0d2_notebook.py
```

不要直接编辑生成的 `.ipynb`。
