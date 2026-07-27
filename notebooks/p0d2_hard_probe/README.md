# P0-D2H 困难 Probe 压力测试 Colab

本目录提供独立、只读的 P0-D2H 入口：

- `p0d2_hard_probe_colab.ipynb`
- `build_p0d2h_notebook.py`

P0-D2H 只读取一个完整 verified 的 P0-D2 budget-match run。它不会重新训练
adapter，不会修改 P0-C、P0-D1 或 P0-D2 产物，也不会自动启动 Narrow Scan 或
多映射训练。

输出使用独立目录：

```text
/content/drive/MyDrive/plasticity-p0d/hard-probe/v1/pipelines/
```

## 冻结规模

- 条件：`full-base`、`early-matched`、`middle-matched`、`late-matched`
- base anchors：`no_write`、`external`
- 24 lessons × 3 seeds
- 16 hard probes/lesson
- 288 read-only adapter units
- 5,376 expected rows

四类 probe 为 `binding_decoys`、`conflict_stack`、`conditional_route` 和
`long_context`。所有结果在真实 Colab aggregate 产生前均为 TBD。

## 执行

1. 选择 GPU runtime；
2. 设置实际 `P0D2_PIPELINE_ATTEMPT` 与 `P0D2_BUDGET_MATCH_ATTEMPT`；
3. 保持 `RUN_HARD_PROBE = False` 完成 preflight；
4. 核对唯一 source manifest、四条件、288 units、5,376 rows 和独立 Drive 路径；
5. 设置 `RUN_HARD_PROBE = True`，从顶部重新运行；
6. 审阅 `summary.json`、`main_table.md` 和 `next_stage_decision.json`。

正常断线会在相同 attempt 中跳过 verified units。任何 `failed` unit 都应保留，并把
`HARD_PROBE_ATTEMPT` 从 `a1` 增加到 `a2`。

## 生成规则

```bash
uv run python notebooks/p0d2_hard_probe/build_p0d2h_notebook.py
```

不要直接编辑生成的 `.ipynb`。
