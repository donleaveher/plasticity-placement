# P0-D LoRA Layer-Locus Colab 实验

本目录包含两个彼此独立的 P0-D 入口：

1. `p0d_band_scan_colab.ipynb`：冻结的 `full/early/middle/late` 主矩阵；
2. `p0d_narrow_scan_colab.ipynb`：仅在 band scan 通过进入门后运行显式层/窗口。

两个 notebook 都只读取已完成的 P0-C v4 Confirmatory manifest，并把结果写入独立的
`/content/drive/MyDrive/plasticity-p0d/lora-locus/v1/pipelines/` namespace。它们不会修改
P0-C 的 manifest、raw records、adapters、calibration 或 aggregate。

## 执行顺序

1. 在 Band Scan notebook 中设置正确的 `P0C_PIPELINE_ATTEMPT` 和
   `P0C_CONFIRMATORY_ATTEMPT`；
2. 保持 `RUN_BAND_SCAN = False` 运行 preflight，核对 source manifest、代码 SHA、
   resolved layers、288-unit plan 和 P0-D Drive 目录；
3. 设置 `RUN_BAND_SCAN = True` 完成运行、verified 检查和聚合；
4. 审阅 `summary.json`、`main_table.md`、gates 和
   `next_stage_candidates.json`；
5. 只有 `eligible_locus_conditions` 非空时，才人工冻结
   `frozen-narrow-conditions.json`；
6. 在 Narrow Scan notebook 中先运行 preflight，再显式打开 `RUN_NARROW_SCAN`。

## Narrow condition 文件

文件必须放在当前 P0-D pipeline 根目录：

```text
frozen-narrow-conditions.json
```

示例：

```json
{
  "stage": "narrow_scan",
  "source_band_run_id": "p0d-band_scan-REPLACE_WITH_REVIEWED_RUN",
  "reference_condition_id": "full",
  "conditions": [
    {
      "condition_id": "full",
      "layer_band": "full",
      "explicit_layers": [],
      "parameter_budget_regime": "fixed_local"
    },
    {
      "condition_id": "layer-11",
      "layer_band": "explicit",
      "explicit_layers": [11],
      "parameter_budget_regime": "fixed_local"
    },
    {
      "condition_id": "window-11-12",
      "layer_band": "explicit",
      "explicit_layers": [11, 12],
      "parameter_budget_regime": "fixed_local"
    }
  ]
}
```

必须把层索引和 `source_band_run_id` 替换为审阅后冻结的真实值。Notebook 不会从 band
结果自动挑选层，也不会自动启动 narrow scan。

## 生成规则

两个 notebook 由 `build_p0d_notebooks.py` 生成。修改公共编排逻辑后运行：

```bash
uv run python notebooks/p0d_lora_locus/build_p0d_notebooks.py
```

不要直接编辑生成的 `.ipynb`。
