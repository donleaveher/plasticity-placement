# 因果可塑性放置实验

该项目实现研究计划的 P0-A/P0-B 模拟管线以及 P0-C 真实模型实验：对同一个 lesson
比较不写入、外部文本记忆、LoRA 参数记忆和二者结合。

模拟后端用于验证环境生成、配对干预、分支隔离和日志协议；P0-C 使用真实 PEFT/LoRA、
oracle external prompt injection、严格动作解析、逐 unit Drive 恢复和配对聚合。

## 环境安装

```bash
uv sync
```

## 运行检查

```bash
uv run pytest
uv run ruff check .
```

## 运行最小试验

```bash
uv run plasticity-pilot --units-per-cell 5 --horizon 20
```

结果写入 `artifacts/pilot.jsonl`。每一行对应一个“实验单元 × 记忆载体”，同一实验单元的四个分支共享未来任务面板。

## LoRA 训练

安装训练依赖：

```bash
uv sync --extra train
```

训练全层 LoRA：

```bash
uv run plasticity-train-lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/examples/lessons.jsonl \
  --output artifacts/adapters/demo-full \
  --layer-band full
```

训练中间层 LoRA：

```bash
uv run plasticity-train-lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/examples/lessons.jsonl \
  --output artifacts/adapters/demo-middle \
  --layer-band middle \
  --use-4bit
```

训练数据采用 JSONL，每行必须包含字符串字段 `prompt` 和 `completion`。训练损失只计算 completion 部分，prompt 词元会被掩码。Colab 上可用 `uv sync --extra train --extra colab` 安装量化训练依赖。

## LoRA 写入评估

对同一组探针依次运行基础模型、加载 LoRA 后的模型，以及禁用 LoRA 后的回滚模型：

```bash
uv run plasticity-evaluate-lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter artifacts/adapters/demo-middle \
  --probes data/examples/probes.jsonl \
  --output artifacts/evaluations/demo-middle \
  --use-4bit
```

评估生成 `probe_results.jsonl` 和 `evaluation_summary.json`，分别保存逐探针输出以及按阶段、类别汇总的准确率。确定性解码保证回滚输出可以与基础模型逐项比较。

## P0-C 真实四载体实验

先编译 frozen lesson bank：

```bash
uv run plasticity-p0c prepare --output artifacts/p0c-smoke
```

在 GPU 环境运行 2-lesson smoke：

```bash
uv run plasticity-p0c run \
  --output artifacts/p0c-smoke \
  --tier smoke \
  --use-4bit
```

运行完成后聚合结果：

```bash
uv run plasticity-p0c aggregate --output artifacts/p0c-smoke
```

`manifest.json` 中所有 adapter unit 必须为 `verified`，且 rollback exact match 为 1。
`aggregate` 会再次强制检查完整 seed 集合、arm/probe 数量、run/model revision 和
adapter hash；部分运行不能生成正式 summary。
随后在 development lessons 上校准配置：

```bash
uv run plasticity-p0c calibrate \
  --output artifacts/p0c-calibration \
  --use-4bit
```

该命令按预注册 successive-halving 方案训练 60 个 development adapters，并生成
`calibration_report.json`。校准专用路径只执行用于选择的 N/P/rollback，并跨候选复用
No-write 输出，不重复运行 E/B。只有 `selected_config` 非空时，才使用新的输出目录运行
pilot：

```bash
uv run plasticity-p0c run \
  --output artifacts/p0c-pilot \
  --tier pilot \
  --use-4bit \
  --calibration-config artifacts/p0c-calibration/calibration_report.json
```

Pilot 和 Confirmatory tier 都强制要求 provenance-compatible calibration report；
Confirmatory 自动要求三个训练种子。Colab 入口为
[`notebooks/p0c_colab.ipynb`](notebooks/p0c_colab.ipynb)。

Colab 首次运行会把当前远程分支解析成 commit SHA 并写入 Drive；后续重连使用同一
detached commit。若代码、GPU、CUDA 或关键依赖发生变化，原输出目录会拒绝继续混跑。
失败 unit 的产物保持不可变，不会在下次执行时自动重新训练覆盖。

## P0-D LoRA layer-locus

P0-D 从一个完整 verified 的 P0-C Confirmatory `manifest.json` 冻结 24-lesson cohort、
模型 revision、训练 seeds 和校准配置。Band scan 固定运行
`full/early/middle/late × 24 lessons × 3 seeds`，即 288 个 adapter units；No-write 和
External 每条 lesson 只运行一次。

先检查解析后的条件矩阵：

```bash
uv run plasticity-p0d plan \
  --output artifacts/p0d-band-scan \
  --source-manifest /path/to/p0c-confirmatory/manifest.json \
  --stage band_scan
```

运行并聚合：

```bash
uv run plasticity-p0d run \
  --output artifacts/p0d-band-scan \
  --source-manifest /path/to/p0c-confirmatory/manifest.json \
  --stage band_scan

uv run plasticity-p0d aggregate \
  --output artifacts/p0d-band-scan \
  --bootstrap-samples 10000
```

P0-D 使用独立的 condition-aware manifest，unit identity 为
`condition × lesson × seed`。Aggregate 会拒绝 partial/mixed matrix，先在 lesson 内
等权汇总 seeds，再执行 lesson-clustered paired bootstrap、retention/locus gates 和
`fact_mapping × act_k2` secondary diagnostic。它不会自动选择或启动 narrow scan。

Colab 入口：

- [`notebooks/p0d_lora_locus/p0d_band_scan_colab.ipynb`](notebooks/p0d_lora_locus/p0d_band_scan_colab.ipynb)
- [`notebooks/p0d_lora_locus/p0d_narrow_scan_colab.ipynb`](notebooks/p0d_lora_locus/p0d_narrow_scan_colab.ipynb)

详细运行和 frozen narrow-condition 规则见
[`notebooks/p0d_lora_locus/README.md`](notebooks/p0d_lora_locus/README.md)。

## P0-D2 参数预算匹配

P0-D2 用独立的 `plasticity-p0d2` CLI 检验 P0-D1 的局部 band 劣势是否来自较少的
trainable parameters。冻结矩阵同时重跑 Full、三个原始 band 和三个 matched band：

```bash
uv run plasticity-p0d2 plan \
  --output artifacts/p0d2-budget-match \
  --source-manifest /path/to/p0c-confirmatory/manifest.json

uv run plasticity-p0d2 run \
  --output artifacts/p0d2-budget-match \
  --source-manifest /path/to/p0c-confirmatory/manifest.json

uv run plasticity-p0d2 aggregate \
  --output artifacts/p0d2-budget-match \
  --bootstrap-samples 10000
```

正式规模为 7 conditions × 24 lessons × 3 seeds = 504 adapter units。Matched 条件从
P0-C calibration 的 rank/alpha 冻结为三倍值，并保持 alpha/rank；runner 对每个
matched unit 相对 `full-base` 验证实际 trainable parameter count，默认容差为 1%。

Colab 入口为
[`notebooks/p0d2_budget_match/p0d2_budget_match_colab.ipynb`](notebooks/p0d2_budget_match/p0d2_budget_match_colab.ipynb)；
设计、决策门和恢复规则见
[`docs/p0d2-budget-match-protocol.md`](docs/p0d2-budget-match-protocol.md)。

## P0-D2H 困难 Probe 压力测试

P0-D2H 不重新训练，只读复用完整 verified P0-D2 run 中的 `full-base` 和三个 matched
conditions。它对每个 lesson 编译 16 个困难 probes，覆盖多干扰绑定、冲突堆叠、
条件路由和长上下文稀释：

```bash
uv run plasticity-p0d2h plan \
  --output artifacts/p0d2h-hard-probe \
  --source-manifest /path/to/p0d2-budget-match/manifest.json

uv run plasticity-p0d2h run \
  --output artifacts/p0d2h-hard-probe \
  --source-manifest /path/to/p0d2-budget-match/manifest.json

uv run plasticity-p0d2h aggregate \
  --output artifacts/p0d2h-hard-probe \
  --bootstrap-samples 10000
```

冻结规模为 288 个 read-only adapter units 和 5,376 行结果。Aggregate 同时报告困难
准确率、相对原 P0-D2 TG 的退化、late-vs-full 的配对韧性以及 seed/lesson type/category
稳定性。它不会自动启动窄扫描或多映射训练。

Colab 入口为
[`notebooks/p0d2_hard_probe/p0d2_hard_probe_colab.ipynb`](notebooks/p0d2_hard_probe/p0d2_hard_probe_colab.ipynb)；
预注册规则见
[`docs/p0d2h-hard-probe-stress-protocol.md`](docs/p0d2h-hard-probe-stress-protocol.md)。

## 项目结构

```text
├── data/examples/             # 小型示例数据
├── docs/                      # 项目设计文档
├── notebooks/                 # Colab 启动与结果分析
├── src/plasticity_placement/
│   ├── simulation/            # 可控环境与四载体实验
│   ├── evaluation/            # Base、LoRA 与回滚探针
│   ├── training/              # LoRA 训练后端
│   ├── p0c/                   # 真实四载体编译、运行、恢复和聚合
│   ├── p0d/                   # LoRA layer-locus 条件、恢复和聚合
│   ├── p0d2/                  # 参数预算匹配、容量救援和决策门
│   └── p0d2h/                 # 只读困难 probe 编译、压力评估和诊断门
├── tests/                     # 单元测试
└── artifacts/                 # 生成结果，不纳入版本控制
```

完整边界说明见 [`docs/project-layout.md`](docs/project-layout.md)。

第一个真实载体实验（P0-C）的 Colab 设计、lesson/probe 规范、运行分级、统计分析和
go/no-go 条件见 [`docs/p0c-colab-protocol.md`](docs/p0c-colab-protocol.md)。
24-lesson × 3-seed Confirmatory 主结果、当前可支持的主张以及“先做 LoRA layer-locus、
再以 GRPO/RLVR 补充”的后续路线见
[`docs/p0c-confirmatory-results-and-next-experiments.md`](docs/p0c-confirmatory-results-and-next-experiments.md)。
