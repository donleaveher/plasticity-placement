# 因果可塑性放置实验

该项目实现研究计划的 P0-A/P0-B 最小实验管线：对同一个实验单元克隆四个分支，比较不写入、外部记忆、参数记忆和混合记忆。

当前的参数记忆是一个确定性的模拟后端，用于先验证环境生成、配对干预、分支隔离、回滚和日志协议。真实 LoRA 后端将在这套接口稳定后接入。

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

## 项目结构

```text
├── data/examples/             # 小型示例数据
├── docs/                      # 项目设计文档
├── notebooks/                 # Colab 启动与结果分析
├── src/plasticity_placement/
│   ├── simulation/            # 可控环境与四载体实验
│   ├── evaluation/            # Base、LoRA 与回滚探针
│   └── training/              # LoRA 训练后端
├── tests/                     # 单元测试
└── artifacts/                 # 生成结果，不纳入版本控制
```

完整边界说明见 [`docs/project-layout.md`](docs/project-layout.md)。
