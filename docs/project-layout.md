# 项目结构

```text
plasticity-placement/
├── data/
│   └── examples/              # 可提交的小型示例数据
├── docs/                      # 设计与运行文档
├── notebooks/                 # Colab 入口和交互式分析
├── src/plasticity_placement/
│   ├── simulation/            # 可控环境、记忆载体和 P0 模拟实验
│   ├── evaluation/            # LoRA 写入、迁移、干扰和回滚评估
│   ├── training/              # PEFT/LoRA 数据、配置与训练后端
│   ├── p0c/                   # P0-C lesson compiler、真实四臂、恢复和聚合
│   └── p0d/                   # P0-D layer-locus matrix、恢复、gates 和聚合
├── tests/                     # 不依赖大模型下载的单元测试
├── artifacts/                 # 本地结果、适配器和检查点；不提交版本库
├── pyproject.toml
└── uv.lock
```

## 边界约定

- `simulation/` 不导入 PyTorch、Transformers 或 PEFT，默认 `uv sync` 后即可运行。
- `training/` 只在调用训练入口时加载重型依赖。
- `p0c/` 的 compiler、manifest 和 analysis 保持轻量；modeling/runtime 延迟导入重型依赖。
- `p0d/` 只读 verified P0-C Confirmatory manifest，复用 compiler/evaluator/trainer，
  但维护独立的 condition-aware manifest、raw/adapter 目录和 aggregate。
- `notebooks/` 不保存核心业务逻辑，只调用已安装的命令行入口。
- `data/examples/` 仅保存可公开、体积小且可复现的输入样例。
- `artifacts/` 保存生成结果，通过 `.gitignore` 排除。
