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
│   └── training/              # PEFT/LoRA 数据、配置与训练后端
├── tests/                     # 不依赖大模型下载的单元测试
├── artifacts/                 # 本地结果、适配器和检查点；不提交版本库
├── pyproject.toml
└── uv.lock
```

## 边界约定

- `simulation/` 不导入 PyTorch、Transformers 或 PEFT，默认 `uv sync` 后即可运行。
- `training/` 只在调用训练入口时加载重型依赖。
- `notebooks/` 不保存核心业务逻辑，只调用已安装的命令行入口。
- `data/examples/` 仅保存可公开、体积小且可复现的输入样例。
- `artifacts/` 保存生成结果，通过 `.gitignore` 排除。
