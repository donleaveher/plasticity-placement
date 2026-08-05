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
│   ├── p0d/                   # P0-D layer-locus matrix、恢复、gates 和聚合
│   ├── p0d2/                  # P0-D2 budget-match、恢复、容量对比和聚合
│   ├── p0d2h/                 # P0-D2H read-only hard-probe stress diagnostic
│   ├── p0d2hc/                # P0-D2H-CAL base-only oracle/scale calibration
│   ├── p0d2hfc/               # P0-D2H-CAL-FC full-string forced choice
│   ├── p0d2hcrd/              # Base-only route/retrieval decomposition
│   ├── p0d2hrr/               # Completed route-remediation LoRA pilot and audits
│   ├── p0d2hcpr/              # Composition-preserving remediation pilot
│   ├── p0d2hrtb/              # Frozen-adapter route-transfer bridge audit
│   ├── p0d2hrsh/              # Route-state handoff audit
│   └── p0d2hrab/              # Two-valid-slot receipt/action binding audit
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
- `p0d2/` 不修改 P0-D1 frozen semantics；它为每个 condition 冻结 rank/alpha/modules，
  并维护独立 `p0d2-*` config、manifest、run ID、budget validation 和 aggregate。
- `p0d2h/` 只读 verified P0-D2 manifest、summary 和 adapter bundles；它维护独立的
  hard-probe compiler、`p0d2h-*` identity/manifest/raw results 和 diagnostic gates，
  不包含训练入口。
- `p0d2hc/` 只读 verified P0-D2H-R manifest、summary 和 hard-probe bank；它只加载
  base model，运行 `no_write/external/answer_copy_oracle`，不读取 adapter 或提供训练入口；
  其 invalid-output audit 仅重解析 verified raw text，并要求独立输出目录。
- `p0d2hfc/` 只读完整 verified P0-D2H-CAL manifest、2,304 raw rows 及其传递来源；
  它在 CPU preflight 冻结四候选 token 边界，再以 candidate-only `sum_logprob`
  作为 primary endpoint；旧 strict generation 只作为配对 secondary metric，且该
  package 不发现/加载 adapter、不训练、不启动 narrow scan 或下一阶段。
- `p0d2hcrd/` 只接受 exact completed P0-D2H-CAL-FC source，并沿其冻结 hash
  验证 P0-D2H-CAL、P0-D2H-R hard bank 与 P0-D2 compiled bank；它以新
  counterbalanced bank 分别运行两候选 `route_only`、四候选 `retrieval_only` 和
  四候选 `combined`。所有 status 仅用于定位 routing/retrieval/composition，
  `training_complexity_review_eligible` 永远为 false。
- `p0d2hrr/` 只接受同一 exact frozen source，以独立合成 route bank 运行单一
  固定 LoRA pilot。训练前必须采纳绑定 preregistration hash 的外部批准；正式
  aggregate 同时要求原 forced-choice gate 与 route/retrieval/combined guardrails，
  且永不自动启动 1/4/8 scan、narrow scan 或 RLVR。
- `p0d2hcpr/` 绑定 verified RR1 与其 same-runtime 失败证据，从干净 base 运行
  route/retrieval/combined 等权 rehearsal。资格评测在同一模型对象内逐 prompt
  OFF→ON，使用 lesson-cluster CI 与 tie-robust bounds；即使成为 candidate 也不自动
  授权 1/4/8。
- `p0d2hrtb/` 绑定失败的 CPR q2 与 immutable cpr2 adapter，运行 inference-only
  prompt-factor bridge；不训练且不改变 1/4/8 gate。
- `p0d2hrsh/` 绑定 complete RTB/q2 artifacts，先做 CPU replay qualification，再在
  同一 runtime 中评分 frozen slot/direct/receipt-A/receipt-B prompts，并派生
  predicted/oracle/wrong-slot handoff；不重分类 RTB、不训练且不改变 1/4/8 gate。
- `p0d2hrab/` 绑定 completed RSH 与其 upstream artifacts，先生成 oracle/wrong-receipt
  四象限诊断，再以两个有效 verified mappings 交叉 receipt A/B 和 canonical/swapped
  binding；同一 runtime OFF→ON，只推理、不重分类 RSH 且不改变 1/4/8 gate。
- `notebooks/` 不保存核心业务逻辑，只调用已安装的命令行入口。
- `notebooks/p0d2h_route_remediation/` 以生成器维护 Colab；默认仅运行
  preregistration/preflight，并把外部授权采纳、单次训练、locked evaluation
  和 aggregate 分成互斥的显式阶段。
- `data/examples/` 仅保存可公开、体积小且可复现的输入样例。
- `artifacts/` 保存生成结果，通过 `.gitignore` 排除。
