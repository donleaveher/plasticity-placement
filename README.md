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

Colab 默认执行正式评估：metadata preflight 与 formal run 分块显示；formal run 先完成
一次带进度的 504-adapter CPU/Drive 完整性校验，再让一份常驻 CUDA 的 base model
顺序切换 288 个只读 adapter。若只需检查路径和冻结矩阵，可把
`RUN_FORMAL_EVALUATION` 设为 `False`。

Colab 入口为
[`notebooks/p0d2_hard_probe/p0d2_hard_probe_colab.ipynb`](notebooks/p0d2_hard_probe/p0d2_hard_probe_colab.ipynb)；
预注册规则见
[`docs/p0d2h-hard-probe-stress-protocol.md`](docs/p0d2h-hard-probe-stress-protocol.md)。

## P0-D2H-CRD 路由/检索拆分

P0-D2H-CRD 是 base-only 诊断，不训练或加载 adapter。它只接受已冻结的
P0-D2H-CAL-FC formal source，将 1.5B canary 的条件路由任务拆成
`route_only`、`retrieval_only` 和 `combined`：

```bash
uv run plasticity-p0d2hcrd plan \
  --output artifacts/p0d2h-route-decomposition \
  --source-manifest /path/to/p0d2h-forced-choice/manifest.json

uv run plasticity-p0d2hcrd audit \
  --output artifacts/p0d2h-route-decomposition \
  --source-manifest /path/to/p0d2h-forced-choice/manifest.json

uv run plasticity-p0d2hcrd run \
  --output artifacts/p0d2h-route-decomposition \
  --source-manifest /path/to/p0d2h-forced-choice/manifest.json

uv run plasticity-p0d2hcrd aggregate \
  --output artifacts/p0d2h-route-decomposition
```

冻结规模为 24 个原子 lesson units、1,536 decisions 和 5,376 candidate
sequences。所有输出 status 只定位 routing、retrieval 或 composition bottleneck；
无论结果如何，均不授权训练、adapter scan 或自动下一阶段。

Colab 入口为
[`notebooks/p0d2h_route_decomposition/p0d2h_route_decomposition_colab.ipynb`](notebooks/p0d2h_route_decomposition/p0d2h_route_decomposition_colab.ipynb)；
冻结协议见
[`docs/p0d2h-route-retrieval-decomposition-protocol.md`](docs/p0d2h-route-retrieval-decomposition-protocol.md)。

## P0-D2H-RR 路由修复 LoRA pilot

P0-D2H-RR 是 P0-D2H-CRD 得到 `route_bottleneck_supported` 后的独立、预注册
修复实验。它生成 480 条 route-only/route-and-copy counterfactual 训练样本和
96 条 group-disjoint dev 样本，只允许一个固定的 rank-8 LoRA 配置。`plan`
不会授权训练；`train` 必须先采纳由输出目录之外签发、绑定精确
`preregistration_sha256` 的批准文件。

```bash
uv run plasticity-p0d2hrr plan \
  --output artifacts/p0d2h-route-remediation \
  --source-manifest /path/to/p0d2h-forced-choice/manifest.json \
  --spec configs/p0d2hrr-route-remediation-pilot-v1.json

uv run plasticity-p0d2hrr status \
  --output artifacts/p0d2h-route-remediation
```

授权后，`train`、`evaluate` 和 `aggregate` 必须分步运行。正式评测完整重跑
scale-canary forced-choice 与 CRD 两个 locked panel，并同时要求原 calibration
gate、`conditional_route >= 0.75`、route 改善以及 retrieval/combined 无退化。
即使通过，也只进入人工 1/4/8 training-complexity review，不会自动启动 scan
或 RLVR。Colab 入口为
[`notebooks/p0d2h_route_remediation/p0d2h_route_remediation_colab.ipynb`](notebooks/p0d2h_route_remediation/p0d2h_route_remediation_colab.ipynb)；
其默认模式只运行代码/源工件锁定与 CPU preflight，授权采纳、训练、locked
evaluation 和 aggregate 均需分别显式开启。完整协议见
[`docs/p0d2h-route-remediation-lora-pilot-protocol.md`](docs/p0d2h-route-remediation-lora-pilot-protocol.md)。

### P0 paired follow-up audit

完成的 RR run 只能通过独立、只读的 P0 audit 与原 NF4 base CRD 做逐行比较：

```bash
uv run plasticity-p0d2hrr audit-p0 \
  --output /path/to/verified-route-remediation \
  --base-crd-output /path/to/verified-nf4-base-crd \
  --analysis-output /independent/path/to/p0-analysis \
  --experiment-code-revision-lock /path/to/rr-pipeline/code_revision.txt \
  --base-crd-code-revision-lock /path/to/base-crd-pipeline/code_revision.txt \
  --historical-preregistration-sha256 ae456ee26d9ac12c1ba35c0372574d7ae7f276eabba9800ae9af3360f6868986 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260730
```

该命令验证完整 provenance，配对 1,152 条 forced-choice 和 1,536 条 CRD
decisions，输出四格转移、signed expected-candidate margin、lesson-clustered
bootstrap、作为次要敏感性检查的 exact McNemar/Holm、保守 tie bounds 和
route/retrieval/combined matched-cell 分析。base 与 adapter 来自两个历史运行，
因此这里只报告配对差异和关联，不声称 adapter 因果效应；同 session 因果重评分
需要另行预注册。它不会修改 source、改变 gate 或授权训练。独立 CPU-only Colab 为
[`notebooks/p0d2h_route_remediation_p0_audit/p0d2h_route_remediation_p0_audit_colab.ipynb`](notebooks/p0d2h_route_remediation_p0_audit/p0d2h_route_remediation_p0_audit_colab.ipynb)。
该审计只能验证当前 artifact graph；若同一路径曾报告不同 preregistration hash，
仍需外部 append-only 日志解释，不能由当前文件自洽性自动消除。

### Same-runtime 结论与 CPR-v1

RR1 的单模型对象 OFF→ON 复评已完成。OFF replay sentinel 精确通过；adapter 将
`route_only` 从 `0.5052` 提升到 `0.6458`，但 external `conditional_route` 仅从
`0.6354` 到 `0.6562`，同时 `combined` 从 `0.7904` 降到 `0.7305`，其配对
lesson-cluster 95% CI 为 `[-0.1042, -0.0182]`。retrieval-only 保持 `1.0000`。
因此 RR1 失败，历史 gate 不变，1/4/8 继续冻结。

P0-D2H-CPR-v1 是一个新的、独立授权的 composition-preserving remediation：从
干净 base 创建 adapter，不 resume RR1；训练 bank 对 route-only、retrieval-only、
combined 三类任务等权 rehearsal，并在每组内平衡 target slot、slot order 与 answer
panel order。固定配置只训练一次、只保留 final checkpoint，然后在原 locked 96 条
conditional-route 和完整 1,536 条 CRD 上做同-runtime OFF→ON 资格复评。

```bash
uv run plasticity-p0d2hcpr plan \
  --output /new/path/composition-remediation-cpr1 \
  --source-output /path/to/verified-route-remediation-rr1 \
  --same-runtime-summary /path/to/same-runtime-sr1/summary.json \
  --source-code-revision-lock /path/to/rr-pipeline/code_revision.txt \
  --spec configs/p0d2hcpr-composition-preserving-remediation-v1.json
```

通过 CPR-specific checks 也只会产生 `reading_qualification_candidate_review_required`；
它仍需人工 review 和 formal gate recheck，不会自动授权 1/4/8。Colab 入口见
[`notebooks/p0d2h_composition_preserving_remediation/p0d2h_composition_preserving_remediation_colab.ipynb`](notebooks/p0d2h_composition_preserving_remediation/p0d2h_composition_preserving_remediation_colab.ipynb)，
也可[直接在 Google Colab 打开](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_composition_preserving_remediation/p0d2h_composition_preserving_remediation_colab.ipynb)。
协议见
[`docs/p0d2h-composition-preserving-remediation-protocol.md`](docs/p0d2h-composition-preserving-remediation-protocol.md)。

CPR-v1 的 cpr2 已完成单次训练；第一次 q1 资格复评在完成全部评分后因分析字段契约
错误、且在结果落盘前终止，因此没有可解释的 q1 决策。一次性、同 adapter、同 locked
panel 的 q2 恢复入口见
[`notebooks/p0d2h_cpr_qualification_recovery/p0d2h_cpr_qualification_recovery_colab.ipynb`](notebooks/p0d2h_cpr_qualification_recovery/p0d2h_cpr_qualification_recovery_colab.ipynb)，
也可[直接在 Google Colab 打开](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_cpr_qualification_recovery/p0d2h_cpr_qualification_recovery_colab.ipynb)。
该恢复不重训、不调参且不授权 1/4/8。

q2 进一步显示 cpr2 在 route-only 上达到 `0.9922`，却使 external
conditional-route 从 `0.6667` 降至 `0.5625`。为区分 route grammar、slot lexicon、
external payload 与 route-to-action composition，项目新增一个冻结 adapter、只推理的
`2×2×2` route-transfer bridge audit。Colab 入口见
[`notebooks/p0d2h_route_transfer_bridge/p0d2h_route_transfer_bridge_colab.ipynb`](notebooks/p0d2h_route_transfer_bridge/p0d2h_route_transfer_bridge_colab.ipynb)，
也可[直接在 Google Colab 打开](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_transfer_bridge/p0d2h_route_transfer_bridge_colab.ipynb)。
协议见
[`docs/p0d2h-route-transfer-bridge-protocol.md`](docs/p0d2h-route-transfer-bridge-protocol.md)。
该审计不能重开 CPR-v1、授权训练或授权 1/4/8。

RTB 已通过一次外部授权的 zero-artifact identical retry 完成。严格 zero-tie gate
因 26 个 primary tie 触发 `scoring_integrity_failed`，但全部 tie 都是
expected-compatible、margin 为零；最坏 tie resolution 下八个 slot-readout cell 的
adapter 增益仍全部为正。完全 external slot 从 `0.5208` 升至 `0.9896`，direct action
却从 `0.6354` 降至 `0.5521`，而 forced-slot action 保持在 `0.9271`。

route-state handoff audit 已完成。保守 handoff rescue 为
`+0.1458 [0.0417, 0.2396]`，predicted chain 与 oracle 在 adapter ON 下均为
`0.9271`；但 oracle-minus-wrong specificity 仅为
`0.1354 [0.0729, 0.2083]`，低于预注册 `0.20`，因此结论为
`handoff_not_supported`。原 wrong slot 是没有唯一有效替代映射的 advisory decoy。
原 Colab 入口见
[`notebooks/p0d2h_route_state_handoff/p0d2h_route_state_handoff_colab.ipynb`](notebooks/p0d2h_route_state_handoff/p0d2h_route_state_handoff_colab.ipynb)，
也可[直接在 Google Colab 打开](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_route_state_handoff/p0d2h_route_state_handoff_colab.ipynb)。
协议见
[`docs/p0d2h-route-state-handoff-protocol.md`](docs/p0d2h-route-state-handoff-protocol.md)。
该审计不改变 RTB decision，训练和 1/4/8 仍未授权。

receipt/action binding audit 已完成，结论为 `binding_not_supported`。adapter ON 的
selected-binding accuracy 为 `0.6406 [0.6042, 0.6771]`，未通过 `0.75` accuracy gate；
canonical/swapped 分别为 `0.6354/0.6458`。causal specificity
`0.2812 [0.2083, 0.3542]` 通过，且 OFF→ON 净增益为 `+0.0625`，说明存在局部因果
响应，但尚未形成稳定的 receipt/action binding。原 Colab 入口见
[`notebooks/p0d2h_receipt_action_binding/p0d2h_receipt_action_binding_colab.ipynb`](notebooks/p0d2h_receipt_action_binding/p0d2h_receipt_action_binding_colab.ipynb)，
也可[直接在 Google Colab 打开](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_receipt_action_binding/p0d2h_receipt_action_binding_colab.ipynb)。
协议见
[`docs/p0d2h-receipt-action-binding-protocol.md`](docs/p0d2h-receipt-action-binding-protocol.md)。
该审计不重分类 RSH、不训练且不授权 1/4/8。

RAB error-topology audit 已完成。其 cell transitions 为 `C→C=90`、`C→W=21`、
`W→C=33`、`W→W=48`；`single_action_locked` 从 `30/48` 降至 `10/48`，但
`receipt_invariant` 从 `3/48` 升至 `13/48`，`partial_mixed` 从 `13/48` 升至
`23/48`。selected-minus-counterfactual margin 的 ON−OFF 均值为
`+0.8145 [0.6907, 0.9251]`。这说明 adapter 打破了大量全局 action locking，但尚未形成
稳定的四格 binding。原 Colab 入口见
[`notebooks/p0d2h_rab_error_topology/p0d2h_rab_error_topology_colab.ipynb`](notebooks/p0d2h_rab_error_topology/p0d2h_rab_error_topology_colab.ipynb)，
也可[直接在 Google Colab 打开](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_error_topology/p0d2h_rab_error_topology_colab.ipynb)。
协议见
[`docs/p0d2h-rab-error-topology-protocol.md`](docs/p0d2h-rab-error-topology-protocol.md)。
该诊断不加载模型、不推理、不训练、不改变 RAB decision，且不授权 1/4/8。

下一步 CPU-only localization reader 将完整读取该 RABD 的 summary、factor slices、192
条 cell records 和 48 条 unit records，定位 21 个 `C→W`、48 个 `W→W`、taxonomy
migrations 与 transition-conditioned margins。Colab 入口见
[`notebooks/p0d2h_rab_error_localization/p0d2h_rab_error_localization_colab.ipynb`](notebooks/p0d2h_rab_error_localization/p0d2h_rab_error_localization_colab.ipynb)，
也可[直接在 Google Colab 打开](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_rab_error_localization/p0d2h_rab_error_localization_colab.ipynb)。
协议见
[`docs/p0d2h-rab-error-localization-protocol.md`](docs/p0d2h-rab-error-localization-protocol.md)。
该 reader 只做 descriptive localization，不改变历史状态，也不授权训练或 1/4/8。

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
│   ├── p0d2h/                 # 只读困难 probe 编译、压力评估和诊断门
│   ├── p0d2hc/                # Base-only oracle/scale calibration
│   ├── p0d2hfc/               # Full-string forced-choice calibration
│   ├── p0d2hcrd/              # Base-only routing/retrieval/composition 拆分
│   ├── p0d2hrr/               # Authorized route-remediation LoRA pilot
│   ├── p0d2hcpr/              # Composition-preserving remediation 与资格复评
│   ├── p0d2hrtb/              # Frozen-adapter route-transfer bridge audit
│   ├── p0d2hrsh/              # Route-state handoff audit
│   ├── p0d2hrab/              # Two-valid-slot receipt/action binding audit
│   ├── p0d2hrabd/             # CPU-only RAB paired error-topology diagnostic
│   └── p0d2hrabdl/            # CPU-only RAB error-localization reader
├── tests/                     # 单元测试
└── artifacts/                 # 生成结果，不纳入版本控制
```

完整边界说明见 [`docs/project-layout.md`](docs/project-layout.md)。

第一个真实载体实验（P0-C）的 Colab 设计、lesson/probe 规范、运行分级、统计分析和
go/no-go 条件见 [`docs/p0c-colab-protocol.md`](docs/p0c-colab-protocol.md)。
24-lesson × 3-seed Confirmatory 主结果、当前可支持的主张以及“先做 LoRA layer-locus、
再以 GRPO/RLVR 补充”的后续路线见
[`docs/p0c-confirmatory-results-and-next-experiments.md`](docs/p0c-confirmatory-results-and-next-experiments.md)。
