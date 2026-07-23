# P0-C Colab 实验协议：同一 Lesson 的真实四载体比较

- **版本：** 0.1
- **日期：** 2026-07-23
- **模式：** design + result-template
- **目标会议：** ICLR
- **硬件边界：** 单个 Colab GPU，允许中断后从 Google Drive 恢复
- **结果状态：** 本文只设计实验；所有结果格均为 `TBD`

## 1. 实验定位

P0-C 不是完整的 recurrence–volatility 相图，也不检验最佳 LoRA 层。它只回答：

> 对完全相同的 canonical lesson，外部文本记忆与全层 LoRA 是否形成不同的目标泛化、迁移、局部干扰和回滚行为？

这个实验是后续工作的资格门：

- 如果 External 与 Parametric 行为基本等价，只剩 token/GPU 成本差，停止“载体因果规律”主张；
- 如果两者存在可重复的收益—干扰差异，再进入动态环境实验；
- 只有动态环境中 Parametric 确有优势，才值得扫描写入层位置。

## 2. 预注册问题与假设

### RQ1：有效写入

在不注入 lesson 文本的条件下，LoRA 是否能提高 held-out target generalization？

$$
\Delta_{P,N}^{TG}
=
TG(\text{Parametric})-TG(\text{No-write})
$$

### RQ2：载体差异

Parametric 与 oracle External 是否具有不同的泛化—干扰轮廓？

$$
\Delta_{P,E}^{TG}=TG(P)-TG(E),
\qquad
\Delta_{P,E}^{IR}=IR(P)-IR(E)
$$

其中 $TG$ 越高越好，$IR$ 是相对 base 的无关任务退化，越低越好。

### RQ3：互补或重复

Both 是否提供超过较好单载体的行为价值？

$$
\Delta_{B,\max(P,E)}^{TG}
=TG(B)-\max(TG(P),TG(E))
$$

### RQ4：经验异质性

不同 lesson 是否出现 Parametric 优于 External 和 External 优于 Parametric 的双向差异？

P0-C 只寻找“存在异质性的先兆”，不训练路由器。

### 预注册主假设

- H0a：Parametric 不能改善 held-out target generalization；
- H0b：Parametric 与 External 的行为轮廓处于预设等价区间；
- H0c：所有 lesson 的最佳单载体相同。

任何“程序型更适合参数、事实型更适合外部”的结论均为探索性结果，不能由本轮小样本直接确认。

## 3. Claim–Evidence Matrix

| Claim | Reviewer question | Evidence | Workload | Baselines | Metric | Result |
|---|---|---|---|---|---|---|
| LoRA 写入真实有效 | 是否只是 base 本来就会？ | 独立 screening + held-out probes | nonce lessons | No-write | $TG$, exact accuracy | TBD |
| 两种载体行为不同 | 是否只是成本或检索错误？ | 同 lesson、同 base、oracle retrieval | 全部 lesson | External, Parametric | $TG$, transfer, $IR$ | TBD |
| Both 有/无互补 | 是否只是重复暴露？ | 同 adapter + 同 canonical note | 全部 lesson | P, E, Both | complement gain | TBD |
| 差异可复现 | 是否依赖训练随机性？ | 3 training seeds | confirmatory set | paired arms | seed variance, CI | TBD |
| 比较公平 | LoRA 是否获得了更多表达形式？ | exposure-matched ICL diagnostic | 差异最大的 lessons | ICL-$K$ | $TG$, tokens | TBD |
| 回滚可信 | adapter 关闭后是否恢复？ | base vs disabled-adapter exact output | 全部 probes | Base | exact-match | TBD |

## 4. 模型和默认配置

### 主模型

`Qwen/Qwen2.5-0.5B-Instruct`

选择小模型是为了让真实多 adapter 实验在 Colab 上可反复运行。P0-C 不据此声称跨模型普适性。

### 固定推理设置

- greedy decoding；
- `temperature=0`；
- `do_sample=False`；
- `max_new_tokens=8`；
- 所有 arm 使用同一 chat template；
- 同一 probe 的 system/user 指令除 memory block 外逐字相同；
- 首轮动作输出限制为一个 JSON 字段或一个动作 token；
- 生成与解析失败单独计为 invalid，不可静默按错误动作替代。

### 固定 LoRA 结构

- layer band：`full`；
- target modules：`q_proj`, `v_proj`；
- dropout：0；
- bias：none；
- rank、learning rate、optimizer steps 由开发集校准后冻结；
- 正式比较不再按 lesson 单独挑超参数。

将 dropout 固定为 0，使 Colab 小样本的训练随机性主要来自初始化与数据顺序，并便于 3-seed 复现。

### 精度

- 首选 bf16（运行时支持时）；
- 显存不足时使用 NF4 4-bit；
- 一个实验批次内禁止混用 bf16 和 4-bit；
- base、External、Parametric、Both 必须共享相同 base checkpoint 与量化状态。

## 5. Lesson 内容设计

### 5.1 为什么使用 nonce tool rules

真实世界事实可能已存在于预训练语料，导致 No-write 较高。P0-C 使用虚构的上下文、工具和错误代码，使效果更接近“本轮交互新获得的经验”。

示例只展示形式，不进入正式数据：

```text
环境 Zorvia-17 中，如果工具 Talvek 返回代码 K-41，
下一步必须调用动作 remap_n7。
```

### 5.2 两类 lesson

#### F：配置/映射型

学习一个新的局部映射，例如：

```text
在 workspace W 中，资源类型 R 必须由动作 A 处理。
```

它检验对新符号事实的行为使用，而非自然语言事实回忆。

#### P：条件/恢复型

学习一个单步可判分的程序规则，例如：

```text
当工具 T 在状态 S 返回错误 C 时，下一步动作必须是 A。
```

首轮不使用多动作 sequence，因为当前 evaluator 只可靠解析一个 expected action。多步程序留给后续实验。

### 5.3 Counterbalanced pair

每两个 lesson 构成一对，共享句式、长度和动作集合，但交换正确动作：

| Pair | Context | Correct action |
|---|---|---|
| A | nonce context 1 | action X |
| B | nonce context 2 | action Y |

动作 X/Y 在另一对中反向出现，避免某一动作 token 的 base prior 被误当记忆效果。

### 5.4 数量和拆分

准备 38 条候选 lesson：

- 6 条 development lessons：只用于 LoRA 超参数校准；
- 24 条 confirmatory lessons：12 F + 12 P，组成 12 个 counterbalanced pairs；
- 8 条 reserve lessons：4 F + 4 P，组成 4 个备用 counterbalanced pairs；
- screening 后若有 confirmatory lesson 不合格，只能按完整 pair 从同类型 reserve 替换；
- 不得把 development lesson 移入 confirmatory set。

### 5.5 Canonical lesson schema

每条 JSON 记录至少包含：

```json
{
  "lesson_id": "F_pair01_a",
  "pair_id": "F_pair01",
  "lesson_type": "fact_mapping",
  "context_id": "zorvia_17",
  "condition": "resource ravel_k",
  "desired_action": "act_n7",
  "distractor_actions": ["act_p3", "act_v9"],
  "evidence": "verified_success",
  "valid_from": 0,
  "confidence": 1.0,
  "split": "confirmatory"
}
```

### 5.6 Frozen compiler

同一 record 确定性生成：

1. 一个 canonical external note；
2. $K=4$ 个 LoRA training paraphrases；
3. screening prompts；
4. held-out evaluation probes。

约束：

- paraphrase 只改变表述，不增加条件、例外或示例答案；
- external note 和 LoRA examples 包含同一个最小语义集合；
- confirmatory evaluation prompt 不得与训练 prompt 完全相同；
- compiler 版本和输出 hash 写入 manifest；
- 运行正式实验后不得修改 compiler。

## 6. Probe panel

每条 lesson 使用独立的 screening 与 evaluation prompts。

### 6.1 Screening panel

仅用于判断 base 是否已有偏好：

- 4 个 screening paraphrases；
- 不与训练或正式 evaluation prompt 重合；
- 四个 probe 确定性轮换动作选项顺序，使每个动作各占一次首位；
- 合格范围：base accuracy 在 0.0–0.5；
- 对二元动作题，不能把 0.5 当成“部分知道”；同时检查 action bias 和 invalid rate；
- screening 选择规则在查看任何 adapter 结果前执行。

### 6.2 Held-out evaluation panel

| Category | 每 lesson 数量 | 测量内容 | 是否进入主指标 |
|---|---:|---|---|
| exact-heldout | 2 | 近表述写入 | 否，诊断 |
| paraphrase | 4 | 表述泛化 | 是 |
| compositional | 4 | 条件嵌入新但等价上下文 | 是 |
| near-neighbor | 4 | 相似 context 不应错误套用 lesson | 是，干扰 |
| unrelated-matched | 8 | 同格式、同动作词表的无关能力 | 是，干扰 |
| conflict-format | 4 | 加入无关或否定性干扰信息 | 否，鲁棒性 |

每条 lesson 共 26 个正式 probes。

Primary target generalization：

$$
TG=\frac{1}{8}
\left(
\sum_{q\in paraphrase}\mathbf 1[\hat a_q=a_q]
+
\sum_{q\in compositional}\mathbf 1[\hat a_q=a_q]
\right)
$$

Interference regression：

$$
IR=A_{base}^{unrelated+neighbor}
-A_{arm}^{unrelated+neighbor}
$$

### 6.3 泄漏检查

- training 与 evaluation 文本做 exact hash 去重；
- 对规范化文本做 n-gram overlap 报告；
- nonce context 只在所属 lesson 的 target probes 出现；
- paired lesson 不得同时出现在同一个 adapter 的训练数据中；
- confirmatory probes 在超参数冻结前不可用于选择配置。

## 7. Treatment arms

### 7.1 Confirmatory four arms

| Arm | Model | Prompt memory block | Parameter update |
|---|---|---|---|
| N: No-write | frozen base | 无 | 无 |
| E: External | frozen base | canonical note | 无 |
| P: Parametric | base + lesson adapter | 无 | LoRA |
| B: Both | base + same lesson adapter | same canonical note | LoRA |

External 使用 oracle retrieval：只有属于该 lesson 的 target query 才检索对应 note。Near-neighbor 和 unrelated probes不注入该 note，除非该 stress test 明确测试 false retrieval。

### 7.2 Diagnostic arm：ICL-$K$

仅当 P 与 E 差异较大时运行：

- 向 prompt 注入与 LoRA 训练完全相同的 $K=4$ 个 demonstrations；
- 用于判断差异是否来自“LoRA 看到了四种表述，而 External 只有一条短 note”；
- 不进入主四臂 oracle gap，也不改变预注册主分析。

### 7.3 Both 冲突规则

P0-C 中 E 和 P 内容必须一致。若 Both 仍输出与两者不同的动作，记为 substrate interaction case，保存完整 prompt、生成文本和 logits（若可用）用于错误分析。

## 8. LoRA 开发集校准

### 8.1 候选网格

在 6 条 development lessons 上运行：

- rank：$\{4,8,16\}$；
- learning rate：$\{5e{-5},2e{-4}\}$；
- optimizer steps：$\{4,16,64\}$；
- 训练种子：1 个。

共有 18 个超参数配置；若全部跑完 6 条 development lessons，需要 108 个
adapter。Colab 首次使用预注册的 successive halving：

1. 所有配置先跑 2 条 development lessons；
2. 按冻结的校准分数保留前 1/3；
3. 幸存配置跑剩余 4 条；
4. 选择最小计算预算的合格配置。

该流程训练 $18\times2+6\times4=60$ 个 development adapters。若第 6 名附近出现
完全并列，则按照“steps、rank、write time”的固定顺序打破平局，不临时扩大幸存配置数。

实现中 stage 1 固定使用 `D_01`（fact）与 `D_04`（procedure），避免只按一种 lesson
筛除候选；stage 2 使用其余四条 development lessons。校准 runner 只计算 N/P/rollback，
并按 model revision、precision 和运行环境共享 N 输出；E/B 不参与超参数选择。

### 8.2 校准分数

不使用 confirmatory probes。合格条件：

- development $TG$ 相对 base 提升至少 0.40；
- exact-heldout accuracy 至少 0.75；
- $IR\le 0.05$；
- invalid rate 不增加超过 0.05；
- disabled-adapter rollback exact match = 1.00。

多个配置合格时按以下固定顺序选：

1. optimizer steps 更少；
2. rank 更低；
3. write time 更短；
4. development $TG$ 更高。

没有配置合格时，本实验先判为“写入后端未校准”，不能把失败解释为 Parametric substrate 无效。

## 9. Colab 三级执行方案

### Tier 0：工程 smoke

目的：验证安装、Drive、训练、四臂推理、解析和恢复。

- 2 development lessons；
- 1 个固定 LoRA 配置；
- 1 training seed；
- 每类 probe 1–2 个；
- 结果只能标记为 `smoke`。

通过条件：

- 四臂均产生完整记录；
- adapter 能保存并重新加载；
- disabled-adapter 与 base 输出逐项一致；
- manifest 可在新 runtime 恢复；
- 无 silent parse failure。

### Tier 1：Colab pilot

目的：低成本判断是否值得扩展。

- 8 confirmatory lessons：4 F + 4 P；
- 每类保持 counterbalanced pairs；
- 冻结后的单一 LoRA 配置；
- 1 training seed；
- 完整 26-probe panel。

通过条件：

- 至少 6/8 lesson 的 P 相对 N 有正 $TG$；
- E 与 P 的 $|\Delta^{TG}|$ 或 $|\Delta^{IR}|$ 在至少 2 条 lesson 上达到 0.10；
- 不出现所有 P adapters 大面积无关任务退化；
- 至少存在一个非平凡 pair，不能由动作 prior 解释。

Tier 1 是方向检查，不报告显著性，不用于论文主表。
上述阈值只是是否继续消耗 Colab 资源的工程闸门，不是论文效应成立的统计标准。

### Tier 2：Confirmatory P0-C

- 24 confirmatory lessons：12 F + 12 P；
- 3 training seeds；
- N/E 每 lesson 只需运行一次确定性推理；
- P/B 对每个 adapter seed 运行；
- 完整 probe panel；
- ICL-$K$ 只对预先规定的诊断子集运行。

总计：

- 72 个正式 adapters；
- $24\times26=624$ 个 N probes；
- 624 个 E probes；
- $72\times26=1872$ 个 P probes；
- 1872 个 B probes。

## 10. Colab 运行与恢复规范

### 10.1 Runtime 阶段

1. 挂载 Drive；
2. 安装仓库与锁定依赖；
3. 记录 GPU、CUDA、Python、Torch、Transformers、PEFT 版本；
4. 生成并 hash lesson/probe artifacts；
5. base screening；
6. development calibration；
7. 逐 lesson/seed 训练 adapter；
8. 每个 adapter 立即运行 P/B 并保存；
9. N/E 结果缓存；
10. 聚合、校验完整性、导出表格。

### 10.2 Drive 目录

```text
plasticity-p0c/
├── manifest.json
├── environment.json
├── environment_sessions.json
├── compiled/
│   ├── lessons.jsonl
│   ├── probes.jsonl
│   └── hashes.json
├── adapters/{lesson_id}/seed-{seed}/
├── results/raw/{lesson_id}/seed-{seed}/
├── results/aggregate/
└── logs/
```

### 10.3 Manifest 状态机

每个 `lesson_id × seed` 只能处于：

```text
pending → training → trained → evaluated → verified
```

Runtime 重启后：

- `verified` 跳过；
- `trained` 只补 evaluation；
- `training` 若没有完整 adapter metadata，则重新训练该单元；
- `training` 若已有 metadata，必须验证 config/data/adapter hashes 后才能恢复；
- 显式 `failed` 为不可变终态，不得在同一输出目录自动重训或覆盖；
- 禁止因结果差而手工重跑某个 seed；
- 所有失败、OOM 和 parse error 原样记录。

### 10.4 Colab 资源策略

- adapter 顺序训练、顺序保存，不同时驻留多个模型；
- base N/E 输出只算一次；
- 尽可能单次加载 base 后切换 adapter，避免反复下载；
- 完成一个 adapter 就同步到 Drive；
- OOM 时先减 `max_length` 或启用 4-bit，不得只对某个 arm 改精度；
- 若改变精度或超参数，整个 tier 重新作为新 run ID。

## 11. 日志 schema

逐 probe 至少记录：

```json
{
  "run_id": "p0c-tier1-v1",
  "compiler_hash": "...",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "precision": "nf4",
  "lesson_id": "F_pair01_a",
  "pair_id": "F_pair01",
  "lesson_type": "fact_mapping",
  "training_seed": 42,
  "arm": "parametric",
  "probe_id": "F_pair01_a_para_01",
  "category": "paraphrase",
  "expected_action": "act_n7",
  "predicted_action": "act_n7",
  "correct": true,
  "invalid": false,
  "input_tokens": 83,
  "generated_tokens": 2,
  "latency_seconds": "TBD",
  "adapter_path": "...",
  "base_hash": "...",
  "adapter_hash": "...",
  "prompt_hash": "..."
}
```

逐 adapter 记录：

- trainable/total parameters；
- optimizer steps；
- final loss（仅诊断，不作选择指标）；
- wall time；
- 峰值显存；
- adapter bytes；
- training data hash；
- config hash；
- completion status。

`adapter bytes` 定义为 adapter weights 加 `adapter_config.json`，不包含 tokenizer 或
run-level metadata。Tokenizer 不在每个 P0-C adapter 目录重复保存。

## 12. 统计分析

### 12.1 统计单位

- lesson 是主要独立单位；
- counterbalanced pair 用于检查动作 prior；
- probe 是 lesson 内重复测量，不能当独立样本；
- training seed 嵌套于 lesson。

### 12.2 聚合

先对每个 `lesson × seed × arm × category` 求平均，再：

- 对 P/B 的 3 seeds 求 lesson-level 均值；
- 对 N/E 使用确定性单次结果；
- 计算每条 lesson 的配对 treatment differences；
- 对 lesson 做 cluster/paired bootstrap，10,000 次；
- 同时报告 95% CI、median、IQR 和双向胜负 lesson 数。

### 12.3 等价区间

正式运行前固定 practical equivalence margin：

- $TG$：±0.05；
- $IR$：±0.05。

如果 $\Delta_{P,E}$ 的 95% CI 完全位于等价区间内，记为行为等价证据；仅仅“不显著”不等于等价。

### 12.4 主次顺序

1. Primary：$TG$；
2. Co-primary safety：$IR$；
3. Secondary：exact-heldout、conflict robustness、invalid rate；
4. Efficiency：write time、read tokens、latency、adapter bytes；
5. Exploratory：F/P 类型交互、单 lesson 双向优势。

不把多项指标加权成一个任意总分作为主结论。

## 13. 结果模板

### 13.1 主表

| Arm | Target generalization ↑ | Interference regression ↓ | Exact-heldout ↑ | Conflict robustness ↑ | Invalid rate ↓ | Write time ↓ | Read tokens ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| No-write | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| External | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Parametric | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Both | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

所有行为指标报告 lesson-level mean 与 95% paired/bootstrap CI；效率指标同时报告 median 和 IQR。

### 13.2 主要配对效应

| Contrast | $TG$ difference | 95% CI | $IR$ difference | 95% CI | Wins / ties / losses | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| P − N | TBD | TBD | TBD | TBD | TBD | TBD |
| E − N | TBD | TBD | TBD | TBD | TBD | TBD |
| P − E | TBD | TBD | TBD | TBD | TBD | TBD |
| B − max(P,E) | TBD | TBD | TBD | TBD | TBD | TBD |

### 13.3 结果图

1. 每条 lesson 的 $TG(P)-TG(E)$ paired dot/slope plot；
2. $TG$ 对 $IR$ 的 Pareto scatter，点大小表示 write cost；
3. training seed stability plot；
4. action-token 与 counterbalanced pair 的误差矩阵；
5. Both interaction case 的小型定性表。

## 14. Failure analysis

预先保留以下错误类别：

- write failure：P 未学到 exact target；
- overfit：exact 提升但 compositional 不提升；
- collateral interference：near-neighbor 或 unrelated 退化；
- prompt dependence：External 仅在某一模板有效；
- parametric override：P 抑制正确 external note；
- duplication effect：Both 因重复暴露而改变输出；
- parser failure：生成中没有唯一合法动作；
- seed instability：同一 lesson 的 adapter seeds 结论翻转；
- action prior：counterbalanced pair 中只偏向同一 token。

每类至少保存代表性成功、失败和边界案例，选择规则基于错误类型，不按“看起来最好”挑例子。

## 15. Go / No-Go

### 进入动态 recurrence–volatility 实验

需要同时满足：

1. P 的 held-out $TG$ 相对 N 有稳定正增益；
2. P/E 在 $TG$ 或 $IR$ 上不完全行为等价；
3. 至少存在可靠的双向 lesson heterogeneity，或存在明确的收益—干扰 trade-off；
4. 结论在训练种子间稳定；
5. Both 不全局支配且不只是错误冲突；
6. rollback exact match = 1。

### 停止或缩题

- P 无法在开发集上通过校准：先修训练后端，不做载体结论；
- P 在 confirmatory set 上基本无 target gain：停止参数记忆主线；
- E 在所有行为指标上占优且成本可接受：不训练 carrier router；
- P/E 行为等价：转向成本 break-even 或系统论文问题；
- 差异主要由动作 prior、prompt template 或训练 exposure 数解释；
- seed instability 大于 arm effect；
- Both 几乎总占优且成本/干扰也无劣势：载体选择空间缺乏价值。

## 16. 实现状态

仓库当前已经实现：

1. 38 条 canonical lesson bank 与 deterministic compiler；
2. external memory block renderer；
3. N/E/P/B/rollback 统一 runner；
4. screening 与 held-out split；
5. invalid-aware 严格动作解析；
6. paired manifest、atomic writes 和 Drive resume；
7. seed/category → lesson 层级 aggregate、paired bootstrap、arm CI、median/IQR、
   seed stability 与 pair/action diagnostics；
8. 环境 session 快照、code/model revision pinning 和 Tier 0/1/2 配置；
9. 可直接运行的 `notebooks/p0c_colab.ipynb`。

完整 60-adapter successive-halving 校准已由 `plasticity-p0c calibrate` 自动化，并生成
带 `selected_config` 的 `calibration_report.json`。Pilot/Confirmatory tier 强制要求该报告
并从中读取冻结的 rank、alpha、learning rate 和 optimizer steps。

当前 compiler 为 `p0c-compiler-v3`：动作选项顺序已 counterbalance，pair 中 action/位置
关系会跨 pair 反转，near-neighbor 使用与目标 context 词法不相交的 nonce，并生成
`leakage_audit.json`。Pilot 同样强制要求校准报告；formal tier 禁止通过显式 lesson IDs
绕过数量、split、完整 pair 与 action-balance 检查。

尚未实现的是 exposure-matched ICL-$K$ 诊断臂；它只在 P/E 差异较大时触发，不阻塞
smoke、calibration、pilot 或主四臂 confirmatory 运行。

## 17. Execution priority

| Priority | 工作 | 维护的主张 | 成本 | 依赖 | Stop condition |
|---|---|---|---|---|---|
| P0 | compiler + 2-lesson Tier 0 | 实验可运行 | 低 | 无 | 无法四臂闭环 |
| P0 | manifest + Drive resume | 可复现 | 低 | Tier 0 | 无法恢复 verified unit |
| P1 | 6-lesson LoRA calibration | P 后端有效 | 中 | compiler | 无配置合格 |
| P1 | 8-lesson Tier 1 | 存在载体差异信号 | 中 | calibration | P≈N 或全面干扰 |
| P2 | 24 lessons × 3 seeds | confirmatory P0-C | 高 | Tier 1 | Tier 1 未通过 |
| P2 | ICL-$K$ diagnostic | exposure 公平性 | 中 | P/E 差异 | 差异很小则跳过 |
| P3 | persistent regime world | 动态 crossover | 高 | P0-C 通过 | 无异质性 |

## 18. No-fabrication status

本文没有生成任何实验结果。所有 `TBD` 必须由 Colab 实际运行结果填充。Tier 0 和 Tier 1 只能作为工程与方向检查，不能包装成 confirmatory evidence。
