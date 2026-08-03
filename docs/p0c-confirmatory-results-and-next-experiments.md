# P0-C Confirmatory 结果与后续实验路线

- **结果状态：** 24-lesson、3-seed Confirmatory P0-C
- **实验日期：** 2026-07-25
- **解释版本：** 2026-07-28
- **后续实验状态：** P0-D2 budget match、P0-D2H-R hard probe、
  P0-D2H-CAL 和 invalid-output audit 已完成；P0-E 尚未进入
- **主线结论：** 等参数 late placement 优势通过 hard-probe 配对门，但训练复杂度实验
  仍被 calibration category gate 阻塞；下一步先做 base-only format-stable calibration

## 1. 结果来源与证据边界

本次 Confirmatory 汇总来自 `p0c-confirmatory-58feadfc39`：

- 24 条 lesson，每条使用训练种子 41、42、43；
- 12 条 `fact_mapping` 和 12 条 `procedure_recovery`；
- fact 部分由 `F_pair01/04/06` 与预先冻结的 reserve pairs
  `RF_pair01/04/05` 组成，procedure 部分使用 `P_pair01–06`；
- 共 6,864 条 probe-level 记录；
- 模型权重解析到
  `7ae557604adf67be50417f59c2c2f167def9a775`；
- 汇总文件 SHA-256：
  `4d9b1ba4a26e6f0f60ed6f4571ab66ca80f3dac696924884c0a956c7bf4439c1`。

P0-C 的 `aggregate` 只接受完整 lesson × seed unit 集合，并要求所有 base arms 和 adapter
units 均为 `verified`、rollback exact match 为 1、adapter hash 与 model revision 一致。
因此该 `summary.json` 的成功生成也验证了这些聚合前置条件。不过，正式归档时仍应同时
保存原始 `manifest.json`、calibration report、raw probe records 和聚合目录，不能只保留
summary。

主分析先在每条 lesson 内等权汇总三个 adapter seeds，再以 lesson 为配对和 bootstrap
聚类单位。因此统计样本是 24 条 lessons，不是 72 个相互独立的 adapters；seed 维度用于
稳定性诊断，不能作为伪重复扩大样本量。

当前证据只适用于冻结的 Qwen2.5-0.5B-Instruct、4-bit QLoRA、全层 `q_proj/v_proj`
LoRA、当前 nonce lesson bank 和 oracle external memory。它不是跨模型、跨任务或真实在线
agent 环境的普适结论。

## 2. 四臂主结果

| Arm | Target generalization mean [95% CI] ↑ | Exact-heldout mean [95% CI] ↑ | Conflict robustness mean [95% CI] ↑ | Interference regression mean [95% CI] ↓ | Invalid rate mean [95% CI] ↓ |
|---|---:|---:|---:|---:|---:|
| No-write | 0.2448 [0.1875, 0.3073] | 0.2292 [0.1042, 0.3542] | 0.1354 [0.0521, 0.2396] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| External | 0.9010 [0.8438, 0.9531] | 1.0000 [1.0000, 1.0000] | 0.6146 [0.4896, 0.7500] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| Parametric | 0.8663 [0.7743, 0.9410] | 0.9236 [0.8403, 0.9931] | 0.5764 [0.4514, 0.7014] | 0.0289 [-0.0347, 0.0949] | 0.0000 [0.0000, 0.0000] |
| Both | 0.9983 [0.9948, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9271 [0.8438, 0.9896] | 0.0289 [-0.0347, 0.0949] | 0.0000 [0.0000, 0.0000] |

成本上，External 不需要训练和 adapter 存储，但平均输入约 104 tokens，而 No-write 和
Parametric 约为 83 tokens。Parametric/Both 每条 lesson 的 adapter 中位数约 1.044 MiB，
写入时间中位数约 5.38 秒，记录的训练峰值显存中位数约 0.902 GiB。推理延迟中位数约从
No-write/External 的 0.223 秒增加到 Parametric/Both 的 0.271 秒。这里 External 的成本
只覆盖本实验的 prompt tokens，不包括真实系统中的记忆生成、索引、检索、存储和维护。

## 3. 配对效应

| Contrast | Target mean [95% paired bootstrap CI] | IR mean [95% paired bootstrap CI] ↓ | Target wins / ties / losses | 解释 |
|---|---:|---:|---:|---|
| P − N | +0.6215 [0.5313, 0.7031] | +0.0289 [-0.0347, 0.0949] | 24 / 0 / 0 | LoRA 参数写入在每条 lesson 上都优于不写入 |
| E − N | +0.6563 [0.5938, 0.7135] | 0.0000 [0.0000, 0.0000] | 24 / 0 / 0 | Oracle external memory 同样稳定有效 |
| P − E | −0.0347 [-0.1042, 0.0295] | +0.0289 [-0.0347, 0.0949] | 5 / 8 / 11 | 不能声称 P 优于 E，也未建立预设 ±0.05 内的实用等价 |
| B − best single | +0.0573 [0.0156, 0.1076] | +0.0324 [-0.0012, 0.0741] | 7 / 16 / 1 | Both 平均更高，但 16/24 lessons 已与最佳单载体打平 |

P − N 的 interference regression 均值为 +0.0289，区间
[-0.0347, 0.0949]；方向和不确定性都不足以支持“LoRA 必然造成额外干扰”。P − E 在
interference difference 上有 10 条为正、4 条为零、10 条为负，也不支持统一方向。

## 4. 异质性与稳定性

### 4.1 Reserve-pair 披露

Confirmatory 的 fact cohort 包含 3 个原 confirmatory pairs
（`F_pair01/04/06`）和 3 个冻结 reserve pairs（`RF_pair01/04/05`）；即 6/12 fact
lessons、6/24 全部 lessons 来自 reserve split。procedure cohort 的 `P_pair01–06`
全部来自原 confirmatory split。

Reserve pairs 在任何 adapter outcome 产生前已经由 v4 compiler 冻结。实际选择先使用
No-write screening 排除 base 已知或高 invalid 的 lesson，再在完整同类型 pairs 中按
action balance、reserve 数量和 pair ID 做确定性选择；没有根据 P/E/B 结果事后挑选。
这保住了配对设计和动作平衡，但样本仍是 **screening-conditioned cohort**。因此：

- 主效应和区间只条件化到这 24 条被选 lessons，不能表述为从开放任务分布随机抽样；
- reserve 不是处理变量，不能把 confirmatory-vs-reserve 的事后差异解释为数据分布效应；
- P0-D 必须从本次 manifest 冻结并复用相同 selected lesson IDs，不再重新 screening，
  否则层位置比较会混入 cohort 改变。

### 4.2 Lesson 类型

| Arm | Fact TG | Procedure TG |
|---|---:|---:|
| No-write | 0.2292 | 0.2604 |
| External | 0.8021 | 1.0000 |
| Parametric | 0.7951 | 0.9375 |
| Both | 0.9965 | 1.0000 |

Parametric 在 procedure recovery 上的平均表现高于 fact mapping。External 在 procedure
recovery 上达到上界，因此该类型当前几乎没有剩余空间用于证明 Parametric 优势。

### 4.3 Action token 与 `fact_mapping × act_k2`

Parametric 的 target generalization 按动作分别为：

- `act_k2`: 0.6597；
- `act_n7`: 0.9167；
- `act_p3`: 0.9167；
- `act_v9`: 0.9722。

`act_k2` 的弱点主要集中在 fact lessons：
`F_pair04_a=0.4583`、`F_pair06_a=0.2500`、
`RF_pair05_b=0.2917`，而三个 procedure `act_k2` lessons 均不低于 0.9583。因此当前
证据更像 `fact_mapping × act_k2` 的交互，而不是 `act_k2` 对所有任务都存在统一失败。
这个模式只有 3 条 fact `act_k2` lessons，当前没有冻结的 interaction test 或独立复现。
它可以生成 P0-D 的预先指定 secondary analysis，但在新 lessons 或第二模型复现前不能
升级为确认性机制结论。

### 4.4 Seed stability

Parametric target generalization 的 24 条 lessons 中：

- 14 条在三个种子间完全一致；
- 23 条的种子间 range 不超过 0.125；
- 只有 `P_pair01_b` 的 range 为 0.5（0.5、0.5、1.0）。

因此整体 arm effect 不是由普遍 seed instability 驱动，但
`P_pair01_b` 必须保留为失败/边界案例。这里的 range 是描述性诊断；由于 No-write 和
External 没有 adapter training seed，不能把它解释为四个 arms 的对称方差比较。Both
又接近上界，P arm 的 seed stability 是本阶段更有辨识力的检查。

## 5. 当前可以和不可以支持的主张

### 可以支持

1. 在当前受控设置中，真实 QLoRA 参数写入能稳定改变 held-out behavior；它不是仅保存了
   adapter 而未改变模型行为的空写入。
2. Parametric 和 External 相对 No-write 的配对 TG 区间均明确高于 0；P − N 在 24/24
   lessons 上为正。
3. External 的平均 TG 略高于 Parametric，但 P − E 区间跨零；当前既不能证明优劣，也
   不能证明二者实用等价。
4. Both 在平均 TG 和 conflict robustness 上最高，但大量 lessons 已达到或接近上界，
   并且 Both 同时承担 external context 与 adapter 两类成本。
5. 参数写入存在值得继续研究的 lesson/action 异质性，尤其是 fact `act_k2` cases。

### 不可以支持

1. 不能声称 LoRA 普遍优于 external memory，或 external memory 普遍优于 LoRA。
2. 不能仅凭 Both 接近 1.0 就声称双写是最优策略；当前尚未测长期存储、检索、并发 adapter、
   累积干扰和持续运行成本。
3. 不能声称已经找到最佳 Transformer 层；P0-C 固定使用全层 LoRA。
4. 不能声称动态 router 已被证明必要；当前只观察到异质性候选，还没有 out-of-sample
   routing gain。
5. 不能外推到更大模型、自然语言开放回答、真实工具执行、长时序 agent 或 RL 训练。

## 6. 阶段判断

P0-C 通过“参数写入是否值得继续研究”的资格门，但不直接跳到复杂动态 router。当前最
高信息增益的下一步是隔离参数写入位置：

> **主线：先完成 LoRA layer-locus/depth；补充：再检验 GRPO/RLVR 下结论是否保持。**

这一顺序避免同时改变优化目标和可训练参数位置，也避免把 GRPO 带来的变化误解释为
layer placement 效果。

## 7. 下一主实验：P0-D LoRA Layer Locus

### 7.1 研究问题

在保持 lesson、监督目标、LoRA target modules 和评估 probes 不变时，参数记忆效果是否
集中在 Transformer 的特定深度？

### 7.2 第一阶段：预注册 band scan

从 P0-C manifest 读取并冻结同一组 24 个 selected lesson IDs 和 seeds
`41/42/43`，不重新按新的 base screening 选择 cohort。主矩阵比较：

| 条件 | 可训练位置 | 作用 |
|---|---|---|
| No-write | 无参数更新、无 memory block | 写入增益锚点；每 lesson 运行一次 |
| External | 无参数更新、oracle note | 载体行为参考；每 lesson 运行一次 |
| LoRA-full | 全部层的 `q_proj/v_proj` | 当前 P0-C 参考 |
| LoRA-early | 前部层 | 深度对照 |
| LoRA-middle | 中部层 | 深度对照 |
| LoRA-late | 后部层 | 深度对照 |

每个 LoRA condition 运行 24 lessons × 3 seeds，共 288 个 adapter units；N/E 不按
condition 或 seed 重复。Parametric 是 layer-locus 的主 arm，Both 不进入 band scan
主矩阵，以免 P0-C 中接近上界的表现掩盖层差异。不能只测试 middle；即使 middle 是
自然的先验候选，也必须保留 early/late，避免把先验变成选择偏差。

正式 P0-D 应在同一 P0-D 代码和环境下重新运行 LoRA-full，不能把 P0-C 的 full 数值直接
当作主对照；P0-C 结果只用于 sanity check。这样可避免把 runner 或 provenance 变化误作
层位置效应。

`early/middle/late` 必须根据模型实际 `num_hidden_layers` 确定性三分，并在 manifest 中
同时记录标签和解析后的完整层索引。不能只保存 `middle` 这样的语义标签。

### 7.3 第二阶段：显式单层或窄窗口扫描

只有在 band scan 满足第 7.5 节的 locus 进入门后，才在获胜 band 内运行显式单层或
等宽窄窗口。候选层应由规则生成，例如获胜 band 的首层、中层、末层以及相邻 band 的
边界层；也可扫描获胜 band 的全部单层，但必须在查看任何窄扫描 target result 前把候选
索引、窗口宽度、seeds 和停止条件写入冻结 condition manifest。

同一 24-lesson cohort 上的窄扫描只用于探索性定位。正式的“层 X 或窗口 X 最佳”主张
至少需要一个未参与 band/窄扫描选择的 held-out lesson bank，或预先指定的第二模型规模
复现。若窄扫描使用现有失败 cases 选层，再在相同 cases 上报告增益，不构成确认性证据。

### 7.4 公平性与主指标

- 固定模型 revision、lesson compiler、训练 paraphrases、训练 seeds 和 probe 集合；
- 首先保持相同 LoRA rank/alpha，回答“相同局部配置写在何处更有效”；
- 固定 `q_proj/v_proj`、optimizer steps、learning rate、数据顺序规则和推理配置；
- 同时报告 trainable parameter count；同 rank 的 full 与 band 条件参数量不同，所以
  这是“固定局部 LoRA 配置”的位置比较，不是参数预算匹配比较；
- 若需参数预算匹配，必须作为独立 secondary matrix，在 development lessons 上冻结
  rank/alpha 规则并报告实际 trainable parameters，不能按 target results 为每个 band
  单独调参；
- 主要 estimands 为每个 band 相对 full 的 lesson-paired TG/IR difference，以及相对
  No-write 的写入增益；95% CI 对 lesson clusters 做 paired bootstrap，不能把 288 个
  adapter units 当成独立样本；
- 同时报告 exact-heldout、conflict robustness、invalid、rollback、seed stability、
  adapter 大小、训练时间、峰值显存、推理延迟和 trainable parameter count；
- 把 `fact_mapping × act_k2` 写成预先指定的 secondary interaction diagnostic，并同时
  报告其样本量和跨 seed 方向；它不单独触发确认性 layer claim。

### 7.5 决策门

- **运行有效门：** 所有预期 condition × lesson × seed units 均为 `verified`，adapter
  hash/model revision/precision 一致，rollback exact match 为 1；否则不得生成正式汇总。
- **保留收益门：** 对 band $b$，预注册实用差值
  $\Delta_b^{TG}=TG(b)-TG(full)$。若其 lesson-clustered 95% CI 下界不低于
  `-0.05`，才称该 band 在预设 margin 内保留 full 的主要 TG 收益。
- **locus 进入门：** 除保留收益外，候选 band 还需相对至少一个其他 band 的 paired TG
  区间完整高于 0，且方向不是由单一 seed、单一 lesson type 或一个异常 lesson 独占，
  才进入显式层/窄窗口扫描。
- **稀疏但非唯一：** 若一个或多个 bands 保留 full 收益，但 band 间区间均跨 0，只能
  主张较少可训练层可能足够；不能主张已经定位唯一最佳深度，也不进入结果驱动的单层挑选。
- **所有 bands 落后：** 先运行冻结的参数预算匹配或 rank/module ablation，区分“需要
  分布式写入”与“band 容量不足”；不能直接把 full 优势解释为跨层协同机制。
- **局部异质性：** 若差异只出现在 `fact_mapping × act_k2` 或现有失败 lessons，必须在
  新 lessons 复现后才升级为主张。

该实验必须新建独立 Colab 子目录和 notebook，不修改或复用 P0-C v4 notebook 作为
实验入口。建议目录为 `notebooks/p0d_lora_locus/`。

## 8. 后续补充：P0-E GRPO/RLVR

GRPO 不作为当前主实验，而作为 layer-locus 之后的优化目标稳健性补充。这里更准确的
名称是 RLVR：使用可验证动作奖励；除非实际引入人类偏好、reward model 和对应流程，
否则不称为经典 RLHF。

参考的[单层 RL 工作](https://arxiv.org/abs/2607.01232)冻结其余网络、直接更新选定
Transformer 层的原始参数，并用 full-parameter RL 作为参照；它不是 LoRA 单层训练。
因此我们的补充实验必须把“训练目标”和“参数载体”分开：

| 优化目标 | Adapter 参数 | 原始模型参数 |
|---|---|---|
| Supervised | full LoRA；best-band LoRA | 非主线 |
| GRPO/RLVR | full LoRA；best-band LoRA | best single layer；full-parameter reference |

建议先实现最小 GRPO-LoRA pilot：

1. 使用相同 lesson 语义和 held-out probes；
2. 用严格动作解析构造 exact-match reward；
3. 在 development prompts 上验证同组 rollouts 中存在非退化 reward variance；若每组
   reward 全为 0 或全为 1，group-relative advantage 无法提供有效学习信号；
4. 固定 rollout、update、token、KL/reference policy 与评估预算；
5. 训练 prompt/reward 只读取训练 split 的 canonical target，held-out probe answer
   不能进入 rollout prompt、reward metadata 或调参日志；
6. 比较 supervised LoRA 与 GRPO-LoRA 的 TG、IR、冲突鲁棒性和训练稳定性；
7. 单独检查 invalid action、模式坍缩、KL 漂移、reward hacking 和跨 seed 失败。

只有在资源允许且需要对齐单层 RL 文献时，才增加“原始单层参数 GRPO”和
“full-parameter GRPO reference”。当前 4-bit QLoRA 权重被冻结，不能把 NF4 权重直接
当作论文中的可训练原始层；这部分需要独立的 BF16/FP16 训练实现和显存评估。

GRPO/RLVR 的进入条件是：

- P0-D 已完成冻结矩阵，并得到可复现的层位置结论或明确的无差异/不确定结果；
- reward parser、泄漏检查和 reward-hacking negative tests 通过；
- development pilot 中 group reward 非退化，invalid rate 和 KL 没有越过预注册停止线；
- Colab 显存和运行时 dry run 能支持 rollout、reference policy、optimizer state 和恢复；
- supervised/GRPO 条件冻结相同语义数据和可比较的 update/token/evaluation budget；
- 新实验使用独立目录，例如 `notebooks/p0e_grpo_rlvr/`，不与 P0-C/P0-D 输出混跑。

GRPO/RLVR 最终只回答“层位置结论是否跨监督学习和奖励优化保持”，不替代 P0-C 的载体
比较，也不应在没有 full-parameter RL reference 时声称复现了单层全参数 RL 方案。

## 9. 推荐执行顺序

1. 归档 P0-C Confirmatory 的 manifest、calibration、raw records、aggregate 和环境快照；
2. 完成 P0-D 独立 runner、聚合与 Colab 入口，先做 early/middle/late/full band scan；
3. 根据预注册门限决定是否做显式单层扫描；
4. 在 held-out lessons 或第二模型规模上确认 layer-locus；
5. 最后新建 P0-E GRPO/RLVR notebook，作为优化目标稳健性补充；
6. layer-locus 与 GRPO 证据稳定后，再决定是否投入 recurrence–volatility router。

Exposure-matched ICL-$K$ 仍可作为 P0-C 的独立诊断并行补做，但 P − E 当前既未建立优劣
也未建立等价；它不应改变 P0-D 的冻结 cohort、层条件或主要 estimands。

这一路线把当前主张限制在已验证证据上，同时让每个新实验只引入一个主要变量。

## 10. 推荐顺序第 2 步的代码完成标准：P0-D

这里的“第 2 步”指第 9 节的 **P0-D band scan**；P0-D 内部的第二阶段窄扫描只有通过
第 7.5 节的 locus 进入门后才执行。现有通用 LoRA trainer 已支持
`full/early/middle/late/explicit`，但 P0-C runner 仍把 `layer_band=full` 和
lesson × seed manifest 写死，因此不能只在旧 notebook 中追加一个 `--layer-band` 参数。

### 10.1 必须实现的配置与运行身份

- 新建 P0-D config/CLI/runner；可复用 compiler、probe evaluator 和 LoRA trainer，
  但不能更改 P0-C frozen protocol 的含义；
- condition schema 至少包含 `full/early/middle/late/explicit`、解析后的层索引、
  `target_modules`、rank、alpha、learning rate、steps 和 parameter-budget regime；
- run ID 必须覆盖代码 hash、模型 revision、compiler/probe hashes、selected lesson IDs、
  seeds 和完整 condition matrix；
- unit identity 必须是 **condition × lesson × seed**。现有只含 lesson × seed 的
  `RunManifest` 不能无修改复用；
- P0-C manifest 只作为只读 cohort/provenance 输入；P0-D 不能改写 P0-C manifest、
  calibration、raw records 或 aggregate。

### 10.2 必须记录和验证的产物

每个 adapter unit 至少记录：

- condition ID、semantic band、`num_hidden_layers`、resolved layer indices；
- trainable/total parameter count、adapter bytes、write time、peak memory 和 precision；
- model revision、training-data/config/adapter SHA-256、compiler/probe hashes；
- raw Parametric/rollback probe rows、strict parser fields 和 rollback exact-match；
- immutable state transition；`failed` unit 保留错误并要求新 attempt，不得静默覆盖。

正式 aggregate 必须拒绝：

- 任一缺失或额外的 condition × lesson × seed unit；
- condition、seed、precision、model revision、selected layers 或 hashes 混跑；
- 非 `verified` unit、rollback 不等于 1、重复 probe 或缺失 probe；
- 根据结果临时改变 condition matrix 后继续写入原目录。

### 10.3 聚合与结果表

- 先保留 condition × lesson × seed 的 category metrics，再在 lesson 内等权汇总 seeds；
- 对 `band-full`、`band-No-write` 做 lesson-paired contrasts 和 lesson-clustered bootstrap；
- 输出 wins/ties/losses、seed range、lesson type、action token、
  `fact_mapping × act_k2` secondary diagnostic；
- 同表报告 TG、IR、exact、conflict、invalid、trainable parameters、时间、显存、
  adapter 大小和延迟；
- 实现第 7.5 节 `-0.05` retention margin 和 locus gate，但只输出 gate status，
  不自动挑选窄扫描层或启动下一阶段；
- 生成机器可读 `summary.json`、人读主表和冻结的下一阶段候选条件文件；所有未运行结果
  保持 `TBD/pending`，不得自动补值。

### 10.4 独立 Colab 与 Drive 目录

新增 `notebooks/p0d_lora_locus/`，至少包含：

```text
notebooks/p0d_lora_locus/
├── README.md
├── build_p0d_notebooks.py
├── p0d_band_scan_colab.ipynb
└── p0d_narrow_scan_colab.ipynb
```

Notebook 应由生成器维护，不直接手改生成文件。Band scan 与 narrow scan 使用不同的
stage attempt/output 目录；Drive 路径必须使用 P0-D 自己的 pipeline namespace，不复用
P0-C 的 `PIPELINE_ATTEMPT` 或 `notebooks/p0c_v4/`。每个 notebook 要在 GPU 工作前打印
并冻结代码 SHA、模型 revision、condition matrix、selected lesson hash、环境 fingerprint
和目标 Drive 目录，支持按 verified unit 零工作恢复。

P0-E 同理使用 `notebooks/p0e_grpo_rlvr/` 和独立 Drive namespace。不得把 P0-E rollout、
reference policy、optimizer checkpoint 或 reward logs 写入 P0-D/P0-C 目录。

### 10.5 最低测试与验收

代码完成的最低门槛是：

1. layer resolution 测试覆盖不能整除 3 的层数、显式层越界/重复和标签—索引一致性；
2. config/run ID 测试证明任一 condition、层索引、seed、hash 改变都会产生新 run；
3. manifest 测试覆盖 condition-aware resume、failed immutability 和 partial/mixed
   matrix rejection；
4. aggregate 测试用合成数据验证 lesson 内 seed 汇总、paired contrast、bootstrap、
   retention/locus gate 和异质性分组；
5. 一个不下载大模型的 fake backend smoke 完成 N/E + 四个 LoRA conditions + rollback
   闭环，并验证第二次运行不改 verified adapter 的 mtime；
6. notebook generator 测试验证 JSON、独立 stage、Drive namespace、预期 unit count
   `288` 和禁止引用 P0-C 输出目录；
7. `pytest`、Ruff、notebook JSON validation、CLI help/load 和 `git diff --check` 全部通过。

这一代码完成标准后来已经落实，并扩展为 P0-D2 参数预算匹配与 P0-D2H read-only hard
probe。当前真实结果和新的进入边界见下一节；本节保留为实现标准和历史预注册记录。

## 11. P0-D2H-CAL 完成后的路线更新

### 11.1 已完成阶段

| 阶段 | 当前状态 | 关键结论 |
|---|---|---|
| P0-D1 band scan | 完成 | 暴露出 layer locus 与 LoRA 参数量共同变化的解释混杂 |
| P0-D2 parameter-budget match | 完成 | 冻结七条件矩阵；等参数 late/full/early/middle 条件可用于理论比较 |
| P0-D2H-R hard probe | 完成 | late-minus-full `+0.1076 [0.0781, 0.1380]`，`hard_locus_robust`；suite quality 未通过 |
| P0-D2H-CAL | 完成 | 0.5B `oracle_failed`；1.5B `oracle_pass_external_failed`；cross-scale 为 `mixed_scale_result` |
| Invalid-output audit | 完成 | 1.5B external semantic `0.9167`，但 `conditional_route=0.6667` |
| P0-D2H-CAL-FC | 完成 | 1.5B external forced-choice `0.9089`，但 `conditional_route=0.6354`；无 eligible model |
| Conditional-route post-hoc audit | 完成 | 34/35 external errors 在 paired oracle 正确；错误受 variant/action/order 混杂 |
| P0-D2H-CRD route/retrieval decomposition | 完成 | retrieval 正常；剩余缺口集中于 routing，精度 tie 已解释 |
| P0-D2H-RR route remediation | 完成、失败 | route-only 提升但 conditional-route 未达 0.75，combined 退化 |
| RR same-runtime OFF/ON audit | 完成 | sentinel 通过；adapter 的 route 增益与 combined 退化均在同一 runtime 重现 |
| P0-D2H-CPR composition-preserving remediation | cpr2 已训练；q1 因结果落盘前的分析字段错误无决策；单次 q2 恢复待运行 | 同一 immutable adapter 与 locked panel；门槛不变；1/4/8 保持冻结 |
| P0-E GRPO/RLVR | 未进入 | 当前 calibration gate 不允许开始 |

P0-D2H-R 在同一 0.5B 模型、固定 24-lesson cohort、固定 LoRA 参数预算上得到：

- `late-matched` hard accuracy `0.4905 [0.4227, 0.5616]`；
- `full-base` hard accuracy `0.3828 [0.3290, 0.4410]`；
- paired late-minus-full `+0.1076 [0.0781, 0.1380]`；
- 三个 seed、两种 lesson type 和 leave-largest-effect-out safeguard 均为正；
- `binding_decoys` 仍是共同 floor，因此 locus 主张只限于其余三个 non-floor hard
  categories；
- suite-quality gate 因 0.5B external `0.4297` 与 binding floor 而失败。

完整结果和主张边界见
[`P0-D2H-R Hard-Probe Stress Protocol`](p0d2h-hard-probe-stress-protocol.md)。

### 11.2 Calibration 与 invalid-output diagnosis

P0-D2H-CAL 只包含 base-only `no_write`、`external` 和
`answer_copy_oracle`，没有 parametric/LoRA arm：

| Model | No-write | External | Oracle | Frozen status |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 0.2526 | 0.4297 | 0.7891 | `oracle_failed` |
| Qwen2.5-1.5B | 0.1276 | 0.6901 | 0.9948 | `oracle_pass_external_failed` |

冻结 cross-scale 结果是：

```text
cross_scale_status = mixed_scale_result
next_stage_status = calibration_followup_required
eligible_model_ids = []
```

post-hoc audit 在不重新推理、不修改 raw rows 和旧 gate 的前提下进一步发现：

- 1.5B external strict accuracy `0.6901 [0.6562, 0.7240]`；
- conservative semantic accuracy `0.9167 [0.8906, 0.9427]`；
- semantic-minus-strict `+0.2266 [0.1953, 0.2552]`；
- `87/89` external invalids 是唯一正确 action 加额外文本；
- `long_context` 从 strict `0.1875` 恢复到 semantic `1.0000`；
- `conditional_route` 只从 `0.5729` 恢复到 `0.6667`。

因此 1.5B 的总体 external strict failure 主要包含格式不服从，但不能把所有错误归因于
格式；`conditional_route` 仍有独立的语义/组合缺口。0.5B 没有 invalid row 可恢复，
oracle 仍为 `0.7891`，所以它的失败也不是格式 parser 单独造成。

详细结果见
[`P0-D2H-CAL protocol`](p0d2hc-oracle-calibration-protocol.md)和
[`invalid-output audit`](p0d2hc-invalid-output-audit.md)。

### 11.3 Format-stable forced-choice calibration

P0-D2H-CAL-FC 对四个冻结 action 做 candidate-only full-string conditional
scoring，结果显示：

- 1.5B external `0.9089 [0.8828, 0.9349]`，明显高于 no-write
  `0.1745 [0.1276, 0.2240]`；
- 1.5B oracle `0.9948 [0.9870, 1.0000]`，说明 forced-choice endpoint 本身
  能稳定表达正确 action；
- 1.5B external 的 binding、conflict、long-context 均为 `1.0000`，但
  `conditional_route=0.6354`，低于预先冻结的 `0.75` category floor；
- 0.5B external 只有 `0.4349 [0.3724, 0.5052]`，oracle 也只有
  `0.7812 [0.7214, 0.8411]`；
- 两个模型均为零 tie/error/non-finite、零 sum/mean disagreement，且
  provenance/token audit 通过。

因此新 cross-scale 状态是
`scale_improvement_without_full_calibration`，`eligible_model_ids=[]`。1.5B
的格式瓶颈已被显著移除，但 routing/composition 缺口仍未通过冻结 gate；0.5B 的失败
也不能归因于自由生成格式。下一步仅允许对 1.5B external 的 96 条
`conditional_route` row 做只读、post-hoc 误差审计，不得自动开始训练或 narrow
scan。

完整结果见
[`P0-D2H-CAL-FC results`](p0d2h-format-stable-calibration-results.md)。

随后对 1.5B external 的全部 96 条 `conditional_route` row 做只读审计。48 个
verified unit、2,304 个 decision row、9,216 个 candidate score 的本地完整性复算
全部通过。35 个 external error 中有 34 个在 paired oracle 正确，且 error 分布于
20/24 lessons。最强的 post-hoc 交互是
`procedure_recovery × variant 3 = 2/12`，同时存在 `act_v9` 过选、
`act_p3` 少选和 candidate position 2 过选；这些因素受冻结 probe schedule 混杂，
不能解释为单一因果机制。详细结果见
[`conditional-route post-hoc audit`](p0d2hfc-conditional-route-posthoc-audit.md)。

### 11.4 当前可支持与不可支持的新增主张

当前证据新增支持：

1. 在等 LoRA 参数预算下，late placement 相对 full placement 的优势能通过 repaired、
   untruncated hard-probe 配对检验；
2. 该相对优势在 conditional routing、conflict rejection 和 long-context retrieval
   中存在，但 binding-decoy category 仍不可用于确认性 layer-locus 主张；
3. 1.5B base canary 在 format-stable forced-choice endpoint 上达到 external
   `0.9089`，而自由生成的 exact-token 格式会显著低估这一能力；
4. 1.5B 的 remaining error 集中到 `conditional_route=0.6354`，不能用总体
   external 结果或其他三个 ceiling category 掩盖。

当前证据仍不支持：

1. 把 late placement 称为跨模型、跨深度或跨任务的普适机制；
2. 把 post-hoc semantic recovery 当成旧 strict gate 已通过；
3. 声称 1.5B 已解决整个 hard suite，或追溯放宽 `conditional_route` category gate；
4. 立即开始 LoRA narrow scan、1/4/8 mappings-per-adapter、GRPO/RLVR 或 router；
5. 将模型规模或 hidden-layer 深度认定为已经证明的唯一因果解释。

### 11.5 更新后的推荐执行顺序

1. 冻结并归档 P0-D2H-R、P0-D2H-CAL 和 invalid-audit 的 manifest、raw rows、
   aggregate、environment 与 provenance；
2. 归档已完成的 P0-D2H-CAL-FC formal output；其冻结结果是无 eligible model；
3. 归档已完成的 1.5B external `conditional_route` 只读错误审计；旧 endpoint 与
   gate 不变；
4. 归档已完成的 CRD、RR1 与 same-runtime 证据；RR1 不得继续训练或复用 checkpoint；
5. 若获得独立授权，只运行固定 CPR-v1：从干净 base 训练一次，并执行一次 locked
   same-runtime qualification；
6. 当前不得评审或执行 1/4/8 mappings-per-adapter 训练复杂度实验；即使 CPR-v1
   成为 qualification candidate，也必须先人工 review 并重新检查 formal gate；
7. 最后才考虑 held-out layer-locus 复现、P0-E GRPO/RLVR 和
   recurrence–volatility router。

已完成阶段的实现 prompt 保留于
[`P0-D2H format-stable calibration next-session prompt`](p0d2h-format-stable-calibration-next-session-prompt.md)；
正式结果与当前 follow-up 边界见
[`P0-D2H-CAL-FC results`](p0d2h-format-stable-calibration-results.md)。
拆分实验的冻结定义、矩阵和诊断边界见
[`P0-D2H-CRD protocol`](p0d2h-route-retrieval-decomposition-protocol.md)。
