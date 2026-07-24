# P0-C 当前实验结果解释：从工程闭环到 8-Lesson Pilot

- **结果日期：** 2026-07-24
- **实验阶段：** Tier 0 Smoke + Development Calibration + Tier 1 Pilot
- **结论状态：** 方向性 Go/No-Go 证据，不是 confirmatory evidence
- **模型：** `Qwen/Qwen2.5-0.5B-Instruct`
- **代码版本：** `dea54bb8f17d9b93ec1e9bc4fb17eb3b9f642562`
- **训练种子：** 42（Pilot 仅 1 个 seed）
- **运行环境：** Colab A100；4-bit；实验产物持久化到 Google Drive

本文解释 P0-C 当前已经完成的 Smoke、LoRA 校准和 8-lesson Pilot。完整预注册设计见
[`p0c-colab-protocol.md`](p0c-colab-protocol.md)，实现与验证状态见
[`p0c-implementation-report.md`](p0c-implementation-report.md)。

> **一句话结论：** 参数写入在 Pilot 中相对 No-write 产生了稳定的目标泛化增益，
> External 与 Parametric 呈现了值得继续检验的行为差异，且没有出现系统性 invalid
> 或大面积干扰；因此 Tier 1 判定为 **GO**，可以进入 24 lessons × 3 seeds 的
> Confirmatory P0-C。这个结论还不能支持统计显著性、跨模型泛化、动态载体切换或参数
> 写入层位置主张。Both 的满分首先应视为当前 lesson/probe 难度造成的 ceiling-effect
> 警告，而不是 dual-write 已经被证明全局最优。

## 1. P0-C 在整个研究中的位置

完整研究问题是：对于同一条 Agent 经验，应该不保存、写入外部文本记忆、写入模型参数，
还是同时写入两者？如果参数记忆有真实价值，后续才进一步研究应该写入哪些 Transformer
层或模块。

P0-C 是这条研究路线的最小资格实验。它固定参数写入算子为全层 LoRA，只比较四种载体：

| Arm | 含义 | Prompt 中的记忆 | 参数更新 |
|---|---|---|---|
| N: No-write | 不保存经验 | 无 | 无 |
| E: External | 外部文本记忆 | Oracle-retrieved canonical note | 无 |
| P: Parametric | 参数记忆 | 无 | Lesson-specific LoRA |
| B: Both | 双写 | Canonical note | 同一个 Lesson LoRA |

P0-C 不测试 recurrence–volatility 动态相图，也不测试 early/middle/late layer placement。
它只检查：

1. LoRA 是否真正写入了 held-out target behavior；
2. External 和 Parametric 是否具有不同的收益—干扰轮廓；
3. Both 是互补、重复，还是产生冲突；
4. 差异是否可能仅由 action-token prior、解析错误或 base 已知答案解释。

## 2. 已完成的实验链路

### 2.1 Tier 0 Smoke

两条 development lessons 的 N/E/P/B/rollback 闭环均完成：

- 两条 base arms 均为 `verified`；
- 两个 adapter units 均为 `verified`；
- disabled-adapter rollback exact-match rate 均为 `1.0`；
- adapter 保存、加载、推理、严格解析和 Drive manifest 恢复均通过。

Smoke 只证明实验管线能够按设计运行，不构成载体效果证据。

### 2.2 Development Calibration

校准使用预注册的 18 组配置：

- rank：4、8、16；
- learning rate：`5e-5`、`2e-4`；
- optimizer steps：4、16、64。

Successive halving 共训练 60 个 development adapters：

- Stage 1：18/18 个候选完成，36/36 adapters verified；
- Stage 2：6/6 个幸存候选完成，24/24 adapters verified；
- 0 个 adapter 失败；
- 4/6 个 Stage 2 候选通过全部校准门槛。

按照预注册的 “更少 steps → 更低 rank → 更短 write time → 更高 development TG”
顺序，冻结配置为：

| Hyperparameter | Value |
|---|---:|
| rank | 4 |
| alpha | 8 |
| learning rate | `2e-4` |
| optimizer steps | 16 |
| target modules | `q_proj`, `v_proj` |
| layer band | full |
| dropout | 0 |

校准在本次 A100 会话中约耗时 10.7 分钟。确切 model revision、precision、环境快照和
报告 hash 保存在 `calibration_report.json`，正式复现时应以该文件为准。

### 2.3 Tier 1 Pilot

Pilot 使用 8 条 lessons、1 个训练 seed 和完整 26-probe panel。实际选择为：

- Fact mapping：`F_pair06_a/b`、`RF_pair01_a/b`；
- Procedure recovery：`P_pair01_a/b`、`P_pair03_a/b`。

其中 6 条来自 confirmatory split，2 条来自预先构造的 reserve split。引入 reserve pair
是为了在 base screening 后仍保持完整 counterbalanced pairs 和 action-token balance，
不是根据 Pilot 结果事后挑选。四个 action token 在 8 条 lessons 中各出现 2 次。

Pilot 的训练、四臂推理、rollback 和聚合均完成；8 个 adapter units 全部 verified。
本次 A100 会话中的 Pilot 单元格约运行 6 分钟。

## 3. Pilot 主结果

### 3.1 四臂行为与成本

下表报告 lesson-level 聚合均值及 bootstrap 95% CI。效率指标为聚合结果中的均值；原始
summary 还保留 median 和 IQR。

| Arm | Target generalization ↑ | 95% CI | Interference regression ↓ | 95% CI | Exact held-out ↑ | Conflict robustness ↑ | Invalid rate ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| No-write | 0.172 | [0.109, 0.250] | 0.000 | [0.000, 0.000] | 0.250 | 0.156 | 0.000 |
| External | 0.906 | [0.812, 1.000] | 0.000 | [0.000, 0.000] | 1.000 | 0.594 | 0.000 |
| Parametric | 0.797 | [0.594, 0.953] | 0.010 | [-0.104, 0.146] | 0.938 | 0.562 | 0.000 |
| Both | 1.000 | [1.000, 1.000] | 0.010 | [-0.104, 0.146] | 1.000 | 0.969 | 0.000 |

主要成本观测：

| Arm | Median latency (s) | Mean input tokens | Write time (s) | Adapter bytes | Peak memory |
|---|---:|---:|---:|---:|---:|
| No-write | 0.223 | 83.2 | 0.0 | 0 | 0 |
| External | 0.223 | 104.0 | 0.0* | 0 | 0 |
| Parametric | 0.269 | 83.2 | 5.72 | 1,094,753 | 969 MB |
| Both | 0.268 | 104.0 | 5.72 | 1,094,753 | 969 MB |

\* External 的 `write_time=0` 是当前 runner 的记账方式，表示没有参数训练；它没有覆盖真实
生产系统中的记忆生成、索引、检索、存储和维护成本。因此不能据此声称 External 写入没有
成本。

### 3.2 配对处理效应

| Contrast | Target difference | 95% CI | Target W/T/L | IR difference | 95% CI |
|---|---:|---:|---:|---:|---:|
| P − N | +0.625 | [0.484, 0.750] | 8 / 0 / 0 | +0.010 | [-0.104, 0.146] |
| E − N | +0.734 | [0.672, 0.797] | 8 / 0 / 0 | 0.000 | [0.000, 0.000] |
| P − E | −0.109 | [-0.266, 0.031] | 1 / 3 / 4 | +0.010 | [-0.104, 0.146] |
| B − best single | +0.062 | [0.000, 0.156] | 2 / 6 / 0 | +0.042 | [-0.031, 0.156] |

这里的 W/T/L 以 contrast 左侧 arm 的 target generalization 为准。例如 P − E 的
1/3/4 表示：P 在 1 条 lesson 上优于 E，3 条相同，4 条低于 E。

## 4. 如何解释这些结果

### 4.1 RQ1：参数写入有效吗？

当前 Pilot 给出明确的方向性肯定信号：

- P 的 target generalization 为 0.797，而 N 为 0.172；
- P − N 的平均 target gain 为 +0.625；
- 8/8 lessons 上 P 均优于 N；
- P 的 exact held-out 从 N 的 0.250 提高到 0.938；
- invalid rate 仍为 0；
- 平均 interference regression 仅为 0.010。

因此，当前 LoRA 后端不是“adapter 被保存了但行为没有改变”的空写入。至少在该模型、该
lesson bank、该冻结配置和该 seed 下，参数写入确实改变了 held-out behavior。

这仍然只是 Pilot 证据。正式结论需要 24 lessons × 3 seeds 检查效果是否在训练随机性下
稳定。

### 4.2 RQ2：External 和 Parametric 是否不同？

Pilot 中 E 的平均 target generalization 比 P 高 0.109。逐 lesson 看，P 有 1 胜、3 平、
4 负；根据 per-lesson pair table，至少 5/8 lessons 的 `|TG(P)-TG(E)|` 达到 0.10。
这超过了 Pilot 预注册的 “至少 2 条 lesson 存在非平凡载体差异” 门槛。

但是，P − E 的 95% CI 为 `[-0.266, 0.031]`，仍跨过 0。正确解释是：

- 当前结果提供 “E 在该 Pilot 中方向上更强” 的信号；
- 当前样本不足以声称 E 在统计上优于 P；
- P − E 的 target CI 和 IR CI 都没有完全落入预设的 ±0.05 等价区间，因此也没有
  practical equivalence 证据；
- “没有等价证据” 不等于已经证明二者不等价。

External 是 oracle retrieval 上界：相关 target query 总能获得正确 canonical note。
因此 E 的高表现不能直接外推到带检索错误、索引噪声和上下文竞争的真实系统。

### 4.3 RQ3：Both 是互补还是重复？

B 在本轮 Pilot 中达到：

- target generalization = 1.000；
- exact held-out = 1.000；
- conflict robustness = 0.969；
- invalid rate = 0。

相对每条 lesson 的最佳单载体，B 的平均 target gain 为 +0.062，2 胜、6 平、0 负。这说明
External note 与 Parametric adapter 在少数 lessons 上可能互补，而且没有观察到由内容
冲突导致的系统性失败。

但当前最直接的替代解释是 **lesson/probe 对同时获得两份一致记忆的 B 来说过于简单**：

- 输出空间只有 4 个离散 action tokens；
- canonical lesson 是确定性的局部规则，正确载体一旦被读取就容易完成；
- E 使用 oracle retrieval，不存在漏检和错检；
- E 与 P 写入的内容完全一致，不包含过期、矛盾或部分可靠的记忆；
- 每个 adapter 只写入一条 lesson，没有累积写入造成的容量竞争；
- target evaluation 是短程决策，不要求长轨迹中的状态维护和多步组合。

在这种设置下，B 相当于获得两份一致证据，达到 1.0 更可能说明当前 RQ3 manipulation
缺乏区分度，而不是说明双写策略已经解决了载体选择问题。增加 lesson 数量和训练 seed
只能缩小不确定性，不能自动消除每条 lesson 的任务天花板。

但 B 同时承担：

- External 的额外 input tokens；
- Parametric 的训练时间、adapter 存储和 GPU memory；
- 双份记忆的生命周期与一致性维护成本。

因此，本轮结果对 Both 的正确结论是 **不可判定**：既不能把满分解释为可靠互补，也不能
直接把它作为削弱载体选择假设的反例。Confirmatory 应保留原始 B arm 作为可复现性检查，
同时在正式运行前预注册一个单独报告的 hard-B diagnostic panel，例如多步组合规则、长程
程序状态、固定上下文预算、受控错检/过期记忆和多 lesson 累积写入。只有 B 在这些有区分度
的条件下仍稳定不劣，并且真实成本可接受，研究问题才应转向 dual-write 的成本 break-even、
一致性维护和动态回滚。

### 4.4 RQ4：差异是 action-token prior 吗？

不是由单个 action token 独占驱动。每个 action 在 Pilot 中恰好对应 2 条 lessons，且 P 在
四个 action 上均优于 N：

| Desired action | N target | P target | P − N |
|---|---:|---:|---:|
| `act_k2` | 0.188 | 0.625 | +0.438 |
| `act_n7` | 0.125 | 0.875 | +0.750 |
| `act_p3` | 0.125 | 0.750 | +0.625 |
| `act_v9` | 0.250 | 0.938 | +0.688 |

这排除了“P 只学会始终输出某一个 action token”的简单解释。

同时，action 之间仍有明显异质性：P 从 `act_k2` 的 0.625 到 `act_v9` 的 0.938，相差
0.313。最突出的是 `F_pair06`：同一 counterbalanced pair 的两个 P target 分别为 0.25
和 1.00，pair 内跨度为 0.75。这可能来自 lesson 内容、action token、训练随机性或它们的
交互，需要在 Confirmatory 的多 seed 分析中重点检查。

## 5. Pilot Go/No-Go 判定

预注册的四项 Pilot 门槛均满足：

| Gate | Observed evidence | Decision |
|---|---|---|
| 至少 6/8 lessons 的 P 相对 N 有正 TG | 8/8 | Pass |
| 至少 2 条 lessons 的 E/P `|ΔTG|` 或 `|ΔIR| ≥ 0.10` | 至少 5/8 的 `|ΔTG| ≥ 0.10` | Pass |
| 不出现所有 P adapters 大面积无关任务退化 | P mean IR = 0.010；无系统性 invalid | Pass |
| 至少一个非平凡 pair，且不能由 action prior 解释 | 完整 pairs；四 action 平衡且均有 P − N 正增益 | Pass |

因此：

> **Tier 1 Pilot decision: GO to Confirmatory P0-C.**

这个 GO 只表示继续投入实验资源是合理的，不表示主假设已经得到 confirmatory 支持。

## 6. 当前结果能支持什么

在限定语境下，可以说：

1. P0-C 工程闭环、严格解析、manifest 恢复和 rollback 已通过；
2. 冻结的 LoRA 后端在 Pilot 中产生了强且一致方向的 P − N target gain；
3. Oracle External 和 Parametric 在 lesson level 呈现了值得继续检验的差异；
4. 当前没有系统性 invalid output 或大面积 collateral interference；
5. action-token counterbalancing 排除了单一 action prior 作为全部效果来源；
6. Both 在本轮接近上界，但更可能暴露了当前 lesson/probe 的 ceiling effect；它暂时不能
   区分可靠互补与任务过于简单。

## 7. 当前结果不能支持什么

当前不能声称：

1. E 在统计上显著优于 P；
2. E 与 P 已经被正式证明不等价；
3. 事实型经验应该写入 External、程序型经验应该写入 Parametric；
4. 最佳载体会随复用率或环境漂移发生因果切换；
5. 某个 Transformer 层区间是最佳写入位置；
6. 结果可跨模型、跨任务或跨随机种子泛化；
7. Both 是全局最优策略；
8. External 在现实系统中的写入和读取成本为零。

## 8. 主要局限与审稿风险

1. **样本量小。** Pilot 只有 8 条 lessons，bootstrap CI 较宽。
2. **只有一个训练 seed。** 目前无法区分稳定 carrier effect 与 adapter 随机性。
3. **筛选后样本。** Lessons 经过 base screening，不是从全部任务分布随机抽样。
4. **使用 reserve lessons。** 实际 Pilot 为 6 条 confirmatory + 2 条 reserve；正式报告
   必须透明说明，并确保 Confirmatory 的选择规则在运行前冻结。
5. **Oracle External。** 没有真实 retrieval errors，E 是理想化上界。
6. **Exposure 可能不完全匹配。** P 训练看到了多种表述，而 E 使用 canonical note；
   差异较大的 lessons 应运行预注册的 exposure-matched ICL-K diagnostic。
7. **模型规模有限。** 0.5B 模型适合 Colab 资格实验，但不支持跨规模主张。
8. **Both 存在任务天花板。** 当前离散 action、oracle retrieval、同内容双写和单 lesson
   adapter 使 B 很容易满分；原始 panel 对 RQ3 的区分力不足，不能把 `B=1.0` 直接解释为
   dual-write 全局支配。
9. **成本核算不完整。** 当前没有真实 external store/retrieval 的端到端成本。
10. **尚未进行显著性或多 seed 稳定性结论。** Pilot 只报告 effect direction、CI 和
    wins/ties/losses，不应包装成论文主表证据。

## 9. Confirmatory 实验需要回答什么

下一阶段保持代码、模型 revision、量化状态、校准配置和 probe compiler 冻结，运行：

- 24 lessons：12 fact + 12 procedure；
- 3 training seeds；
- 72 个正式 adapters；
- N/E 每 lesson 运行一次；
- P/B 对每个 seed 运行；
- lesson-clustered paired bootstrap，10,000 次；
- rollback exact match 必须保持 1.0。

Confirmatory 的重点不是简单复现四臂均值，而是回答：

1. P − N 的正 target gain 是否跨 seed 稳定；
2. P − E 的 target/IR CI 是否支持差异、等价或仍然不确定；
3. P 胜 E 和 E 胜 P 的双向 lesson heterogeneity 是否可靠；
4. `F_pair06` 式 pair 内极端差异是否在其他 seed 重现；
5. action-token effect 是否小于 carrier effect；
6. B 在原始 panel 和预注册 hard-B diagnostic panel 上是否仍近乎全局不劣，以及其收益
   是否足以覆盖双写成本；
7. 最大 P/E gap 是否能被 exposure-matched ICL-K 解释。

hard-B panel 不应根据 Pilot 中具体失败案例临时挑题。更稳妥的做法是先定义难度维度和固定
生成规则，再重新编译全部 hard probes，并将其作为独立 secondary/diagnostic analysis；
若修改原始 primary endpoint 或 lesson bank，则应升级协议版本并重新运行 Pilot。

只有 Confirmatory 同时显示参数记忆有效、P/E 差异稳定、异质性跨 seed 可复现，并且
hard-B diagnostic 对双写仍有足够区分度，才应进入 recurrence–volatility 动态环境和
layer-locus 扫描。

## 10. 建议用于论文的保守表述

在 Confirmatory 完成前，可在内部进展报告中使用：

> In an eight-lesson, single-seed pilot with action-balanced counterfactual pairs,
> calibrated parametric memory improved held-out target generalization over no-write
> on all eight lessons (mean paired gain 0.625). Oracle external memory achieved a
> higher mean target score than parametric memory, while the paired P–E interval
> remained wide and crossed zero. The pilot therefore supports scaling to the
> preregistered multi-seed confirmatory study, but does not establish statistical
> superiority, practical non-equivalence, or cross-model generality.

不应使用：

> Parametric and external memory are significantly different, proving that agents
> require a learned memory-substrate router.

后一句超出了当前 Pilot 的样本量、统计证据和实验范围。
