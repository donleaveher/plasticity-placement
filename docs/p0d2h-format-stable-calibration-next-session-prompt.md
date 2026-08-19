# Next-session prompt: P0-D2H format-stable calibration

Copy the text below into a new Codex session.

---

你现在位于仓库：

```text
/Users/tourbillion/Documents/ICLR/agent_research/plasticity-placement
```

请实现下一阶段 **P0-D2H-CAL-FC: Base-Only Format-Stable Forced-Choice
Calibration**。开始前先检查当前分支、`git status` 和日志；基线至少应包含提交
`ba54377`（P0-D2H-CAL Invalid-Output Audit）。保留用户已有改动，不要自行 commit
或 push。

## 先读这些文件

完整阅读并以它们为事实来源：

- `docs/p0d2h-hard-probe-stress-protocol.md`
- `docs/p0d2hc-oracle-calibration-protocol.md`
- `docs/p0d2hc-invalid-output-audit.md`
- `docs/project-layout.md`
- `notebooks/README.md`
- `src/plasticity_placement/p0d2hc/`
- `notebooks/p0d2h_calibration/`
- `notebooks/p0d2h_invalid_audit/`
- 相关 `tests/test_p0d2hc_*.py`

按仓库现有规范使用 `planning-with-files` 和 `daily-coding`；先把当前任务写入
`task_plan.md`，实现过程中持续更新 `notes.md`。不要启用 subagent。

## 已验证事实

P0-D2H-R 在 0.5B 模型、固定 24-lesson cohort、等参数 LoRA 条件上得到：

- `late-matched` hard accuracy `0.4905`；
- `full-base` hard accuracy `0.3828`；
- paired late-minus-full `+0.1076 [0.0781, 0.1380]`；
- frozen locus label 为 `hard_locus_robust`；
- suite quality 因 0.5B external `0.4297` 和 binding floor 而失败。

P0-D2H-CAL 是 base-only 的三臂实验
`no_write/external/answer_copy_oracle`，没有 LoRA、parameter arm 或训练：

- 0.5B oracle `0.7891`，`oracle_failed`；
- 1.5B oracle `0.9948`，通过；
- 1.5B external strict `0.6901`，invalid `0.2318`；
- frozen cross-scale status 为 `mixed_scale_result`；
- `eligible_model_ids=[]`，不得自动开始训练或 narrow scan。

post-hoc invalid audit 覆盖 2,304 行，发现：

- 1.5B external semantic accuracy `0.9167 [0.8906, 0.9427]`；
- semantic-minus-strict `+0.2266 [0.1953, 0.2552]`；
- `87/89` external invalids 是唯一正确 action 加额外文本；
- external `long_context`：strict `0.1875`，semantic `1.0000`；
- external `conditional_route`：strict `0.5729`，semantic `0.6667`；
- 说明总体失败主要含格式问题，但 conditional routing 仍有真实语义/组合缺口；
- 这些 post-hoc 指标不能让旧 strict gate 追溯通过。

## 新实验要回答的问题

在不依赖自由生成格式的情况下，base model 是否能从四个冻结 action 中选出正确项？
具体区分：

1. `answer_copy_oracle` 是否能在 format-stable endpoint 上接近上界；
2. `external` 是否明显优于 `no_write`；
3. 1.5B 的整体 external selection 是否通过，同时
   `conditional_route` 是否仍失败；
4. 0.5B 与 1.5B 的差异是否仍支持 scale/capacity bottleneck；
5. 结果是否足以进入“另行评审”的 1/4/8 mappings-per-adapter 设计。

## 冻结实验边界

实现一个全新的 package、manifest/schema、CLI、tests、notebook 目录和 Drive
namespace。建议命名：

```text
src/plasticity_placement/p0d2hfc/
notebooks/p0d2h_forced_choice/
docs/p0d2h-format-stable-calibration-protocol.md
/content/drive/MyDrive/plasticity-p0d/
  hard-probe-forced-choice/v1/pipelines/
```

必须满足：

- 只读复用 verified P0-D2H-R hard-probe bank 和完整 P0-D2H-CAL source run；
- 固定模型为原 0.5B source revision 和 1.5B canary revision；
- 固定 24 lessons × 16 probes × 3 arms × 2 models = 2,304 decision rows；
- 三臂仍为 `no_write`、`external`、`answer_copy_oracle`；
- base-only；禁止发现、加载或训练 adapter；
- 禁止 parametric/LoRA 条件、narrow scan、GRPO/RLVR 和 1/4/8 training；
- 不修改 P0-D2H-R、P0-D2H-CAL 或 invalid-audit 的 raw rows、manifest、gate 和输出；
- 不自动启动任何下一阶段。

## Primary endpoint：四候选 full-string conditional scoring

不要让模型自由生成 primary prediction。对每条 prompt 的四个冻结 allowed actions
逐一计算 teacher-forced conditional log-likelihood：

1. 使用与原 calibration 完全一致的 chat template 和 prompt rendering；
2. candidate continuation 必须冻结 canonical leading whitespace；
3. 只累计 candidate token 的 log probabilities，不累计 prompt token；
4. primary score 冻结为 candidate sequence 的 `sum_logprob`；
5. 同时记录 `mean_logprob`、candidate token IDs/count、score margin 和排名，作为
   tokenizer/长度敏感性诊断；
6. primary prediction 是最高 `sum_logprob` 的 candidate；
7. non-finite score、无法验证 prompt/candidate 拼接、或并列最高分必须形成显式
   invalid/error，不能静默 tie-break；
8. action 顺序、expected action 和 prompt hash 必须与 source row/probe 完全一致。

每个 decision raw row 至少记录：

- source run/manifest/raw-row identity 和 hashes；
- model ID/name/immutable revision；
- lesson/probe/category/arm；
- expected action、ordered allowed actions；
- 每个 candidate 的 token IDs、token count、sum/mean logprob；
- predicted action、top-1/top-2 margin、correct、tie/error status；
- prompt hash、untruncated token count、environment/session provenance。

旧 strict-generation accuracy 只作为 source-linked secondary format metric 展示，
不要重定义或覆盖它。可以按稳定 row key 与 P0-D2H-CAL strict raw row 配对，报告
forced-choice minus strict，但新 endpoint 仍须拥有独立结果文件。

## 预先冻结的 metrics 和 gate

使用 lesson-clustered bootstrap（默认 10,000 次）报告：

- forced-choice accuracy 及 95% CI；
- model × arm × category accuracy；
- `external - no_write` paired difference 及 95% CI；
- `oracle - external` difference；
- top-1/top-2 score margin；
- sum-score 与 mean-score prediction disagreement rate；
- tie/error/non-finite rate；
- 与旧 strict generation 的 paired difference（secondary）。

在任何正式结果产生前，把以下 gate 写进 config identity、manifest 和 protocol；
不得看结果后调阈值：

- prompt/provenance/token audit 全部通过，零 silent truncation；
- tie/error/non-finite rate = `0`；
- oracle overall accuracy ≥ `0.95`；
- oracle every-category accuracy ≥ `0.90`；
- external overall accuracy ≥ `0.85`；
- external every-category accuracy ≥ `0.75`；
- no-write overall accuracy ≤ `0.35`；
- `external - no_write` paired 95% CI lower bound > `0`。

只有单个模型通过全部条件时，才可输出
`training_complexity_review_eligible`。这个状态只表示可以人工评审下一套冻结设计；
`automatic_training_started=false` 和
`automatic_narrow_scan_started=false` 必须始终成立。

若 overall external 通过但 `conditional_route` 低于 `0.75`，状态必须明确为
`category_calibration_failed`，不得用其他三个 ceiling category 掩盖。

## Provenance、恢复和输出

沿用现有项目的不可变约束：

- 在 formal scoring 前做完整 source、prompt、candidate-token audit；
- manifest identity 覆盖 source hashes、两个 model revisions、prompt renderer、
  candidate-scoring version、primary score definition、thresholds 和 code hash；
- 每个 model × lesson unit 原子写入；正常 Colab 断线可验证后 resume；
- `failed` unit 不得在原 attempt 覆盖，修复后必须使用新 attempt；
- 每个 model 只加载一次并常驻 CUDA，按 lesson/arm/probe 批量评分；
- 提供 progress heartbeat，明确哪些单元格是 CPU preflight，哪个单元格开始占用 GPU；
- aggregate 前再次验证完整 2,304-row matrix 和所有 hashes；
- 输出 manifest、environment/source sessions、raw rows、summary JSON、Markdown
  主表和 next-stage decision；
- notebook 只做编排，核心逻辑全部在 `src/`。

## Colab 和文档

生成独立 Colab：

```text
notebooks/p0d2h_forced_choice/p0d2h_forced_choice_colab.ipynb
```

同时提供 notebook generator 和 README，并把 Colab URL 加入 README/protocol：

```text
https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/p0d2h_forced_choice/p0d2h_forced_choice_colab.ipynb
```

更新：

- `docs/project-layout.md`
- `notebooks/README.md`
- 新 protocol 文档
- `pyproject.toml` CLI entry（如需要）

不要把新实验追加到旧 notebook。

## 测试与验收

至少覆盖：

- candidate-only logprob mask 正确，不计入 prompt token；
- canonical leading whitespace 与 token concatenation audit；
- four-candidate ranking、margin、tie/non-finite handling；
- sum/mean score 分离；
- source provenance/hash mismatch 拒绝；
- 完整矩阵行数与重复/缺失 row 拒绝；
- gate 的边界值和 `conditional_route` category failure；
- 不存在 adapter discovery/training code path；
- resume、immutable conflict 和 failed-attempt 语义；
- notebook 从 generator 可重复生成，cell 分块清楚；
- GPU formal cell 前不得加载模型到 CUDA。

完成后运行 focused tests、full repository tests、Ruff、notebook regeneration 和
`git diff --check`。最后给出：

1. 实现内容；
2. 科学边界；
3. 测试结果；
4. Colab URL；
5. 当前 `git status`；
6. 明确说明未 commit/未 push。

如果实现过程中发现 full-string scoring 与当前 runtime 的 chat-template/token 边界
不兼容，先用最小 tokenizer fixture 和真实 Qwen tokenizer 的 CPU audit定位，并把问题
写入 `task_plan.md`；不要退回 post-hoc parser，也不要改旧 gate。

---
