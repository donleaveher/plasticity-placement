# Notebook 目录

Notebook 只负责 Colab 环境初始化、调用项目命令、进度检查和结果可视化。核心编译、训练、
推理、恢复与聚合逻辑必须保留在 `src/` 中，避免单元格执行顺序影响可复现性。

第一个真实四载体实验的 Colab 预注册协议见
[`docs/p0c-colab-protocol.md`](../docs/p0c-colab-protocol.md)。Notebook 只编排已经实现在
`src/` 中的 canonical compiler、四臂 runner、manifest/resume 和聚合脚本。

## 强制组织规则

从 v4 开始，**每个新实验必须新建一个独立的 Colab notebook 文件**。
不得继续在已有 notebook 末尾追加新的实验、诊断臂或后续研究阶段。

以下情况均视为新实验，必须创建新文件：

- 新的执行阶段，例如 Confirmatory；
- 新的诊断实验，例如 exposure-matched ICL-K 或 hard-B；
- 新的研究问题，例如 recurrence–volatility 或 layer-locus；
- 修改 primary endpoint、lesson bank、compiler、模型、精度或冻结训练配置后的重跑。

新 notebook 应满足：

1. 文件名明确表达实验与版本，例如 `p0c_confirmatory_v4_colab.ipynb`；
2. 能独立完成 Drive 挂载、代码 revision 恢复、依赖安装和 provenance 检查；
3. 只运行一个实验，不依赖先执行其他 notebook 中的内存变量或隐藏单元格；
4. 明确声明 calibration/input 目录和本实验的独立 output 目录；
5. 包含运行开关、进度检查、失败诊断和聚合入口；
6. 将可复用实验逻辑放入 `src/`，notebook 内只保留编排代码；
7. 产生正式结果后冻结文件；若实验定义变化，创建新 notebook 或升级文件版本，而不是
   静默修改原实验入口。

## 当前入口

- [`p0c_v4/p0c_smoke_v4_colab.ipynb`](p0c_v4/p0c_smoke_v4_colab.ipynb)：只运行
  2-lesson Smoke v4；
- [`p0c_v4/p0c_calibration_v4_colab.ipynb`](p0c_v4/p0c_calibration_v4_colab.ipynb)：
  只运行 60-adapter
  Development Calibration v4，启动前验证 Smoke v4；
- [`p0c_v4/p0c_pilot_v4_colab.ipynb`](p0c_v4/p0c_pilot_v4_colab.ipynb)：只运行
  8-lesson Pilot v4，
  启动前验证 Smoke 和 Calibration v4；
- [`p0c_v4/p0c_confirmatory_v4_colab.ipynb`](p0c_v4/p0c_confirmatory_v4_colab.ipynb)：
  只运行 24 lessons × 3 seeds Tier 2 Confirmatory v4，启动前验证 Pilot v4 并要求显式
  确认 Pilot `GO`。

[`p0c_colab.ipynb`](p0c_colab.ipynb) 是 v4 拆分所依据的组合式 provenance/workflow
基线；[`p0c_confirmatory_colab.ipynb`](p0c_confirmatory_colab.ipynb) 是旧版
Confirmatory 入口。二者均保留用于历史审计，不再作为 v4 执行入口。

四个 v4 notebook 由
[`p0c_v4/build_p0c_v4_notebooks.py`](p0c_v4/build_p0c_v4_notebooks.py) 从同一模板生成，
确保 checkout、fingerprint、日志、manifest 校验和恢复策略一致。修改公共编排逻辑时应
先改生成器并重新生成全部 v4 文件，不要单独手改某一个生成文件。

P0-D layer-locus 已使用独立目录实现：

- [`p0d_lora_locus/p0d_band_scan_colab.ipynb`](p0d_lora_locus/p0d_band_scan_colab.ipynb)：
  `full/early/middle/late` × 24 lessons × 3 seeds；
- [`p0d_lora_locus/p0d_narrow_scan_colab.ipynb`](p0d_lora_locus/p0d_narrow_scan_colab.ipynb)：
  仅接受人工冻结的显式层/窗口条件。

生成、恢复和 Drive namespace 规则见
[`p0d_lora_locus/README.md`](p0d_lora_locus/README.md)。

P0-D2 参数预算匹配同样使用独立目录：

- [`p0d2_budget_match/p0d2_budget_match_colab.ipynb`](p0d2_budget_match/p0d2_budget_match_colab.ipynb)：
  7 conditions × 24 lessons × 3 seeds，共 504 adapter units；
- [`p0d2_budget_match/README.md`](p0d2_budget_match/README.md)：冻结矩阵、实际参数预算
  验证、恢复和 Drive namespace。

P0-D2H 困难 probe 诊断使用另一个独立目录：

- [`p0d2_hard_probe/p0d2_hard_probe_colab.ipynb`](p0d2_hard_probe/p0d2_hard_probe_colab.ipynb)：
  只读复用 verified P0-D2 adapters，运行 16 probes/lesson；
- [`p0d2_hard_probe/README.md`](p0d2_hard_probe/README.md)：四类 difficulty、
  5,376-row matrix、恢复和解释边界。

后续 ICL-K、hard-B、recurrence–volatility 和 P0-E GRPO/RLVR 仍必须各自创建新目录和
notebook。

## Drive 目录规则

- 不同实验必须使用不同输出目录；
- 同一实验正常断线且没有 `failed` unit 时，可从原目录恢复；
- 出现不可变 `failed` unit 时，保留原目录用于审计，修复后使用新的 run 目录；
- 不得删除失败记录后在原目录覆盖重跑；
- 模型、精度、代码、compiler、calibration 或超参数变化时必须使用新的 run 目录；
- v4 notebook 会根据 experiment code hash 和冻结模型/运行设置自动生成
  `/content/drive/MyDrive/plasticity-p0c/v4/pipelines/<pipeline-attempt>/runs/`
  下的 provenance 目录，并在其中使用 `smoke-a1`、`calibration-a1`、`pilot-a1`、
  `confirmatory-a1` 等 phase-attempt 目录；
- 四个阶段必须使用同一个 `PIPELINE_ATTEMPT`；首次执行会在该 pipeline 目录锁定精确
  code revision 和 model revision，后续 notebook 自动复用；
- 如果代码、模型或冻结设置改变，创建新的 `PIPELINE_ATTEMPT`，不得改写原 pipeline
  的 revision lock；
- P0-D2 使用
  `/content/drive/MyDrive/plasticity-p0d/budget-match/v1/pipelines/`，不得复用
  P0-D1 `lora-locus/v1` 的 pipeline 或 stage attempt；
- P0-D2H 使用
  `/content/drive/MyDrive/plasticity-p0d/hard-probe/v1/pipelines/`，只读引用
  P0-D2 manifest/summary/adapters，不得向 P0-D2 source stage 写入文件；
- 不同阶段允许由 Colab 分配不同 GPU；每次会话的 GPU、CUDA、Python、依赖版本和
  environment fingerprint 都会单独记录，不参与跨阶段目录寻址；
- 同一阶段在相同环境中的正常断线保持 attempt 名不变即可恢复；如果同一阶段恢复时
  GPU 或关键环境改变，或发生不可变 `failed` unit，则保留旧目录并递增该阶段 attempt，
  例如从 `pilot-a1` 改为 `pilot-a2`。

每个 notebook 将 lesson/seed adapter、原始结果和 manifest 写入对应的 Google Drive
目录。只有 smoke 的所有 unit 达到 `verified` 后，才应启动 60-adapter development
calibration；校准产生非空 `selected_config` 后，才能运行 Pilot 或 Confirmatory。

每个 v4 notebook 都在安装阶段把 Git branch 解析成精确 commit，并以 detached HEAD
运行；也可通过 `REQUESTED_CODE_REVISION` 显式指定 commit。它们同时冻结模型 revision、
代码 hash、运行设置和关键环境快照，将每次 Colab 会话写入 `source_sessions.json`，
并把本阶段 stdout/stderr 保存到对应 Drive 目录。代码、模型或冻结设置变化必须使用
新的 pipeline attempt；阶段运行环境变化则使用新的 stage attempt，不会与旧输出混用。
