# Notebook 目录

Notebook 只负责 Colab 环境初始化、调用项目命令和结果可视化。核心环境、训练与实验逻辑必须保留在 `src/` 中，避免单元格执行顺序影响可复现性。

第一个真实四载体实验的 Colab 预注册协议见
[`docs/p0c-colab-protocol.md`](../docs/p0c-colab-protocol.md)。Notebook 只编排已经实现在
`src/` 中的 canonical compiler、四臂 runner、manifest/resume 和聚合脚本。

当前按实验阶段提供独立入口：

- [`p0c_colab.ipynb`](p0c_colab.ipynb)：历史 Smoke、Development Calibration 和 Pilot
  入口；
- [`p0c_confirmatory_colab.ipynb`](p0c_confirmatory_colab.ipynb)：独立的 24 lessons ×
  3 seeds Tier 2 Confirmatory 入口。

每个 notebook 将 lesson/seed adapter、原始结果和 manifest 写入对应的 Google Drive
目录。只有 smoke 的所有 unit 达到 `verified` 后，才应启动 60-adapter development
calibration；校准产生非空 `selected_config` 后，才能运行 Pilot 或 Confirmatory。

Notebook 首次运行会把代码 commit 固定到 Drive 的 `code-revision-v2.txt`；重连不会
自动漂移到分支的新 HEAD。`smoke-v2`、`calibration-v2` 和 `pilot-v2` 对应 compiler v3，
不要与旧输出目录混用。

Confirmatory notebook 不复用上述 revision 指针，而是从所选成功
`calibration_report.json` 的 selected candidate 环境快照中恢复精确 Git commit，并在
创建实验 manifest 前核对 code hash。每次新的 Confirmatory 尝试使用新的
`CONFIRMATORY_RUN` 目录名；发生不可变 `failed` unit 时保留原目录用于审计，不覆盖重跑。
