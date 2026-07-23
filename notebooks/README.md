# Notebook 目录

Notebook 只负责 Colab 环境初始化、调用项目命令和结果可视化。核心环境、训练与实验逻辑必须保留在 `src/` 中，避免单元格执行顺序影响可复现性。

第一个真实四载体实验的 Colab 预注册协议见
[`docs/p0c-colab-protocol.md`](../docs/p0c-colab-protocol.md)。Notebook 只编排已经实现在
`src/` 中的 canonical compiler、四臂 runner、manifest/resume 和聚合脚本。

当前入口为 [`p0c_colab.ipynb`](p0c_colab.ipynb)。默认运行 `smoke` tier，并将每个
lesson/seed 的 adapter、原始结果和 manifest 写入 Google Drive。只有 smoke 的所有 unit
达到 `verified` 后，才应启动 60-adapter development calibration；校准产生非空
`selected_config` 后，才能使用新的输出目录运行 `pilot`。

Notebook 首次运行会把代码 commit 固定到 Drive 的 `code-revision-v2.txt`；重连不会
自动漂移到分支的新 HEAD。`smoke-v2`、`calibration-v2` 和 `pilot-v2` 对应 compiler v3，
不要与旧输出目录混用。
