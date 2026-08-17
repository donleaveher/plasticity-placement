[Open this notebook in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/pathmem_g1_p0/pathmem_g1_p0_colab.ipynb)

# PathMem G1 / P0 Colab workflow

Notebook: `pathmem_g1_p0_colab.ipynb`

运行顺序：

1. 保持控制单元默认值，运行全部单元；只验证 G0 并显示 G1/P0 计划，不训练。
2. 选择支持原生 BF16 的 NVIDIA GPU（推荐 A100；L4 也可），在唯一的控制单元设置 `USE_GOOGLE_DRIVE=True`、`RUN_G1=True`，并填写作为责任人标识的 `APPROVER`。
3. G1 全部门槛通过并生成 `g1_authorization.json` 后，保持同一个 `RUN_LABEL` 和 `APPROVER`，改为 `RUN_G1=False`、`RUN_P0=True`，再次运行全部单元。

G1 使用 12 个 `interface_dev` 项的 24 个 current-only anchors。P0 只有在采用并验证 G1 授权后才会建立 4 个 `smoke` 项的 40 个核心 DAG 节点和 4 个技术重复，共 44 个训练 artifact。中断后使用相同控制值重新运行会复用已验证 artifact；已冻结内容不允许静默覆盖。

注意：上述 Colab URL 只有在本目录及相关实现被提交并推送到 `agent/add-lora-evaluation` 分支后，才会从 GitHub 正常打开。
