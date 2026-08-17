[Open this notebook in Google Colab](https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/agent%2Fadd-lora-evaluation/notebooks/pathmem_g1_p0/pathmem_g1_p0_colab.ipynb)

# PathMem G1 / P0 Colab workflow

Notebook: `pathmem_g1_p0_colab.ipynb`

运行顺序：

1. 保持控制单元默认值，运行全部单元；只验证 G0 并显示 G1/P0 计划，不训练。
2. 只诊断失败的 Recipe A 时，保持两个运行开关为 `False`，设置 `USE_GOOGLE_DRIVE=True`、`RUN_LABEL="pathmem-v1"`；结果单元会读取旧 artifact，不会训练。
3. 保留已经失败的 Recipe A 目录；选择支持原生 BF16 的 NVIDIA GPU（推荐 A100；L4 也可），使用新的 `RUN_LABEL`，在唯一的控制单元设置 `USE_GOOGLE_DRIVE=True`、`RUN_G1=True`、`G1_RECIPE="B"`，并填写作为责任人标识的 `APPROVER`。
4. G1 全部门槛通过并生成 `g1_authorization.json` 后，保持同一个 `RUN_LABEL` 和 `APPROVER`，改为 `RUN_G1=False`、`RUN_P0=True`，再次运行全部单元。

Recipe B 固定为 rank 16、alpha 32、学习率 `2e-4`、每事件 16 个 optimizer steps；它保持 Recipe A 的 `alpha/r` 缩放和冻结的 step exposure，只把 LoRA 条件容量加倍。G1 使用 12 个 `interface_dev` 项的 24 个 current-only anchors。P0 只接受同一 Recipe B 产生并验证的 G1 授权，随后建立 4 个 `smoke` 项的 40 个核心 DAG 节点和 4 个技术重复，共 44 个训练 artifact。中断后使用相同控制值重新运行会复用并重新验证已有 artifact；已冻结内容不允许静默覆盖。

结果单元会用 CPU 运行 `diagnose-g1`，从现有 G1 结果生成动作混淆矩阵、候选概率边际、base→adapter 配对变化和 unrelated correctness transitions。它是只读诊断，不改变 gate，也不能授权 P0。

注意：上述 Colab URL 只有在本目录及相关实现被提交并推送到 `agent/add-lora-evaluation` 分支后，才会从 GitHub 正常打开。
