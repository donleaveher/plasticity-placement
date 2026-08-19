# P0-C v4 Colab 实验

本目录只保存 P0-C v4 的四个独立实验入口及其共享生成器。按以下顺序逐个运行：

1. `p0c_smoke_v4_colab.ipynb`
2. `p0c_calibration_v4_colab.ipynb`
3. `p0c_pilot_v4_colab.ipynb`
4. `p0c_confirmatory_v4_colab.ipynb`

四个 notebook 必须使用相同的 `PIPELINE_ATTEMPT`，但可以在不同 Colab Runtime 和不同
GPU 上执行。共享 pipeline 由精确代码 revision、模型 revision 和冻结设置识别；每个阶段
另外保存自己的环境记录和 stage attempt。

这些 notebook 由 `build_p0c_v4_notebooks.py` 生成。修改公共编排逻辑时，应修改生成器并
重新生成全部四个文件，不要只编辑某个生成后的 notebook。

完整组织规则和恢复策略见 [`../README.md`](../README.md)。
