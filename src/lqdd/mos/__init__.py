"""MOS 预测后端。

帧级 MOS 总分由无参考画质模型直接预测（不再硬编码 per-distortion 扣分）。
按 ``ReportConfig.mos_model`` 选择实现：

- ``clip_iqa``  — 见 ``lqdd.mos.clip_iqa``（基于 pyiqa + CLIP-IQA，默认，需可选依赖）
                  依赖缺失 / 权重下载失败 / 推理异常时 overall_mos=null 并给出原因，不回退默认分
- ``internal``  — 预留，需自行实现 ``lqdd.mos.internal_model``
"""
