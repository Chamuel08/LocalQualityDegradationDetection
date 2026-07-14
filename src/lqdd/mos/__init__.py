"""MOS 预测后端。

按 ``ReportConfig.mos_model`` 选择实现：
- ``rule``      — 见 ``lqdd.report.generator.compute_mos`` 的衰减公式（默认，零依赖）
- ``clip_iqa``  — 见 ``lqdd.mos.clip_iqa``（基于 pyiqa + CLIP-IQA，需可选依赖）
- ``internal``  — 预留，需自行实现 ``lqdd.mos.internal_model``
"""
