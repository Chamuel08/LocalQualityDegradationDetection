# 局部画质劣化检测 — 评测与效果说明（飞书文档模板）

> 将本文复制到飞书，替换 `【】` 占位符并粘贴截图。  
> GitHub 仓库仅提供代码与内置小样例；**本文档承载完整效果证明**。

---

## 1. 项目一句话

离线 badcase 单帧 → **局部 mask 定位** + **数值 Evidence** + **MOS 归因**；V1 用 Agent 编排，灰区可选 VLM 确认。

- 代码仓库：【填写 GitHub 链接】
- 作者：【姓名】

---

## 2. 架构（可贴架构图或 decision_trace 截图）

```
输入单帧 → GlobalScan → Router → EdgeBleed / Compression
         → [灰区 VLM Confirm] → LLM Judge → [Round 2] → HTML/JSON 报告
```

**要点**：

- 无参考图 diff，单帧 NR 启发式 + 可配置阈值
- 输出 `region_mask_rle` 不规则区域，非粗矩形框
- HTML：彩色轮廓 + 图下 legend

---

## 3. 方案 A — 单张真实帧验证（肉眼）

| 样本 | 说明 | MOS | 检出 | 截图 |
|------|------|-----|------|------|
| 干净源帧 | 【如 TaoLive 干净帧】 | 【4.5】 | 无 | 【粘贴】 |
| 真实压缩 | 【如 H.264 CRF36】 | 【4.35】 | compression mild | 【粘贴】 |
| 合成劣化 | 【如 compression_hf #000000】 | 【4.25】 | texture_loss | 【粘贴】 |

**可视化**：优先展示 **contour_fill**（半透明填充 + 轮廓 + legend）。

【在此粘贴三张对比图或 HTML 截图】

---

## 4. 方案 B — 有 GT 批量评测（数字）

数据集：【简述：真实底图 + 程序化合成 + manifest bbox】  
支持类型：compression / green_spill（v0.1 仅两类）

| 方法 | 支持类 recall | IoU@0.3 | 干净误报 |
|------|--------------|---------|----------|
| lqdd v0.1 | 【%】 | 【%】 | 【%】 |
| Oracle 上界 | 100% | 100% | 0% |

**口径**：分类型报告，不把 blur/mosaic 等未支持类型混入总准确率。

【粘贴 summary 表截图】

---

## 5. 方案 C — 目录批量（规模）

| 目录 | 张数 | 有检出 | 平均 MOS | 说明 |
|------|------|--------|----------|------|
| 干净源帧目录 | 【38】 | 【x/38】 | 【】 | 误报率 |
| 合成劣化目录 | 【79】 | 【】 | 【】 | 类型分布 |

---

## 6. 方案 D — V1 Agent + VLM

| 项 | 结果 |
|----|------|
| 样本 | 【单张灰区帧】 |
| 无 Ollama | VLM skip，与 v0.1 一致 |
| 有 Ollama | 【vlm_reasoning 截图 / trace】 |
| Judge | 【consistent / round2】 |

【粘贴 Agent trace 或 VLM reasoning 截图】

---

## 7. 可视化方案选型

GitHub `examples/viz_styles/` 提供内置小样例对比：

| 方案 | 说明 | 推荐 |
|------|------|------|
| contour_only | 仅彩色轮廓 | 轻量 |
| contour_fill | 轮廓 + 半透明填充 + legend | **作品集推荐** |

【粘贴两种风格对比图】

---

## 8. 边界与诚实说明

- 当前 v0.1 **仅支持** 绿边溢色、压缩伪影两类
- `texture_var_reference` 等为配置常数，非参考图 diff
- 合成 GT 精确，与真实 badcase 仍有 domain gap
- 实时流不在范围，离线单帧

---

## 9. 复现（仅本人本地，不写绝对路径到 GitHub）

评测数据与 `benchmark/runs/` 保留在本地，通过本飞书文档展示结果；公开仓库使用 `data/sample/` 即可复现 demo。

```bash
# 公开 demo（面试官用 GitHub 即可）
python scripts/generate_demo_assets.py
open examples/demo_report.html
```

---

## 10. 联系方式

【邮箱 / 微信 / 其他】
