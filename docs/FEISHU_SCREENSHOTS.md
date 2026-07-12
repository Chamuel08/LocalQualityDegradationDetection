# 飞书文档截图清单（本地素材）

> 素材目录：`benchmark/runs/feishu_export/`（已 gitignore，不上传 GitHub）  
> 生成命令：`python scripts/export_feishu_assets.py`  
> 需本地 `config.yaml` 配置 Ollama（`qwen2.5vl:7b` + `qwen2.5:3b`）

---

## 一、文档结构建议（飞书目录）

1. 封面：项目一句话 + GitHub 链接  
2. 架构图 / decision_trace  
3. **方案 A** 三张真实帧（定性）  
4. **方案 B** 指标表（定量）  
5. **方案 C** 批量统计（规模）  
6. **方案 D** VLM 对比（Agent 增值）  
7. 边界与诚实说明  

模板正文：`docs/FEISHU_EVALUATION.template.md`  
**本次预填版（直接复制到飞书）**：`docs/FEISHU_EVALUATION.filled.md`（gitignore）  
副本：`benchmark/runs/feishu_export/FEISHU_EVALUATION.filled.md`

---

## 二、方案 A — 粘贴清单

| 飞书小节 | 本地文件 | 说明 |
|----------|----------|------|
| 干净源帧 | `feishu_export/scenario_a/clean_source_input.png` | 原图 |
| 干净源帧 | `feishu_export/scenario_a/clean_source_overlay.png` | 应无 mask |
| 干净源帧 | 打开 `clean_source.html` 截图 Evidence 表 | MOS 4.5 |
| 合成劣化 | `synth_000000_input.png` + `synth_000000_overlay.png` | 橙色压缩 mask |
| 合成劣化 | `synth_000000.html` | texture_loss Evidence |
| 真实 CRF36 | `real_crf36_input.png` + `real_crf36_overlay.png` | 轻度 block |
| 真实 CRF36 | `real_crf36.html` | mild compression |

索引：`feishu_export/scenario_a/manifest.json`

---

## 三、方案 B — 粘贴清单

| 飞书小节 | 本地文件 |
|----------|----------|
| 指标总表 | `feishu_export/summaries/summary.txt` |
| 详细说明 | `feishu_export/summaries/RUN_SUMMARY.md` 中 B 节 |

**口述要点**：

- 支持类检出 **100%**（76/76 有 compression 检出）  
- IoU@0.3 **0%**（定位仍弱，诚实写进文档）  
- manifest 内 3 张 clean **仍 100% 误报**（与方案 A 干净帧表现不一致，可注明 manifest clean 子集更严）

原始 JSON：`benchmark/runs/scenario_b/results.json`

---

## 四、方案 C — 粘贴清单

数据来自 `benchmark/runs/scenario_c/`（JSON 仅本地）。

建议飞书表格：

| 目录 | 张数 | 有检出 | 平均 MOS | 备注 |
|------|------|--------|----------|------|
| taolive 干净 | 38 | 36/38 | 4.11 | 2 张零检出含 `10001737` |
| synthetic | 79 | 79/79 | 4.09 | compression 为主 |

可选截图：任选 1 张 `taolive_clean/*.json` 与 `synthetic/000000.json` 的 degradations 字段对比。

---

## 五、方案 D — VLM 对比（重点）

| 飞书小节 | 本地文件 | 说明 |
|----------|----------|------|
| v0.1 无 VLM | `feishu_export/scenario_d/legacy_v01_overlay.png` | conf ≈ 0.60 |
| Agent+VLM | `feishu_export/scenario_d/agent_vlm_overlay.png` | conf ≈ 0.75 |
| 对比数据 | `feishu_export/scenario_d/compare.json` | 一键复制数字 |
| VLM 推理 | `agent_vlm.html` 或 JSON 中 `vlm_reasoning` | 中文 reasoning 截图 |
| 决策链 | `agent_vlm.json` → `decision_trace` | `vlm_confirmed` + `judge_uncertain` |

**对比话术**：

```
灰区置信度 0.60 → 触发 VLM Confirm
VLM：确认背景块状压缩伪影（conf 0.85）
融合后置信度 0.60 → 0.75，MOS 保持 4.25
Judge：uncertain（未触发 Round 2）
```

同屏对比 HTML：

- `benchmark/runs/scenario_d/legacy_v01.html`  
- `benchmark/runs/scenario_d/agent_vlm.html`  

---

## 六、GitHub vs 飞书分工（给面试官）

| GitHub | 飞书本文档 |
|--------|------------|
| 代码 + `examples/` 小样例 | 真实 TaoLive / CRF / 合成集效果 |
| `pytest` 可复现 | B/C/D 跑分与截图 |
| 架构 spec | VLM reasoning 实录 |

README 飞书链接填本文档发布 URL。

---

## 七、本地复现命令速查

```bash
# 1. Ollama 模型（与 config.yaml 一致）
ollama list   # 需 qwen2.5vl:7b, qwen2.5:3b

# 2. 重跑 B/C/D（已完成可跳过）
python benchmark/run_eval.py --manifest /path/to/manifest.json \
  --output benchmark/runs/scenario_b/results.json

# 3. 导出飞书素材包
python scripts/export_feishu_assets.py

# 4. 打开素材目录
open benchmark/runs/feishu_export/scenario_a/
open benchmark/runs/feishu_export/scenario_d/
```
