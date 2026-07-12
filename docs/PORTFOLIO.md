# 作品集说明（GitHub 公开版）

本仓库面向面试官提供 **可克隆、可运行、可测试** 的代码与最小 demo。  
**完整评测效果、真实业务帧对比、方案 A–D 跑分** 见飞书文档（在根目录 README 填写链接）。

## 仓库里有什么

| 内容 | 路径 | 说明 |
|------|------|------|
| 检测核心 | `src/lqdd/` | GlobalScan、EdgeBleed、Compression、Agent、VLM 接口 |
| CLI | `detect.py` | 单张 / 批量 JSON·HTML |
| 内置样例图 | `data/sample/frames/` | 合成小图，零外部数据依赖 |
| 交互 demo | `examples/demo_report.html` | mask 轮廓 + legend + Evidence |
| 分场景报告 | `examples/reports/` | edge / block / normal 三份 HTML |
| 可视化对比 | `examples/viz_styles/` | contour_only vs contour_fill |
| 评测脚本 | `benchmark/run_eval.py` | 需自备 manifest，结果不提交 |
| 规格 | `specs/` | v0.1 / V1 Agent 契约与用例 |

## 仓库里没有什么（刻意不包含）

- 真实直播抽帧、TaoLive 路径、CRF 对比缓存
- 本地 `benchmark/runs/` 批量 JSON/HTML 跑分
- 面试速查、内部设计稿（`.gitignore`）
- 任何个人机器绝对路径

## 面试官 3 分钟体验路径

```bash
git clone https://github.com/Chamuel08/LocalQualityDegradationDetection.git
cd LocalQualityDegradationDetection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml
python scripts/generate_synthetic_samples.py   # 若无 sample 图
python scripts/generate_demo_assets.py
python detect.py --image data/sample/frames/block/block_01.png \
  --mode fast --legacy-fixed --output /tmp/report.html
open examples/demo_report.html
```

## 能力对应关系

| 能力 | 代码证据 |
|------|----------|
| 局部画质分析 | `detectors/` + `region_mask_rle` + HTML mask 预览 |
| 可解释性 | `Evidence`（method/metric/value/threshold/detail） |
| Agent 编排 | `agent/orchestrator.py` + `decision_trace` |
| VLM 应用 | `vlm/confirm.py`（灰区；需 Ollama，见 README） |

## 飞书文档建议结构

复制 [`FEISHU_EVALUATION.template.md`](FEISHU_EVALUATION.template.md) 到飞书后填入截图与链接。
