# LocalQualityDegradationDetection

**无参考、可解释的局部画质劣化检测**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

给一张图（或一段视频），自动找出**哪里画质变差了、差在哪、为什么**——输出像素级定位、数值证据、画质评分和可回溯的决策过程。不需要原始清晰图作对比，全程本地推理，不依赖云端 API。

## 它解决什么问题

传统画质评估的两个痛点：① 只给一个总分，不知道差在哪；② 需要原始参考图做 diff，实际业务里几乎拿不到。本项目在**无参考**前提下，定位**局部**劣化区域（压缩块效应 / 模糊 / 马赛克 / 色带 / 绿幕溢色 / 面部 / 发丝 / 背景 / 手部），给出 bbox + 像素 mask + 数值证据 + 根因，再用本地 LLM+VLM 的 ReAct Agent 自主决定是否做视觉二次确认。

## 30 秒上手

```bash
git clone https://github.com/Chamuel08/LocalQualityDegradationDetection.git
cd LocalQualityDegradationDetection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp config.example.yaml config.yaml

# 无需 Ollama，直接出 JSON
python detect.py --image docs/demo/case_blur.png --legacy-fixed
```

跑完你会得到：`overall_mos`（画质分）+ `degradations[]`（每条含 bbox / mask / evidence / root_cause）+ `decision_trace`（可回溯决策链）。加 `--config config.yaml` 走 V1 ReAct Agent（需 Ollama），加 `--output report.html` 出可视化报告。

## 核心能力

### 可解释归因

九个专用 CV 检测器（压缩块效应 / 模糊 / 马赛克 / 色带 / 绿幕溢色 / 发丝 / 面部 / 背景 / 手部）定位局部劣化区域，每条 `degradations[]` 给出三层可解释结构：**L1 evidence**（数值判定）→ **L2 root_cause**（根因）→ **L3 vlm_reasoning**（VLM 语义确认 + UX 影响）。归因与 MOS 解耦——MOS 只是帧级总分（CLIP-IQA 无参考预测），归因看 `degradations[]`。

### VLM/LLM 自主决策

ReAct Agent 以 LLM 为决策中心，观察 CV 结果后**自主判断**是否调 VLM / 是否补检，而非硬编码阈值触发。VLM 三件套：

- `vlm_analyze` — 对低置信检测项做视觉二次确认
- `vlm_discover` — 全帧主动扫描，发现 CV 盲区语义异常（如 AI 生成多指）
- `vlm_caption` — 整帧自然语言画质总结

辅以 `scenario_attribution`（劣化 → 转码/直播/推荐/AIGC 场景 + 修复建议）；Ollama 不可用时 Agent 自动规则降级。

### 视频时序能力（V2 新增）

`VideoClipRunner` 逐帧跑单帧 pipeline + 帧间 `TemporalFlicker` 聚合，不改单帧接口。时序建模三件套：

- **C1 运动补偿**：Farneback 光流对齐后残差能量，区分真闪烁 vs 镜头运动
- **C2 时序 SSIM**：相邻补偿帧 SSIM，衡量时序一致性
- **C3 局部闪烁热力图**：分块残差 → 热力图 + 带 bbox 的局部段，定位「闪在哪」

### 架构

**V1 ReAct Agent（默认）**

```
GlobalScan → 9 detectors → ReAct Agent Loop → Report
                                   │
                    ┌──────────────▼──────────────┐
                    │  Observe: CV 检测结果         │
                    │  Think:   LLM 推理（决策中心）│
                    │  Act:     自主选择工具调用    │
                    │    ├─ vlm_analyze   (VLM确认) │
                    │    ├─ vlm_discover  (VLM主动发现) │
                    │    ├─ vlm_caption   (VLM画质描述) │
                    │    ├─ rerun_detector(重检)    │
                    │    └─ accept       (终止)     │
                    └──────────────────────────────┘
```

**v0.1 基线（`--legacy-fixed`）**：`GlobalScan → 9 detectors → Report`，无 Agent / VLM，全量确定性运行。

### 检测器

| 检测器 | 目标 | 方法 | 成熟度 |
|--------|------|------|--------|
| `edge_bleed` | 边缘 / 溢色 | 绿通道偏置 + ΔE | 主力 |
| `compression_artifact` | 压缩块效应 | DCT 8×8 边界 + Laplacian 纹理损失 | 主力 |
| `blur_artifact` | 主体模糊 | 前景 ROI Laplacian | 主力 |
| `mosaic_artifact` | 马赛克 / 像素化 | 下采样–上采样块一致性 | 主力 |
| `banding_artifact` | 色带 | 梯度量化台阶 | 主力 |
| `background_artifact` | 背景劣化 | 背景块效应 / 色彩漂移 | 主力 |
| `hair_texture` | 发丝细节损失 | FFT 高频能量比 | 主力 |
| `face_artifact` | 面部过曝 / 模糊 | 亮度 + Laplacian | 主力 |
| `hand_anomaly` | 手部几何异常 | 启发式 ROI + 边缘密度；可选 mediapipe 关键点 | **实验性** |

> `hand_anomaly` 的 CV 路径多指/粘连未实现；多指等语义异常可由 `vlm_discover` 经 VLM 主动发现，写入 `agent_meta.vlm_discover_findings`。

## 进阶用法

### CLI

```bash
source .venv/bin/activate

# 单帧 — V1 Agent（需 Ollama）出 HTML / JSON
python detect.py --image docs/demo/case_blur.png --config config.yaml --output report.html
# v0.1 基线（无 Agent / VLM，无需 Ollama）
python detect.py --image docs/demo/case_blur.png --legacy-fixed --output report_v01.json
# 批量
python detect.py --image-dir /path/to/frames/ --legacy-fixed --output-dir reports/
# 视频（V2）— 均匀抽帧 + TemporalFlicker 聚合，输出 VideoClipReport JSON
python detect.py --video docs/demo/clip_jimeng_degraded.mp4 --legacy-fixed --max-frames 16 --output /tmp/video_report.json
python detect.py --video docs/demo/clip_jimeng_degraded.mp4 --config config.yaml --max-frames 16 --output /tmp/video_report_v1.json
```

| 参数 | 说明 |
|------|------|
| `--image PATH` | 单张输入图像 |
| `--image-dir PATH` | 图像目录（非递归） |
| `--video PATH` | 视频文件 → 多帧 V2 clip 流程 |
| `--max-frames N` | 视频模式：最大抽帧数（默认 8；命中短时劣化窗口建议 16+） |
| `--config PATH` | YAML 配置（默认读 `config.yaml`） |
| `--legacy-fixed` | v0.1 固定流水线，跳过 Agent 层 |
| `--output PATH` | 输出文件（`-` = stdout JSON；`.html` = HTML 可视化；视频仅 JSON） |
| `--output-dir PATH` | 批量输出目录 |
| `--metadata PATH` / `--ignore-regions PATH` | 可选 JSON 侧车文件 |
| `--frame-id ID` / `--verbose` | 覆盖 frame_id / 详细日志 |

安装后也可直接用 `lqdd` 入口（等价 `python detect.py`）。

### Python API

```python
import cv2
from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.agent.orchestrator import AgentOrchestrator

config = load_config()
frame = cv2.imread("path/to/frame.png")
fi = SingleFrameInput(frame=frame, frame_id="my_frame", mode="fast")

report = AgentOrchestrator(config).run(fi)      # 需 Ollama
print("MOS:", report.overall_mos)
for step in (report.agent_meta or {}).get("agent_steps", []):
    print(f"Step {step['step']}: [{step['action']}] {step['thought']}")
```

无需 Ollama 的降级：`AgentOrchestrator(config, judge_client=RuleBasedJudgeClient(config.agent))`（规则降级）或 `MockJudgeClient()`（测试）。v0.1 基线用 `FastPipeline(config).run(fi)`。

**视频多帧**：`VideoClipRunner` 是单帧 pipeline 的外层包装，逐帧 `pipeline.run()` + 帧间 `TemporalFlicker`，返回 `VideoClipReport`（逐帧报告 + 闪烁摘要 + 聚合 MOS），不改单帧接口。

```python
from lqdd.pipeline.agent_pipeline import AgentPipeline
from lqdd.pipeline.video_clip_runner import VideoClipRunner, sample_frames_from_video

runner = VideoClipRunner(AgentPipeline(config))   # 也可用 FastPipeline
frames = sample_frames_from_video("input.mp4", max_frames=8)
result = runner.run(frames, clip_id="clip_001")
print(result.aggregate_mos, result.worst_frame_mos, result.worst_frame_index)
print(result.flicker_result.is_flickering, result.degradation_summary)
```

> `TemporalFlicker` 依赖多帧，不进单帧 `ALL_DETECTOR_NAMES`，仅在 `VideoClipRunner` 层调用。

**自定义 Agent**：实现 `JudgeClient` 接口（`review` / `decide`）即可替换决策后端（如 OpenAI 兼容模型），传给 `AgentOrchestrator(config, judge_client=...)`。

### 图形界面（`lqdd-gui`）

基于 Gradio + pywebview 原生窗口，复用 pipeline 与 mask 叠加渲染，覆盖单帧与 V2 视频输入。

```bash
pip install "lqdd[gui]"      # gradio + pywebview（可选）
lqdd-gui                     # pywebview 原生窗口（默认）
lqdd-gui --browser           # 浏览器回退（http://127.0.0.1:7860）
```

结果区展示：mask 叠加预览 + MOS/严重度 + 劣化列表 + Agent 决策轨迹 + `vlm_discover` 主动发现 + VLM 画质描述 + 业务场景归因 + 视频 flicker 聚合（C1/C2/C3）+ 完整 JSON。

### 打包可执行文件（`build/`）

```bash
bash build/build.sh           # 产出 dist/lqdd-gui/lqdd-gui
./dist/lqdd-gui/lqdd-gui      # 启动原生窗口
```

`app.spec` 用 `collect_all` 收集 gradio/pywebview/cv2，**排除** torch/pyiqa/mediapipe（体积太大）；**不含** Ollama / 模型权重，需另装。

### 配置

复制 `config.example.yaml` 为 `config.yaml`（已 gitignore），按需调阈值/Agent/VLM 参数：

```yaml
agent:
  enabled: true
  max_rounds: 2               # Agent 最大循环步数 = max_rounds × 3
  max_detectors_per_frame: 9
vlm:
  provider: ollama
  model: qwen2.5vl:7b          # VLM 视觉确认 / 主动发现
  max_calls_per_frame: 3
judge:
  provider: ollama
  model: qwen2.5:1.5b          # Agent 决策 LLM
report:
  mos_model: "clip_iqa"        # clip_iqa（默认）/ internal（预留）
```

环境变量覆盖：`OLLAMA_HOST` / `LQDD_VLM_MODEL` / `LQDD_JUDGE_MODEL` / `LQDD_AGENT_ENABLED=0`（关 Agent，等效 v0.1）。

## 参考

### 结果示例

所有素材均为本项目独立生成（AI 合成人像 + 程序化劣化），不依赖第三方数据集。`docs/demo/` 含干净源图与 7 种劣化版本：

![干净合成人像](docs/demo/synthetic_portrait.png)

**V1 ReAct Agent 示例（`case_blur`）**：`face_artifact` 原始置信 0.68，LLM Agent 自主触发 VLM 确认，融合后置信升至 0.782，`agent_driven_vlm: true`（VLM 由 LLM 自主触发，非硬编码路由）：

```json
{
  "overall_mos": 3.673, "severity": "moderate",
  "degradations": [{
    "detector": "face_artifact", "degradation_type": "face_blur",
    "confidence": 0.782, "bbox": [315, 255, 393, 522],
    "evidence": {"metric": "face_laplacian_var", "value": 10.96, "threshold": 85.0,
                 "detail": "面部 Laplacian 方差 11 ≤ 85（偏糊）"},
    "root_cause_hypothesis": {"cause": "generation_artifact", "confidence": 0.45},
    "vlm_reasoning": {"vlm_confidence": 0.85, "fusion_decision": "agree"}
  }],
  "agent_meta": {"agent_steps": [
    {"step": 1, "action": "vlm_analyze", "reason": "置信度 0.68 不足，需要 VLM 确认"},
    {"step": 2, "action": "accept", "reason": "VLM 已确认，结果可信"}
  ]}
}
```

每条劣化是三层可解释结构：**L1 evidence**（数值判定）→ **L2 root_cause**（根因）→ **L3 vlm_reasoning**（VLM 语义确认 + UX 影响 + 融合）。MOS 与归因解耦：归因看 `degradations[]`，MOS 只是帧级总分（CLIP-IQA），未装 pyiqa/权重下载失败时 `overall_mos=null` + `mos_unavailable_reason`。

**V2 视频流程验证（`clip_jimeng_degraded.mp4`）**：10s 人像视频（即梦生成 + 脚本在 3s/7s 注入边缘抠图痕迹 + 全程低码率压缩），`--video --max-frames 16`：

```
aggregate_mos=2.731 | worst_frame_mos=2.452 @ idx=5 | flicker=True (ratio=0.867)
degradations: blockiness×16, blur×16, mosaic×16, banding×5, green_spill×3, background_artifact×10, face_blur×2
```

- `green_spill` 仅命中 idx 4/5（≈3s）与 idx 11（≈7s），恰好对应注入抠图痕迹的窗口；**最差帧正是 green_spill 帧**。
- `blockiness/blur/mosaic` 全 16 帧出现（全程压缩），`green_spill` 只在 3s/7s（局部时序异常）——全程压缩 vs 局部异常可区分。
- TemporalFlicker：C1 运动补偿残差 `mean=10.98/max=16.89`、C2 时序 SSIM `0.8593`、C3 局部段含 critical bbox `[0,224,720,1056]` max_delta `91.96`（人像区抠图痕迹的帧间跳变）。

V1 Agent 模式逐帧独立决策，动作空间含 `accept` / `vlm_discover` / `vlm_analyze` / `rerun_detector`。

### 项目结构

```
├── detect.py                 # CLI 入口（单图 / 批量 / 视频）
├── config.example.yaml       # 配置模板
├── src/lqdd/
│   ├── detectors/            # 九个劣化检测器
│   ├── agent/                # ReAct Agent 编排（orchestrator / judge_client / prompts / router）
│   ├── vlm/                  # Ollama VLM 客户端 + 确认 + CV/VLM 融合
│   ├── mos/                  # MOS 后端（CLIP-IQA）
│   ├── temporal_flicker/     # V2 帧间闪烁（C1/C2/C3）
│   ├── attribution/          # 业务场景归因
│   ├── pipeline/             # fast / agent / video_clip_runner
│   ├── ui/                   # lqdd-gui（gradio + pywebview）
│   └── report/               # JSON / HTML 报告（含 compute_mos）
├── build/                    # PyInstaller 打包
├── docs/demo/                # 演示素材（合成人像 + 劣化 + 视频）
├── benchmark/                # 批量评测脚本
├── scripts/                  # benchmark 生成 / demo 资源 / VLM prompt 消融
├── tests/                    # 单元 / 集成 / 契约测试
└── specs/                    # 设计文档与契约（spec-kit 驱动开发）
```

### 项目状态

| 模块 | 状态 | 说明 |
|------|------|------|
| v0.1 固定流水线 | ✅ | GlobalScan → 9 检测器 → Report（`--legacy-fixed`） |
| V1 ReAct Agent | ✅ | LLM 自主决策循环；工具 vlm_analyze / vlm_discover / vlm_caption / rerun_detector / accept；规则降级 |
| V2 视频 + 主动发现 | ✅ | `VideoClipRunner` 多帧 + `TemporalFlicker`（C1/C2/C3）；`vlm_discover` VLM 全帧主动发现 |
| 业务场景归因 / VLM 画质描述 | ✅ | `scenario_attribution` / `quality_caption` |
| MOS（CLIP-IQA） | ✅ | 无参考感知预测，不可用时 null + 原因 |
| CLI + JSON/HTML 报告 | ✅ | 单图 / 批量 / 视频 / schema 校验 |
| GUI + 可执行文件打包 | ✅ | `lqdd-gui` + PyInstaller |

**已知限制**：`banding` 检出率偏低（阈值保守）；`hand_anomaly` 实验性（CV 多指未实现，靠 `vlm_discover`）；干净帧 `compression/blur/mosaic` 有交叉误报；Deep Mode 未实现（`--mode deep` 返回 exit 2）；Agent 小模型 `qwen2.5:1.5b` 推理质量有限，可换 `qwen2.5:7b`。

详见 [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md)。

### 评测（可选，本地数据）

本仓库不含评测图像或 GT mask。如需可复现指标，本地生成合成 benchmark：

```bash
python scripts/generate_benchmark_dataset.py \
  --input $LQDD_DATA_DIR/source_frames --output $LQDD_DATA_DIR/synthetic_benchmark \
  --samples-per-type 5 --clean-count 8
python benchmark/run_eval.py --manifest $LQDD_DATA_DIR/synthetic_benchmark/manifest.json --output benchmark/runs/results.json
```

> `benchmark/run_eval.py` 走 v0.1 `--legacy-fixed` 路径用于可复现基线；评测 V1 Agent 用不带 `--legacy-fixed` 的 `detect.py`，对比 `decision_trace` / `agent_meta.agent_steps`。

### 开发

```bash
pytest tests/ -m "not vlm" -q    # 依赖 VLM 服务的测试标记 @pytest.mark.vlm，默认跳过
```

### 文档

| 文档 | 说明 |
|------|------|
| [`specs/USE_CASE_BADCASE.md`](specs/USE_CASE_BADCASE.md) | Badcase 场景与边界 |
| [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md) | v0.1 / V1 / V2 路线 |
| [`specs/001-v0-fast-mvp/`](specs/001-v0-fast-mvp/) | v0.1 契约与 schema |
| [`specs/002-v1-agent-layer/`](specs/002-v1-agent-layer/) | V1 ReAct Agent / VLM 契约 |
| [`specs/003-v2-video-and-discover/`](specs/003-v2-video-and-discover/) | V2 视频 + `vlm_discover` 契约 |
| [`specs/004-algorithm-deepening/`](specs/004-algorithm-deepening/) | 时序建模升级 + 多模态画质归因契约 |

## License

[Apache License 2.0](LICENSE)



