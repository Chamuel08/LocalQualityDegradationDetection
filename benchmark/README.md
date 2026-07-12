# Benchmark（可选）

本目录提供 **有 GT manifest 时的批量评测脚本**，代码可公开，**评测数据与跑分结果不随 GitHub 分发**。

详细效果截图、真实直播帧评测与方案 A–D 说明见作品集飞书文档（README 中的链接）。

## 前置

- 自备 `manifest.json` + 对应图像目录（合成或私有数据集）
- 本仓库 `.venv` 已 `pip install -e ".[dev]"`

## 运行

```bash
python benchmark/run_eval.py \
  --manifest /path/to/your/manifest.json \
  --output benchmark/results.json
```

`benchmark/results.json` 与 `benchmark/runs/` 已在 `.gitignore` 中，不会提交到 GitHub。

## 输出

- 终端：分层指标摘要（支持类 recall、IoU@0.3、干净误报等）
- `--output`：逐样本明细 JSON

## 模块

| 文件 | 作用 |
|------|------|
| `run_eval.py` | 主评测入口 |
| `baselines.py` | noop / 整图 blockiness / random / oracle |
| `metrics.py` | IoU、recall |
| `type_mapping.py` | GT 类型 → lqdd 检测器映射 |
