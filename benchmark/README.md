# Benchmark

批量评测脚本：给定 GT manifest，对 lqdd 输出计算 recall、IoU@0.3、干净帧误报等指标。

**数据与跑分结果不随仓库分发**（manifest 需自备；`benchmark/runs/`、`benchmark/results*.json` 已 gitignore）。

## Usage

```bash
python benchmark/run_eval.py \
  --manifest /path/to/manifest.json \
  --output benchmark/results.json
```

## Modules

| File | Role |
|------|------|
| `run_eval.py` | 主评测入口 |
| `run_eval_tier0.py` | Tier-0 合成数据快速评测 |
| `baselines.py` | noop / blockiness / random / oracle |
| `metrics.py` | IoU、recall |
| `type_mapping.py` | GT 类型 → 检测器映射 |
