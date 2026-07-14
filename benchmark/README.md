# Benchmark

批量评测：GT manifest（`synthetic_benchmark`）→ lqdd → recall / IoU@0.3 / clean FPR。

**数据不随 lqdd 仓库分发**；在本地 `~/data/` 生成（见 `scripts/generate_benchmark_dataset.py`）。

## 1. 生成 GT 集

```bash
cd /path/to/LocalQualityDegradationDetection
export LQDD_DATA_DIR=/path/to/data
export LQDD_SOURCE_DIR=$LQDD_DATA_DIR/source_frames
python scripts/generate_benchmark_dataset.py \
  --output $LQDD_DATA_DIR/synthetic_benchmark \
  --samples-per-type 5 \
  --clean-count 8
```

## 2. 全量评测

```bash
python benchmark/run_eval.py \
  --manifest $LQDD_DATA_DIR/synthetic_benchmark/manifest.json \
  --output benchmark/runs/results.json
```

## 3. Smoke（按 GT 类型）

```bash
python benchmark/run_eval_smoke.py
```

## Modules

| File | Role |
|------|------|
| `run_eval.py` | 主评测（recall / IoU / FPR） |
| `run_eval_smoke.py` | 按 `primary_type` 快速对齐 |
| `run_eval_tier0.py` | **已废弃**（原 data/sample/frames） |
| `type_mapping.py` | GT 类型 → detector |
| `metrics.py` / `baselines.py` | 指标与基线 |

## GT 类型（v2）

`edge_compression`, `block`, `blur`, `mosaic`, `banding`, `overexposure`, `green_spill`, `hair_texture`, `clean`

合成逻辑：`~/data/degradation/synthesize.py`（feather / compression_hf + lqdd GlobalScan ROI）
