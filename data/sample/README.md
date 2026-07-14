# Benchmark data (GT)

**主测试集不在本仓库内生成。** 请使用本地 `~/data/degradation` 生成带 pixel mask + bbox 的评测集。

## 生成 GT benchmark（推荐）

```bash
# 在 lqdd 仓库
export LQDD_DATA_DIR=/path/to/data
export LQDD_SOURCE_DIR=$LQDD_DATA_DIR/source_frames
python scripts/generate_benchmark_dataset.py \
  --output $LQDD_DATA_DIR/synthetic_benchmark \
  --samples-per-type 5 \
  --clean-count 8
```

输出：

```
~/data/synthetic_benchmark/
├── images/          劣化图
├── masks/           GT mask（像素值 = class_id）
└── manifest.json    bbox + primary_type + source
```

## 跑批量评测

```bash
python benchmark/run_eval.py \
  --manifest $LQDD_DATA_DIR/synthetic_benchmark/manifest.json \
  --output benchmark/runs/results.json
```

## 与旧 `data/sample/frames/` 的关系

`data/sample/frames/` **已废弃**（刻意降质 demo 图，无 GT，不用于 benchmark）。

- **Benchmark 主力**：`synthetic_benchmark`（由 `~/data/degradation` 合成）
- **Demo 可视化**：`docs/demo/`（仓库自带，AI 生成合成人像 + 程序化劣化）

## GT 类型 ↔ lqdd detector

见 `benchmark/type_mapping.py`。v2 覆盖：`edge_compression`, `block`, `blur`, `mosaic`, `banding`, `overexposure`, `green_spill`, `hair_texture` + clean 负样本。
