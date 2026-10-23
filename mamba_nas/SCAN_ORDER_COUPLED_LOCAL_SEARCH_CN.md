# Scan-order 搜索与耦合 Local Search

当前实现新增三个不使用 Diversity Archive 的变体：

```text
scan_order
scan_order_local_search
scan_order_coupled_local_search
```

## 搜索空间

新增基因：

```text
scan_order ∈ {time_major, patch_interleaved}
```

`time_major` 复用旧 Tokenizer；`patch_interleaved` 对每个通道独立切 patch，再按 patch 优先、通道交错排列：

```text
[B,L,C] → [B,C,T,P] → [B,C,T,D]
       → [B,T,C,D] → [B,T×C,D]
```

例如：

```text
patch0-channel0 → patch0-channel1 → patch1-channel0 → patch1-channel1
```

非法组合：

```text
point + patch_interleaved
```

合法组合为：

```text
time_major + point/patch8/patch16
patch_interleaved + patch8/patch16
```

旧空间为 864 个 Genome；新条件空间为 1440 个 Genome。旧 Genome 缺少 `scan_order` 时自动按 `time_major` 读取，且 `time_major` 哈希保持旧值。

## 模型和 MACs

普通 time-major 的 token 数为：

```text
ceil(sequence_length / patch_size)
```

patch-interleaved 的 token 数为：

```text
ceil(sequence_length / patch_size) × input_channels
```

mask 会复制到同一时间 patch 的所有通道 token，padding 区域不会影响 logits。四组 scan-order 消融统一使用 `--hypervolume-space scan_order`，确保 Hypervolume reference 一致。

## Local Search 变体

| 变体 | scan order | 一阶 LS | 耦合 LS | Archive |
|---|---:|---:|---:|---:|
| `scan_order` | 是 | 否 | 否 | 否 |
| `scan_order_local_search` | 是 | 是 | 否 | 否 |
| `scan_order_coupled_local_search` | 是 | 是 | 是 | 否 |

一阶邻域优先顺序：

```text
scan_order → tokenizer → direction → pooling
→ num_blocks → d_model → d_state → d_conv → expand
```

耦合邻域恰好改变两个基因，其中一个必须是 `scan_order`：

```text
scan_order + tokenizer
scan_order + direction
scan_order + d_conv
```

默认预算：

```text
Local Search = 20% offspring
一阶邻域 = 70%
耦合邻域 = 30%
```

跨代累计分配离散槽位，避免整数四舍五入导致实际比例偏离。邻域不足时依次回退到一阶邻域和普通 NSGA-II variation，不增加总候选预算。

## 输出

`local_search.csv` 会记录：

```text
neighbor_type
changed_gene
parent_scan_order
child_scan_order
coupled_partner
```

`generation_diversity.csv` 会记录：

```text
scan_order_coverage
scan_order_entropy
time_major_pareto_count
patch_interleaved_pareto_count
single_local_children
coupled_local_children
single_pareto_entry_rate
coupled_pareto_entry_rate
single_mean_delta_f1
coupled_mean_delta_f1
single_mean_hv_contribution
coupled_mean_hv_contribution
```

## 运行

```bash
bash scripts/NAS/run_scan_order_coupled_ablation.sh
```

默认数据集：

```text
Heartbeat
SpokenArabicDigits
PEMS-SF
```

默认配置：

```text
holdout
NAS seeds = 2021 2022 2023
paper budget
max_unique_candidates = 384
自动 refit = 1
```

dry-run：

```bash
NAS_DRY_RUN=1 NAS_SEEDS="2021" \
bash scripts/NAS/run_scan_order_coupled_ablation.sh
```

结果图表：

```bash
python -m mamba_nas.plot_scan_order_ablation \
  --input runs --output results/scan_order_ablation
```

## 测试

已覆盖：Genome 兼容、1440 条件空间、非法组合、交错 token 顺序、padding 隔离、MACs、合法交叉/变异、耦合邻居、70/30 预算、新变体配置、synthetic smoke 和 resume。

当前完整测试结果：

```text
107 passed
```
