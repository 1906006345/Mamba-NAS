# MambaTSC-NSGA2 模块

本目录是一个独立的时间序列分类神经架构搜索（NAS）流水线，不会导入或修改原有的
`run.py`、`models/`、`exp/` 或 `data_provider/` 训练流水线。

生产模型使用 `mamba_ssm.Mamba`。CPU 单元测试会注入一个小型、确定性的 mixer，用于验证架构、
掩码、搜索算子、数据隔离和导出流程；该 mixer 不是实际的搜索后端。

搜索阶段只读取 `{dataset}_TRAIN.ts`。明确命名的 `load_test_for_refit` 函数只有在代表性架构
选定之后才会被调用，因此 TEST 数据不会参与架构搜索。

## 搜索验证策略

默认使用原有的确定性 80/20 holdout 划分：

```bash
python -m mamba_nas.cli search \
  --dataset Heartbeat \
  --budget paper \
  --run-id heartbeat-paper-holdout \
  --validation-strategy holdout
```

如需启用共享的分层 K 折验证，可显式指定：

```bash
python -m mamba_nas.cli search \
  --dataset Heartbeat \
  --budget paper \
  --run-id heartbeat-paper-kfold \
  --validation-strategy kfold \
  --cv-folds 5
```

所有候选架构使用同一份持久化 fold 划分。每个 fold 都会重新训练一个模型，并且只使用该 fold
的训练子集拟合归一化参数。候选架构的适应度由拼接后的 out-of-fold 预测计算 Macro-F1 得到。
恢复任务时不能更改验证策略。

代表性架构角色为 `high_f1`、`knee` 和 `low_macs`。重训练流程仍兼容旧实验结果中的
`high_accuracy` 和 `low_cost` 字段。

## 搜索变体与多样性产物

搜索引擎与搜索变体是两个独立配置。当前正式对比只保留 `random` 和 `nsga2`；`search_algorithm` 仍支持：

```text
random：固定 seed 的独立随机 Genome 采样
nsga2：默认 Vanilla NSGA-II
moead：基于 Tchebycheff 分解的 MOEA/D（保留接口，但不再纳入正式对比脚本）
```

例如：

```bash
python -m mamba_nas.cli search \
  --dataset Heartbeat --budget smoke --seed 2021 \
  --search-algorithm moead --search-variant baseline
```

每代会同时保存 `search/generation_diversity.csv` 和
`search/convergence_metrics.csv`，其中包含 Hypervolume、HV-AUC、候选评价数、训练时间、
Pareto size 以及参考前沿可用时的 GD、IGD、IGD+ 和 epsilon 指标。

Random、NSGA-II 和 `archive_local_search` 的正式对比脚本为：

```bash
bash scripts/NAS/run_search_algorithm_comparison.sh
```

脚本默认对所有数据集使用 holdout，并固定候选预算和 NAS seeds。若需要单独对某些数据集使用
K-fold，可设置 `NAS_KFOLD_DATASETS`，例如：

```bash
NAS_KFOLD_DATASETS="Heartbeat" bash scripts/NAS/run_search_algorithm_comparison.sh
```

当前对比脚本默认使用 `-algcmp-v2` run-id 后缀，用于区分旧的
`archive_local_search` 结果；该版本统一 Random、Vanilla NSGA-II 和
`archive_local_search` 的唯一 offspring、generation 和候选预算协议。

`local_search_ratio` 敏感性实验脚本为：

```bash
bash scripts/NAS/run_local_search_ratio_sensitivity.sh
```

默认测试 `0.10、0.20、0.30、0.40` 四个比例，数据集为 Heartbeat、
SpokenArabicDigits 和 PEMS-SF，不包含 UWaveGestureLibrary。可通过以下变量覆盖：

```bash
NAS_RATIOS="0.10 0.20 0.30" \
NAS_SEEDS="2021 2022 2023" \
NAS_REFIT=1 \
bash scripts/NAS/run_local_search_ratio_sensitivity.sh
```

默认的 `baseline` 是原始 NSGA-II 流程。下面的可复现实验变体保持目标空间中的 Pareto 排名
不变，只改变生存选择和 offspring 的结构多样性机制：

| 变体 | 决策空间拥挤度 | 多样性档案 | Pareto 局部搜索 |
| --- | --- | --- | --- |
| `baseline` | 否 | 否 | 否 |
| `dual_crowding` | 是 | 否 | 否 |
| `diversity_archive` | 否 | 是 | 否 |
| `dual_space` | 是 | 是 | 否 |
| `local_search` | 否 | 否 | 是 |
| `archive_local_search` | 否 | 是 | 是 |
| `archive_local_search_v2` | 否 | 是 | 是（Macro/Micro + 分层多起点） |
| `full` | 是 | 是 | 是 |
| `scan_order` | 否 | 否 | 否（启用 scan-order 空间） |
| `scan_order_local_search` | 否 | 否 | 是（一阶邻域） |
| `scan_order_coupled_local_search` | 否 | 否 | 是（一阶 + 耦合邻域） |
| `scan_order_dual_crowding` | 是 | 否 | 否（目标+决策空间拥挤） |

## Scan-order 搜索与耦合局部搜索

Scan-order 变体在旧 8 维 Genome 后增加：

```text
scan_order ∈ {time_major, patch_interleaved}
```

`time_major` 保留原 Tokenizer；`patch_interleaved` 对每个通道独立切
`patch8/patch16`，再按同一时间 patch 的不同通道依次排列。为避免逐点展开导致序列爆炸，
`point + patch_interleaved` 是非法组合。旧搜索空间仍为 864，新条件空间为 1440。

完整方法使用 20% offspring 进行局部搜索，其中 70% 为合法一阶邻域，30% 为：

```text
scan_order + tokenizer
scan_order + direction
scan_order + d_conv
```

的耦合邻域，不启用 Diversity Archive。正式四组消融脚本为：

```bash
bash scripts/NAS/run_scan_order_coupled_ablation.sh
```

默认运行 Heartbeat、SpokenArabicDigits、PEMS-SF，NAS seeds 为
2021/2022/2023，并自动 refit。只检查命令可运行：

```bash
NAS_DRY_RUN=1 NAS_SEEDS="2021" \
  bash scripts/NAS/run_scan_order_coupled_ablation.sh
```

实验完成后生成图表：

```bash
python -m mamba_nas.plot_scan_order_ablation \
  --input runs --output results/scan_order_ablation
```

例如，在普通 holdout 协议下运行完整方法：

```bash
python -m mamba_nas.cli search \
  --dataset Heartbeat --budget paper --run-id heartbeat-holdout-full \
  --validation-strategy holdout --search-variant full
```

每个搜索任务都会写入 `search/variant.json` 和 `search/generation_diversity.csv`。
算法对比或离线参考前沿生成后，还会写入 `search/convergence_metrics.csv` 和
`search/reference_front.csv`；后者只用于离线 GD/IGD/IGD+ 统计，不参与搜索决策。
启用多样性档案时，还会写入 `search/diversity_archive.csv`；启用局部搜索时，会写入
`search/local_search.csv`。

`archive_local_search_v2` 是改造后的组合：使用 Macro/Micro Archive，并按全局 Pareto、
结构档案和当前种群分层选择局部搜索种子，默认局部搜索比例为 0.20。它额外写入：

```text
search/macro_archive.csv
search/micro_archive.csv
```

三数据集正式消融脚本为：

```bash
bash scripts/NAS/run_single_and_revised_combo_ablation.sh
```

脚本默认会检查 `manifest.json`、`search_completed` 和 `pareto_front.csv`，已经完成的搜索直接跳过；
如果对应的搜索已完成但 refit 缺失，则只执行 refit。需要强制重新运行时设置：

```bash
NAS_SKIP_COMPLETED=0 bash scripts/NAS/run_single_and_revised_combo_ablation.sh
```

如果需要把 SpokenArabicDigits 和 PEMS-SF 也改为 5-fold K-fold，可使用新的环境变量；建议加上
`NAS_RUN_TAG`，避免与之前的 holdout run-id 重叠：

```bash
NAS_KFOLD_DATASETS="Heartbeat SpokenArabicDigits PEMS-SF" \
NAS_CV_FOLDS=5 \
NAS_RUN_TAG="-cv5" \
NAS_SEEDS="2021 2022 2023" \
NAS_REFIT=1 \
bash scripts/NAS/run_single_and_revised_combo_ablation.sh
```

只先检查命令：

```bash
NAS_DRY_RUN=1 \
NAS_KFOLD_DATASETS="Heartbeat SpokenArabicDigits PEMS-SF" \
NAS_RUN_TAG="-cv5" \
NAS_SEEDS="2021" \
bash scripts/NAS/run_single_and_revised_combo_ablation.sh
```

生成的 run-id 形如：

```text
spokenarabicdigits-kfold-cv5-baseline-seed2021
pems-sf-kfold-cv5-archive_local_search_v2-slotsfix-seed2021
```

修复局部搜索槽位回退后的 v2 默认使用 `slotsfix` run-id 后缀，避免覆盖旧实验：

```text
heartbeat-kfold-archive_local_search_v2-slotsfix-seed2021
```

新版本先按结构生态位严格选择多起点 seed；未访问邻居不足时，允许同生态位备用 seed；仍不足的
槽位由普通 variation 补齐。每代和累计记录：

```text
requested_local_slots
actual_local_children
fallback_local_children
fallback_variation_children
local_search_slot_fill_rate
local_search_cumulative_positive_f1_rate
local_search_cumulative_pareto_entry_rate
local_search_cumulative_mean_hv_contribution
```

该脚本固定 Heartbeat 使用 5-fold K-fold，SpokenArabicDigits 和 PEMS-SF 使用 holdout，
并依次运行 baseline、两个单模块、原始组合和改造组合。

恢复任务时，如果验证策略、fold 数量、seed 或搜索变体开关与已保存的 manifest 不一致，程序会
拒绝恢复。Python API 既可以传入命名的 `search_variant`，也可以直接设置三个机制开关；
直接组合的开关会以 `custom` 记录，保证实验可复现。

启用决策空间拥挤的变体可以设置目标空间与决策空间权重，两者之和必须为 1：

```bash
python -m mamba_nas.cli search \
  --dataset Heartbeat --budget paper --run-id heartbeat-dual-w02 \
  --search-variant dual_crowding \
  --objective-weight 0.8 --decision-weight 0.2
```

使用 `scripts/NAS/run_full_and_ablation.sh` 时，可通过环境变量设置：

```bash
NAS_OBJECTIVE_WEIGHT=0.8 NAS_DECISION_WEIGHT=0.2 \
  bash scripts/NAS/run_full_and_ablation.sh all
```

权重会保存到 `manifest.json` 和 `search/variant.json`；恢复搜索时权重必须与原任务一致。

## 效率指标

搜索候选保存统一的解析效率字段：

```text
parameters
macs
flops = 2 × macs
```

最终 refit 和传统对比模型的 `result_classification.txt` 还会记录 profiler FLOPs/MACs、
batch=1 的中位/P90 延迟、吞吐量和 GPU 峰值显存。由于 fused Mamba CUDA kernel 可能不暴露
完整 FLOPs，`profile_flops_status` 会明确标记是否可用，不能用不完整 profiler 值替代解析估计。
NAS refit 还会生成 `final/efficiency_summary.csv`，用于直接汇总三个代表架构及其 refit seeds。

可使用以下脚本生成搜索算法和效率图表：

```bash
python -m mamba_nas.plot_search_algorithm_comparison \
  --input runs --output results/search_algorithm_comparison
```

## 查看已保存的实验产物

可以使用 `visualize_run.py`，传入想要查看的文件或数据集目录。脚本会自动选择图表类型并写入
PNG 文件，不会直接打印原始实验参数。

```bash
# 数据集总览：候选架构、Hypervolume、最终角色和代表性架构
python -m mamba_nas.visualize_run \
  --input runs/heartbeat-paper-kfold/Heartbeat \
  --output /tmp/heartbeat_overview.png

# 候选架构搜索散点图：Macro-F1 与 MACs
python -m mamba_nas.visualize_run \
  --input runs/heartbeat-paper-kfold/Heartbeat/search/evaluations.csv

# 某一代的 Hypervolume 历史
python -m mamba_nas.visualize_run \
  --input runs/heartbeat-paper-kfold/Heartbeat/search/generations/generation_0038.json

# 同样支持 fold 指标、训练历史、预测结果和最终汇总
python -m mamba_nas.visualize_run --input <path/to/metrics.json>
python -m mamba_nas.visualize_run --input <path/to/history.csv>
python -m mamba_nas.visualize_run --input <path/to/val_predictions.npz>
python -m mamba_nas.visualize_run --input <path/to/summary.csv>
```

如果不指定 `--output`，输入文件会在原目录旁生成同名 `.png` 文件；输入数据集目录时，输出文件
名为 `run_overview.png`。

对于 `evaluations.csv`，所有已完成的候选架构会显示为灰色点，Pareto 前沿会显示为红色点，
并绘制一条保持形状的平滑视觉辅助曲线。相邻 `representatives.json` 中的代表性架构使用不同
的颜色和标记：`high_f1` 为蓝色星形，`knee` 为绿色菱形，`low_macs` 为橙色加号。
红色点仍然是真实的离散架构，平滑曲线仅用于视觉展示。

## 生成消融实验展示图

当 Heartbeat 和 SpokenArabicDigits 的 2021 seed 搜索完成后，可以用下面的脚本生成组会展示
图。搜索阶段图读取各 run 的 `search/` 目录；如果存在 `final/summary.csv`，还会同时汇总
refit 后的 TEST 指标：

```bash
python -m mamba_nas.plot_ablation_search \
  --input runs \
  --output results/nas_ablation_2021 \
  --seed 2021
```

输出包括：

```text
search_convergence_hypervolume_2021.png  # Hypervolume 收敛曲线
search_quality_diversity_2021.png        # HV、结构覆盖率、熵、Genome 距离等
pareto_front_ablation_2021.png           # 六种变体的 Pareto 前沿
ablation_summary_table_2021.png          # 组会用汇总表
ablation_search_summary_2021.csv         # 对应的精确数值
refit_test_performance_2021.png           # TEST F1/Accuracy（refit seeds 均值±标准差）
refit_test_f1_macs_2021.png               # TEST F1–MACs 权衡
refit_generalization_gap_2021.png         # 搜索验证 F1 到 TEST F1 的差距
refit_test_summary_table_2021.png         # TEST 指标精确汇总表
ablation_refit_test_summary_2021.csv      # refit TEST 精确数值汇总
```
