# 部署感知三目标 NAS 与超网对照

本实验在 NVIDIA L20 上用单请求实测成本比较三组目标，验证树编码候选的短训排名，并比较树编码独立短训与单路径超网的完整搜索流程。新实验使用原有多尺度树空间、标准交叉和变异。Heartbeat 使用正向扫描，PEMS-SF 和 SpokenArabicDigits 保留三种扫描方向。所有阶段只使用原始 TRAIN 的内部训练、验证划分。

## 在 Conda 环境运行

先激活安装了 PyTorch CUDA、官方 Mamba 和项目依赖的 Conda 环境。命令从仓库根目录执行。数据默认来自 `data/UEA/<数据集>/<数据集>_TRAIN.ts`，可以用 `--data-root` 指定其他目录。

```bash
# 查看预算。不需要数据或 GPU，也不创建结果目录。
bash mamba_tree_nas/scripts/run_deployment_nas.sh compare --dry-run

# CPU 合成数据流程测试，输出到默认 smoke 目录或指定的空目录。
bash mamba_tree_nas/scripts/deployment_nas_smoke.sh

# 用真实 Heartbeat 和 L20 验收全部阶段。
bash mamba_tree_nas/scripts/deployment_nas_gpu_acceptance.sh

# 在三级研究中分别启动需要的正式阶段。
bash mamba_tree_nas/scripts/run_deployment_nas.sh benchmark --execute
bash mamba_tree_nas/scripts/run_deployment_nas.sh objectives --execute
bash mamba_tree_nas/scripts/run_deployment_nas.sh rank --execute
bash mamba_tree_nas/scripts/run_deployment_nas.sh compare --execute
bash mamba_tree_nas/scripts/run_deployment_nas.sh status
bash mamba_tree_nas/scripts/run_deployment_nas.sh report --execute
```

`benchmark` 先冻结对应数据集的输入面板和 HV 尺度，`objectives` 和 `compare` 使用这份尺度。两者共用独立短训的三目标运行。`rank` 可单独运行，只创建架构面板、独立训练缓存和排序结果。各训练入口默认预览，加 `--execute` 才运行。`status` 读取结果，`report --execute` 写出汇总和图。

默认结果目录为 `mamba_tree_nas/work/deployment_nas/<profile>/`。`--dataset` 可以重复指定数据集，`--seed` 可筛选已配置的搜索种子。运行 `status` 或 `report` 时，筛选作用于终端展示，导出报告仍覆盖该目录的全部结果。中断后重跑相同命令，搜索从已提交的候选与随机状态继续，超网从最近完成的 epoch 继续。评分和测量缓存可复用。更换配置、代码、数据协议或 GPU 环境时，使用新的 `--output`。

| 配置档 | 数据集 | 架构面板 | 搜索种子 × 候选数 | 种群大小 | 短训上限 / 充分训练 / 超网预训 |
|---|---|---:|---:|---:|---:|
| smoke | 默认 Heartbeat 合成数据 | 6 | 1 × 6 | 3 | 2 / 3 / 2 |
| pilot | 默认真实 Heartbeat | 8 | 1 × 8 | 4 | 8 / 12 / 3 |
| paper | Heartbeat、PEMS-SF、SpokenArabicDigits | 每数据集 32 | 每数据集 3 × 800 | 200 | 30 / 100 / 30 |

paper 档每次搜索包含 200 个初始候选和三轮各 200 个新候选，在累计 200、400、600、800 个候选时复核。每个检查点每组最多选 3 个候选复核，最终每次搜索的复核前沿最多包含 12 个架构。

目标对照最多 27 次搜索、21,600 个逻辑候选，排序最多 576 次独立训练。超网对照涉及 18 次搜索，其中 9 次树编码搜索可直接复用目标对照的三目标结果。统一复核最多 648 次独立训练，重复架构和同协议评分通过缓存去重。dry-run 显示预算上限。如已有旧预算的 paper 结果，请为新实验指定新的 `--output`，并从该目录的 `benchmark` 阶段开始。

## 指标口径

- **部署目标。** 三组分别用验证 Macro-F1 与 MACs、验证 Macro-F1 与实测延迟、验证 Macro-F1 与实测延迟及显存进行搜索。三组的统一 HV 采用 `(1−F1, 延迟 / 固定尺度, 显存 / 固定尺度)`。延迟和显存尺度取预先固定面板中各自最大值的两倍。三维参考点为 `(1.1, 1.1, 1.1)`，二维 F1–延迟投影使用前两坐标。报告中的 native HV 是各自的目标空间，仅用于同口径观察。
- **推理测量。** 每个数据集按内部 TRAIN 的序列长度选择 P10、P50、P90 和最长样本。同长度以原始样本索引最小者为代表。主延迟先将每种输入各轮样本合并求同步前向 P50，再对不同输入等权平均。CUDA allocated 峰值以载入模型前为基线，逐输入测量并取最大值。P95、CUDA event 延迟、由已预处理的 CPU 输入传至 GPU 并取回 logits 的请求延迟、批量吞吐均为辅助指标。测量使用 PyTorch eager 和独立子网，不含数据预处理、网络通信和服务排队。
- **架构排序。** 同一批树架构分别以短训及固定充分训练独立拟合。两种协议使用相同的三个训练种子，各取最佳验证 F1 后按种子平均。Kendall τ-b 和 Spearman 保留并列，top-5 名额按 F1 后按哈希确定。逐种子分数、充分训练的标准差和未成功配对数随结果保留。
- **收敛。** 树编码和超网使用相同初始架构和候选预算。在线曲线保留各自原始评分。统一复核曲线在每个检查点最多各取三个新架构，用充分独立训练重新评分和累积。它描述固定复核配额内找到的解，候选覆盖数量随结果导出。两条时间轴分别记录实际墙钟时间与按原始测量、评分成本归属后的时间。超网预训练计入搜索，完整复核成本单列。HV-AUC 为从零开始的阶梯曲线面积除以观察时长或候选数，配对时间指标取共同观察时长。

`smoke` 的 CPU 混合器、计时和占用值仅用于验证流程，真实 CUDA 显存与 CUDA event 值由 GPU 验收提供。

## 结果入口

| 目录 | 内容 |
|---|---|
| 数据集的 `benchmark/` | 逐架构及逐输入指标、倍率配对、排序相关、冻结后的尺度 |
| 数据集的 `rank/` | 逐架构短训和充分训练 F1、逐种子明细、排序汇总 |
| 数据集的 `search/<组>/<seed>/` | 候选与恢复状态、在线曲线、公共三维 Pareto 前沿、复核架构与曲线 |
| 数据集的 `supernet/<seed>/` | 最近的超网训练检查点、权重组激活统计和子网导出核验 |
| `report/` | 在线和复核曲线 CSV、逐运行汇总、按搜索种子配对的 HV 差值及图 |

GPU 验收脚本运行真实 Mamba 前反向和小预算搜索，并检查实测字段、同初始种群、复核曲线和子网导出。真实部署性能以这套协议在目标 L20 上测得的结果为准。
