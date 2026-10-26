# Mamba 功能树 NAS：独立实现

本目录独立完成 UEA 多变量时序分类的功能树 NAS，不导入旧 `mamba_nas`，也不更改旧方法。当前方法是**真实标签训练与验证选择 + 初始化 CE 梯度旁路观测**，不使用教师表示。测试专用 mixer 仅验证工程流程，不能提供 Mamba 的性能证据。

## Conda 服务器安装与同步

所有入口使用当前环境的 `python`；**不依赖 uv**。开发机器的 uv 只是临时依赖管理工具。同步源代码与 `config.yaml` 即可，排除 `work/`、`__pycache__/`、`runs_smoke/` 和 `paper_exports_smoke/`。正式运行的 `runs/` 可按需要单独传输以恢复任务。

如果已有可工作的官方 Mamba Conda 环境，优先复用，避免替换 CUDA/PyTorch 二进制栈：

```bash
conda activate 你的现有环境
cd /path/to/Mamba-NAS
python -m pip install -r mamba_tree_nas/requirements.txt
python -m mamba_tree_nas verify-environment --gpu-test
```

新环境提供可选模板（Python 3.11 / PyTorch 2.5.1 / CUDA runtime 12.1；**尚未在本项目 GPU 上验收，不是兼容性保证或锁文件**）：

```bash
conda env create -f mamba_tree_nas/environment.yml
conda activate mamba-tree-nas
# 先有 GPU PyTorch，再装与服务器 CUDA / 编译工具链匹配的官方扩展：
python -m pip install causal-conv1d --no-build-isolation
MAMBA_KEEP_CUDA_BUILD=TRUE python -m pip install mamba-ssm --no-build-isolation
python -m mamba_tree_nas verify-environment --gpu-test
```

PyTorch 的 Conda CUDA 12.1 组合见[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。Mamba 安装与 CUDA 扩展开关见[官方仓库](https://github.com/state-spaces/mamba#installation)。若需要本地编译，还需要匹配的 CUDA toolkit / nvcc；Conda 的 runtime 并不等于完整编译器。旧版 Mamba 的安装开关可能不同，以所选版本说明为准。安装成功后建议 `conda env export > mamba_tree_nas/runs/environment-resolved.yml` 并保存 `python -m pip freeze`；每个实际 run 也自动记录环境版本。

Windows 可以在已有 PyTorch CPU 环境执行单元测试和 smoke；生产入口仅允许 Linux/WSL2 + NVIDIA + 官方 `mamba_ssm.Mamba`，不会静默使用替代 mixer。

## 一键入口

在项目根目录运行。唯一超参数来源是 `config.yaml`，三档为 `smoke` / `pilot` / `paper`。默认非 smoke 命令使用 `paper`。`--set dotted.path=value` 可临时覆盖已有配置键，最终解析值完整保存到 manifest 并参与恢复检查。路径相对项目根目录解析。

```bash
# 环境、自带测试、CPU 小预算完整流程（合成数据）
python -m mamba_tree_nas verify-environment
python mamba_tree_nas/scripts/run.py test
python -m mamba_tree_nas smoke --run-id first-smoke

# 原始数据：可直接使用同一份 UEA .ts 文件；下载不是搜索的一部分
python -m mamba_tree_nas download --suite primary --dry-run
python -m mamba_tree_nas download --suite primary
# 已有数据目录可在 config.yaml 修改 data.root，或 --set data.root=/data/UEA

# 正式预算先预览，命令不会启动训练
python mamba_tree_nas/scripts/run.py primary --profile paper --dry-run --with-refit
python mamba_tree_nas/scripts/run.py uea10 --profile paper --dry-run --with-refit

# 单次搜索，然后 refit；这些命令需在服务器主动执行
python -m mamba_tree_nas search --profile pilot --dataset Heartbeat --variant tree_direction --search-seed 2021
python -m mamba_tree_nas refit --run mamba_tree_nas/runs_pilot/Heartbeat/tree_direction/seed_2021/main

# 首批三数据集 / 全部 UEA10 / 五组对照：矩阵从 YAML 读取
python mamba_tree_nas/scripts/run.py primary --profile paper --with-refit
python mamba_tree_nas/scripts/run.py uea10 --profile paper --with-refit
python mamba_tree_nas/scripts/run.py comparisons --dataset Heartbeat --profile paper --dry-run

# 观察、恢复与导出
python -m mamba_tree_nas status
python mamba_tree_nas/scripts/run.py tensorboard --profile paper --port 6006
python -m mamba_tree_nas search --profile pilot --dataset Heartbeat --variant tree_direction --search-seed 2021 --resume
python -m mamba_tree_nas refit --run mamba_tree_nas/runs_pilot/Heartbeat/tree_direction/seed_2021/main --resume
python -m mamba_tree_nas report --root mamba_tree_nas/runs --output mamba_tree_nas/paper_exports
```

可以用 `search ... --with-refit` 连续运行两阶段。`run-suite` 默认仅搜索，指定 `--with-refit` 才重训练。suite 的 `--dataset` / `--variant` / `--search-seed` 用于筛选矩阵；`--run-id` 区分实验目录。`refit --run` 始终读取搜索时的完整配置，不能临时更改协议。更改配置后应使用新的 run-id。

### 中断后直接重跑原命令

`search`、`run-suite`、`smoke` 和 `refit` 的命令行入口默认使用 `--existing auto`，无需额外添加 `--resume`：

- 已完成：检查原配置、核心源码和数据协议后打印 `[SKIP]`，复用完成结果，不重新训练。
- 已中断或状态仍为 running：打印 `[RESUME]`；搜索复用已提交候选，当前未完成候选从固定初始化重跑；refit复用已完成校准和seed，仅补齐剩余训练。
- 尚未开始或只有空目录：打印 `[START]`，继续创建运行。批量任务中已经完成的搜索仍会检查是否需要补做refit。

例如服务器上中断后，使用相同配置、数据目录、搜索seed和run-id再次执行原命令即可：

```bash
python mamba_tree_nas/scripts/run.py primary --profile paper --with-refit
# 先查看每个运行预计执行 start / resume / skip，不启动训练：
python mamba_tree_nas/scripts/run.py primary --profile paper --with-refit --dry-run
```

原有 `--resume` 仍接受，行为与默认自动策略相同。需要“发现已有运行就报错”的严格模式时，使用 `--existing error`。自动策略不会覆盖配置不一致、源码不一致、状态损坏或缺失manifest的旧结果，也不会绕过运行锁。直接调用Python核心函数时，原有显式 `resume=True` 约定保持不变。

paper、pilot、smoke 默认分别输出到 `runs/`、`runs_pilot/`、`runs_smoke/`，报告目录同样分离。查看 pilot 或 smoke 时请传对应 `--profile`；禁止将不同训练预算的结果直接合并。

服务器小预算真实 GPU 验收：`bash mamba_tree_nas/scripts/gpu_acceptance.sh`。此脚本会读取 Heartbeat TRAIN，运行官方 Mamba 前反向以及 3 候选搜索→校准→refit→报告，需预先准备数据。它不代表正式预算结果。本次实现阶段不自动执行此脚本或正式实验。

## 空间与对照

| 实验组 | 拓扑 | 扫描 | 算法 |
|---|---|---|---|
| chain_forward | 串行 RES(OP) | F | NSGA-II |
| tree_forward | 完整有界树 | F | NSGA-II |
| chain_direction | 串行 RES(OP) | F / B / FB_shared | NSGA-II |
| tree_direction | 完整有界树 | F / B / FB_shared | NSGA-II |
| tree_direction_random | 完整有界树 | F / B / FB_shared | 同分布随机搜索 |

默认最多 3 个 OP 叶子、3 层组合节点。OP 深度为 0。OP = RMSNorm → Mamba → Dropout，无自身残差；RES(x)=x+F(x)，SEQ(x)=G(F(x))，PAR(x)=(F(x)+G(x))/2。叶子的方向、d_state、d_conv、expand 独立。全局 tokenizer / d_model / pooling 共用；time_major 固定。三个残差块串行组合的最大深度为 3，在两类空间中都合法。

FB_shared 在一个叶子内共用同一套 Mamba 参数做两次扫描并平均，跨叶子不共享。B 仅反转有效前缀，输出恢复位置。每个节点后重新清除 padding；patch 支持不足整段的尾部。PAR 对所有分支都计成本。结构 JSON 严格解码、无 eval；PAR 顺序规范化，SEQ 保留顺序；格式版本纳入 SHA256。初始与随机搜索按声明递归分布生成，整树越界则拒绝重采样。链基线从 1–3 层均匀采样。交叉、变异后均检查约束；不能产生足够唯一候选时明确中止，不偷偷更换为其他搜索策略。

## 搜索与数据协议

UEA 文件路径为 `<data.root>/<dataset>/<dataset>_TRAIN.ts` 和 `..._TEST.ts`。支持非等长样本与数值 timestamp .ts；同一样本各通道长度必须一致，否则要求显式对齐，不静默截断。每个样本各通道单独线性插值；端点延续，全缺失通道置零。类别映射根据 TRAIN 字符串标签固定排序。

固定分层 80/20、split seed2021；类别稀少导致无法分层则报错。归一化仅在内部 TRAIN 的有效时间点拟合每通道均值/标准差，无 padding 统计。所有方法共享划分、归一化与代理样本索引。原始 TRAIN SHA256 和所有索引保存在 split/manifest。搜索与代理不读取 TEST。

搜索 seed 2021/2022/2023 只控制演化随机性；候选训练固定11001；代理21001/21002/21003；校准41001；refit31001/31002/31003。默认 AdamW、FP32、无增强和调度。官方 fused CUDA 算子可能存在不可避免的非确定性，因此启用 PyTorch deterministic warn-only 并记录环境，不宣称跨硬件逐位复现。

NSGA-II 使用非支配排序、拥挤距离、二元锦标赛和父子精英环境选择。每个候选用真实训练后的最佳验证 Macro-F1 评价，最小化 `(1−Macro-F1, log1p(MACs))`。最佳 checkpoint 依次比较更高 F1、更低 CE、更早 epoch，仅在内存保留。候选唯一预算包含显式失败项，避免失败计算被隐藏。代理失败单列，不等同训练失败；连续候选/后端失败达到阈值就中止。

从全程真实成功候选的 Pareto 集选 high_f1、knee、low_macs。knee 使用固定成本上界归一化后距理想点的距离；平局按 MACs、hash。随机搜索同样训练与选代表，不引入代理指导。

## CE 梯度观测定义

均衡 batch 只使用内部训练子集，大小 `min(n_train,max(32,2K))`；每类无放回轮询采样，索引固定。每候选训练前独立构建三个模型，保留官方初始化、关闭 dropout、开启 autograd，CE backward 后不进行优化器更新。保存 Python / NumPy / Torch CPU/CUDA RNG 并恢复，正式训练重新使用固定 seed 和独立 DataLoader generator。

分别在全部可训练参数和树主干参数统计：CE、`||g||₂`、`sqrt(sum(g²)/n)`、`sum|gθ|`、`mean|gθ|`，以及 `abs(<θ,g>)/max(||θ||₂||g||₂,epsilon)`。本首版的原始乘积明确为 **gradient_L2 × absolute_cosine**；聚合保存三个初始化的各项均值及 `log10(mean(product))`。这是无教师的 CE 参数梯度观测定义，不宣称等价复现 GeoBridge 教师表示代理或已经验证其有效性。接近零、非有限、缺失梯度均单独标记；无效乘积为 null，不赋予“更好/更差”惩罚。

报告保留带符号 Spearman、正序/逆序 top10% 命中率，以及成本中位数分层和树形分组诊断。这些是离线观察，不能用于本轮反选公式、方向或架构，也不能把不同超参数样本的均值当作受控消融。CE 代理不参与采样、排序、淘汰与代表选择。

## 校准、refit 与统计

每个选定的唯一架构先在同一内部划分校准轮数（最多100、至少10、patience15），固定seed41001，优化器与最终训练一致。冻结最佳 epoch 后，在完整 TRAIN 重新拟合归一化，分别进行三个 refit。每次训练全部结束后才加载并评估 TEST；TEST 不选择结构、epoch、代理公式或“最好 seed”。多个角色选择同一 hash 时只训练一次并记录角色 aliases。

每个角色先对一次搜索的三个 refit seed 求均值，再对三个搜索均值报告均值及样本标准差（ddof=1）；训练内部标准差另列。只有一个搜索时 between std 为 null。不同 run-id 重复同一搜索 seed 不允许作为独立搜索合并；报告拒绝混合数据/评估协议与 test/production 后端。

## 指标与论文表图

分类保留 Macro-F1（选择）、Accuracy（UEA 可比）、固定类别集合的 Balanced Accuracy、逐类 P/R/F1、混淆矩阵、CE；不存在预测的类别按零处理。采用 Accuracy 与公开基准对齐的依据见[多变量 Bake Off](https://www.timeseriesclassification.com/mtsc_bakeoff.php)。搜索评估严格区分验证与 TEST，遵循[NAS Best Practices](https://www.jmlr.org/papers/v21/20-056.html)的重复实验与报告原则。

效率输出实际参数、解析 MACs/FLOPs、搜索总时间、代理时间、候选训练时间、校准/refit时间、batch1延迟中位/P90、固定 batch 吞吐、GPU 峰值 allocated memory。延迟取最长 TRAIN 样本，在模型 eval、warmup、CUDA 同步后测量；采样次数和 batch 都在 YAML。CPU smoke 标记 test 后端。MACs 在最大 TRAIN 长度估计，包含 tokenizer、Mamba 投影/卷积/近似扫描/门控、pooling/head，省略 norm、bias、残差/融合逐元素运算与 kernel 开销；FLOPs=2MACs 是解析约定，不是完整 profiler 测量。

HV 的误差范围固定为[0,1]；成本 log1p 上界由完整比较空间的最大宽度/状态/卷积/expand/叶数/双向次数及 tokenizer 上限推导，五组共享，不随前沿变化；参考点(1.1,1.1)。规则参见[pymoo HV说明](https://pymoo.org/getting_started/part_4.html)。HV-AUC 对候选数和累计搜索时间使用右连续阶梯积分，同时给面积/横轴跨度。失败仍消耗候选数与时间。无预声明联合经验参考前沿时不输出 IGD。拓扑覆盖报告已观察的规范树型数量、频数与自然对数熵；它不是“已穷尽空间比例”。

`report` 可从 JSON 重建 CSV 与 PNG/矢量PDF，无需 TensorBoard。核心产物：

| 文件/图 | 用途 |
|---|---|
| results.csv / `<dataset>_results.csv` | 各组、各角色嵌套seed均值与跨搜索std |
| within_search.csv / refit_metrics.csv | 一次搜索内训练波动、完整单次指标和效率 |
| search_summary.csv | 搜索预算、失败率、时间、HV-AUC、拓扑熵 |
| candidate_metrics.csv / candidate_epochs.csv | 可追溯的候选标量和逐epoch记录 |
| pareto.png/pdf / hv.png/pdf | 验证前沿、按候选数和时间的HV |
| candidate_training / refit / calibration | 代表架构训练、校准与最终曲线 |
| proxy_diagnostic / proxy_diagnostics.csv | 梯度代理散点、双方向排名和分层诊断 |
| architecture_*.png/pdf | OP/RES/SEQ/PAR树结构与叶配置 |
| confusion / test_per_class | 每个唯一架构与seed的分类细节 |

建议论文主表为三次搜索独立重复的角色结果，附录展示各seed、成本、完整失败率和计算预算。应在正式论文前完成真实 GPU 验收、全部预算与外部基线复现；不能用 smoke 指标形成科学结论。

外部基线使用 `--external-baselines table.csv`，格式参考 `external_baselines.example.csv`。必须提供 source、protocol_description、provenance。`literature` 单独标记可比性未确定；`same_protocol` 还必须附带与导出 search_summary 匹配的 evaluation_fingerprint。表格不自动混合排名或宣称直接可比。

## 预算、存储与恢复

首批3数据集 × 5组 × 3搜索 = 45次搜索；paper预算最多17,280个唯一候选，每个最多30epoch且另有3次代理前反向。代表选择最多135次校准、405次完整 TRAIN refit；角色重复会减少数量。UEA10 为150次搜索、57,600候选、最多450校准/1,350refit。实际 GPU 小时应由 pilot 实测，代码不虚构预计耗时。

每 run 保存 manifest（完整解析配置/指纹）、环境、数据划分、候选 JSON、逐epoch标量、最新恢复状态、代表架构和 refit 指标。候选 JSON 与 search_state 原子写入；先提交候选后推进状态，崩溃窗口恢复会复用候选缓存。未完成候选重新使用相同初始化训练，不保存优化器中间状态。保存待评价子代、父代与 RNG，保证恢复不重新抽样；进程级锁避免两个进程写同一run。CUDA 的数值非确定性边界见上。

manifest 还保存核心源码指纹。搜索恢复和 refit 必须使用同一核心实现；修改核心算法应创建新run，保留旧代码版本。异常退出可记录截至最后状态提交的耗时；系统强制终止在最后提交之后的未提交耗时无法完整恢复，不能将其当作精确能耗审计。

不保存所有候选权重、激活、梯度张量、每代模型或优化器历史。每次搜索每个唯一代表架构只保留首个 refit seed 的权重，其余seed仅有指标和压缩预测（样本索引、真标签、预测类别、概率，无logits）。权重包含结构、state_dict、归一化、类别与协议指纹。重复角色共用文件。`runs/`、TensorBoard、权重和开发缓存被本目录gitignore忽略；轻量 `paper_exports/` 可单独归档；原方法产物不清理。

保留模型可用 `from mamba_tree_nas.inference import load_retained_model` 恢复，返回 eval 模型、完整TRAIN拟合的 normalizer 和类别名称；输入必须继续使用有效前缀 mask。训练峰值显存与推理画像峰值分别记录。数据集级 comparison 图叠加各组独立搜索的HV与前沿，refit误差线使用搜索之间的标准差。

## 验收范围

测试覆盖树边界/往返/hash/遗传合法性，RES/SEQ/PAR语义、有效前缀反向、共享参数、patch尾部与padding不变性，数据隔离、RNG/正式训练隔离、零/非有限梯度，固定类别分类指标、HV/成本、嵌套seed、外部协议、原子提交/缓存/恢复冲突，以及CPU端到端搜索→refit→报告和权重数量。真实Mamba GPU测试在不满足Linux+NVIDIA时明确skip。

当前本地验收结果与限制记录在 `VALIDATION.md`。没有在 Windows 进行真实 Mamba 训练或正式实验。
