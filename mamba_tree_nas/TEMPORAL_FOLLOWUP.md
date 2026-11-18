# 时序变异后续实验（独立入口）

本入口对应复跑审计 → 邻域普查 → 受控诊断 → 三组配对搜索。
默认覆盖 Heartbeat、PEMS-SF、SpokenArabicDigits；默认只预览，不启动训练。
原 `run_temporal_mutation.sh` 和原搜索、训练、refit 实现保持不变。

## Conda 一键入口

在服务器激活已有 Conda 环境，从项目根目录运行：

```bash
conda activate mamba-tsc-nas

# 先检查预算；下面四条都不会训练/落盘
bash mamba_tree_nas/scripts/run_temporal_followup.sh audit
bash mamba_tree_nas/scripts/run_temporal_followup.sh census
bash mamba_tree_nas/scripts/run_temporal_followup.sh diagnose
bash mamba_tree_nas/scripts/run_temporal_followup.sh search --with-refit
```

`PYTHON=/path/to/conda/env/bin/python` 可替换解释器；不依赖 uv。
脚本在启动 Python 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
这只是复现设置之一，不保证官方 Mamba 全部算子逐位确定。

确认每阶段结果后依次运行：

```bash
# P0：36次固定架构训练，每次30轮，每个重复使用独立Python进程，不加载TEST
bash mamba_tree_nas/scripts/run_temporal_followup.sh audit --execute

# P1：只枚举邻域，不训练，也不加载TEST
bash mamba_tree_nas/scripts/run_temporal_followup.sh census --execute

# P2：每数据集最多12个真正可配对且样本不同的父代，最多540次训练
bash mamba_tree_nas/scripts/run_temporal_followup.sh diagnose --execute

# P3：三组 × 三数据集 × 五搜索seed × 96候选；默认不refit
bash mamba_tree_nas/scripts/run_temporal_followup.sh search --execute

# 方法冻结后，复用完成的搜索并追加预算内代表的refit
bash mamba_tree_nas/scripts/run_temporal_followup.sh search --execute --with-refit

bash mamba_tree_nas/scripts/run_temporal_followup.sh status
bash mamba_tree_nas/scripts/run_temporal_followup.sh report --execute
```

只有 `search --with-refit --execute` 的最终 refit 流程会加载 TEST。
审计、普查、诊断都不使用 TEST，不按本轮TEST挑父代或训练轮数。
阶段间需要由研究者查看结果再推进；没有自动启动全部正式实验的脚本。

## 路径、筛选与配置

默认只读来源：`mamba_tree_nas/work/temporal_mutation/search/paper/`。
可指定 `--source-root mamba_tree_nas/runs`，读取已完成、方向及尺度配置匹配的
多尺度树搜索。读取 `manifest.json`、`search_state.json`、`candidates/*.json`，
不读取源结果的refit/TEST汇总。源TRAIN文件、划分或候选成本不匹配时明确报错。
父代哈希去重时保留确定顺序中的首次记录，不从同哈希的多次评分中挑最好值。

```bash
bash mamba_tree_nas/scripts/run_temporal_followup.sh census \
  --source-root /workspace/results/temporal_mutation/search/paper \
  --data-root /workspace/data/UEA --execute

# 单数据集、单新seed；不会调度传统方法、旧串并树等实验
bash mamba_tree_nas/scripts/run_temporal_followup.sh search \
  --dataset PEMS-SF --seed 52021 --execute
```

统一参数在 `config.yaml` 的 `temporal_followup`：审计架构数/seed/重复数/轮数、
诊断父代数/子代数/seed/轮数、成本窗、五个新搜索seed、三组矩阵与预算。
`paper/pilot/smoke` 的差异也集中在该节。优化器、代理、refit仍使用原公共配置。

```bash
# 修改协议务必指定新的输出目录；默认同配置自动恢复
bash mamba_tree_nas/scripts/run_temporal_followup.sh search \
  --output mamba_tree_nas/work/temporal_followup/window30 \
  --set temporal_followup.window=0.30 --execute
```

默认输出：`mamba_tree_nas/work/temporal_followup/<profile>/`。
与只读来源目录重叠会被拒绝。配置、环境或实现指纹改变也会拒绝复用旧输出。
`--seed` 只筛选配置中的新搜索seed；需要新seed时先修改配置或使用 `--set`。
`--dry-run` 可显式预览；与 `--execute` 互斥。

## 每个阶段验证什么

- **audit**：默认选择验证高分的预算内代表和一个不同的低成本架构。
  比较同seed独立进程的初始权重hash、实际样本顺序hash、每轮首batch的logits/
  loss/梯度/更新后权重hash以及验证曲线。只保存摘要，不保存权重或梯度张量。
  中断的单次训练按同初始化重跑，完成的重复直接跳过。
- **census**：按10%/15%/30%成本窗统计single与成本匹配整叶随机编辑、
  redistribute与随机双叶编辑的覆盖、邻域交并比、等价率和拒绝原因。
  single使用总MACs窗，双叶使用主干MACs窗，两者分开标注。
- **diagnose**：按tokenizer/叶数/拓扑分层打乱父代，抽满真正有合法邻居的名额；
  同邻域、同子代样本排除并计数。不足名额直接报告，不拿无邻居父代填充。
  每条轨迹固定30轮，同时记录最佳验证与第30轮终点；以最佳验证为主。
  对训练seed及子代先在父代内平均，再汇总父代配对差值。
- **search**：原始变异original、成本匹配随机编辑matched_random、单叶尺度编辑single。
  后两组具有相同的single可行性门槛、总MACs窗和回退规则，在叶变异分支直接调度，
  不再额外乘1/6。默认名义触发率约0.3×0.3=9%，实际提交率必须看日志。
  本阶段不启用redistribute。基础搜索与局部编辑使用不同随机流，均保存状态；
  无合法局部邻居时，不额外消耗局部随机流，按原整叶变异回退。

## 公共评分、恢复与存储

`evaluations/<dataset>/<key>.json` 的身份含完整架构hash、数据/划分/归一化、
训练超参、候选seed、代理配置与seed、模型/训练代码及环境；不含搜索组或搜索seed。
同架构同训练身份在三组使用完全相同的已完成评分。
不导入旧实验的“最好评分”；当前入口第一份完成的新评估是公共评分。
固定训练seed的一份缓存只是共同评分映射，不能消除训练seed间的方差。

搜索按候选提交标量、代与两个随机流状态。中断候选从同seed重跑，完成候选复用。
连续三次训练/后端失败终止，错误写入候选；代理异常保留代理自己的状态。
首代逐哈希、状态与验证评分检查一致性。

预算内代表按验证F1、MACs、hash选择；未命中成本上限不refit。
`shared_refits/<dataset>/<hash前16位>/` 的manifest验证完整hash，跨组/搜索seed
复用同架构的轮数校准及三个refit结果；各搜索只保存 `refit_alias.json` 链接。
同架构同refitseed不作为额外训练重复，每个唯一架构只保存首refitseed权重。
缩短外层目录是为避免Windows路径过长，不改变完整hash与身份校验。

## 文件与指标查看

| 路径 | 含义 |
|---|---|
| `protocol.json` | 完整解析配置、实现与环境指纹 |
| `audit/plan.json`、各架构的 `seed_*_repeat_*.json` | 复跑计划与独立进程轨迹摘要 |
| `audit/comparisons.csv` | 同seed重复的首次偏离位置及F1差值 |
| `census/neighborhoods.csv`、`summary.csv` | 每父代邻域与覆盖汇总 |
| `diagnose/plan.json` | 实际可配对父代、剔除原因与缺额 |
| `diagnose/children.csv`、`parent_seed_pairs.csv` | 子代及父代-seed配对明细 |
| `diagnose/parents.csv`、`summary.csv` | 已在父代内平均seed后的诊断结果 |
| `search/<dataset>/<arm>/seed_*/main/` | 架构、逐epoch、搜索状态和TensorBoard |
| `reports/search_runs.csv`、`search_paired.csv` | 逐搜索及同seed策略配对差值 |
| `reports/search_summary.csv` | 先平均refit，再汇总搜索间均值/标准差 |
| `reports/search_hv.csv` | 按候选数、实际时间和归属评估时间的HV趋势 |
| `reports/status.json` | 搜索完成度、阶段状态、失败原因及别名去重后的训练数 |

`report` 默认只在终端显示，`report --execute` 才写入CSV/JSON。
`status`只读。已完成搜索的重入仍会校验身份，不能凭目录存在就跳过校验。

```bash
tensorboard --logdir mamba_tree_nas/work/temporal_followup/paper --port 6006 --bind_all
```

观察search/hv、Pareto大小、实际新训练数、cache_hits、累计时间、mutation提交数，
以及候选/审计/诊断/refit的loss与F1。
缓存命中缩短墙钟时间不能算算法加速：同时报告逻辑候选数、真实训练数、缓存命中率
及evaluation_cost_seconds（归属训练+代理成本）。HV-AUC按左连续阶梯积分，HV(0)=0，
报告除以观察候选数的值；未完成搜索的AUC不能与完整96候选直接比较。

## CPU与GPU验收

```bash
# CPU流程、恢复和共享refit；只使用合成数据与测试mixer
bash mamba_tree_nas/scripts/temporal_followup_smoke.sh

# Linux + NVIDIA + 官方mamba；默认Heartbeat/pilot，需已有源搜索与原始TRAIN/TEST
bash mamba_tree_nas/scripts/temporal_followup_gpu_acceptance.sh \
  mamba_tree_nas/work/temporal_followup/gpu_acceptance \
  /workspace/data/UEA /workspace/results/temporal_mutation/search/paper

# 不使用bash也可在Windows/Conda直接运行
python -m pytest mamba_tree_nas/tests/test_temporal_followup.py -q --basetemp=mamba_tree_nas/work/pf
python -m mamba_tree_nas.temporal_followup search --profile smoke --dataset Heartbeat --execute
```

CPU smoke仅验证流程，不代表真实Mamba性能或CUDA确定性。GPU验收需在服务器实际运行。
不自动重跑旧方法，不自动删除旧实验产物，不自动扩大至384候选。
