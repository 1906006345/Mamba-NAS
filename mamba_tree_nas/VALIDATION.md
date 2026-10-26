# 实现验收记录

日期：2026-10-26。范围仅为本目录；原有方法及其文件未改动。

## 已执行

- Windows，Python 3.12.13，PyTorch 2.14.1 CPU 临时测试环境。
- `python -m pytest mamba_tree_nas/tests -q`：**51 passed, 2 skipped**。最后一轮耗时11.48秒；临时文件放在本目录忽略的 `work/` 下。
- `python -m mamba_tree_nas smoke --run-id verified-v1`：完成5个唯一候选、每候选3次初始化代理、真实合成标签CE训练和验证选择、轮数校准、三种子refit、报告导出。
- 本次smoke的三个代表角色选中同一架构：共用1次校准、3次refit，仅保存1份权重和3份压缩预测。已验证保留权重可重新加载并复现对应预测指标。
- 验证TensorBoard事件实际含HV、时间、Pareto大小、拓扑数量/熵、full/trunk代理各项及CE；候选与refit的逐epoch曲线分别写入事件目录。
- 实际导出CSV、PNG和矢量PDF；人工检查HV图布局，修正测试标识与轴标签重叠。合成图保留显著的测试mixer标识。
- 首批矩阵dry-run：45次搜索、最多17,280候选、51,840次代理初始化、135次校准、405次refit。UEA10矩阵测试验证150次搜索。
- 五组对照均有小预算搜索测试。额外验证启用/关闭代理得到相同候选序列、目标值和代表选择。
- 验证候选提交后的崩溃缓存窗口、候选级恢复、配置冲突拒绝、suite恢复继续创建未开始运行、连续后端失败中止、代理失败单独记录、角色别名、权重数量与嵌套seed汇总。

最终smoke输出：

```text
mamba_tree_nas/runs_smoke/SyntheticSmoke/tree_direction/seed_2021/verified-v1/
mamba_tree_nas/paper_exports_smoke/smoke/
```

这些本地运行和缓存路径被gitignore忽略，不能作为正式实验结果；同步服务器无需复制。开发期间较早的smoke快照可能使用旧核心源码指纹，应以 `verified-v1` 为当前验收快照。

## 明确未验证

- 本机无NVIDIA CUDA，**真实官方Mamba前反向与真实Mamba功能树变长测试共2项skip**。
- 没有在真实UEA数据上进行GPU搜索、正式预算、完整五组对照或正式refit。
- Conda环境模板未在服务器实际创建；模板不是锁文件。服务器已有工作环境时优先仅安装本目录requirements，保留现有Torch/CUDA/Mamba栈。
- 合成数据及测试mixer结果只证明流程可运行，不证明架构效果、CE代理相关性、Mamba延迟或科学优越性。

## Conda服务器验收

无需uv。在项目根目录、激活已有Conda环境后：

```bash
python -m pip install -r mamba_tree_nas/requirements.txt
python -m mamba_tree_nas verify-environment --gpu-test
python -m pytest mamba_tree_nas/tests -q
# 预先放好Heartbeat原始.ts数据后，小预算真实Mamba全链路验收：
bash mamba_tree_nas/scripts/gpu_acceptance.sh
# 确认资源预算，不启动正式任务：
python -m mamba_tree_nas run-suite --profile paper --with-refit --dry-run
```

真实GPU验收通过后，再主动启动pilot及正式矩阵。各run保存完整配置、源码指纹、数据划分与环境信息，便于后续论文审计。

## 自动跳过与中断恢复回归验收（2026-10-26）

命令行现默认 `--existing auto`：重复原命令会检查并复用完成运行，恢复中断运行，启动尚未创建的运行。`--resume` 保持兼容，`--existing error` 提供原有严格拒绝行为。此次只更改命令行编排、启动器说明及测试/文档，核心源码指纹未改变。

执行 `python -m pytest mamba_tree_nas/tests/test_cli_reuse.py mamba_tree_nas/tests/test_pipeline.py -q`，结果 **29 passed, 2 skipped**（11.18秒，GPU条件不满足的2项仍跳过）。覆盖混合状态批量运行、原命令重复执行、候选级恢复、refit剩余seed恢复、完成文件内容及修改时间不变、严格模式、配置冲突、损坏/孤立状态拒绝与dry-run无训练。

另对原 `verified-v1` 快照实际执行无 `--resume` 的独立refit命令，成功输出 `[SKIP]`；5份候选JSON、3份refit结果JSON及1份权重的内容和修改时间均未改变。原快照dry-run输出 search=skip / refit=skip，核心源码指纹匹配。
