# V100 全模型 megakernel 交接

日期：2026-10-29。任务按用户要求暂停，转为交接。

**完整 64 层的单 token 前向已通过短程等价检查。性能对比尚未执行。prefill 扩展尚未验收。代码未提交、未合并、未部署。**

## 先从这里接手

1. 保留已下载权重和远端锁环境，避免重新下载。
2. 先复测当前源码的单 token 路径。最近新增了 token 循环，不能沿用旧结果证明新代码。
3. 再测完整 27B 的 B=4、长上下文与质量指标。
4. 修复并运行 benchmark。最后再优化 prefill 和接入 engine。

## 用户已确认的范围

- 模型：Qwen3.8-27B；实现：TileLang；设备：`ssh v100`。
- 对照：tileRL 原调度与全模型 megakernel。
- FP4 权重可从魔搭下载；负载自行选择，两组严格对齐。
- 用户接受本任务所需改动，不需重复确认文件数。
- 主助手直接执行，不使用子代理。
- 编译、检查、测试都在远端运行；Python 使用 uv。
- 单层融合、CUDA Graph 都不算全模型单 launch。
- 不改动其他服务、全局系统设置或参考仓库。

## 代码和设备位置

- 主仓库：`/Users/bytedance/code/tileRL`。
- 工作分支：`feat/v100-megakernel-direct`。
- 基线提交：`2a6c4506`，主仓库 main 保持此版本。
- 改动在下列独立 worktree，**分支名不包含未提交文件**：

```text
/Users/bytedance/Library/Application Support/DoubaoWork/Default/.doubaowork/agent_mode/workspace/.sessions/38450066931986434/agents/m_0cw3KPZV9ca/worktree
```

- 远端源码：`/home/chenkailun.c/tilerl-megakernel-direct`。
- 远端物理路径：`/data00/home/chenkailun.c/tilerl-megakernel-direct`。
- 设备：单张 Tesla V100-SXM2-32GB，卡号 0，sm70。
- 驱动：535.261.03；编译器路径：`/usr/local/cuda-12.4/bin`。
- 交接检查时：本任务下载、前向、benchmark 均无运行进程；显存 0 MiB。
- 其他会话的下载和服务不属于本任务，不得停止。

## 权重与依赖

权重来源：`unsloth/Qwen3.8-27B-NVFP4`，魔搭。

```text
/home/chenkailun.c/models/Qwen3.8-27B-NVFP4-modelscope
```

来源清单保存于 `source-manifest.json`。下载使用逐文件固定 revision。
全部文件已通过 SHA256 检查。两份 safetensors 已通过头部长度与索引检查。
`VERIFIED.json` 记录 1968 个索引张量。

- 主权重：`model.safetensors`，22,568,192,096 B。
- 主权重 SHA256：`c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05`。
- MTP 权重：`model_mtp.safetensors`，849,400,392 B。
- MTP SHA256：`1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe`。
- 发行版混合 FP4、FP8、BF16。不能称为全部权重都是 FP4。
- 结构与项目配置一致：64 层，16 个完整注意力层，48 个 GDN 层。
- 当前范围是语言模型前向，不包含视觉编码器或 MTP 推测解码。

独立锁环境位于 `scripts/v100-env`。默认仓库 `uv.lock` 未改。

- torch：2.5.1+cu121，包含 sm70。
- TileLang：0.1.13。
- safetensors：0.8.0；tokenizers：0.23.1。
- huggingface-hub：1.28.0。
- 环境已安装；CUDA 张量计算已验证。
- `fetch_torch.py` 获取原始 wheel，并校验锁定 SHA256。
- `pyproject.toml` 使用本地 wheel 路径；重建环境时先运行此脚本。

## 实测结果与覆盖边界

### 完整 27B：仅短程单 token 路径通过

证据：远端 `full-model-gate2.log`。

- 完整 64 层，B=1，连续 6 个 teacher-forced token。
- 两组共享同一组加载后的权重，分别维护状态。
- `torch.testing.assert_close(rtol=1e-2, atol=1e-2)` 全部通过。
- 最大 logits 绝对误差：0.008036613464355469。
- 6 个位置的 top-1 全部相同。
- profiler 记录 1 个 GPU 事件：`megakernel_kernel`。
- 退出码 0。
- trace：`megakernel-launch-trace.json`。

该结果没有证明完整 27B 的 B=4、长上下文、PPL、MMLU、needle 或自由生成质量。
输入是随机 token，不是语言质量评测集。
profiler 在已推进状态上额外重放了一次前向，只用于计数；它不能代替状态等价检查。

### 其他门禁

- 跨块同步：80 个 block，17 轮，3 次重复，结果精确；错误结果负控制被拒绝。
  证据：`probe_megakernel_grid.py`。
- 融合 FP4 linear 与 residual add：对原 sm70 算子，最大误差 0。
  证据：`ops-probe.log`。
- FP8 按需解码：对旧 BF16 materialized 权重面，最大误差 2.9802322387695312e-08。
  融合与原调度两个实现都通过。证据：`fp8-probe.log`。
- 混合精度小模型：B=4、连续 12 token，最大误差 0.0020003914833068848，top-1 全同；1 个 GPU kernel。
  证据：`mixed-gate.log`。
- CPU 对照小模型：连续 6 token，最大误差 0.0016471520066261292。
  证据：`mega-cpu.log`。
- 现有 CPU 回归：65 passed、5 skipped。
  证据：`cpu-gates-hidden.log`，范围为 `test_ops_parity.py` 和 `test_weights.py`。
- Ruff 曾对当时的新内核与共享路径通过；后来又修改了 loader、prefill 和 probe，**最新全量检查待跑**。

**速度、峰值显存、KV 容量变化、prefill/decode FLOPs 变化均未实测。没有提速结论。**
同形状、同精度下没有删除注意力或模型层，FLOPs 不应因减少 launch 自动下降。

## 当前实现

新增文件均在源码快照中，保持仓库相对路径。

- `kernels_mega.py`：persistent 指令解释核，固定驻留网格，每条指令后执行 `T.sync_grid()`。
  覆盖 embedding、norm、linear、SiLU、residual、RoPE、完整注意力、GDN、切片和 KV 写入。
- `mega.py`：构造模型指令、绑定权重地址、分配状态；提供 `step`、CPU 对照和尚未验收的 `prefill`。
- `mega_reference.py`：CPU 指令对照实现。
- `mega_loader.py`：逐张量加载；使用现有格式转换函数，保留 FP4/FP8 权重。
- `kernels_fp8_sm70.py`：原调度路径的 FP8 按需解码，计算权重面保持旧 BF16 舍入。
- `bench_megakernel.py`：待验证的 decode 对照脚本。
- `probe_megakernel_*.py`：同步、算子、FP8、模型门禁。
- `download_megakernel_weights.py`：来源清单、Range 下载、SHA256 和 safetensors 校验。
- `scripts/v100-env`：独立依赖定义、锁文件及 torch wheel 获取脚本。

已修改的旧文件：

- `backend.py`：加入 `TILERL_SM70_NATIVE_FP8=1` 分支；保留 FP8 存储，调用新 sm70 算子。
- `registry.py`：注册该算子。
- `model.py`：serving 加载避免创建无用 FP8 master；FP8 拼接改为 uint8 字节视图再恢复 dtype。

### 基线已经发生变化，必须单独披露

旧整库加载在主机内存压力下退出 137；`old-load.log` 为空。
退出 137 本身不证明 CUDA OOM。现场主机内存逼近上限，没有进入 GPU 前向。

当前 64 层门禁使用逐张量 loader，两组共用。
原调度臂还启用了 native FP8 存储路径。
所以当前对照是**原调度 + 共同适配**对**megakernel + 共同适配**。
不能将这组结果写成未修改 main 的端到端性能比较。

## 最近改动与已知问题

1. **prefill 尚未验证。**
   最近把 `token_step` 循环放进单 kernel，并新增 `MegaForward.prefill`。
   循环按 token 串行处理整个模型，每 token 仍计算 lm_head；它不是高效批量 GEMM prefill。
   不能以一份 prompt 只有一个 launch 就宣称达到性能目标。

2. **本地与远端不完全相同。**
   当前远端 `kernels_mega.py` 已有 token 循环；远端 `mega.py` 尚无 `prefill` 方法。
   64 层门禁进程在最后扩展前启动；运行时已导入的代码与磁盘新代码可能不同。
   保存原日志，但先复测最新源码。

3. **benchmark 尚未执行。**
   脚本只测 decode；共同 prefill 由 baseline 完成。
   KV/state 复制后做两组换序测试；质量检查失败会阻断计时。
   尚未核对 GDN active window 的 parity、完整 KV 复制和 snapshot 内存峰值。
   `TILERL_SM70_NATIVE_FP8=1` 必须由运行环境设置。

4. **调试模式有遗漏。**
   `probe_megakernel_model.py --debug` 把任何非零 linear mode 送入 FP4 对照。
   FP8 的 mode=-1 也会进入该分支，需要修复。
   debug 会复制全部 buffers，禁止直接对 27B 使用，避免显存不足。

5. **下载续传需要整理。**
   首次顺序下载的 prefix 续传成功。
   分段下载若中断，文件 size 不代表全部 range 都完成；需要 range 完成清单。
   现有完整权重已通过总 SHA256，不受该问题影响。

6. **生产接入尚未做。**
   Model.forward、Engine 的 submit/poll、paged KV、prefix 生命周期尚未接入 megakernel。
   当前独立 kernel 使用自己的连续 KV/state；不能声称已完成生产调度替换。
   性能优化、最小化整理、完整 pytest、bench 条目、CHANGELOG、PR 均未完成。

7. **开发辅助文件可在整理时删除。**
   `.expand_mega_tokens.py`、`.megakernel-environment-note.py` 和未使用的 `run_megakernel_job.py` 不应作为生产依赖。

## 已验证入口

以下命令均在远端运行。不要使用旧 `scripts/v100.sh` 的默认源码目录覆盖其他会话。

```bash
ssh v100
cd /home/chenkailun.c/tilerl-megakernel-direct
export PATH=/usr/local/cuda-12.4/bin:$PATH
export TMPDIR=$HOME/tmp
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

~/.local/bin/uv run --project scripts/v100-env --frozen python -u \
  scripts/probe_megakernel_model.py \
  --checkpoint /home/chenkailun.c/models/Qwen3.8-27B-NVFP4-modelscope \
  --stream-load --tokens 6 --profile
```

上述入口曾运行成功；最新源码应重新运行，不把旧日志覆盖。

```bash
CUDA_VISIBLE_DEVICES='' TILERL_TARGET=cpu \
  ~/.local/bin/uv run --project scripts/v100-env --frozen \
  pytest tests/test_ops_parity.py tests/test_weights.py -q
```

CPU 目标必须隐藏 CUDA。旧 `memory.measured_peak_bytes` 只检查 CUDA 可用性，
会把 CPU backend.device 传给 CUDA 内存查询；未隐藏时有两个测试失败。
该旧问题未修改。

## 下一位执行者的验收顺序

- 先对最新文件做 Ruff 和单 token 回归；补可失败的模型状态负控制。
- 对完整 27B 执行 B=1/B=4、不同位置和连续状态验证，比较 logits、KV、GDN state 和 window。
- 对多 token `prefill` 做串行参考与原批量 prefill 对照，检查正确性和实际 launch。
- 修复并跑 `bench_megakernel.py`。建议先 B=1/context=512，再 B=4 和 4096。
- 每组使用相同 token、权重、精度、缓存容量；排除编译时间，重复换序并增加独立进程对照。
- 先报告质量指标，再报告容量、KV 与 prefill/decode FLOPs；目标与实测分开。
- 用真实 profile 决定优化项，当前重点候选是串行 attention、低并行度和反复权重读取。
- 最后完成 engine 接入、全量门禁、bench 记录和最小化整理，再提交自己的文件。

## 交接文件

- 本文：`docs/megakernel-v100-handoff.md`，用文本编辑器或 Markdown 预览打开。
- 源码和证据：`megakernel-v100-handoff-source.zip`，解压到新的临时目录。
- 快照的 `source/` 保存当前改动；`changes.patch` 只包含三个已跟踪旧文件的修改。
  新文件必须从快照复制，不能只应用 patch。
- `evidence/` 保存远端门禁日志、trace、权重来源清单和校验结果。
- 快照不含模型权重、wheel、虚拟环境或编译缓存。

## 交接核验

文档格式和四项关键状态检查通过。
源码包逐项通过 SHA256 回读检查。
压缩包通过 CRC 检查。关键源码和日志齐全。
语言评分为 80%，短句项未通过。
评分包含代码路径和命令；不声称语言检查全部通过。

交接检查仅覆盖文档、快照和现场状态。
性能优化任务仍未完成。
