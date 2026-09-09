# 架构评审：删什么、建什么、什么叫 AI friendly

数字基准：origin/main @ `fa1bcec`（2026-09-09）。scripts/ = 236 个 .py / 35,367 行；src+kernels = 20,707 行；tests = 18,344 行。验证装置（scripts + tests）= 53,711 行，是系统的 2.6 倍。装置自己也遵守这条规律：xdist 把 pytest 段加速 6.4x，job wall 只快 1.77x（2 个落地前 run 对 9 个落地后 run 的中位数，27 实测）——加速一个环节，瓶颈就换人（uv sync、ruff、串行分布式 gates 不在 xdist 手里）。

## 开头：产生速度大于删除速度

89 个 `probe_*.py`，最后修改全部落在 8 天内（2026-09-02 起）：

```bash
git ls-files 'scripts/probe_*.py' | xargs -I{} git log -1 --format=%ad --date=short -- {} | sort | uniq -c
```

输出（fa1bcec）：09-02 ×3、09-03 ×25、09-04 ×9、09-05 ×6、09-06 ×12、09-07 ×11、09-08 ×22、09-09 ×1 = 89 个 / 8 天 ≈ **11 个/天**。这改变了「舍弃更多东西」的答案形状：一次性清扫会在一周内被填满。**删除是配套，不是主解。主解是让一次新测量的默认入口比写一个新脚本更便宜**（B1）。先删，是因为删除给 B1 腾出位置和注意力。

## 1. 删什么（按删除成本从低到高）

### D1 — step 5 的 22 个脚本（成本：接近零）

`docs/bench-inventory.md` 已判定：Keep 33，Delete 22（10 bench 含 `bench_smoke.py` 移 CI、4 ab、4 sweep/matrix、4 launcher）。判定已记录，这是执行不是评审。

规则（写进 step 5 PR 描述）：**先迁移唯一出处的值，再删**。三条件全过才删：判定已记录 + 不是任何在用值的唯一出处 + 可复现或不再测量。

### D2 — probe_ 家族：9 个候选（成本：低；判据必须可跑）

可跑判据：

```bash
git ls-files 'scripts/probe_*.py' | while read f; do b=$(basename "$f" .py);
  git grep -l "$b" -- docs/ CHANGELOG.md .github/ scripts/ >/dev/null || echo "$f"; done
```

fa1bcec 上输出 9 个：probe_fla_hostcost、probe_fla_parity、probe_group16_mechanism、probe_gsm8k_lengths、probe_iso_frames、probe_kernel_gaps、probe_math_boxed、probe_state_scan_port、probe_w8_e2e。

「近 30 天有 commit 碰过」可跑，但对 probe_ 无效——89 个全部本周碰过，它一个都删不掉。「主线工具」是形容词，不能跑。上面的判据能失败：它输出了 9 个名字。每个候选仍须过 D1 的三条件；判据只产生名单，不做判决。

**2026-09-09 更正 — 上面这条判据漏掉了 CI 实际执行脚本的那条路径，先跑下面这条。** 它按脚本名 `git grep`，而 `tests/test_main_selfchecks.py` 的 `_hermetic_scripts()` 用 `glob("*.py")` 加 AST 收集 `scripts/`：一个 hermetic 且 `__main__` 里带 assert 的脚本会被 CI 跑，而树里任何地方都不出现它的名字。名字搜索看不见模式引用，且正对照救不了这一类——换一个脚本名去搜同样是 0，对照会和错误答案一起点头。

`origin/main` 上被 CI 跑的三个：`gate_noise_floor.py`、`probe_group16_mechanism.py`、`sm70_tile_occupancy.py`。第二个就在上面那份 9 人名单里。

所以候选先过这一条，它一票否决：

```bash
python3 - <<'EOF'
import ast, pathlib
def asserting_main(p):
    try: tree = ast.parse(p.read_text())
    except SyntaxError: return False
    return any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)
               and any(isinstance(x, ast.Assert) for x in ast.walk(n)) for n in tree.body)
for p in sorted(pathlib.Path("scripts").glob("*.py")):
    if not any(k in p.read_text() for k in ("import torch", "get_backend", "build_engine")) \
       and asserting_main(p):
        print(p.name)
EOF
```

一般形式：**判断一个文件有没有被用到，先问消费者是怎么命名它的输入的。** 按名字引用才能用名字搜索；glob、动态 import、`getattr`、字符串拼出来的键、按文件名约定收集——每一种都是名字搜索找不到的引用。唯一可靠的读法是跑一遍消费者自己的过滤器，或者读它的运行日志（CI 日志把十个 gate 逐个打印了名字）。同一天 `ci.yml:59` 的 `tests/*_world[0-9].py` 也是这样被漏掉的，代价是一个 PR 差点删掉训练侧梯度平均的唯一门。

### D3 — `--devices` / DataParallelEngine（成本：中；论据是错误类，不是性能）

**不要用性能论据。** 7.54x（wins/2026-08-29-data-parallel-scales）量的是 8 个独立进程对一张卡的聚合，从来没有量过 `--devices`——27 已撤回这个用法。

真证据：

```bash
git grep -l "DataParallelEngine\|serve --devices" origin/main -- docs/experience
```

→ 11 篇，其中 9 篇是 errors/，十天之内（08-30 至 09-07）。病因（errors/2026-09-05-the-clamp-read-a-getattr-default.md:19）：`DataParallelEngine`（parallel.py:27）手写转发 seam 方法，**没有 `__getattr__`**——每加一个 seam 方法就有一次静默漏掉的机会，已经漏过至少六次（limits、submit 的 prefix blocks、clip norm、SSE handler、socket 关闭、health 锁）。`parallel.py` 的 `limits` property 现在存在——那是修复后的现场，不是活着的 bug；论据不需要它现在坏着，需要的是「漏一个方法的默认后果是静默」这个结构。

删掉失去什么：多卡 serving。为什么不值：项目形状是一卡一进程（CLAUDE.md），README 头条数字全部单卡；一个手写转发层的成本是每个 seam 方法一次静默 400，而它的收益从来没被量过。

删法：serve 的 `--devices`、parallel.py 的 DP 类一起删。**两个同名的东西不删**：`generate --devices` 是每卡一进程的 fan-out（替代路径本身，不是 wrapper）；`tests/dp_world4.py` 的 "dp" 是训练侧梯度平均（`cli._shard` + `train.train_step` + `Backend.dp_reduce`），守的是训练侧的门（mean 不是 sum、reduce 在 clip 之前、各 rank apply 顺序一致），和服务端副本的 DataParallelEngine 是两个机制——删掉它一个活着的机制就没有门了。**多卡 serving 若要回来，门槛是一个表面全等测试**：枚举 Engine 的方法集，断言 wrapper 全部转发，漏一个 CI 就红。没有这个测试，不要重建。

### D4 — `--patience-mode` raw（成本：零，9b 在删）

确认删除，不要复活。

### 明确不删：spec decode 和 prefix cache

我带着「H4：spec 和 prefix 互斥，删一个」的假设来，驳倒了：互斥已经是 build-time 的 ValueError（engine.py:447，drafter 接 trunk aux 层 + prefix cache 直接拒绝），且错误消息本身写明 `the failure would look like a weak drafter, not a bug`——作者已经把这个失败模式写在拒绝的理由里。不诚实的地方在 README 没写明不组合，不在引擎。两条都是活跃主线（spec = 09-07 优先级，KV tiers = 09-05 方向）。处置：gate 保留 + 一个钉死 gate 的测试 + README 写明不组合。删活跃主线是 ckl 的产品决定，评审不替。

## 2. 建什么：只有一件

### B1 — `tilerl bench <name>`：一个入口 + 注册表

现状：236 个脚本各自解析参数、各自打印、数字靠人抄进 wins/。新测量的默认动作是「写第 237 个脚本」——12 个/天就是这么来的。B1 让默认动作变成「注册表里加一行 metric + 一个 collector」，store 已经在了（docs/experience/bench/measurements.jsonl + scripts/benchrec.py，#350/#360 已合）。

**成本：** 第一批覆盖 7 个 README 引用的 metric 的 feeder（decode_tok_s、prefill_tok_s、decode_agg_tok_s、spec_goodput_ratio、gsm8k_pct、mmlu_pct、ssd_restart_speedup）。每个迁移 = 包一层 collector + 定 floor/shape，约 1–2 人时/个，合计 1–1.5 人日。注册表已有 14 个 metric，这 7 个的 schema 都在。

**eval 没有独立入口，必须单独说：** gsm8k 评分（`eval.gsm8k_accuracy`）活在 `_train_adapters`（cli.py:655）里面，没有 `tilerl eval`——不跑一次 training 就量不到 gsm8k_pct，而它是注册表里权重最高的 metric（0.94）。B1 必须包含一个独立 eval arm（`tilerl bench gsm8k` 或 `tilerl eval`），否则权重最高的度量永远只能作为训练的副作用被收集。

不建它明天怎么坏：今晚灌数之后 store 有行、注册表有 metric，但下一个新测量仍然没有入口——速率不变，三周后 apparatus 到 3 倍。

## 3. AI friendly：定义 + 三条纪律换成结构

**定义：agent 没有默契。** 人靠「我们都知道」工作的地方，agent 靠「写错了会怎样」工作。AI friendly 不是提示词写得好，是**让错误在结构上不可能**——写错的东西在写的时候被拒绝，而不是在评审时被发现。

树里已有三条纪律在靠人维持，每条都有结构替代：

1. **warm.compiles：22/23 个采集点硬写 "warm"，没有一个测量它**——stats() 没有 compiles 计数器。纪律「记得标 warm」→ 结构：compile 路径加 per-engine 计数器 + stats() key，collector 读窗口差值。半天，排在明天。计数器落地后 warm.state 从声明变成可校验。
2. **commit sha 手填 → 校验器**（已落地）：benchrec.validate() 查 40-hex + `git cat-file -e`；`git_dirty()` 对未知状态 raise——从不把 unknown 渲染成 clean。纪律「别填错 sha」变成写时拒绝。
3. **README 可复现 → 生成**（进行中）：README 每个数字必须映射到注册表 metric，否则标 external；映射是生成的，不是手抄的。今天的逆向审计抓到一个单位错误（351.8 被映射成秒，实为 tokens/correct）——手抄的映射会漂移，生成的不会。

4. **绿不是评审：「合并前必须读 diff」→ 冻结表面测试。** 2026-09-09 晚上，一个删除 `--deterministic` 的 PR 双绿、CLEAN、无冲突地合入——删除明明白白写在 diff 里，合并者只看了绿勾。纪律在一晚合 20 个 PR 时必然失效。结构替代：**一个测试枚举 CLI 的全部 flag（`--help` 输出），删掉任何一个都会让 CI 红，除非同一个 PR 显式修改期望集合**——删除于是变成一个 deliberate、可见的动作。它和接线测试是一对：接线测试保证 flag 做事（`--deterministic` 复活时补的那个），冻结表面保证 flag 不会无声消失。边界：挡得住表面删除，挡不住 flag 还在但语义漂移——那是接线测试的活。（已落地：`tests/test_docs_links.py::test_cli_surface_is_frozen`，2026-09-09；直接读 parser，与 `--help` 输出逐项一致。）

评审纪律本身同构：不说「建议考虑」；每个论断带 file:line 或可跑命令；驳论先于立论——这份文档的 H4 就是被驳掉的删项，流程起作用了。
