# TA 任务中断诊断

先运行项目状态入口：

```bash
cd /Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents
.venv/bin/python -m tradingagents.analysis_runner status --ticker 688507
```

按输出的 `log_path` 读取本次日志，核对 `run_id`、`analysis_date`、`exit_reason`、`report_path` 和 `delivery_status`。不要把旧的同股票报告当作本次结果。

- `running`：任务锁仍由 supervisor 或分析 worker 持有。不要重复启动、按固定分钟数杀进程。
- `succeeded`：worker 确认分析成功，且存在非空 decision.md；飞书发送仍需独立检查 `delivery_status=sent`。
- `partial`：已有可核验报告，但托管未完整结束。核对是否已发送，不要发送“没有报告”的误报或自动再跑一份。
- `failed`：看退出码和本次日志，普通 Python 异常应有 traceback。
- `interrupted`：进程均停止且没有可恢复结果，退出原因未知。检查工具或系统的对应时间记录。

`signal:SIGKILL` 仅证明信号 9，不足以区分内存压力、工具清理、手工终止或其他系统原因。高 swap、低空闲内存、无 traceback 都不能单独证明 OOM。

旧启动链的两个已知问题：

1. OpenClaw 托管 shell 结束后会清理仍存活的进程组，`nohup ... &` 及子 shell 后台不能保证独立存活。
2. `subprocess.run(timeout=...)` 超时会终止子进程。捕获 TimeoutExpired 后不可说“仍在运行”。540 秒不足以完成历史约 20—30 分钟的 TA 分析。

按 SKILL.md 使用 OpenClaw 内层 shell exec 的 `background: true`、`timeoutSeconds: 0` 参数。外层 JavaScript 编排工具的 code/title schema 不是 shell exec 的 schema；通过 `catalog.search("exec shell command background")` 返回条目的 `input` schema 确认参数。`API.read("tools/exec.d.ts")` 在部分运行时返回 `content: null`，不代表工具不支持。

不得自动关闭用户浏览器、Node 或无关应用，不更换模型、不跳过分析师、不降低报告标准来掩盖运行问题。整组进程或机器停止后，本地通知不保证还能发出；后续状态查询会识别中断。
