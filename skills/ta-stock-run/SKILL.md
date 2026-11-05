---
name: ta-stock-run
description: 用户通过飞书要求 TA 分析股票，或询问 TA 跑完了吗、分析是不是挂了、为什么没有报告时使用。
---

# TA 单股票分析

项目：`/Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents`。
使用该项目 `.venv/bin/python`，沿用项目 `.env`、模型配置和 main.py 内的报告／飞书交付流程。

## 先查状态

调用 exec 运行以下短命令，替换股票代码：

```bash
cd /Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents
.venv/bin/python -m tradingagents.analysis_runner status --ticker 688507
```

- `running`：已有同标的任务，仅报告状态，不重复启动。
- `succeeded`：分析完成。用返回的 `report_path` 定位报告；只有 `delivery_status=sent` 才可声称飞书发送成功。
- `failed`：读取该次 `log_path` 与 `exit_reason`，说明失败，不让用户继续等待。
- `partial`：已有可核验报告，但进程异常结束；核对 `report_path` 与 `delivery_status`，不要声称完全成功，也不要声称没有报告。
- `interrupted`：托管进程与分析进程均不在运行，且未写入终态；杀因未知，不得直接归因 OOM。
- `not_found`：没有新入口的运行记录。对旧任务检查 `/tmp/ta_<ticker>.log` 及对应进程，避免启动重复任务。

查询旧报告不等于重新分析。用户明确要求重新分析时，即使有历史 succeeded 记录，也可启动新任务。
本项目当前共享 LLM 调用日志，只允许一份 TA 分析在运行；入口遇到任意标的正在分析会返回 `already_running`。

## 启动：必须用 OpenClaw 原生独立后台模式

区分外层 JavaScript 编排工具与内层 shell exec：外层常只有 `code/title`，后台参数属于内层 shell exec。

代码模式中先用 `catalog.search("exec shell command background")` 查询工具；当前运行时返回条目的 `input` 字段是 JSON schema 字符串，直接读取它确认参数。若 `API.read("tools/exec.d.ts")` 返回 `content: null`，改读目录返回的 schema，不要反复读取空文件。`catalog.search(...).describe()` 的空结果也不等于工具不支持。

确认内层 exec 的 `background` 与 `timeoutSeconds` 参数后，执行以下编排代码（若直接暴露 shell exec，传入同一个参数对象）：

```javascript
const result = await exec({
  command: ".venv/bin/python -u -m tradingagents.analysis_runner run --ticker 688507 --rounds 2",
  workdir: "/Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents",
  background: true,
  timeoutSeconds: 0,
  title: "TA 股票分析 688507"
});
text(result);
```

股票代码必须来自用户。默认分析日期为上海时区当天；历史分析使用 `--date YYYY-MM-DD`。
多空辩论默认 `--rounds 2`，用户明确要求三轮时设为 3。

`background: true` 把整条前台命令交给独立服务托管。`timeoutSeconds: 0` 关闭整任务超时；单次 API 超时仍由项目配置控制。
不要设置 `awaitResults: true`。不要在 shell 命令后加 `&`，不要用 nohup、setsid 或 subprocess.run(timeout=...) 包装。
不要修改 main.py 的 `_TICKER`／辩论轮数；入口把参数直接交给现有分析函数。
不要使用 `capture_output=True` 将完整分析日志积压到内存。

如果当前 exec schema 缺少独立后台参数，明确说明运行环境不兼容；不要猜测参数名或退回会被回收的 shell 后台方式。

## 启动验收与用户回复

在另一个短 exec 调用中查询一次 status，确认 `running` 或已经结束。
`running` 只代表已托管，回复股票、运行编号、分析日期、当前阶段及日志路径；若 exec 返回 `sessionId`，一并保留。历史分析约 20—45 分钟，实际随模型和数据源变化，不据此设置整任务超时。
`already_running` 表示本次没有启动，先查询正在运行的任务，不宣称新任务已接单。
`succeeded` 与 `delivery_status` 分开判断：分析成功但推送未确认时，明确说“报告已生成，飞书送达未确认”，不得误报成“仍在运行”。

启动后不持续轮询，不占用当前会话等待模型，不新建重复分析。用户询问进度时再查 status。
成功推送沿用 main.py，普通分析失败也由 main.py 通知；分析子进程异常退出时 runner 补发失败通知。整组进程或宿主机停止后，status 能识别 interrupted，但不能保证当时仍有进程可发送通知。

## 异常诊断

先看 `exit_reason` 和 `log_path`：

- `signal:SIGKILL`：进程被信号 9 终止，需结合系统或工具记录判断原因。单凭静默退出、swap 较高不能证明 OOM。
- `exit:1`：看完整错误堆栈，可能是项目或上游服务错误。
- `supervisor_and_worker_not_running`：未取得最终退出码，先标记中断，检查 OpenClaw／系统日志。

未经用户授权，不关闭浏览器、Node 或其他无关进程，不清理用户内存和缓存，不改模型或降低分析质量来掩盖启动失败。
