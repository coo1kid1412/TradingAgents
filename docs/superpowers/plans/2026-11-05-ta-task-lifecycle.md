# 飞书调用 TA 任务中断修复

## 事故证据

- 最近一次实际分析标的是 688507，分析日期 2026-11-04。
- 项目四份当天运行日志仅有启动横幅；`/tmp/ta_688507.log` 停在 Phase 1/13 与首个 LLM 输入记录，没有分析异常堆栈、最终报告或成功推送。
- OpenClaw 历史记录显示：先使用 nohup 后台子进程；随后尝试 macOS 不存在的 setsid；最后用 `subprocess.run(..., timeout=540)` 包装完整分析，同时 exec timeoutSeconds=600。
- 当前安装的 OpenClaw exec schema 明确支持 `background: true`（独立服务、脱离请求停止）及 `timeoutSeconds: 0`（禁用整任务超时）。托管进程代码会回收命令退出后仍存活的进程组，普通 shell 后台不适合此场景。
- 540 秒超时与正常 20—30 分钟分析不兼容；不能把 TimeoutExpired 后已经被 subprocess.run 杀掉的子进程描述为“仍在运行”。
- 另有内存压力较高的历史观测，但没有对应 TA 进程的系统杀因记录。不能认定 OOM，也不应自动关闭用户浏览器或 Node 进程。

## 修复

1. `tradingagents.analysis_runner` 提供 run/status 入口，沿用现有 `.env`、分析图、报告与推送逻辑，不改股票分析决策。
2. 股票、日期和辩论轮数通过参数传入，不再编辑 main.py 的个人配置。
3. 使用 OpenClaw 原生独立后台执行 runner，无整任务固定超时。每个任务持久化独立 output.log 和 status.json，保留真实退出码／信号。
4. 共享锁阻止重复分析；锁由 supervisor 与 worker 共同持有，避免 supervisor 异常消失后重复启动。状态查询使用任务锁，不依赖可能复用的 PID。
5. 报告生成与飞书发送分别记录；无非空 decision.md 不得标记成功。普通主流程失败不重复推送，异常退出补发失败通知。
6. main.py 的失败通知优先采用 `TA_RUN_LOG` 的本次真实日志路径，保留旧入口默认路径。
7. 版本化 TA 技能，部署到实际 OpenClaw Workshop 技能及旧跨运行时技能，避免错误启动指令再次被选用。

## 验证

先添加失败回归，再实现入口。使用真实 Python 子进程验证成功落盘、信号 9、引导失败、普通失败、缺失报告、重复启动、失联任务及飞书独立状态，不消耗模型额度。

- runner 的 13 项真实子进程回归通过；tests 目录 unittest 共 27 项通过，另有 14 项独立函数检查通过，共 41 项。
- 通过本机 OpenClaw agent 的真实编排工具完成 45 秒假分析。使用 inner exec `background=true/timeoutSeconds=0`；agent 返回后任务仍存活，最终运行到 Phase 13 并落盘，exit:0。此测试不调用 TA 模型、不发送飞书。
- 假任务 run_id：20261105_212250_488782_68104609；记录目录 `/tmp/ta-openclaw-lifecycle-smoke-20261105/logs/analysis_runs/`。
- 已核验 Workshop 实际技能、诊断参考及旧跨运行时别名与仓库版本字节一致，旧文件备份于 `/tmp/ta-stock*-before-20261105.md`。
- 独立代码审查发现并补测：supervisor 输出断管不能让存活 worker 提前失败；supervisor 消失后可从 worker 结果恢复成功；已生成／发送报告后异常退出标记 partial，避免“没有报告”误报。
- 真实关闭 supervisor stdout 管道的测试先复现 Python 退出刷新导致 exit 120，再通过重定向失效 stdout 描述符修复；任务成功不再被托管层误判为失败。
- 真实 688507（2026-11-05）回归尚未启动：自动审批要求本次具体标的分析及飞书推送的明确授权，已向用户补问。不得将无模型冒烟等同于真实投研报告交付成功。
