---
name: ta-stock
description: 用户要求 TA 分析 A 股、美股或港股，或者询问 TA 分析失败、没有报告、任务状态时使用。
---

# TA 股票分析入口

先读取并遵循项目维护的完整运行规则：

`/Users/lailixiang/WorkSpace/QoderWorkspace/TradingAgents/skills/ta-stock-run/SKILL.md`

该规则使用 `.venv` 的 `tradingagents.analysis_runner` 入口，以及 OpenClaw 原生 exec 的 `background: true`、`timeoutSeconds: 0` 参数。
不得改用 nohup、shell `&` 或固定整体超时，不编辑 main.py 的股票常量。
状态以独立任务记录为准；不要把旧报告或 PID 存在当作本次成功，不擅自关闭其他应用。
