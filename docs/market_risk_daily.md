# 开盘前市场风险任务

`python -m tradingagents.harness.market_risk_daily --market a_share` 生成并保存 A 股快照；
`--market us` 对应美股。飞书推送优先使用 `.env` 中的 `FEISHU_MARKET_RISK_WEBHOOK`；如果没有配置 webhook，会复用既有的 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_USER_OPEN_ID` 通过飞书 Open API 私聊发送文本。两种方式都没配置时，快照仍会落库，但推送会记录为失败。

建议按当地市场开盘前运行：

```cron
# A 股：上海时间工作日 08:30
30 8 * * 1-5 cd /path/to/TradingAgents && .venv/bin/python -m tradingagents.harness.market_risk_daily --market a_share

# 美股：纽约时间工作日 08:30；请在使用该 cron 的主机/调度器中指定 America/New_York 时区
30 8 * * 1-5 cd /path/to/TradingAgents && .venv/bin/python -m tradingagents.harness.market_risk_daily --market us
```

可用 `--date YYYY-MM-DD --dry-run` 验证数据链路；dry-run 不写数据库也不推送。休市日自动跳过，不生成伪预测。

如果用 crontab，注意 cron 不一定加载交互式 shell 环境；脚本会自动读取项目根目录 `.env`，所以通常不需要在 crontab 里重复 export 飞书变量。
