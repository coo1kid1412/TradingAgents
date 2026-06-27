# launchd 持久调度（替代 macOS cron，根治笔记本睡眠漏跑）

## 背景

`daily_update`（真值采集）原用 macOS **crontab** 调度。问题：cron 在笔记本**睡眠/关机**时
**不触发、也不补跑** → 真值采集被饿死（2026-06-27 周 review 实测：7 天只跑了 1 次、
最长落后 14 天、T+30 零数据）。

**launchd 的 `StartCalendarInterval` 会在机器下次唤醒时补跑一次错过的任务**，根治这个问题。

## 安装（在项目根目录的会话里用 `!` 前缀执行，或自己终端跑）

```bash
# 1. 安装 plist 到用户 LaunchAgents
cp deploy/launchd/com.tradingagents.daily-update.plist ~/Library/LaunchAgents/

# 2. 加载（-w 写入开机自启；旧 daily_update cron 行需移除，避免双跑，见第 4 步）
launchctl load -w ~/Library/LaunchAgents/com.tradingagents.daily-update.plist

# 3. 确认已加载
launchctl list | grep tradingagents

# 4. 从 crontab 移除旧的 daily_update 行（保留 weekly_review / market_risk）
crontab -l | grep -v 'tradingagents.harness.daily_update' | crontab -

# 立即手动跑一次验证（可选）
launchctl start com.tradingagents.daily-update
tail -20 harness_data/daily_update.log
```

## 卸载 / 回滚

```bash
launchctl unload -w ~/Library/LaunchAgents/com.tradingagents.daily-update.plist
rm ~/Library/LaunchAgents/com.tradingagents.daily-update.plist
# 如需回退 cron：重新把 daily_update 行加回 crontab
```

## 注意

- 仍只在机器**开机**时运行；长期关机期间不会跑，但开机唤醒会补跑一次（cron 不会）。
- `weekly_review`（周六 10:00）和 `market_risk_daily`（工作日 8:30/20:30）目前仍在 cron，
  有同样的睡眠漏跑问题；如需要可照此模式各加一个 plist。
- 真正长期稳定需要一台常开的机器（小服务器/NAS/云）跑这些 cron——launchd 只是缓解。
