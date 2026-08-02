# 大盘骤跌预警 V1 运维手册

## 1. 报告先看什么

报告第一屏按操作优先排列，依次看：

1. **灯号与立即操作**：橙灯表示“提前防守”，红灯表示“风险确认”。
2. **入场门、新增仓位上限、持仓动作**：这是当下可执行的仓位约束。
3. **数据截至时间与可靠度**：数据不是扫描时点可见、可靠度不足时，不应把报告当成有效红灯。
4. **前三条触发规则**：用于解释这次晋级由哪些可观测信号造成。

`规则分数 0-10` 只用于规则强度审计，**不是暴跌概率**。影子模型和 MiniMax M3 只提供对照与情景解释，均不能改变规则灯号、入场门或仓位上限。

### 灯号含义

| 灯号 | 含义 | 报告中的操作口径 |
|---|---|---|
| 绿灯 | 未发现显著骤跌结构 | 按原策略执行 |
| 黄灯 | 风险升温但未达到推送门槛 | 盘中不推送 |
| 橙灯 | 早期脆弱性与压力信号形成组合 | 暂停追高，收紧新增仓位，检查高波动持仓 |
| 红灯 | 硬触发或风险延续得到确认 | 停止新增风险，优先降低高弹性和高拥挤暴露 |
| UNKNOWN | 核心数据或系统不可用 | 不等于安全，也不等于红灯；先处理数据故障 |

盘中只在首次橙灯、橙灯升级红灯或直接红灯时推送；黄灯、同级维持和风险恢复保持静默。盘前报告每天固定发送一次，便于看到完整状态，而不是只看异常。

## 2. 调度口径

- A 股盘前：每个交易日 `08:30`。
- A 股盘中：`09:35` 起每 10 分钟至 `11:25`；`13:05` 起每 10 分钟至 `14:55`。
- cron 每 5 分钟唤醒一次，runner 只在上述格点执行，避免错过边界时间。
- 美股 V1：仅按纽约当地时间 `08:30` 盘前影子运行，不启用规则生产通知，也不进入个股硬门控。
- 规则快路径不调用 LLM。发生有效橙/红推送后，M3 才在慢路径补充解释，最长等待 90 秒。

## 3. 数据与权限

A 股盘中规则需要两类当前数据：指数实时行情和全市场横截面实时行情。Tushare 5000 积分不代表账户一定拥有全市场 `rt_k` 权限，因此上线前必须运行权限探针。

```bash
.venv/bin/python scripts/probe_market_warning_data.py \
  --market a_share \
  --json \
  --output harness_data/models/market_warning/rule-v1-data-smoke.json
```

重点检查 JSON 中：

- `index_realtime_ready`：指数行情属于当天且陈旧不超过 5 分钟。
- `rt_k_permission`：必须为 `available`。
- `realtime_breadth_coverage_pct`：必须不低于 80%。
- `realtime_breadth_staleness_minutes`：必须不高于 5 分钟。
- `stk_limit_available`：当日跌停价数据可用。
- `latest_completed_trade_date`：盘前基线使用的最近已完成交易日。
- `ready` 与 `failures`：是否满足通知数据门槛及精确阻断原因。

权限不足、覆盖率不足或数据陈旧时，系统不会把 T-1 收盘数据冒充盘中现价，也不会触发依赖市场宽度或跌停占比的红灯。连续 3 个应执行时点失败时，飞书收到的是“预警系统数据故障”，该消息不代表市场红灯。

## 4. 上线前检查

### 4.1 检查生产门槛

```bash
.venv/bin/python -m tradingagents.harness.market_warning.readiness \
  --mode rule_v1/notify
```

只有输出 `ready: true` 才允许安装。通知门槛同时要求规则历史评估、实时数据探针、清单校验和及 100 轮快路径基准通过。

### 4.2 检查时点，不执行分析

```bash
.venv/bin/python -m tradingagents.harness.market_warning.runner \
  --market a_share \
  --mode rule_v1 \
  --at 2026-08-03T09:35:00+08:00 \
  --dry-run
```

有效格点应只返回一个 `due`；非交易日或非格点返回空列表。

### 4.3 预览和安装调度

```bash
.venv/bin/python scripts/install_market_warning_rule_v1.py \
  --mode rule_v1/notify \
  --dry-run
```

```bash
.venv/bin/python scripts/install_market_warning_rule_v1.py \
  --mode rule_v1/notify \
  --yes
```

安装器只激活 `rule_v1/notify`，不会激活 `rule_v1/gate`。重复安装不会生成重复 cron 项。

## 5. 日常排查

### 查询最近运行

```bash
sqlite3 harness_data/tradingagents.db \
  "SELECT market,as_of_time,session_slot,mode,status,error_class,ROUND(latency_ms,1),llm_calls FROM market_warning_runs ORDER BY id DESC LIMIT 20;"
```

快路径正常记录应满足 `mode=rule_v1`、`status=success`、`llm_calls=0`。`overlap_skipped` 表示前一轮仍持有租约，不代表市场风险。

### 解除过期租约

租约默认 8 分钟自动失效。只有确认对应分析进程已经结束、且租约时间确实过期后，才手工清理：

```bash
sqlite3 harness_data/tradingagents.db \
  "DELETE FROM market_warning_leases WHERE expires_at <= datetime('now');"
```

不要在进程仍运行时删除租约，否则可能形成重复扫描和重复数据调用。

### 停用通知并卸载调度

```bash
.venv/bin/python scripts/install_market_warning_rule_v1.py --uninstall
```

该命令只删除本 V1 生成的 cron 区块并停用 A 股 `notify`，不删除其他周期任务，也不改动 `gate` 状态。

## 6. 10 个交易日灰度审计

V1 首次部署只运行通知灰度，个股分析硬门控保持关闭。至少完成 10 个 A 股交易日后，逐日检查：

- 应执行格点是否完整，失败和数据陈旧是否被明确记录。
- 首次橙灯、橙转红、直接红灯是否只推送一次。
- 月均新橙/红次数是否仍不超过历史评估预算。
- 单一危机样本对真阳性的贡献是否不高于 50%。
- 告警后的 1 日和 3 日结果是否与历史口径一致。
- 个股报告是否只读取同一规则版本、同一清单校验和且数据为 `fresh` 的通知结果。

审计满足后先检查硬门控 readiness：

```bash
.venv/bin/python -m tradingagents.harness.market_warning.readiness \
  --mode rule_v1/gate \
  --soak-sessions 10
```

`ready: true` 只代表具备评估激活资格，不会自动打开硬门控。应在人工复核 10 日审计记录后再单独激活；安装器始终禁止自动启用 `gate`。

## 7. 当前 V1 边界

- A 股规则生产通知：需完成真实权限探针与 readiness 后方可安装。
- A 股个股硬门控：默认关闭，等待 10 个交易日灰度审计。
- 美股：保持盘前影子模式，不发送规则生产告警。
- 历史日线评估只证明日级信号有效性，不能描述为分钟级实盘验证；分钟级表现需在后续交易日持续审计。
