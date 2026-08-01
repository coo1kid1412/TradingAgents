# A股与美股大盘骤跌概率预警

该模块估计 A 股和美股未来 1 个、3 个交易日的宽基骤跌概率。它用于限制短期新增风险，不修改个股一年期评级或目标价，也不自动下单。

## 报告怎么看

每份报告先看最上方两行：

1. 灯号与“立即操作”给出当前最重要的短期动作。
2. `入场门` 决定能否开新仓，`新增仓位上限` 是市场级上限，个股规则可以更严格。
3. 概率表同时列出模型概率和历史基准率。概率是风险估计，不是“必然下跌”的断言。
4. `FIRST_SHOCK` 表示尚未进入明显回撤后的第一段冲击；`CONTINUATION` 表示市场已经处于回撤后的脆弱阶段。
5. “相比上一份”用于判断风险是升级、维持还是进入恢复观察。
6. “主要驱动”只展示前三个可审计模型贡献项。
7. “M3 情景校验”展示结构化场景、因果链和反向证据，不保存 Think 过程。
8. 最后检查数据状态、可靠度、特征版本、模型版本和校准版本。

## 灯号与动作

| 灯号 | 短期含义 | 入场门 | 新增仓位上限 |
|---|---|---|---:|
| GREEN | 校准概率接近历史基准 | OPEN | 100% |
| YELLOW | 风险抬升，避免追高 | OPEN | 100% |
| ORANGE | 风险显著抬升，仅允许条件单 | CONDITIONAL | 3% |
| RED | 硬触发或极高相对概率 | WAIT | 0% |
| UNKNOWN | 数据或模型不足，无法可靠判断 | WAIT | 0% |

`UNKNOWN` 不是低风险，也不等同于已经确认的 `RED`。它表达“当前无法可靠测量”，因此新增仓位按失败关闭处理，但原因必须写成数据或模型不足，不能编造市场压力。

## 推送规则

- 每个交易日当地时间 08:30 固定生成并推送盘前报告。
- A 股盘中在 `09:35-11:25`、`13:05-14:55` 每五分钟评估。
- 美股盘中从交易所开盘后五分钟评估到收盘前五分钟；夏令时、冬令时和提前收盘由交易所日历决定。
- 盘中 GREEN/YELLOW、风险维持和恢复过程静默落库。
- 首次 ORANGE/RED 以及 ORANGE 升 RED 才推送。
- 幂等键由市场、当地交易日、盘前时点或五分钟桶、灯号、状态变化和模型版本组成。重复进程不能重复推送。

## 美股影子限制

Yahoo 单一来源的美股盘中结果标记为 `SHADOW`。报告可以用于观察，但不会覆盖美股个股的生产硬门控。只有数据源冗余和稳定性达到生产标准后，才能单独评审解除影子限制。

## 手动运行

只查看某个固定时点是否应执行，不访问行情、模型或飞书：

```bash
.venv/bin/python -m tradingagents.harness.market_warning.runner \
  --market a_share --at 2026-08-03T08:30:00+08:00 --dry-run

.venv/bin/python -m tradingagents.harness.market_warning.runner \
  --market us --at 2026-07-06T09:35:00-04:00 --dry-run
```

实际运行时去掉 `--dry-run`。只有在同一评估时点的通知已经失败、需要重试同一个幂等键时才加 `--force`；它不会重发已经成功的提醒。

## 模型训练与晋级

运行时只加载注册表中已激活、校验和一致、特征版本匹配且没有数据穿越的 1 日/3 日模型。没有合格激活模型时输出 `UNKNOWN`，不会在运行时偷偷训练。

训练、评估和晋级命令：

```bash
.venv/bin/python -m tradingagents.harness.market_warning.training train \
  --start 2000-01-01 --test-end 2026-07-31 --version market-warning-v1

.venv/bin/python -m tradingagents.harness.market_warning.training promote \
  --version market-warning-v1
```

只有 Brier、AUPRC、校准误差、危机集中度和月度提醒预算全部通过，模型才允许激活。

## 文件与日志

- 报告：`reports/market_warning/<market>/<YYYY-MM-DD>/<HHMM>-<slot>.md`
- 原始点时缓存：`harness_data/market_warning/raw/`
- 模型：`harness_data/models/market_warning/`
- 日志：`harness_data/logs/market_warning.log`
- 审计表：`market_warning_feature_snapshots`、`market_warning_predictions`、`market_warning_reasoning`、`market_warning_decisions`、`market_warning_alerts`、`market_warning_model_registry`

## 定时任务与回滚

安装脚本只增加一条五分钟唤醒任务，并在安装前打印和确认：

```bash
scripts/install_market_warning_cron.sh
```

回滚时从 `crontab -e` 删除包含 `tradingagents.harness.market_warning.runner` 的一行。代码回滚不会删除历史审计记录；如需停用模型，应在模型注册表中取消激活，而不是删除模型文件或历史预测。
