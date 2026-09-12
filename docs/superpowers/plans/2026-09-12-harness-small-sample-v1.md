# Harness 少样本 V1 实施与验收

**Goal:** 在不增加 LLM 费用、不改变策略判断的情况下，把旧命中率周报改为可追溯的质量复盘。

**Architecture:** 原日更继续采集；新质量模块只读归档数据，生成规则问题、案例和证据差异；原周报入口复用原飞书发送器。

**Tech Stack:** 项目 `.venv`、Python 标准库、SQLite、现有飞书 shell 适配。

**Spec:** `docs/superpowers/specs/2026-09-12-harness-small-sample-v1-design.md`。

## 已落地

- [x] `quality_review.py`：区分报告/股票日期/跨期限结果行；近七天按上海日期统计。
- [x] 对参考价、摘要、WAIT 仓位、跟踪缺失、到期不明、采集失败和旧评价口径进行规则审计。
- [x] 形成关联报告 ID、案例路径、修复建议与回归验收条件。
- [x] `weekly_review.py`：替换默认周报内容；原 cron 命令仍适用，无新增调度。
- [x] 证据差异持久化：同证据不提示重复深度复盘，预览/发送失败不推进通知基线。
- [x] `review.py`：旧统计保留手工诊断，但明确不是真实交易收益、不构成显著性或调参依据。
- [x] 新增 24 项标准库测试，并验证原 5 项周报及 4 项旧统计回归。

## 验证命令

```bash
.venv/bin/python -m unittest tradingagents.harness.test_quality_review -q
.venv/bin/python -m tradingagents.harness.test_weekly_review
.venv/bin/python -m tradingagents.harness.test_backtest_metrics
.venv/bin/python -m tradingagents.harness.weekly_review --no-notify --output-dir /tmp/ta-harness-quality-v1-preview --date 2026-09-12
```

当前 `.venv` 无 pytest，本次没有安装新依赖，采用上述聚焦回归。测试中的飞书发送成功/失败均使用替身，不是真实推送。

## 真实数据预览

2026-09-12 只读预览：近七天新增 2 份归档报告、8 条最近采集记录；累计 586 条归档记录，其中历史失败 290 条；已有结果覆盖 291 份报告、215 个股票日期组合，共 1118 条跨期限结果。以上均不是独立交易样本数。

发现 46 条待跟踪结果缺少到期日、20 条旧采集失败。V1 将它们暴露为需核查，而非把它们显示为“全部正常”。不会据此声称已经修复这些数据源或完成到期计算改造。

## 保持不变及后续

- 主分析 `main.py`、报告评级、超参数、已有数据库与原始报告保持不变。
- 无付费模型调用，无行情接口抓取，无真实股票重跑，无新增推送测试。
- 生成时完整配置档案、3 日/一年专用标签、正确市场基准/成交模拟、agent 交接归因、受预算约束的 LLM 提案仍待后续分批落地。
- 下一优先事项：修复到期日和数据状态，再逐步补充生成时配置与证据留档；不为了样本不足而批量跑研报。
