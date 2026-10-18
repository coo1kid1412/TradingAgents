# TLTW 分析失败修复

## 事故定位

- 2026-10-18 09:25:32，OpenClaw 的 TA skill 成功启动项目 `main.py`，模型为 MiniMax-M3。
- 09:39:44，RM 调用 `compute_ic_recommendation`，将 `thesis_breaker_evidence_ids` 传为 `{}`，Pydantic 的可选字符串列表契约拒绝该输入。
- 后续恢复未获得成功的权威工具结果；09:41:17 分析按既有安全规则中止，未进入最终报告交付。
- 日志：`/tmp/ta_TLTW.log`、`logs/run_20261018_092532.log`。
- 此次明确的致命错误是工具参数格式，不是 skill 未启动，也不是已确认的网络、额度或模型审核拒绝。

## 修复边界

1. IC 工具的六个可选证据列表字段，在 Pydantic 输入边界仅把空字典转为新空列表。这是无信息容器的格式兼容，不生成证据，也不影响正常输入的评级计算。
2. 对外工具 JSON schema 仍为字符串数组或 null。非空字典、标量、嵌套对象仍拒绝，证据 ID 合格性验证与硬否决门槛保持不变。
3. 工具抛错、返回 `error` 或名称不存在时，ToolMessage 标为 error。
4. 缺少强制工具成功结果时，恢复提示明确要求修正参数并发起真实 `tool_calls`，不再把该故障泛称为思考正文丢失。
5. 达到工具轮数软上限但尚缺强制工具时，不下达相互矛盾的“禁止任何工具”指令；原有硬上限、续写预算及失败关闭规则保留。
6. 不修改全局模型配置、推理内容清理、ETF 估值策略、上游分析师数据口径或 OpenClaw skill，不提交用户 `main.py` 的 TLTW 配置改动。

## 验证

已先复现失败，再完成离线回归：

```bash
.venv/bin/python -m tradingagents.agents.managers.test_ic_recommendation
.venv/bin/python -m tradingagents.agents.managers.test_entry_timing
.venv/bin/python -m tradingagents.agents.managers.test_decision_attribution_prompts
.venv/bin/python -m tradingagents.agents.managers.test_market_risk_gate
```

- IC 契约 19 项、经理流程及输出契约 82 项、归因提示契约 8 项、市场风险门 4 项，共 113 项通过。
- 包括空对象与空列表结果等价、六个字段兼容、非空错误拒绝、hard_veto 无证据仍失败、工具 schema 不放宽、软上限后修正工具仍能完成报告、权威工具始终缺失仍失败关闭。
- 完整 TLTW 回归尚未执行：自动安全检查要求用户明确授权将该次分析上下文发送给 MiniMax 并允许原有飞书通知流程。不得把离线通过等同于真实报告已生成。
