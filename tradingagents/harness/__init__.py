"""TradingAgents Harness 模块：报告归档 + 真值采集 + 回测分析。

V1 范围：
- archive: 解析 reports/ 目录下的报告，提取 RM_SUMMARY / PM_SUMMARY YAML
  写入 SQLite DB
- truth: 每天采集 T/T+1/T+5/T+30 真实收盘价
- backtest: 计算命中率切片
- review: 生成系统性偏差洞察清单（人工 review）

V1 不包含：自动改 prompt / look-ahead bias 输入端防御
"""

__version__ = "0.1.0"
