import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 确保代理可用（Google Gemini 等国际 API 需要代理）
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")

# 国内数据源绕过代理（AKShare/Tushare/公告 API 等）
os.environ.setdefault(
    "NO_PROXY",
    "*.eastmoney.com,*.sina.com.cn,*.tushare.pro,*.baidu.com,api.tauric.ai,*.akshare.xyz,*.minimaxi.com"
)

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "google"
config["deep_think_llm"] = "gemini-2.5-pro"
config["quick_think_llm"] = "gemini-2.5-flash"
config["max_debate_rounds"] = 1

# A股代码会自动路由到 akshare → tushare → yfinance
# 以下配置仅影响非A股（美股等）的供应商选择
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# A股测试：淳中科技 603516
_, decision = ta.propagate("603516", "2025-03-21")
print(decision)
