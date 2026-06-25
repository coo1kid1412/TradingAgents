-- TradingAgents Harness SQLite Schema
-- 设计要点：所有 DDL 用 IF NOT EXISTS，可重复执行；时间用 ISO-8601 字符串

-- ============================================================================
-- runs: 每次报告生成的元数据 + 时间窗分类
-- ============================================================================
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    company_name TEXT,
    trade_date DATE NOT NULL,
    report_timestamp TEXT NOT NULL,           -- ISO-8601: "2026-05-20T22:49:01"
    report_window TEXT NOT NULL,              -- pre_market / morning / afternoon / post_market
    report_dir TEXT NOT NULL UNIQUE,          -- 报告目录路径，唯一约束防止重复归档
    git_commit TEXT,
    code_version_tag TEXT,                    -- 可选：用户标签
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archive_status TEXT NOT NULL DEFAULT 'archived'  -- archived / partial / failed
);

CREATE INDEX IF NOT EXISTS idx_runs_ticker_date ON runs(ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_runs_window ON runs(report_window);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(report_timestamp);


-- ============================================================================
-- predictions: 从 RM_SUMMARY + PM_SUMMARY YAML 提取的结构化预测
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    run_id INTEGER PRIMARY KEY,

    -- 共用字段（current_price 允许 NULL 以容忍历史报告无 RUN_SUMMARY 的情况）
    current_price REAL,
    style TEXT,
    theme_stage TEXT,
    composite_score REAL,
    momentum_score REAL,

    -- RM 字段
    rm_rating TEXT,
    rm_conviction TEXT,
    target_price_low REAL,
    target_price_mid REAL,
    target_price_high REAL,
    bull_target REAL,
    bull_prob REAL,
    base_target REAL,
    base_prob REAL,
    bear_target REAL,
    bear_prob REAL,
    base_case_expected_return_pct REAL,
    deviation_pct REAL,
    threshold_dn_pct REAL,
    threshold_up_pct REAL,
    -- 评级链审计字段（2026-06 P0：回测分腿归因——regime 闸门/六腿/趋势叠加三路是否有效，数据说话）
    valuation_regime TEXT,
    regime_legs TEXT,
    rating_raw TEXT,
    peg_confidence TEXT,
    overlay_style_adj INTEGER,
    overlay_vote_adj INTEGER,
    overlay_catalyst_adj INTEGER,

    -- PM 字段
    pm_rating TEXT,
    pm_conviction_stars INTEGER,
    pm_invest_judgment TEXT,
    pm_entry_judgment TEXT,
    pm_action_keyword TEXT,
    pm_size_low_pct REAL,
    pm_size_high_pct REAL,
    pm_entry_low REAL,
    pm_entry_high REAL,
    pm_tp1 REAL,
    pm_tp2 REAL,
    pm_tp3 REAL,
    pm_sl_soft REAL,
    pm_sl_hard REAL,
    pm_horizon_months_low INTEGER,
    pm_horizon_months_high INTEGER,
    pm_rating_adjusted_from_rm INTEGER,       -- 0/1 (SQLite 没原生 bool)
    market_risk_level TEXT,
    market_entry_gate TEXT,
    market_position_cap_pct REAL,
    short_term_trend TEXT,
    short_term_confidence TEXT,
    theme_outlook_12m TEXT,

    -- 元
    rm_yaml_parsed INTEGER DEFAULT 0,         -- 0/1: RM_SUMMARY 是否成功解析
    pm_yaml_parsed INTEGER DEFAULT 0,
    parse_warnings TEXT,                      -- 解析时的告警信息（如缺字段）

    FOREIGN KEY (run_id) REFERENCES runs(id)
);


-- ============================================================================
-- outcomes: 真值（每 run 4 行：T / T+1 / T+5 / T+30）
-- ============================================================================
CREATE TABLE IF NOT EXISTS outcomes (
    run_id INTEGER NOT NULL,
    horizon TEXT NOT NULL,                    -- T / T+1 / T+5 / T+30
    target_date DATE,                         -- 该 horizon 对应的真实交易日
    reference_price REAL NOT NULL,            -- 锚定价（predictions.current_price）
    actual_close_at_horizon REAL,             -- pending 时为 NULL
    actual_high_during_horizon REAL,
    actual_low_during_horizon REAL,
    realized_return_pct REAL,                 -- (actual_close - ref) / ref * 100
    signed_pnl_pct REAL,                       -- 按方向取符号的策略盈亏：long/neutral=+r, short=-r

    -- 命中判定
    direction_predicted TEXT,                 -- long / short / neutral
    direction_hit INTEGER,                    -- 0/1/NULL
    tp1_hit INTEGER,                          -- TP1 是否被触达
    sl_hard_hit INTEGER,                      -- 硬止损是否触达
    fetch_status TEXT NOT NULL DEFAULT 'pending',  -- pending / fetched / failed / not_due
    fetched_at TIMESTAMP,
    error_message TEXT,

    PRIMARY KEY (run_id, horizon),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_status ON outcomes(fetch_status);
CREATE INDEX IF NOT EXISTS idx_outcomes_target_date ON outcomes(target_date);


-- ============================================================================
-- backtest_metrics: 聚合统计快照（按需重算，不实时维护）
-- ============================================================================
CREATE TABLE IF NOT EXISTS backtest_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date DATE NOT NULL,
    horizon TEXT NOT NULL,
    group_dimension TEXT NOT NULL,            -- rating / style / theme_stage / conviction / window
    group_value TEXT NOT NULL,
    total_runs INTEGER NOT NULL,
    direction_hits INTEGER NOT NULL,
    direction_hit_rate REAL NOT NULL,
    avg_return_correct REAL,
    avg_return_wrong REAL,
    expectation REAL,                         -- hit_rate * avg_correct + (1-hit_rate) * avg_wrong

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_snapshot ON backtest_metrics(snapshot_date, horizon);


-- ============================================================================
-- price_cache: 本地价格缓存表，避免重复调 tushare/akshare 拉同一段日期数据
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_cache (
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_price_cache_ticker ON price_cache(ticker);
CREATE INDEX IF NOT EXISTS idx_price_cache_date ON price_cache(trade_date);

-- ============================================================================
-- market_risk_snapshots: 开盘前市场风险快照（A 股 / 美股）
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_risk_snapshots (
    market TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    as_of_time TEXT NOT NULL,
    snapshot_version TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    risk_score REAL,
    t_plus_1_bias TEXT NOT NULL,
    entry_gate TEXT NOT NULL,
    position_cap_pct REAL NOT NULL,
    data_status TEXT NOT NULL,
    factor_breakdown_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_market_risk_snapshot_date
    ON market_risk_snapshots(as_of_date);
