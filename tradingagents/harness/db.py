"""SQLite 连接 + schema 初始化。"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# 默认 DB 位置（项目根目录下 harness_data/）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _PROJECT_ROOT / "harness_data" / "tradingagents.db"
_SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def get_db_path(custom_path: Path | str | None = None) -> Path:
    """返回 DB 文件路径。默认在项目根目录下 harness_data/tradingagents.db。"""
    if custom_path is not None:
        return Path(custom_path)
    return _DEFAULT_DB_PATH


def init_db(db_path: Path | str | None = None) -> Path:
    """初始化 DB：建目录、建表（IF NOT EXISTS 幂等）、跑必要的 migration。"""
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = _SCHEMA_FILE.read_text(encoding="utf-8")
    with sqlite3.connect(path) as conn:
        conn.executescript(schema_sql)
        _migrate_schema(conn)
        conn.commit()
    logger.info("Harness DB initialized at %s", path)
    return path


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """轻量 migration：对已存在 DB 添加新列（SQLite 不支持 IF NOT EXISTS for ALTER）。

    每次 init_db 跑一次（幂等：列已存在则跳过）。
    """
    # outcomes 表加 benchmark / relative_return 列
    cols = {r[1] for r in conn.execute("PRAGMA table_info(outcomes)").fetchall()}
    if "benchmark_ticker" not in cols:
        conn.execute("ALTER TABLE outcomes ADD COLUMN benchmark_ticker TEXT")
        logger.info("Migrated: outcomes.benchmark_ticker added")
    if "benchmark_return_pct" not in cols:
        conn.execute("ALTER TABLE outcomes ADD COLUMN benchmark_return_pct REAL")
        logger.info("Migrated: outcomes.benchmark_return_pct added")
    if "relative_return_pct" not in cols:
        conn.execute("ALTER TABLE outcomes ADD COLUMN relative_return_pct REAL")
        logger.info("Migrated: outcomes.relative_return_pct added")

    # predictions 表加评级链审计列（2026-06 P0：回测分腿归因）
    pred_cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    for col, typ in [
        ("valuation_regime", "TEXT"),
        ("regime_legs", "TEXT"),
        ("rating_raw", "TEXT"),
        ("peg_confidence", "TEXT"),
        ("overlay_style_adj", "INTEGER"),
        ("overlay_vote_adj", "INTEGER"),
        ("overlay_catalyst_adj", "INTEGER"),
    ]:
        if col not in pred_cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
            logger.info("Migrated: predictions.%s added", col)


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """获取 DB 连接，自动 commit + close。

    每次连接前都跑 init_db（建表 IF NOT EXISTS + 幂等 migration）：
    原来只在文件不存在时初始化，导致已有库永远吃不到新列 migration
    （实测：predictions 扩列后归档报 no column named valuation_regime）。
    本地 SQLite，executescript 开销可忽略。
    """
    path = get_db_path(db_path)
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # 启用外键约束
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_exists(report_dir: str, db_path: Path | str | None = None) -> bool:
    """检查某报告目录是否已归档。"""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM runs WHERE report_dir = ?", (report_dir,)
        ).fetchone()
        return row is not None
