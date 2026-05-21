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
    """初始化 DB：建目录、建表（IF NOT EXISTS 幂等）。返回 DB 路径。"""
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = _SCHEMA_FILE.read_text(encoding="utf-8")
    with sqlite3.connect(path) as conn:
        conn.executescript(schema_sql)
        conn.commit()
    logger.info("Harness DB initialized at %s", path)
    return path


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """获取 DB 连接，自动 commit + close。"""
    path = get_db_path(db_path)
    if not path.exists():
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
