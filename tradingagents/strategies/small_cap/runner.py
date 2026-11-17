"""Archive inputs before publication; keep delivery receipts separate from research."""

from contextlib import contextmanager
from datetime import datetime, time
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import uuid

from .config import Config, RULE_ASSUMPTIONS, digest
from .reporting import (DeliveryError, DeliveryUnknown, recipient_identity,
                        render_markdown, send_feishu)
from .selector import SHANGHAI, parse_date, select_top10

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "strategy_data" / "small_cap"


def code_version():
    package = Path(__file__).parent
    return "small-cap-source-" + digest({path.name: path.read_text(encoding="utf-8")
                                        for path in sorted(package.glob("*.py"))})


class RunBusy(RuntimeError):
    """Another process already owns this strategy's runtime directory."""


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunBusy("small-cap run already active") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_new(path, value):
    """Atomic no-overwrite publication, including the crash-before-rename case."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                                                             indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _db(root):
    connection = sqlite3.connect(Path(root) / "state.sqlite3", timeout=5)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, data_date TEXT NOT NULL, config_hash TEXT NOT NULL,
            snapshot_path TEXT, snapshot_hash TEXT, report_dir TEXT NOT NULL,
            result_hash TEXT NOT NULL, markdown_hash TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS deliveries (
            run_id TEXT NOT NULL, recipient TEXT NOT NULL, uuid TEXT NOT NULL,
            status TEXT NOT NULL, message_id TEXT, error_code TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, recipient));
    """)
    connection.commit()
    return connection


def _publish_bundle(folder, result, markdown, staging_root):
    folder, staging_root = Path(folder), Path(staging_root)
    if folder.exists():
        raise FileExistsError("report destination already exists")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="report-", dir=staging_root))
    try:
        _write_new(staging / "result.json", result)
        _write_new(staging / "report.md", markdown)
        folder.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, folder)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_archive(row):
    folder = Path(row["report_dir"])
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    markdown = (folder / "report.md").read_text(encoding="utf-8")
    if digest(result) != row["result_hash"] or digest(markdown) != row["markdown_hash"]:
        raise ValueError("archived report integrity check failed")
    if row["snapshot_path"]:
        snapshot = json.loads(Path(row["snapshot_path"]).read_text(encoding="utf-8"))
        if digest(snapshot) != row["snapshot_hash"]:
            raise ValueError("archived snapshot integrity check failed")
    return result, markdown


def _changes(connection, result):
    prior = connection.execute("SELECT * FROM runs WHERE data_date < ? AND config_hash = ? "
                               "ORDER BY data_date DESC, created_at DESC",
                               (result["data_date"], result["config_hash"])).fetchall()
    for row in prior:
        previous, _ = _load_archive(row)
        if previous.get("ranking_complete") is not True:
            continue
        old = {item["ts_code"]: item["rank"] for item in previous["top10"]}
        new = {item["ts_code"]: item["rank"] for item in result["top10"]}
        return {"previous_data_date": previous["data_date"],
                "added": sorted(new.keys() - old.keys()), "removed": sorted(old.keys() - new.keys()),
                "rank_changes": [{"ts_code": code, "from": old[code], "to": new[code]}
                                 for code in sorted(new.keys() & old.keys()) if old[code] != new[code]]}
    return {}


def _deliver(connection, result, markdown, sender, recipient, now):
    try:
        recipient = recipient or recipient_identity()
    except DeliveryError as error:
        return dict(result, delivery_status="failed", delivery_error=error.safe_code)
    # This logical identity survives process/run IDs and identical refreshes.
    identity = digest({"rule_version": result["rule_version"], "data_date": result["data_date"],
                       "snapshot_id": result.get("snapshot_id") or digest(result.get("warnings", [])),
                       "recipient": recipient})
    prior = connection.execute("SELECT * FROM deliveries WHERE run_id = ? AND recipient = ?",
                               (identity, recipient)).fetchone()
    if prior and prior["status"] in {"sent", "unknown", "sending"}:
        status = "unknown" if prior["status"] == "sending" else prior["status"]
        if status != prior["status"]:
            connection.execute("UPDATE deliveries SET status='unknown' WHERE run_id=? AND recipient=?",
                               (identity, recipient))
            connection.commit()
        return dict(result, delivery_status=status, message_id=prior["message_id"])
    target = result.get("target_trade_date")
    if target and now >= datetime.combine(parse_date(target), time.min, SHANGHAI):
        return dict(result, delivery_status="expired", delivery_error="TARGET_CUTOFF_PASSED")
    delivery_uuid = identity[:32]
    connection.execute("INSERT INTO deliveries VALUES (?,?,?,?,?,?,?) "
                       "ON CONFLICT(run_id,recipient) DO UPDATE SET status='sending',error_code=NULL,updated_at=excluded.updated_at",
                       (identity, recipient, delivery_uuid, "sending", None, None, now.isoformat()))
    connection.commit()
    message_id, error_code = None, None
    try:
        receipt = sender(result, markdown, delivery_uuid)
        if receipt.get("status") != "sent" or receipt.get("recipient_hash") != recipient:
            raise DeliveryUnknown("INVALID_DELIVERY_RECEIPT")
        status, message_id = "sent", receipt.get("message_id")
    except DeliveryUnknown as error:
        status, error_code = "unknown", error.safe_code
    except DeliveryError as error:
        status, error_code = "failed", error.safe_code
    except Exception:
        status, error_code = "unknown", "UNEXPECTED_DELIVERY_FAILURE"
    connection.execute("UPDATE deliveries SET status=?,message_id=?,error_code=?,updated_at=? "
                       "WHERE run_id=? AND recipient=?", (status, message_id, error_code, now.isoformat(),
                                                          identity, recipient))
    connection.commit()
    return dict(result, delivery_status=status, message_id=message_id, delivery_error=error_code)


def run_daily(root=DEFAULT_ROOT, source=None, sender=send_feishu, recipient=None,
              data_date=None, notify=False, refresh=False, config=Config(), clock=None):
    now = (clock or (lambda: datetime.now(SHANGHAI)))()
    if now.tzinfo is None:
        raise ValueError("clock must include timezone")
    now = now.astimezone(SHANGHAI)
    requested_date = parse_date(data_date).isoformat() if data_date else now.date().isoformat()
    root = Path(root).resolve()
    with run_lock(root):
        connection = _db(root)
        try:
            cached = connection.execute("SELECT * FROM runs WHERE data_date=? AND config_hash=? "
                                        "ORDER BY created_at DESC LIMIT 1",
                                        (requested_date, config.config_hash)).fetchone()
            if cached and not refresh:
                result, markdown = _load_archive(cached)
                attempts = connection.execute("SELECT COUNT(*) FROM runs WHERE data_date=? AND config_hash=?",
                                              (requested_date, config.config_hash)).fetchone()[0]
                if result["status"] in {"ERROR", "BLOCKED"} and attempts < 3:
                    cached = None
            if not cached or refresh:
                if source is None:
                    from .data_source import TushareSource
                    source = TushareSource()
                try:
                    snapshot = source.collect(data_date=requested_date)
                except Exception as error:
                    snapshot = {"status": "ERROR", "data_date": requested_date,
                                "warnings": [getattr(error, "safe_code", "DATA_COLLECTION_FAILED")]}
                if snapshot.get("status") == "CLOSED":
                    return {"status": "CLOSED", "data_date": requested_date,
                            "top10": [], "delivery_status": "not_sent"}
                run_id = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
                folder = root / "reports" / requested_date / run_id
                snapshot_path, snapshot_hash = None, None
                if snapshot.get("status") == "ERROR":
                    result = {"status": "ERROR", "ranking_complete": False, "top10": [],
                              "warnings": snapshot.get("warnings") or ["DATA_COLLECTION_FAILED"], "config_hash": config.config_hash,
                              "rule_version": config.rule_version, "data_date": requested_date,
                              "as_of_at": now.isoformat(), "target_trade_date": None,
                              "regime": "UNKNOWN", "new_position_policy": "BLOCK",
                              "strategy_mode": "DATA_UNAVAILABLE", "delivery_status": "not_sent"}
                    result.update(mode="forward", rule_assumptions=dict(RULE_ASSUMPTIONS))
                else:
                    snapshot_hash = digest(snapshot)
                    snapshot_path = root / "snapshots" / requested_date / (snapshot_hash + ".json")
                    if snapshot_path.exists():
                        if digest(json.loads(snapshot_path.read_text(encoding="utf-8"))) != snapshot_hash:
                            raise ValueError("snapshot integrity check failed")
                    else:
                        _write_new(snapshot_path, snapshot)
                    result = select_top10(snapshot, config)
                    if result["data_date"] != requested_date:
                        raise ValueError("provider returned another data date")
                    result["snapshot_id"] = snapshot_hash
                    if result["ranking_complete"]:
                        result["changes"] = _changes(connection, result)
                result.update(run_id=run_id, report_dir=str(folder), generated_at=datetime.now(SHANGHAI).isoformat(),
                              code_version=code_version(), exclusions_ref="result.json#/exclusions",
                              snapshot_path=str(snapshot_path) if snapshot_path else None)
                markdown = render_markdown(result)
                _publish_bundle(folder, result, markdown, root / ".staging")
                connection.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                                   (run_id, requested_date, config.config_hash,
                                    str(snapshot_path) if snapshot_path else None, snapshot_hash,
                                    str(folder), digest(result), digest(markdown), now.isoformat()))
                connection.commit()
            send_time = (clock or (lambda: datetime.now(SHANGHAI)))().astimezone(SHANGHAI) if notify else now
            return _deliver(connection, result, markdown, sender, recipient, send_time) if notify else result
        finally:
            connection.close()


def replay_snapshot(snapshot_path, as_of=None, output_dir=None, config=Config()):
    path = Path(snapshot_path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{64}", path.stem) or digest(snapshot) != path.stem:
        raise ValueError("replay requires the original content-addressed snapshot")
    # A replay is exactly the captured decision, not an invented historical view.
    if as_of is not None and as_of != snapshot.get("as_of_at"):
        raise ValueError("replay must retain the original as-of cutoff")
    result = select_top10(snapshot, config)
    result["snapshot_id"] = digest(snapshot)
    if output_dir:
        output = Path(output_dir).resolve()
        _publish_bundle(output, result, render_markdown(result), output.parent / ".small-cap-staging")
    return result
