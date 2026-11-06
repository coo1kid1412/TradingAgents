"""One-node, one-request technical smoke; never starts the analysis graph.

Usage: .venv/bin/python -m tradingagents.technical_smoke --ticker 301171 \
    --date 2026-11-05 --output-dir logs/technical_smoke/a-new-run
"""

import argparse
from collections.abc import Mapping
from datetime import date, datetime, timezone
import json
import re
from pathlib import Path
import sys
from threading import Lock
import time


MAX_TOKENS = 8192
MAX_INPUT_CHARS = 24000
TIMEOUT = 180


def create_llm_client(**kwargs):
    # Lazy imports keep --help and test discovery free of runtime side effects.
    from tradingagents.llm_clients import create_llm_client as factory

    return factory(**kwargs)


def create_compact_market_analyst(llm):
    from tradingagents.agents.analysts.compact_market_analyst import (
        create_compact_market_analyst as factory,
    )

    return factory(llm)


def set_config(config):
    from tradingagents.dataflows.config import set_config as configure

    configure(config)


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _safe_usage(value):
    """Persist token counts only, never arbitrary provider metadata or reasoning."""
    if not isinstance(value, Mapping):
        return {}
    counts = {
        "input_tokens", "output_tokens", "total_tokens", "prompt_tokens",
        "completion_tokens", "reasoning_tokens", "cached_tokens", "audio_tokens",
        "cache_read_input_tokens", "cache_creation_input_tokens", "cache_read",
        "cache_creation", "audio", "text", "reasoning", "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    }
    details = {
        "input_token_details", "output_token_details", "prompt_tokens_details",
        "completion_tokens_details",
    }
    result = {}
    for key, count in value.items():
        if key in counts and type(count) is int and count >= 0:
            result[key] = count
        elif key in details and isinstance(count, Mapping):
            result[key] = _safe_usage(count)
    return result


def _input_chars(value):
    if isinstance(value, str):
        return len(value)
    if hasattr(value, "to_messages"):
        value = value.to_messages()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Smoke input must be text or messages")
    total = 0
    for message in value:
        content = _field(message, "content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += len(json.dumps(content, ensure_ascii=False, allow_nan=False))
        else:
            raise TypeError("Smoke message content must be text or content blocks")
    return total


class OneRequest:
    """Expose only invoke, with a nonrenewable delegate-call budget.

    The delegate must be built with SDK, wall-clock and empty-output retries
    disabled. The legacy metadata key additionally suppresses 1026/1027 retries.
    """

    def __init__(self, delegate):
        self._delegate = delegate
        self._lock = Lock()
        self.request_count = 0
        self.input_chars = None
        self.max_tokens = MAX_TOKENS
        self.usage = {}
        self.finish_reason = None
        self.visible_content = ""

    def _record(self, response):
        metadata = _field(response, "response_metadata", {}) or {}
        body = _field(response, "body", {}) or {}
        for source in (body, metadata, response):
            for key in ("token_usage", "usage", "usage_metadata"):
                self.usage.update(_safe_usage(_field(source, key)))
            reason = _field(source, "finish_reason")
            choices = _field(source, "choices", [])
            if not reason and isinstance(choices, list) and choices:
                reason = _field(choices[0], "finish_reason")
            if isinstance(reason, str) and reason in {
                "stop", "length", "tool_calls", "function_call", "content_filter", "error",
            }:
                self.finish_reason = reason

    def invoke(self, input, config=None):
        # No **kwargs or attribute delegation: callers cannot raise the output
        # ceiling, bind tools, stream, or bypass the guard through another method.
        with self._lock:
            if self.request_count:
                raise RuntimeError("Technical smoke permits only one request")
            self.input_chars = _input_chars(input)
            if self.input_chars > MAX_INPUT_CHARS:
                raise ValueError("Technical smoke input exceeds 24000 characters")
            self.request_count = 1
        bounded_config = {"metadata": {
            "market_warning_disable_compliance_retry": True,
            "market_warning_disable_raw_io_logging": True,
        }}
        try:
            response = self._delegate.invoke(input, config=bounded_config)
        except Exception as error:
            self._record(error)
            raise
        self._record(response)
        content = _field(response, "content")
        if isinstance(content, (str, list)):
            from types import SimpleNamespace
            from tradingagents.llm_clients.base_client import normalize_content

            self.visible_content = normalize_content(SimpleNamespace(content=content)).content
            # Be conservative if an unclosed thinking block follows visible text.
            self.visible_content = re.split(r"(?i)<think\b[^>]*>", self.visible_content, maxsplit=1)[0]
        return response


def run_smoke(ticker, analysis_date, output_dir):
    """Save success/failure statistics in a new directory and return an exit code."""
    output_dir = Path(output_dir).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"Cannot create a fresh output directory ({type(error).__name__}): {output_dir}",
              file=sys.stderr)
        return 1

    started = time.monotonic()
    cache_dir = output_dir / "artifacts"
    result = {
        "status": "failed", "ticker": ticker, "analysis_date": analysis_date,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provider": None, "model": None,
        "request_count": 0, "input_chars": None, "max_input_chars": MAX_INPUT_CHARS,
        "max_tokens": MAX_TOKENS, "timeout": TIMEOUT,
        "max_retries": 0, "wall_clock_max_retries": 0, "empty_output_retry_max_tokens": 0,
        "compliance_retry_disabled": True, "usage": {}, "finish_reason": None,
        "artifact_paths": [], "error": None,
    }
    llm = None
    stage = "arguments"
    try:
        if date.fromisoformat(analysis_date).isoformat() != analysis_date:
            raise ValueError("Expected YYYY-MM-DD")
        if not ticker.strip():
            raise ValueError("Ticker must not be empty")

        stage = "configuration"
        # main loads the existing project .env; do not invent a smoke model.
        from main import _build_config

        config = dict(_build_config())
        result["provider"] = config["llm_provider"]
        result["model"] = config["deep_think_llm"]
        if config["llm_provider"].lower() != "minimax":
            raise ValueError("This bounded smoke is authorized only for MiniMax")
        config["data_cache_dir"] = str(cache_dir)
        config["market_context_max_chars"] = MAX_INPUT_CHARS
        set_config(config)

        stage = "client"
        llm = OneRequest(create_llm_client(
            provider=config["llm_provider"], model=config["deep_think_llm"],
            base_url=config["backend_url"], max_tokens=MAX_TOKENS,
            max_retries=0, wall_clock_max_retries=0, empty_output_retry_max_tokens=0,
            timeout=TIMEOUT,
        ).get_llm_wrapped())
        stage = "node"
        node = create_compact_market_analyst(llm)
        state = {
            "company_of_interest": ticker, "company_name": ticker,
            "trade_date": analysis_date, "capital_flow_yaml": "", "messages": [],
        }
        update = node(state)
        report = update.get("market_report") if isinstance(update, Mapping) else None
        if not isinstance(report, str) or not report.strip() or llm.request_count != 1:
            raise ValueError("Compact node did not produce a nonempty single-request report")
        # SUMMARY/HANDOFF validation belongs to the compact node; never repair
        # its failures by invoking another model or advancing to another agent.
        stage = "report"
        with (output_dir / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(report)
        result["status"] = "success"
    except Exception as error:
        # Provider/parser exception text may contain raw response or reasoning.
        result["error"] = {"type": type(error).__name__, "stage": stage}
        code = getattr(error, "code", None)
        if isinstance(code, str) and code in {
            "incomplete_artifact", "invalid_summary_choice", "missing_judgment", "invalid_asof",
            "conflicting_price", "unknown_fact_reference", "unexpected_tool_calls",
        }:
            result["error"]["code"] = code
        status = getattr(error, "status_code", None)
        if type(status) is int:
            result["error"]["http_status"] = status
    finally:
        if llm is not None:
            result.update(request_count=llm.request_count, input_chars=llm.input_chars,
                          usage=llm.usage, finish_reason=llm.finish_reason)
            result["visible_chars"] = len(llm.visible_content)
            if llm.visible_content:
                with (output_dir / "model_visible.md").open("x", encoding="utf-8") as stream:
                    stream.write(llm.visible_content)
        result["artifact_paths"] = sorted(
            str(path) for path in cache_dir.rglob("*") if path.is_file()
        )
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with (output_dir / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    print(f"Technical smoke {result['status']}: {output_dir / 'result.json'}")
    return 0 if result["status"] == "success" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; existing paths are refused")
    args = parser.parse_args(argv)
    return run_smoke(args.ticker, args.date, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
