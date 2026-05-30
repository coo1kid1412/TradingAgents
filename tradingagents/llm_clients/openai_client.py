import os
import time
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

# Per-process counter for LLM call logging
_llm_call_seq = 0

# Patch appended to input when an output-sensitive rejection (e.g. MiniMax 1027)
# triggers a retry. The phrasing rules in agents/utils/agent_utils.py already
# cover the source-side prevention; this is the second line of defense.
_COMPLIANCE_RETRY_PATCH = """

【追加合规约束（上次输出被审核拦截，重试触发）】
你的上一次输出被供应商输出层合规审核拦截，请重新生成并严格遵守：
- 禁止使用情绪化/煽动性词汇："暴雷/爆雷/崩盘/崩塌/血洗/腰斩/砸盘/做空获利/时间炸弹/引爆/踩雷"
- 涉及下行风险时改用中性表述："显著下行/深度回调/估值大幅压缩/下行幅度 N%"
- 涉及损失估算时改用："潜在下行幅度/下行风险敞口"，禁用"亏多少/巨亏/血亏"
- 保留所有定量分析与数字，仅替换措辞
"""


def _is_output_sensitive_error(err: BaseException) -> bool:
    """Detect MiniMax-style output content moderation rejection (422 + 1027)."""
    msg = str(err)
    return ("new_sensitive" in msg) or ("1027" in msg and "422" in msg)


def _patch_input_for_compliance(input):
    """Append a compliance reminder to the LLM input for retry."""
    if isinstance(input, str):
        return input + _COMPLIANCE_RETRY_PATCH
    if isinstance(input, list):
        from langchain_core.messages import HumanMessage
        return list(input) + [HumanMessage(content=_COMPLIANCE_RETRY_PATCH)]
    if hasattr(input, "content"):
        from langchain_core.messages import HumanMessage
        return [input, HumanMessage(content=_COMPLIANCE_RETRY_PATCH)]
    return input


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        global _llm_call_seq
        _llm_call_seq += 1
        seq = _llm_call_seq

        # Log input before calling
        try:
            _log_llm_input(seq, self.model, input)
            import sys
            sys.stderr.write(f"[LLM #{seq}] Input logged to llm_calls/\n")
            sys.stderr.flush()
        except Exception as e:
            import sys
            sys.stderr.write(f"[LLM #{seq}] Log input failed: {e}\n")
            sys.stderr.flush()

        start = time.time()
        try:
            result = super().invoke(input, config, **kwargs)
        except Exception as e:
            if not _is_output_sensitive_error(e):
                raise
            import sys
            sys.stderr.write(
                f"[LLM #{seq}] Output sensitivity rejected ({type(e).__name__}); "
                f"retrying once with compliance patch...\n"
            )
            sys.stderr.flush()
            patched_input = _patch_input_for_compliance(input)
            try:
                result = super().invoke(patched_input, config, **kwargs)
                sys.stderr.write(f"[LLM #{seq}] Compliance-patch retry succeeded.\n")
                sys.stderr.flush()
            except Exception as retry_err:
                sys.stderr.write(
                    f"[LLM #{seq}] Compliance-patch retry also failed: {retry_err}\n"
                )
                sys.stderr.flush()
                raise
        elapsed = time.time() - start

        # Log output after returning
        try:
            _log_llm_output(seq, self.model, result, elapsed)
        except Exception as e:
            import sys
            sys.stderr.write(f"[LLM #{seq}] Log output failed: {e}\n")
            sys.stderr.flush()

        # 记录到 profiling collector（按 agent 分组耗时）
        try:
            from tradingagents.profiling import record_llm
            prompt_text = input if isinstance(input, str) else str(input)[:3000]
            record_llm(prompt_text, self.model, elapsed)
        except Exception:
            pass

        return normalize_content(result)


def _log_llm_input(seq: int, model: str, input):
    """Save LLM input to a numbered log file for debugging hangs."""
    if isinstance(input, str):
        content = input
    elif hasattr(input, "content"):
        content = input.content
    elif isinstance(input, dict):
        content = str(input.get("content", input))
    else:
        content = str(input)[:5000]

    if len(content) > 20000:
        content = content[:20000] + "\n... [truncated]"

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "llm_calls")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"llm_call_{seq:04d}.txt")

    with open(log_file, "w") as f:
        f.write(f"=== LLM Call #{seq} ===\n")
        f.write(f"Model: {model}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(content)


def _log_llm_output(seq: int, model: str, result, elapsed: float):
    """Append output summary to the log file."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "llm_calls")
    log_file = os.path.join(log_dir, f"llm_call_{seq:04d}.txt")

    content = getattr(result, "content", str(result)[:500])

    with open(log_file, "a") as f:
        f.write("\n\n" + "=" * 60 + "\n")
        f.write(f"Duration: {elapsed:.1f}s\n")
        f.write(f"Response length: {len(content)} chars\n")
        f.write(f"Response preview: {content[:500]}...\n")


# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "temperature",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model, "timeout": self._get_timeout()}

        # Provider-specific base URL and auth
        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs (timeout already set, skip it)
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and key not in llm_kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
