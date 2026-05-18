from abc import ABC, abstractmethod
import logging
import queue
import re
import threading
import time
import warnings
from typing import Any, Optional

from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

# 匹配推理模型（DeepSeek、QwQ 等）的思考链标签
# 支持 <think>...</think> 标准闭合，以及 <think>... 未闭合的情况
_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"^<think>[\s\S]*", re.DOTALL)

def _strip_think_tags(text: str) -> str:
    """移除推理模型输出中的 <think>...</think> 思考链内容。"""
    if "<think>" not in text:
        return text
    # 先处理正常闭合的 <think>...</think>
    text = _THINK_TAG_RE.sub("", text)
    # 再处理未闭合的（整段都是 <think> 开头且无 </think>）
    if text.strip().startswith("<think>"):
        text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def normalize_content(response):
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.

    Also strips <think>...</think> tags from reasoning models (DeepSeek, QwQ, etc.).
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    # 移除推理模型的思考链标签
    if isinstance(response.content, str):
        response.content = _strip_think_tags(response.content)
    return response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    # Default timeout (seconds) for LLM API requests.
    # Prevents indefinite hangs when the API is unresponsive.
    # Can be overridden via kwargs["timeout"] or config["llm_timeout"].
    DEFAULT_TIMEOUT = 180

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def _get_timeout(self) -> int:
        """Return the timeout value from kwargs or the default."""
        return self.kwargs.get("timeout", self.DEFAULT_TIMEOUT)

    def get_provider_name(self) -> str:
        """Return the provider name used in warning messages."""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """Warn when the model is outside the known list for the provider."""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass

    def get_llm_wrapped(self) -> Any:
        """Return the LLM instance wrapped with wall-clock timeout protection.

        背景：langchain → openai SDK → httpx 链路的 timeout 偶尔不透传到底层 socket，
        导致客户端在 TCP ESTABLISHED 但服务端不响应时无限死等（已多次复现卡死 30-40 分钟）。
        本包装在最外层强制 wall-clock 超时，超时后抛 TimeoutError 触发重试或失败。
        """
        return WallClockTimeoutLLM(
            self.get_llm(),
            timeout=self._get_timeout(),
            max_retries=2,
        )


class WallClockTimeoutLLM(Runnable):
    """LLM 包装器：拦截 invoke 用 daemon 线程 + queue 强制壁钟超时。

    继承 `Runnable` 以兼容 langchain LCEL 链式语法（`prompt | wrapped_llm`）。
    Runnable 的其他必需方法（batch/stream/ainvoke/astream）由基类提供默认实现，
    我们只需重写 `invoke`。

    设计与 tushare_vendor._call_with_timeout 一致：
    - 使用 daemon 线程而非 ThreadPoolExecutor（避免 shutdown(wait=True) 阻塞）
    - 超时后线程仍在跑，但因 daemon 标记会在进程退出时被回收
    - 通过 __getattr__ 代理其他属性，保持 callbacks 等 langchain 方法仍可用
    - bind_tools / with_structured_output 显式拦截，返回的也是 wrapped 实例

    超时后行为：
    - 指数退避重试（10s → 20s → ...，最多 max_retries 次）
    - 全部重试失败抛 TimeoutError，让上层 langgraph/调用方决定如何处理
    """

    def __init__(self, llm: Any, timeout: int = 180, max_retries: int = 2,
                 retry_wait_base: int = 10):
        self._llm = llm
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_wait_base = retry_wait_base

    def invoke(self, input: Any = None, config: Optional[RunnableConfig] = None,
               **kwargs: Any) -> Any:
        """带 wall-clock 超时保护的 invoke，超时后指数退避重试。

        Runnable.invoke 签名：invoke(input, config, **kwargs)
        我们透传给原 llm（input 作为第一个位置参数）。
        """
        last_err: Optional[BaseException] = None
        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                wait = self._retry_wait_base * (2 ** (attempt - 1))
                logger.warning(
                    "LLM 调用超时，第 %d/%d 次重试（等待 %ds）",
                    attempt, self._max_retries, wait,
                )
                time.sleep(wait)

            q: queue.Queue = queue.Queue()

            def _worker():
                try:
                    # 透传给原 llm.invoke(input, config, **kwargs)
                    if config is not None:
                        q.put((self._llm.invoke(input, config, **kwargs), None))
                    else:
                        q.put((self._llm.invoke(input, **kwargs), None))
                except BaseException as e:
                    q.put((None, e))

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=self._timeout)

            if t.is_alive():
                # 超时：daemon 线程仍在跑（进程退出时被回收），主流程继续重试
                last_err = TimeoutError(
                    f"LLM 调用 wall-clock 超时（{self._timeout}s，第 {attempt+1} 次尝试）"
                )
                continue

            result, exc = q.get()
            if exc is not None:
                raise exc
            return result

        # 全部重试用尽
        raise last_err if last_err else TimeoutError("LLM 调用超时且无错误信息")

    def bind_tools(self, *args, **kwargs) -> "WallClockTimeoutLLM":
        """拦截 bind_tools，让返回的新 llm 实例也保留 wall-clock timeout 保护。

        原本 langchain `llm.bind_tools([...])` 返回裸的新 ChatModel 实例，
        失去 wall-clock 保护。我们在这里重新包装，保证 tool calling 链路也安全。
        """
        new_bound_llm = self._llm.bind_tools(*args, **kwargs)
        return WallClockTimeoutLLM(
            new_bound_llm,
            timeout=self._timeout,
            max_retries=self._max_retries,
            retry_wait_base=self._retry_wait_base,
        )

    def with_structured_output(self, *args, **kwargs) -> "WallClockTimeoutLLM":
        """同 bind_tools，对 structured_output 也包装。"""
        new_llm = self._llm.with_structured_output(*args, **kwargs)
        return WallClockTimeoutLLM(
            new_llm,
            timeout=self._timeout,
            max_retries=self._max_retries,
            retry_wait_base=self._retry_wait_base,
        )

    def __getattr__(self, name: str) -> Any:
        """代理其他属性到原 llm，让 callbacks 等仍可用。

        注意：bind_tools / with_structured_output 已被显式拦截重新包装，
        其他可能返回新 llm 的方法（如 .bind / .with_config）暂不拦截——
        它们较少使用且不影响核心 tool calling 路径。
        """
        # 避免无限递归：__getattr__ 只在普通属性查找失败时调用
        return getattr(self._llm, name)
