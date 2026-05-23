from abc import ABC, abstractmethod
import logging
import queue
import re
import signal
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

def _can_use_sigalrm() -> bool:
    """判断当前进程能否用 SIGALRM 做 wall-clock 超时。

    限制：
    - Windows 没有 SIGALRM（hasattr 检查）
    - signal handler 只能由主线程注册（current_thread is main_thread 检查）
    """
    if not hasattr(signal, "SIGALRM"):
        return False
    return threading.current_thread() is threading.main_thread()


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

        保护机制（按强度从强到弱，自动选择）：
        1. SIGALRM 信号（Unix only，主线程，OS 层强制中断，不受 GIL 阻塞）
        2. daemon thread + join（跨平台兜底，但 GIL 被 C 扩展持有时 join 可能延迟）

        历史 bug（PID 28850，5/23-24）：daemon thread + join(timeout=300) 实际跑了 6 小时
        才超时，因为 httpx socket read 在 C 层阻塞，GIL 一直未释放，主线程 join 内部的
        Condition.wait() 没被及时唤醒。SIGALRM 由 OS 调度，在 Python 解释器下一次 bytecode
        循环时立即触发 handler 抛 TimeoutError，可靠得多。
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

            try:
                if _can_use_sigalrm():
                    return self._invoke_with_signal(input, config, **kwargs)
                else:
                    return self._invoke_with_thread_join(input, config, **kwargs)
            except TimeoutError as e:
                last_err = e
                continue

        # 全部重试用尽
        raise last_err if last_err else TimeoutError("LLM 调用超时且无错误信息")

    def _invoke_with_signal(self, input: Any, config: Optional[RunnableConfig],
                            **kwargs: Any) -> Any:
        """用 SIGALRM 实现 wall-clock 超时（主线程 + Unix）。

        SIGALRM 由 OS 调度，在 Python 解释器下一次 bytecode 循环就会处理 pending
        signal——即使 httpx socket read 阻塞，read() 系统调用收到信号后返回 EINTR，
        Python 转抛 TimeoutError，主线程立即从 LLM invoke 栈逃出。
        """
        def _alarm_handler(signum, frame):
            raise TimeoutError(
                f"LLM 调用 wall-clock 超时（{self._timeout}s SIGALRM）"
            )

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(self._timeout)
        try:
            if config is not None:
                return self._llm.invoke(input, config, **kwargs)
            return self._llm.invoke(input, **kwargs)
        finally:
            signal.alarm(0)  # 取消 alarm（无论成功失败）
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, TypeError):
                pass  # 老 handler 可能已失效

    def _invoke_with_thread_join(self, input: Any, config: Optional[RunnableConfig],
                                  **kwargs: Any) -> Any:
        """daemon thread + queue 实现 wall-clock 超时（跨平台兜底，但弱保护）。

        ⚠ 已知问题：httpx C 扩展阻塞时 GIL 持有，join(timeout) 内部 Condition.wait()
        可能延迟唤醒。仅作为 SIGALRM 不可用时的退化方案（如非主线程调用）。
        """
        q: queue.Queue = queue.Queue()

        def _worker():
            try:
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
            raise TimeoutError(
                f"LLM 调用 wall-clock 超时（{self._timeout}s thread-join，弱保护）"
            )

        result, exc = q.get()
        if exc is not None:
            raise exc
        return result

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
