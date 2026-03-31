from abc import ABC, abstractmethod
from typing import Any, Optional
import re

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

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass
