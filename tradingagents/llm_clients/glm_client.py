"""智谱 GLM (BigModel) LLM client for TradingAgents.

GLM 提供 OpenAI 兼容接口（https://open.bigmodel.cn/api/paas/v4/），
故直接复用 langchain-openai 的 ChatOpenAI（同 MiniMax 客户端套路）。
文档：https://docs.bigmodel.cn/cn/api/introduction
"""

import os
from typing import Any

from .base_client import BaseLLMClient
from .validators import validate_model
from .openai_client import NormalizedChatOpenAI

_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# .env 里配这个（也兼容智谱官方惯用的 ZHIPUAI_API_KEY）
_GLM_API_KEY_ENVS = ("GLM_API_KEY", "ZHIPUAI_API_KEY")

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "callbacks",
    "http_client", "http_async_client", "temperature",
)


class GLMClient(BaseLLMClient):
    """智谱 GLM provider 客户端（OpenAI 兼容端点）。

    从环境变量 GLM_API_KEY（或 ZHIPUAI_API_KEY）读 key，默认 max_tokens=8192
    （投研报告较长，避免上游默认值截断）。
    """

    def get_llm(self) -> Any:
        llm_kwargs = {
            "model": self.model,
            "base_url": self.base_url or _GLM_BASE_URL,
            "max_tokens": self.kwargs.get("max_tokens", 8192),
            "timeout": self._get_timeout(),
        }

        if "api_key" not in self.kwargs:
            for env in _GLM_API_KEY_ENVS:
                api_key = os.environ.get(env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
                    break

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and key not in llm_kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("glm", self.model)
