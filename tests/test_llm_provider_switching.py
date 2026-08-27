from __future__ import annotations

import os
import unittest
from unittest import mock

import main
from tradingagents.llm_clients.minimax_client import MiniMaxClient
from tradingagents.llm_clients.provider_config import resolve_llm_provider_settings


class LLMProviderSwitchingTests(unittest.TestCase):
    def test_minimax_is_the_safe_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_PROVIDER", None)
            config = main._build_config()

        self.assertEqual(config["llm_provider"], "minimax")
        self.assertEqual(config["deep_think_llm"], "MiniMax-M3")
        self.assertEqual(config["quick_think_llm"], "MiniMax-M3")

    def test_one_provider_setting_switches_all_stock_analysis_models(self) -> None:
        provider_environment = {
            "MINIMAX_MODEL": "MiniMax-M3",
            "MINIMAX_DEEP_MODEL": "MiniMax-M3",
            "MINIMAX_QUICK_MODEL": "MiniMax-M3",
            "DEEPSEEK_DEEP_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_QUICK_MODEL": "deepseek-v4-flash",
        }
        with mock.patch.dict(os.environ, provider_environment, clear=False):
            os.environ["LLM_PROVIDER"] = "minimax"
            minimax = main._build_config()
            os.environ["LLM_PROVIDER"] = "deepseek"
            deepseek = main._build_config()

        self.assertEqual(
            (minimax["llm_provider"], minimax["deep_think_llm"], minimax["quick_think_llm"]),
            ("minimax", "MiniMax-M3", "MiniMax-M3"),
        )
        self.assertEqual(
            (deepseek["llm_provider"], deepseek["deep_think_llm"], deepseek["quick_think_llm"]),
            ("deepseek", "deepseek-v4-pro", "deepseek-v4-flash"),
        )

    def test_provider_specific_overrides_need_no_code_change(self) -> None:
        settings = resolve_llm_provider_settings({
            "LLM_PROVIDER": "minimax",
            "MINIMAX_BASE_URL": "https://minimax.example/v1/",
            "MINIMAX_DEEP_MODEL": "MiniMax-M3-next",
            "MINIMAX_QUICK_MODEL": "MiniMax-M3-fast",
        })

        self.assertEqual(settings.provider, "minimax")
        self.assertEqual(settings.backend_url, "https://minimax.example/v1")
        self.assertEqual(settings.deep_model, "MiniMax-M3-next")
        self.assertEqual(settings.quick_model, "MiniMax-M3-fast")

    def test_minimax_never_reuses_an_openai_key(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-openai-must-not-leak"},
            clear=False,
        ):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaisesRegex(ValueError, "MINIMAX_API_KEY"):
                MiniMaxClient("MiniMax-M3").get_llm()

    def test_cli_uses_global_provider_without_model_prompts(self) -> None:
        from cli.main import _configured_llm_selection

        selection = _configured_llm_selection({
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_DEEP_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_QUICK_MODEL": "deepseek-v4-flash",
        })

        self.assertEqual(selection, {
            "llm_provider": "deepseek",
            "backend_url": "https://api.deepseek.com",
            "shallow_thinker": "deepseek-v4-flash",
            "deep_thinker": "deepseek-v4-pro",
        })

    def test_cli_without_explicit_provider_uses_safe_minimax_default(self) -> None:
        from cli.main import _configured_llm_selection

        self.assertEqual(_configured_llm_selection({}), {
            "llm_provider": "minimax",
            "backend_url": "https://api.minimaxi.com/v1",
            "shallow_thinker": "MiniMax-M3",
            "deep_thinker": "MiniMax-M3",
        })

    def test_global_switch_rejects_provider_without_warning_adapter(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimax/deepseek"):
            resolve_llm_provider_settings({"LLM_PROVIDER": "glm"})


if __name__ == "__main__":
    unittest.main()
