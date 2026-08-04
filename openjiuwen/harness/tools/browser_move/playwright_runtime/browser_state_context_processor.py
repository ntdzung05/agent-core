# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ephemeral current-browser-state injection for browser-agent model calls."""

from __future__ import annotations

import json
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from openjiuwen.core.context_engine import ContextEngine, ContextWindow, ModelContext
from openjiuwen.core.context_engine.processor.base import ContextEvent, ContextProcessor
from openjiuwen.core.foundation.llm import AssistantMessage, BaseMessage, ToolMessage, UserMessage

from .browser_logging import browser_agent_log_warning

_BROWSER_STATE_MESSAGE_NAME = "current_browser_state"
_BROWSER_STATE_METADATA_KEY = "browser_state_context"
_BROWSER_STATE_REFRESH_TOOL_NAMES = frozenset(
    {
        "browser_batch_interact",
        "browser_click",
        "browser_close",
        "browser_custom_action",
        "browser_fill_form",
        "browser_navigate",
        "browser_navigate_back",
        "browser_press_key",
        "browser_select_option",
        "browser_tabs",
        "browser_type",
    }
)


class BrowserStateContextProcessorConfig(BaseModel):
    """Configuration for capturing browser state initially and after mutations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any = Field(exclude=True, repr=False)
    max_dom_chars: int = Field(default=40_000, gt=0)


@ContextEngine.register_processor()
class BrowserStateContextProcessor(ContextProcessor):
    """Inject cached browser state, refreshing it after browser mutations."""

    def __init__(self, config: BrowserStateContextProcessorConfig):
        super().__init__(config)
        self._cached_state: Dict[str, Any] | None = None
        self._seen_refresh_tool_call_ids: set[str] = set()

    @property
    def config(self) -> BrowserStateContextProcessorConfig:
        return self._config

    async def trigger_get_context_window(
        self,
        context: ModelContext,
        context_window: ContextWindow,
        **kwargs: Any,
    ) -> bool:
        del context, context_window, kwargs
        # The hook must run for every model call because this processor's state
        # message is ephemeral. The expensive browser capture inside the hook is
        # gated on the initial call and completed browser-mutating tool calls.
        return True

    async def on_get_context_window(
        self,
        context: ModelContext,
        context_window: ContextWindow,
        **kwargs: Any,
    ) -> tuple[ContextEvent | None, ContextWindow]:
        del kwargs
        source_messages = context.get_messages() if context is not None else context_window.context_messages
        refresh_tool_call_ids = self._completed_refresh_tool_call_ids(source_messages)
        should_refresh = self._cached_state is None or bool(refresh_tool_call_ids - self._seen_refresh_tool_call_ids)
        if should_refresh:
            self._cached_state = await self._capture_state()
        self._seen_refresh_tool_call_ids.update(refresh_tool_call_ids)

        state = self._cached_state if self._cached_state is not None else {}
        current_message = self._build_state_message(state)

        context_window.context_messages = [
            message for message in context_window.context_messages if not self._is_browser_state_message(message)
        ]
        context_window.context_messages.append(current_message)
        return None, context_window

    @classmethod
    def _completed_refresh_tool_call_ids(
        cls,
        messages: list[BaseMessage],
    ) -> set[str]:
        """Return state-invalidating calls that have a corresponding result."""

        completed_call_ids = {
            str(message.tool_call_id)
            for message in messages
            if isinstance(message, ToolMessage) and message.tool_call_id
        }
        refresh_tool_call_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, AssistantMessage):
                continue
            for tool_call in message.tool_calls or []:
                call_id = str(tool_call.id) if tool_call.id else ""
                if call_id in completed_call_ids and cls._is_refresh_tool_name(tool_call.name):
                    refresh_tool_call_ids.add(call_id)
        return refresh_tool_call_ids

    @staticmethod
    def _is_refresh_tool_name(tool_name: str) -> bool:
        return any(
            tool_name == expected or tool_name.endswith(f".{expected}") or tool_name.endswith(f"_{expected}")
            for expected in _BROWSER_STATE_REFRESH_TOOL_NAMES
        )

    async def _capture_state(self) -> Dict[str, Any]:
        try:
            state = await self.config.provider.capture_browser_state()
        except Exception as exc:
            browser_agent_log_warning(
                "[BrowserStateContextProcessor] browser state capture failed: %s",
                exc,
            )
            return {
                "ok": False,
                "error": f"browser state capture failed: {exc}",
                "url": "",
                "title": "",
                "tabs": [],
                "page_position": {},
                "dom": "",
            }

        if not isinstance(state, dict):
            return {
                "ok": False,
                "error": "browser state capture returned a non-object result",
                "url": "",
                "title": "",
                "tabs": [],
                "page_position": {},
                "dom": "",
            }
        return state

    def _build_state_message(self, state: Dict[str, Any]) -> UserMessage:
        text = self._format_state_text(state)
        return UserMessage(
            name=_BROWSER_STATE_MESSAGE_NAME,
            metadata={_BROWSER_STATE_METADATA_KEY: True},
            content=text,
        )

    def _format_state_text(self, state: Dict[str, Any]) -> str:
        dom = state.get("dom")
        if not isinstance(dom, str):
            dom = json.dumps(dom or {}, ensure_ascii=False)
        if len(dom) > self.config.max_dom_chars:
            omitted = len(dom) - self.config.max_dom_chars
            dom = f"{dom[: self.config.max_dom_chars]}\n...[DOM truncated; {omitted} characters omitted]"

        state_header = {
            "ok": bool(state.get("ok")),
            "error": state.get("error"),
            "url": state.get("url") or "",
            "title": state.get("title") or "",
            "tabs": state.get("tabs") or [],
            "page_position": state.get("page_position") or {},
            "dom_error": state.get("dom_error"),
        }
        return (
            "<browser_state>\n"
            "This observation was captured initially or after the latest detected browser mutation and "
            "replaces any previous browser state. It is reused until another state-invalidating browser "
            "tool completes; element references may become stale if the page changes independently.\n"
            f"{json.dumps(state_header, ensure_ascii=False, indent=2)}\n"
            "<dom_snapshot>\n"
            f"{dom or '[DOM snapshot unavailable]'}\n"
            "</dom_snapshot>\n"
            "</browser_state>"
        )

    @staticmethod
    def _is_browser_state_message(message: BaseMessage) -> bool:
        return message.name == _BROWSER_STATE_MESSAGE_NAME or bool(message.metadata.get(_BROWSER_STATE_METADATA_KEY))

    def load_state(self, state: Dict[str, Any]) -> None:
        del state
        self._cached_state = None
        self._seen_refresh_tool_call_ids = set()

    def save_state(self) -> Dict[str, Any]:
        return {}


__all__ = [
    "BrowserStateContextProcessor",
    "BrowserStateContextProcessorConfig",
]
