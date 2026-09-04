#!/usr/bin/env python
# coding: utf-8
# pylint: disable=protected-access

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from openjiuwen.core.foundation.tool import McpServerConfig
from openjiuwen.core.runner import Runner
from openjiuwen.harness.tools.browser_move.playwright_runtime.browser_capabilities import (
    CORE_BROWSER_TOOL_NAMES,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.config import BrowserRunGuardrails
from openjiuwen.harness.tools.browser_move.playwright_runtime.probes import (
    build_interactive_probe_js,
    enrich_interactive_probe_payload,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.runtime import (
    BrowserAgentRuntime,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.runtime_tools import (
    BrowserProbeInteractivesTool,
)
from openjiuwen.harness.tools.browser_move.playwright_runtime.site_profiles import (
    builtin_site_profiles,
)


def _run(coro):
    return asyncio.run(coro)


def _make_runtime() -> BrowserAgentRuntime:
    mcp_cfg = McpServerConfig(
        server_id="test-playwright-runtime",
        server_name="test-playwright-runtime",
        server_path="stdio://playwright",
        client_type="stdio",
        params={"cwd": str(Path.cwd())},
    )

    return BrowserAgentRuntime(
        provider="openai",
        api_key="test-key",
        api_base="https://example.invalid/v1",
        model_name="test-model",
        mcp_cfg=mcp_cfg,
        guardrails=BrowserRunGuardrails(
            max_steps=3,
            max_failures=1,
            timeout_s=30,
            retry_once=False,
        ),
    )


def test_build_interactive_probe_js_contains_high_value_selectors() -> None:
    js = build_interactive_probe_js(
        max_items=25,
        viewport_only=True,
        site_profiles=builtin_site_profiles(),
    )

    assert "button" in js
    assert "a[href]" in js
    assert "input" in js
    assert "[aria-label]" in js
    assert "[data-testid]" in js
    assert "max_items" in js
    assert "viewport_only" in js
    assert "validateSelectorHint" in js
    assert "matches.length === 1 && matches[0] === el" in js
    assert "selector_hint_validated" in js
    assert "actionable" in js
    assert "clickable" in js
    assert "match_count" in js
    assert "visible" in js
    assert "enabled" in js
    assert "generation_id" in js
    assert "matchCount === 1" in js
    assert "document.elementFromPoint" in js
    assert "calendar_date" in js
    assert "sort_tab" in js
    assert "rating_filter" in js
    assert "hotel_destination" in js
    assert "hotel_checkin" in js
    assert "hotel_checkout" in js
    assert "hotel_search_submit" in js
    assert "hotel_filter" in js
    assert "hotel_search" in js
    assert "global_search" in js
    assert "region: classifyRegion" in js
    assert "kind: kind || actionLikelihood" in js
    assert "'[role=\"gridcell\"]'" in js
    assert "'[role=\"tab\"]'" in js
    assert "ariaSnapshotJSON" in js
    assert "ariaSnapshot" in js
    assert "depth: 0" in js
    assert "timeout: 500" in js
    assert "Math.min(8, elements.length)" in js
    assert "browser_snapshot" not in js
    assert "mode: 'ai'" not in js


def test_build_interactive_probe_js_embeds_generation_id() -> None:
    js = build_interactive_probe_js(generation_id="g7")

    assert '"generation_id": "g7"' in js
    assert "selector_hint: clickable ? selectorHint : ''" in js
    assert "selector_hint_validated: clickable" in js


def test_build_interactive_probe_js_expands_search_and_input_queries() -> None:
    js = build_interactive_probe_js(max_items=25, viewport_only=True, query="search")

    assert "queryAliases" in js
    assert "queryMatches" in js
    assert "action_likelihood" in js
    assert "className" in js
    assert "input_type" in js
    assert "搜索" in js
    assert "关键词" in js
    assert 'role="searchbox"' in js
    assert "[placeholder]" in js
    assert "aria-labelledby" in js
    assert "el.labels" in js


def test_enrich_interactive_probe_payload_merges_structured_ax_and_dom_state() -> None:
    payload = {
        "elements": [
            {
                "role": "textbox",
                "accessible_name": "from",
                "action_likelihood": "input",
                "text": "",
                "enabled": True,
                "actionable": True,
                "clickable": True,
                "selector_hint": "#from",
                "selector_hint_validated": True,
                "__ax_selector": "#from",
                "__ax_json": [
                    {
                        "role": "combobox",
                        "name": "Flying from",
                        "expanded": False,
                    }
                ],
                "__dom_ax": {
                    "states": {"required": True, "readonly": False},
                    "value": "Singapore",
                },
            }
        ]
    }

    enrich_interactive_probe_payload(payload)

    element = payload["elements"][0]
    assert element["role"] == "combobox"
    assert element["accessible_name"] == "Flying from"
    assert element["action_likelihood"] == "input"
    assert element["ax"] == {
        "role": "combobox",
        "name": "Flying from",
        "states": {
            "expanded": False,
            "required": True,
            "readonly": False,
        },
        "value": "Singapore",
    }
    assert not any(key.startswith("__") for key in element)
    assert payload["ax_enrichment"] == {
        "status": "complete",
        "attempted": 1,
        "enriched": 1,
        "failed": 0,
    }


def test_enrich_interactive_probe_payload_supports_yaml_and_partial_failure() -> None:
    payload = {
        "elements": [
            {
                "role": "checkbox",
                "__ax_yaml": '- checkbox "Subscribe" [checked] [disabled=false]',
                "__dom_ax": {"states": {"required": True}},
            },
            {
                "role": "button",
                "__ax_yaml": "not: [valid",
                "__ax_selector": "#broken",
            },
        ]
    }

    enrich_interactive_probe_payload(payload)

    assert payload["elements"][0]["ax"] == {
        "role": "checkbox",
        "name": "Subscribe",
        "states": {"checked": True, "disabled": False, "required": True},
    }
    assert "ax" not in payload["elements"][1]
    assert "__ax_yaml" not in payload["elements"][1]
    assert "__ax_selector" not in payload["elements"][1]
    assert payload["ax_enrichment"] == {
        "status": "partial",
        "attempted": 2,
        "enriched": 1,
        "failed": 1,
    }


def test_enrich_interactive_probe_payload_parses_yaml_value() -> None:
    payload = {
        "elements": [
            {
                "role": "textbox",
                "__ax_yaml": '- textbox "Email" [invalid]: not-an-email',
            }
        ]
    }

    enrich_interactive_probe_payload(payload)

    assert payload["elements"][0]["ax"] == {
        "role": "textbox",
        "name": "Email",
        "states": {"invalid": True},
        "value": "not-an-email",
    }


def test_enrich_interactive_probe_payload_disables_unsafe_ax_target() -> None:
    payload = {
        "elements": [
            {
                "role": "button",
                "enabled": True,
                "actionable": True,
                "clickable": True,
                "selector_hint": "#submit",
                "selector_hint_validated": True,
                "__ax_json": [{"role": "button", "name": "Submit", "disabled": True}],
            }
        ]
    }

    enrich_interactive_probe_payload(payload)

    element = payload["elements"][0]
    assert element["disabled"] is True
    assert element["enabled"] is False
    assert element["actionable"] is False
    assert element["clickable"] is False
    assert element["selector_hint"] == ""
    assert element["selector_hint_validated"] is False


def test_build_interactive_probe_js_clamps_max_items() -> None:
    js = build_interactive_probe_js(max_items=999, viewport_only=True)

    assert '"max_items": 100' in js


def test_browser_probe_interactives_tool_invokes_runtime_api() -> None:
    runtime = _make_runtime()
    runtime.probe_interactives = AsyncMock(
        return_value={
            "ok": True,
            "elements": [
                {
                    "id": "e1",
                    "role": "button",
                    "text": "Add to cart",
                    "selector_hint": "button:nth-of-type(1)",
                }
            ],
            "error": None,
        }
    )

    tool = BrowserProbeInteractivesTool(runtime, language="en")

    result = _run(
        tool.invoke(
            {
                "max_items": 200,
                "viewport_only": "false",
                "query": "cart",
            }
        )
    )

    runtime.probe_interactives.assert_called_once_with(
        max_items=40,
        viewport_only=False,
        query="cart",
    )
    assert result.success is True
    assert result.data["elements"][0]["text"] == "Add to cart"


def test_browser_probe_interactives_tool_reports_runtime_error() -> None:
    runtime = _make_runtime()
    runtime.probe_interactives = AsyncMock(
        return_value={
            "ok": False,
            "error": "browser_code_executor_not_ready",
            "elements": [],
        }
    )

    tool = BrowserProbeInteractivesTool(runtime, language="en")

    result = _run(tool.invoke({}))

    assert result.success is False
    assert result.error == "browser_code_executor_not_ready"
    assert result.data["elements"] == []


def test_runtime_probe_interactives_uses_code_executor_and_parses_json() -> None:
    runtime = _make_runtime()
    runtime.ensure_runtime_ready = AsyncMock()
    runtime._code_executor = AsyncMock(
        return_value={
            "ok": True,
            "url": "https://example.com",
            "title": "Example",
            "elements": [
                {
                    "id": "e1",
                    "role": "button",
                    "text": "Search",
                    "region": "global_search",
                    "kind": "search",
                    "selector_hint": "button:nth-of-type(1)",
                    "selector_hint_validated": True,
                    "match_count": 1,
                    "visible": True,
                    "enabled": True,
                    "actionable": True,
                    "clickable": True,
                    "__ax_selector": "button:nth-of-type(1)",
                    "__ax_json": [{"role": "button", "name": "Search the site"}],
                    "__dom_ax": {"states": {}},
                }
            ],
        }
    )

    result = _run(
        runtime.probe_interactives(
            max_items=10,
            viewport_only=True,
            query="search",
        )
    )

    runtime.ensure_runtime_ready.assert_called_once()
    runtime._code_executor.assert_called_once()
    assert result["ok"] is True
    assert result["url"] == "https://example.com"
    assert result["elements"][0]["text"] == "Search"
    assert result["elements"][0]["region"] == "global_search"
    assert result["elements"][0]["kind"] == "search"
    assert result["elements"][0]["ax"] == {
        "role": "button",
        "name": "Search the site",
    }
    assert result["ax_enrichment"] == {
        "status": "complete",
        "attempted": 1,
        "enriched": 1,
        "failed": 0,
    }
    assert result["elements"][0]["target_id"].startswith("t_g")
    assert result["elements"][0]["generation_id"] == result["page_state"]["generation_id"]
    assert "id" not in result["elements"][0]
    assert "selector_hint" not in result["elements"][0]
    assert result["page_state"]["interactives"][0]["target_id"] == result["elements"][0]["target_id"]
    assert result["page_state"]["interactives"][0]["region"] == "global_search"
    assert result["page_state"]["interactives"][0]["kind"] == "search"
    assert result["page_state"]["interactives"][0]["ax"] == result["elements"][0]["ax"]


def test_runtime_probe_interactives_handles_missing_code_executor() -> None:
    runtime = _make_runtime()
    runtime.ensure_runtime_ready = AsyncMock()
    runtime._code_executor = None

    result = _run(runtime.probe_interactives())

    assert result["ok"] is False
    assert result["error"] == "browser_code_executor_not_ready"
    assert result["elements"] == []


def test_explicit_probe_failure_preserves_previous_page_state() -> None:
    runtime = _make_runtime()
    runtime.ensure_runtime_ready = AsyncMock()
    runtime._code_executor = AsyncMock(
        side_effect=[
            {
                "ok": True,
                "url": "https://example.com",
                "title": "Example",
                "elements": [
                    {
                        "role": "button",
                        "text": "Search",
                        "selector_hint": "#search",
                        "selector_hint_validated": True,
                        "match_count": 1,
                        "visible": True,
                        "enabled": True,
                        "actionable": True,
                        "clickable": True,
                    }
                ],
            },
            RuntimeError("probe timeout"),
        ]
    )

    first = _run(runtime.probe_interactives())
    failed = _run(runtime.probe_interactives())

    target_id = first["elements"][0]["target_id"]
    assert failed["ok"] is False
    assert failed["page_state"]["interactives"][0]["target_id"] == target_id
    assert (
        runtime._ensure_page_state().resolve_target(
            generation_id="g0",
            target_id=target_id,
        )
        is not None
    )


def test_runtime_playwright_client_lookup_keys_include_server_name_variants() -> None:
    runtime = _make_runtime()
    runtime._service.mcp_cfg.server_id = "playwright_official_stdio"
    runtime._service.mcp_cfg.server_name = "playwright-official"

    keys = runtime._playwright_client_lookup_keys()

    assert "playwright_official_stdio" in keys
    assert "playwright-official" in keys
    assert "playwright_official" in keys
    assert "playwright" in keys


def test_runtime_unwrap_mcp_text_result() -> None:
    runtime = _make_runtime()

    raw = {
        "content": [
            {
                "type": "text",
                "text": '{"ok": true, "elements": []}',
            }
        ]
    }

    assert runtime._unwrap_mcp_text_result(raw) == '{"ok": true, "elements": []}'


def test_probe_query_uses_exact_match_before_bounded_alias_widening() -> None:
    script = build_interactive_probe_js(query="搜索清华大学")

    assert "exactQueryMatches" in script
    assert "widenedCandidates" in script
    assert "query_widened" in script
    assert "raw.includes(term)" in script


def test_runtime_call_playwright_run_code_unsafe_uses_runner_mcp_tool(monkeypatch) -> None:
    runtime = _make_runtime()
    assert runtime.service.allowed_tool_names == CORE_BROWSER_TOOL_NAMES
    assert "browser_run_code_unsafe" not in runtime.service.allowed_tool_names

    class FakeToolResult:
        success = True
        error = None
        data = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"ok": true, "elements": []}\n'
                        "### Page state\n"
                        "- Page URL: https://example.com/\n"
                        "- Page Snapshot: large snapshot omitted"
                    ),
                }
            ]
        }

    class FakeTool:
        def __init__(self):
            self.inputs = None

        async def invoke(self, inputs):
            self.inputs = inputs
            return FakeToolResult()

    fake_tool = FakeTool()

    async def fake_get_mcp_tool(**kwargs):
        if kwargs.get("name") == "browser_run_code_unsafe":
            return [fake_tool]
        return []

    monkeypatch.setattr(
        Runner.resource_mgr,
        "get_mcp_tool",
        fake_get_mcp_tool,
    )

    result = _run(runtime._call_playwright_run_code_unsafe("async (page) => ({ok: true})"))

    assert fake_tool.inputs == {"code": "async (page) => ({ok: true})"}
    assert result["__browser_compact_rpc__"] is True
    assert result["payload"] == '{"ok":true,"elements":[]}'
    assert result["rpc_metrics"]["tool_name"] == "browser_run_code_unsafe"
    assert result["rpc_metrics"]["transport_response_size_bytes"] > result["rpc_metrics"]["response_size_bytes"]
