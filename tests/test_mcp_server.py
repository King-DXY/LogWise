"""MCP Server（mcp_server.py）的测试。

守护两件事：
1. 工具确实注册成功（mcp 2.x 的 ``MCPServer`` API 与 1.x 的 ``FastMCP`` 不同，
   升级时这里会立刻失败，而不是等用户连不上才发现）
2. 工具调用能真的查到案例库（知识库加载路径正确）
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp", reason="mcp 是可选依赖")


def _import_server():
    from mcp_server import mcp

    return mcp


class TestToolRegistration:
    def test_exposes_both_tools(self):
        mcp = _import_server()
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert names == {"search_fault_cases", "lookup_error_type"}

    def test_search_tool_returns_cases(self):
        mcp = _import_server()
        r = asyncio.run(mcp.call_tool("search_fault_cases", {"query": "No space left on device", "k": 2}))
        text = r.content[0].text
        assert '"matched": 2' in text
        assert "disk-01" in text, "BM25 检索应命中磁盘案例"

    def test_lookup_tool_returns_playbook(self):
        mcp = _import_server()
        r = asyncio.run(mcp.call_tool("lookup_error_type", {"error_type": "disk_full"}))
        text = r.content[0].text
        assert "disk_full" in text
        assert "solution_steps" in text

    def test_unknown_error_type_lists_available(self):
        """查不到时要给出可用类型列表 —— 让调用方下次能查对。"""
        mcp = _import_server()
        r = asyncio.run(mcp.call_tool("lookup_error_type", {"error_type": "no_such_type"}))
        text = r.content[0].text
        assert '"matched": 0' in text
        assert "available_types" in text
