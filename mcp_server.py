"""把历史故障案例库封装成 MCP Server —— 让任何支持 MCP 的 Agent 都能调用。

为什么单独做一个 MCP Server
---------------------------
LogWise 的 3 个工具（见 app/tools.py）原本写死在自己进程里，只有 LogWise 的
Agent 循环能用。MCP（Model Context Protocol）是工具暴露的**标准协议**：
封装之后，Claude Desktop、Cursor、或任何 MCP 客户端都能直接检索这个案例库，
不需要复制代码。

使用方式
--------
本文件是独立进程（stdio 传输），不进 FastAPI 服务：

    python mcp_server.py

在 MCP 客户端（如 Claude Desktop）的配置里登记::

    {
      "mcpServers": {
        "logwise-fault-cases": {
          "command": "python",
          "args": ["D:/LogWise/mcp_server.py"]
        }
      }
    }

暴露的工具
----------
- ``search_fault_cases(query, k)``：BM25 检索历史故障案例
- ``lookup_error_type(error_type)``：按错误类型精确查处置规范

（``get_log_overview`` 依赖当次诊断的日志上下文，不适合作为独立工具暴露，故不包含。）
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from app.tools import lookup_error_type as _lookup_error_type
from app.tools import search_fault_cases as _search_fault_cases

# mcp 2.x：FastMCP 已更名为 MCPServer（装饰器用法不变）
mcp = MCPServer("logwise-fault-cases")


@mcp.tool(description=(
    "在历史故障案例库中检索与关键词相关的案例。"
    "适用于遇到不熟悉的错误码、专有组件名时；"
    "常见故障（磁盘满、空指针、连接被拒）模型本来就会，不需要调用。"
))
def search_fault_cases(query: str, k: int = 3) -> dict:
    """检索历史故障案例。

    参数:
        query: 检索关键词，通常是错误码、异常类名或组件名
        k: 返回案例数，默认 3，最多 5
    """
    return _search_fault_cases(query, k)


@mcp.tool(description="按错误类型标识（fault_type，如 disk_full、oom）精确查找该类故障的标准处置流程。")
def lookup_error_type(error_type: str) -> dict:
    """按错误类型查处置规范。"""
    return _lookup_error_type(error_type)


if __name__ == "__main__":
    # stdio 传输：MCP 客户端通过标准输入输出与本进程通信
    mcp.run()
