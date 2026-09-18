"""解析器单元测试：通用格式 + HDFS 真实格式。"""
from app.parser import parse_log, parse_text

SAMPLE_LOG = """\
2025-09-14 10:23:45,123 INFO [http-nio-8080-exec-1] com.example.OrderService - 用户下单失败: 库存不足

2025-09-14 10:23:46,789 ERROR [main] app.db - 连接数据库超时
Traceback (most recent call last):
  File "app/db.py", line 18, in insert
    raise ConnectionError("connect timeout")
ConnectionError: connect timeout

2025-09-14 10:23:47,012 WARN [scheduler] app.job - 重试任务 3 次仍失败，稍后再次调度
"""

HDFS_LOG = """\
081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
081109 203807 222 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-6952295868487656571 terminating
081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864
"""


def test_generic_parse_count():
    logs = parse_text(SAMPLE_LOG)
    assert len(logs) == 3


def test_generic_fields():
    log = parse_text(SAMPLE_LOG)[0]
    assert log.timestamp == "2025-09-14 10:23:45,123"
    assert log.level == "INFO"
    assert log.module == "http-nio-8080-exec-1"
    assert "库存不足" in log.message


def test_generic_stack_trace():
    log = parse_text(SAMPLE_LOG)[1]
    assert log.level == "ERROR"
    assert "Traceback" in log.stack_trace
    assert "ConnectionError" in log.stack_trace


def test_hdfs_parse_count():
    logs = parse_text(HDFS_LOG)
    assert len(logs) == 3


def test_hdfs_fields():
    log = parse_text(HDFS_LOG)[0]
    assert log.timestamp == "081109 203615"
    assert log.level == "INFO"
    assert log.module == "dfs.DataNode$PacketResponder"
    assert "terminating" in log.message


def test_unknown_format_graceful():
    # 未知格式不应抛异常，整块降级为消息
    log = parse_log("this is not a log at all\njust some text")
    assert log.message == "this is not a log at all\njust some text"
    assert log.level == ""
