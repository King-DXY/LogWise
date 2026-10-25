"""结构化日志：一行一个 JSON，方便 grep、jq，或直接交给采集器。

为什么不用 Python 默认的日志格式
--------------------------------
默认格式是给人看的「时间 - 级别 - logger - 消息」，想按字段过滤就得写正则。
这个项目真正要查的问题都很具体：「哪次请求花了多少钱」「哪个 request_id 报错了」，
这些字段在 JSON 行里可以直接 jq 出来。

什么时候生效
------------
只在 ``app/run.py`` 启动时调用 ``configure()``。测试里不配置 ——
那样 pytest 的输出会被日志淹没，而 logger 默认只处理 WARNING 以上，
INFO 级别的请求日志自然不会出现在测试输出里。
"""
from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    """把 LogRecord 压成一行 JSON。

    额外字段约定放在 ``extra={"data": {...}}`` 里，会被合并到最外层 ——
    这样调用方可以自由加字段，格式器不需要认识每一个业务字段。
    """

    def format(self, record: logging.LogRecord) -> str:
        item: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if isinstance(data, dict):
            item.update(data)
        if record.exc_info:
            # 异常栈很长，截断到 4000 字符，避免一条日志几千行
            item["exc"] = self.formatException(record.exc_info)[:4000]
        return json.dumps(item, ensure_ascii=False)


def configure(level: str = "INFO") -> logging.Logger:
    """配置 ``logwise`` logger 输出 JSON 行。重复调用是幂等的。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("logwise")
    logger.setLevel(level)
    logger.handlers[:] = [handler]
    logger.propagate = False
    return logger
