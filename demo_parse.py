"""演示脚本：解析 sample.log 与 sample_hdfs.log，打印结构化结果。"""
from app.parser import parse_text

for filename in ("data/sample.log", "data/sample_hdfs.log"):
    print("=" * 50)
    print(filename)
    print("=" * 50)
    text = open(filename, encoding="utf-8").read()
    logs = parse_text(text)
    print("解析出日志条数:", len(logs))
    for i, log in enumerate(logs):
        print(f"--- 第 {i + 1} 条 ---")
        print("时间:", log.timestamp)
        print("级别:", log.level)
        print("模块:", log.module)
        print("消息:", log.message[:80])
        if log.stack_trace:
            print("堆栈:", log.stack_trace.replace("\n", " | ")[:120])
    print()
