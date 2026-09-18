"""演示：解析 sample.log 里的 ERROR 日志，调 DeepSeek 生成结构化诊断。"""
import json

from app.diagnose import diagnose
from app.parser import parse_text

text = open("data/sample.log", encoding="utf-8").read()
logs = parse_text(text)
error_log = logs[1]  # 带堆栈的那条 ERROR

print("待诊断日志:")
print(error_log)
print("\n" + "=" * 50)
print("DeepSeek 诊断结果:")
result = diagnose(error_log)
print(json.dumps(result, ensure_ascii=False, indent=2))
