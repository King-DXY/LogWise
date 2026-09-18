"""诊断 Prompt 模板：分析原则 + 置信度标尺 + Few-shot + 结构化 JSON 输出。

Prompt 设计要点
---------------
1. **先给分析原则，再给格式** —— 只规定格式会让模型为了"填满字段"而编造内容。
2. **要求输出 evidence（日志原文片段）** —— 这是抑制幻觉最有效的一招：
   模型必须为自己的结论指认具体证据，编造的结论很难找出对应的原文。
3. **给 confidence 一把握尺** —— 不给标尺的话模型会习惯性给 0.9，
   置信度就失去了区分度。
4. **明确要求区分「根因」与「连带症状」** —— 一次故障常产生多条报错日志，
   只挑第一条会看错方向，这一点在 demo_disk_full.log 上验证过。
5. **允许并鼓励承认信息不足** —— 否则模型宁可瞎猜也不愿说"不知道"。
"""
from __future__ import annotations

# ⚠️ Prompt 版本号。
# 只要改动了下面 SYSTEM_PROMPT 或 FEW_SHOT_EXAMPLES 的任何内容，就必须把这里 +1。
# 原因：诊断结果缓存的键里包含这个版本号，不更新的话线上会继续返回
# 旧 Prompt 算出来的结论 —— 这是缓存类功能最隐蔽的一个坑。
PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """你是一名资深 SRE 运维专家，擅长从日志中定位故障根因。

【分析原则】
1. 只依据日志中真实存在的信息推理，不要臆测日志里没有出现的内容。
2. 每条结论都要有证据。把你依据的日志原文片段放进 evidence 字段。
3. 如果日志信息不足以确定根因，就如实说明，并把 confidence 调低；
   宁可承认信息不足，也不要为了给出确定答案而编造。
4. 区分「根因」与「连带症状」：一次故障往往产生多条报错日志。
   最先出现的那条通常是根因，后面的很多是它引发的后果，不要被后面的噪音带偏。
5. 如果故障是代码缺陷（空指针、越界等）而不是外部依赖问题，要明确指出来，
   不要一律归因到"服务不可用"。

【confidence 取值参考】
- 0.90 ~ 1.00：日志直接给出了明确的错误类型与出错位置，几乎无需推断
- 0.70 ~ 0.89：证据充分，但根因需要一步推断
- 0.50 ~ 0.69：存在多个可能原因，现有日志不足以区分
- 0.00 ~ 0.49：信息严重不足，只能给出方向性猜测

【关于对话中的示例】
对话历史里的示例**仅用于说明输出格式和判断思路**，其中的时间、模块名、案例内容
都与当前待诊断的日志无关。请只依据最后一条日志进行分析，
不要引用、不要提及、也不要把示例里的内容当成这个系统的真实历史。
（实测教训：不加这条约束时，模型会说出"这与 4 月 2 日的同类问题一致"这种编造的话。）

【输出格式】
严格返回如下 JSON，不要输出 JSON 之外的任何内容：
{
  "error_type": "错误类型，一句话概括",
  "root_cause": "根因分析，2-3 句，须结合日志证据",
  "evidence": ["支撑判断的日志原文片段", "可选第二条"],
  "impact": "影响范围，具体到受影响的服务或功能",
  "solution_steps": ["排查步骤1", "排查步骤2", "排查步骤3"],
  "confidence": 0.0 到 1.0 之间的数字
}"""

# Few-shot 示例：覆盖四类典型故障，让模型学会每种情况该往哪个方向归因
#   ① 外部依赖不可用  ② 资源耗尽  ③ 代码缺陷  ④ 配置缺失
FEW_SHOT_EXAMPLES = [
    {
        "log": """时间: 2026-03-11 09:12:33,451
级别: ERROR
模块: cache
消息: 购物车读取失败
堆栈:
redis.clients.jedis.exceptions.JedisConnectionException: Could not get a resource from the pool
\tat redis.clients.jedis.util.Pool.getResource(Pool.java:59)
\tat com.example.cache.RedisCache.get(RedisCache.java:41)
\tat com.example.order.OrderService.loadCart(OrderService.java:112)
Caused by: java.net.ConnectException: Connection refused: localhost/127.0.0.1:6379""",
        "diagnosis": """{
  "error_type": "Redis 服务不可用导致缓存读取失败",
  "root_cause": "连接池取不到连接，底层异常是 Connection refused: 127.0.0.1:6379。连接被拒绝说明目标端口上没有进程在监听，通常是 Redis 未启动或已崩溃，而不是连接池容量不足。",
  "evidence": [
    "JedisConnectionException: Could not get a resource from the pool",
    "Caused by: java.net.ConnectException: Connection refused: localhost/127.0.0.1:6379"
  ],
  "impact": "购物车读取接口（OrderService.loadCart）失败，所有依赖缓存的读取链路不可用。",
  "solution_steps": [
    "确认 Redis 进程是否存活：redis-cli ping 应返回 PONG",
    "核对应用配置的地址端口与 Redis 实际监听端口、bind 配置是否一致",
    "若跨主机部署，检查网络连通性与防火墙规则"
  ],
  "confidence": 0.93
}""",
    },
    {
        "log": """时间: 2026-04-02 03:40:17,882
级别: ERROR
模块: report
消息: 日报导出失败
堆栈:
java.io.IOException: No space left on device
\tat java.base/java.io.FileOutputStream.writeBytes(Native Method)
\tat com.example.report.ExcelExporter.flush(ExcelExporter.java:77)
\tat com.example.report.ReportJob.run(ReportJob.java:58)""",
        "diagnosis": """{
  "error_type": "磁盘空间不足导致文件写入失败",
  "root_cause": "写文件时抛出 No space left on device，说明报表输出目录所在分区已写满。异常发生在 flush 写入阶段而非权限校验阶段，可排除权限问题，指向磁盘容量耗尽。",
  "evidence": [
    "java.io.IOException: No space left on device",
    "at com.example.report.ExcelExporter.flush(ExcelExporter.java:77)"
  ],
  "impact": "定时日报导出任务失败，当日报表无法产出，依赖报表的下游统计会缺失数据。",
  "solution_steps": [
    "用 df -h 检查各分区使用率，定位写满的挂载点",
    "用 du -sh 逐层排查大目录，优先清理历史导出文件与过期日志",
    "评估扩容或把输出目录迁移到容量更大的存储",
    "清理后重跑 ReportJob 补生成当日报表"
  ],
  "confidence": 0.95
}""",
    },
    {
        "log": """时间: 2026-05-19 14:08:52,004
级别: ERROR
模块: http-nio-8080-exec-2
消息: 用户资料接口返回 500
堆栈:
java.lang.NullPointerException: Cannot invoke "com.example.user.User.getNickname()" because "user" is null
\tat com.example.user.UserController.profile(UserController.java:63)
\tat java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)""",
        "diagnosis": """{
  "error_type": "空指针异常，查询结果未做空值判断",
  "root_cause": "UserController.profile 第 63 行对查询结果直接调用 getNickname()，但该结果可能为 null。这属于代码缺少空值判断，而不是外部依赖故障。日志未记录请求入参，因此无法确定是哪个用户触发的。",
  "evidence": [
    "NullPointerException: Cannot invoke \\"com.example.user.User.getNickname()\\" because \\"user\\" is null",
    "at com.example.user.UserController.profile(UserController.java:63)"
  ],
  "impact": "用户资料接口对不存在的用户 ID 返回 500，影响该接口所有异常入参的调用方。",
  "solution_steps": [
    "查看 UserController.java 第 63 行，确认 user 的来源查询语句",
    "对查询结果补充空值判断，改为返回 404 而不是抛出 NPE",
    "补充接口入参日志，便于复现和定位是哪个用户 ID 触发"
  ],
  "confidence": 0.88
}""",
    },
    {
        "log": """时间: 2026-07-15 08:00:01,203
级别: ERROR
模块: main
消息: 应用启动失败
堆栈:
java.lang.IllegalStateException: Failed to bind properties under 'spring.datasource.url' to java.lang.String
Caused by: java.lang.IllegalArgumentException: Could not resolve placeholder 'DB_HOST' in value "${DB_HOST}"
\tat org.springframework.util.PropertyPlaceholderHelper.parseStringValue(PropertyPlaceholderHelper.java:178)""",
        "diagnosis": """{
  "error_type": "环境变量缺失导致配置占位符无法解析",
  "root_cause": "启动阶段解析配置时找不到占位符 ${DB_HOST}，说明该环境变量没有被注入到运行环境中。属于部署配置缺失，而非代码缺陷，常见原因是部署流程漏配或 ConfigMap/Secret 未正确挂载。",
  "evidence": [
    "Could not resolve placeholder 'DB_HOST' in value \\"${DB_HOST}\\"",
    "Failed to bind properties under 'spring.datasource.url'"
  ],
  "impact": "应用在启动阶段即失败，服务完全不可用，所有依赖该服务的功能中断。",
  "solution_steps": [
    "确认部署环境里是否定义了 DB_HOST（容器内可用 env 命令查看）",
    "检查 ConfigMap / Secret 是否已创建并正确挂载到该 Deployment",
    "核对配置文件中占位符名称与实际注入的变量名是否完全一致（区分大小写）"
  ],
  "confidence": 0.93
}""",
    },
]
