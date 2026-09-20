# 接口与防护

## 接口列表

| 接口 | 方法 | 说明 | 花钱吗 |
| --- | --- | --- | --- |
| `/health` | GET | 健康检查（K8s 探针使用，**不加鉴权 / 不计度量 / 不受限流**）；响应里的 `api` 字段兼作接口契约版本握手 | 否 |
| `/parse` | POST | 解析日志，返回结构化字段列表 | 否 |
| `/context` | GET / POST | **预览上下文聚合结果**：哪些日志会被送给模型 | 否 |
| `/diagnose` | POST | 解析 + 诊断，一次性返回 JSON | 是 |
| `/diagnose/stream` | GET / POST | SSE 流式诊断，逐 token 返回 | 是 |
| `/stats` | GET | 运行时度量：请求量 / 耗时 / token / 成本 / 缓存命中率 | 否 |
| `/stats/reset` | POST | 清零统计与缓存（用于做对照实验） | 否 |

交互式文档：<http://127.0.0.1:8000/docs>

> ⚠️ **带文本参数的接口请优先用 POST。** GET 版把日志放在 query string 里，
> 实测长度上限约 **32K 字符**，再长会被 ASGI 服务器直接以 400 拒绝 ——
> 而「日志很长」恰恰是上下文聚合最有价值的场景。POST 版用 JSON body 传，没有这个限制。
> 前端已经统一改用 POST；GET 版保留向后兼容。

## `/diagnose` 的响应结构

响应里带一个 `context` 字段，说明这次实际送了哪些日志给模型：

```json
{
  "log": { "level": "ERROR", "message": "...", "...": "..." },
  "context": {
    "enabled": true,
    "total_parsed": 8,
    "selected": 5,
    "truncated": false,
    "window": 8,
    "max_logs": 20
  },
  "diagnosis": {
    "error_type": "...",
    "root_cause": "...",
    "evidence": ["..."],
    "...": "..."
  },
  "cached": false
}
```

- `log` 是锚点（模型看到的中心）
- `context.selected` 是连同上下文一起送进去的条数
- 聚合策略见 [`EXPERIMENTS.md` §4](EXPERIMENTS.md#4-上下文聚合)

---

## 接口防护

### 鉴权

默认**关闭**（`LOGWISE_API_KEY` 留空），保证本地演示开箱即用。
填上之后，所有业务接口都需要请求头 `X-API-Key`：

```bash
curl -H "X-API-Key: 你的密钥" http://127.0.0.1:8000/stats
```

来自本机回环地址（`127.0.0.1`）的请求可豁免（`LOGWISE_TRUST_LOCALHOST`，默认开启），
所以本地演示页不需要任何改动；而容器里外部请求的源 IP 是 Pod / 节点 IP，
不会被误放行 —— 部署到 K8s 后外部流量必须带 Key。

Key 的比较用固定时间算法，避免通过响应时间猜出密钥。

### 限流

按来源 IP 的**令牌桶**，默认每 60 秒 20 次。超限返回 `429` 并带 `Retry-After` 头。

> 注意：`X-Forwarded-For` 默认**不采信**（`LOGWISE_TRUST_PROXY_HEADERS=false`）。
> 因为该请求头可以被客户端随意伪造，一旦采信，限流和本机豁免都会失效。
> 只有在反向代理 / Ingress 后面才应开启。

### 缓存

同一段日志在 TTL 内不重复调用模型。

**缓存键 = `sha256(归一化日志 + 模型名 + Prompt 版本 + 上下文聚合参数 + 检索结果指纹)`**

每一项都有理由：

- **归一化** 会折叠空白、缩进差异和多空行 —— 用户从不同地方复制的同一段日志能稳定命中
- **模型名与 Prompt 版本** —— 换模型或改 Prompt 后旧缓存自动失效
  （这是缓存类功能最隐蔽的坑，`app/prompts.py` 里的 `PROMPT_VERSION` 就是为此存在的；
  改了 Prompt 却命中旧缓存，你会以为"改了没效果"）
- **聚合参数** —— 窗口 / 上限变了，同样的输入会产生不同的上下文包、进而产生不同结论
- **检索结果指纹** —— 否则 RAG 打开时会命中"旧 Prompt + 旧检索结果"算出来的结论，
  等于你以为在测 RAG，实际跑的还是老路径

缓存是**内存实现，单实例有效**。多副本部署需换成 Redis —— K8s 里 `replicas` 与 HPA
一旦扩到 N 个副本，每个 Pod 各有一份缓存，命中率会打折。

### 运行时度量

每次请求都会记录耗时、token 用量与估算成本，聚合结果在 `GET /stats`。

流式接口通过 `stream_options={"include_usage": True}` 拿到**真实** token 数（非估算，已实测支持）。

成本按**三档单价**计算 —— 缓存命中输入 / 未命中输入 / 输出。
DeepSeek 会自动缓存重复出现的 Prompt 前缀，命中部分单价远低于未命中；
最初按「全部输入 × 全价」计算，**成本高估约 1/3**。

几个容易看错的字段：

| 字段 | 含义 |
| --- | --- |
| `requests.llm_calls` | **真正调用了模型**的次数（含调用失败）。缓存命中不算 —— 它一次模型都没调 |
| `requests.cache_hits` | 被诊断缓存挡下来、没调模型的次数 |
| `requests.cache_hit_rate` | `cache_hits / (cache_hits + llm_calls)`，即「需要诊断的请求里缓存命中占比」 |
| `requests.billable_calls` | **产生了费用**的调用次数（判据是 `cost > 0`，所以调用失败不算） |
| `latency_ms.p95` | 最近保留记录的 95 分位（最近秩法）。注意样本量受 `RECENT_LIMIT` 限制 |

> 单价不硬编码 —— DeepSeek 调价频繁且分高峰/空闲时段。请在 `.env` 里配置
> `DEEPSEEK_PRICE_INPUT` / `DEEPSEEK_PRICE_OUTPUT`，`/stats` 会回显当前使用的单价，
> 方便一眼看出配置是否过时。
> 官方定价：<https://api-docs.deepseek.com/quick_start/pricing>
