# 启动、部署与项目结构

## 一键启动（推荐）

在文件管理器里**双击 `app\run.py`**，或在终端运行：

```bash
python app\run.py
```

启动后浏览器会自动打开 <http://127.0.0.1:8000>。关掉窗口即停止服务。

`app/run.py` 会自动完成这些事：

- 切换到项目自带的 `.venv`（即使被系统全局 Python 启动也能正确切换）
- 检查 Python 版本、依赖库、`.env` 里的 API Key，并给出中文提示
- 端口被占用时（例如 Docker 也在跑）自动改用下一个可用端口
- 服务就绪后自动打开浏览器

可选参数：`--reload`（改代码后自动重启）、`--port 9000`、`--no-browser`

> ⚠️ **改了 Python 代码必须重启服务才生效** —— 代码在启动时载入内存，
> 而 `static/index.html` 是每次请求现读磁盘的。只刷新浏览器会出现
> 「页面是新版、接口是旧版」的错位（表现为 404 / 405）。
> 演示页已内置版本自检，遇到这种情况会在顶部显示提示条。

## 手动启动（本地）

```bash
cd LogWise
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # 填入 DEEPSEEK_API_KEY
uvicorn app.main:app --reload --port 8000
```

- API 文档：<http://127.0.0.1:8000/docs>
- 演示页：<http://127.0.0.1:8000/>

### 环境变量

从 `.env.example` 复制。真实 Key 放在 `.env`（已在 `.gitignore` 中）。

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 诊断模型 |
| `LOGWISE_LLM_TIMEOUT` / `LOGWISE_LLM_CONNECT_TIMEOUT` / `LOGWISE_LLM_MAX_RETRIES` | 模型调用的超时与重试。**务必配置** —— SDK 默认 read=600 秒 + 2 次重试，最坏能挂约 30 分钟 |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` | 向量化（可选，仅向量后端需要） |
| `LOGWISE_API_KEY` | 接口鉴权，留空即关闭 |
| `LOGWISE_RATE_LIMIT` / `LOGWISE_RATE_WINDOW` | 限流阈值 |
| `LOGWISE_CONTEXT_WINDOW` / `LOGWISE_CONTEXT_MAX_LOGS` | 上下文聚合参数（默认 ±8 / 20） |
| `LOGWISE_RAG_MODE` | `fixed`（默认，不检索）/ `dynamic` / `hybrid` |
| `LOGWISE_RETRIEVER_BACKEND` | `bm25`（默认）/ `dense` / `hybrid` |
| `DEEPSEEK_PRICE_INPUT` / `DEEPSEEK_PRICE_OUTPUT` | 成本统计单价 |

## 测试与评测

```bash
pytest                              # 单元测试（289 项全通过）
python eval_hdfs.py                 # 解析器评测（Loghub HDFS_2k，2000 行）
python eval_retrieval.py            # 检索评测（零成本，不调模型）
python eval_retrieval.py --compare  # 三后端并排对比
python eval_diagnosis.py            # 诊断质量评测（会真实调用模型，产生少量费用）
python eval_diagnosis.py --from-git HEAD   # 评测历史版本的 Prompt，用于 A/B 对比
python build_index.py               # 构建向量索引（约 0.005 元）
```

项目里的**脚本全部支持双击运行**（会自动切换到项目虚拟环境，结束后停留让你看结果）：

| 脚本 | 说明 | 花钱吗 |
| --- | --- | --- |
| `app\run.py` | 启动服务 | 否 |
| `demo_parse.py` | 演示：日志解析 | 否 |
| `demo_context.py` | 演示：上下文聚合的新旧差别 | 约 0.002 元 |
| `demo_diagnose.py` | 演示：调 LLM 诊断一条日志 | 约 0.001 元 |
| `eval_hdfs.py` | 解析器评测（Loghub 真实数据） | 否 |
| `eval_retrieval.py` | 检索质量评测（`--compare` 三后端对比） | 否* |
| `eval_diagnosis.py` | 诊断质量评测 | 约 0.04 元 |
| `build_index.py` | 构建向量索引 | 约 0.005 元 |

\* `eval_retrieval.py` 用 `dense` / `hybrid` 后端时会把 query 向量化（已缓存的文本不重复计费）。

> 脚本内部统一用 `app/launcher.py` 做虚拟环境切换，路径也统一锚定到项目根目录，
> 所以「双击」和「在任意目录下用命令行运行」效果一致。

---

## Docker 部署

```bash
docker compose up --build
```

> 注意：`.dockerignore` 已排除 `.env`、测试与示例数据；API Key 通过环境变量注入。
>
> 国内网络需指定依赖源（Dockerfile 已内置 `ARG PIP_INDEX`，默认清华源）：
>
> ```bash
> docker build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple -t logwise-logwise:v2 .
> ```

## Kubernetes 部署

```bash
# 1. 构建镜像并打唯一 tag（重要：见下方说明）
docker build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
             -t logwise-logwise:v2 .

# 2. 创建 Secret（从 .env 注入 API Key）
kubectl create secret generic logwise-secret --from-env-file=.env

# 3. 部署（deployment.yaml 已指向 logwise-logwise:v2）
kubectl apply -f k8s/

# 4. 访问（LoadBalancer 8080）
curl http://localhost:8080/health
```

- HPA：`kubectl get hpa`（CPU 超过 50% 自动扩容，min 1 / max 5）
- 滚动更新：`kubectl rollout restart deployment/logwise`

> ⚠️ **镜像必须打唯一 tag。** Docker Desktop 的 K8s 使用独立的 containerd 镜像缓存，
> `docker build` 出的新镜像不会自动同步进去；在 `imagePullPolicy: IfNotPresent` 下，
> `latest` 这个 tag 会命中旧缓存，表现为"改了代码但 pod 行为没变"。
> 详见 [`PITFALLS.md`](PITFALLS.md)。

---

## 项目结构

```
LogWise/
├── app/
│   ├── run.py           # 一键启动器（双击即可运行）
│   ├── launcher.py      # 虚拟环境切换 + 控制台窗口停留（供各脚本复用）
│   ├── apikey.py        # API Key 读取与「占位符」判定（各调用方共用同一套判据）
│   ├── main.py          # FastAPI 入口 + SSE 流式接口
│   ├── agent.py         # Agent 循环（手写 ReAct：Function Calling + 轮次上限 + trace）
│   ├── agent_langgraph.py # 同一套逻辑的 LangGraph 版（编排层对照，可选依赖）
│   ├── tools.py         # Agent 工具层：工具 Schema + 本地执行 + 统一分发
│   ├── session.py       # 诊断会话：多轮追问（TTL + LRU + 轮数上限）
│   ├── parser.py        # 日志解析器（通用 + HDFS 格式）
│   ├── context.py       # 日志上下文聚合：决定把哪些日志送给模型
│   ├── diagnose.py      # 诊断引擎（同步 + 异步客户端，采集 token 用量）
│   ├── prompts.py       # System Prompt + Few-shot 示例 + PROMPT_VERSION
│   ├── guard.py         # 接口防护：API Key 鉴权 + 按 IP 令牌桶限流
│   ├── cache.py         # 诊断结果缓存（TTL + LRU）
│   ├── retriever.py     # 检索层：BM25 / Dense / Hybrid（RRF 融合），后端可插拔
│   ├── embedding.py     # 向量化客户端（OpenAI 兼容；带落盘缓存与优雅降级）
│   ├── rag.py           # RAG 注入端：把检索到的案例拼进 Prompt
│   └── metrics.py       # 运行时度量：耗时 / token / 成本统计
├── data/                # 示例日志、评测集与知识库
│   ├── sample.log               # 通用格式样例
│   ├── sample_hdfs.log          # HDFS 格式样例
│   ├── HDFS_2k.log              # Loghub 公开数据集（2000 行，用于解析器评测）
│   ├── demo_order_timeout.log   # 演示：数据库连接池耗尽
│   ├── demo_disk_full.log       # 演示：磁盘写满
│   ├── demo_pod_evicted.log     # 演示：Pod 被驱逐（用来看上下文聚合的效果）
│   ├── eval_cases.json          # 诊断质量评测集（38 条人工标注用例，5 档难度）
│   └── fault_cases.jsonl        # 故障案例知识库（56 条「日志 → 诊断」，供检索使用）
├── docs/                # 本目录：接口、部署、实验数据、踩坑记录
├── static/index.html    # 演示页（static/samples/ 为示例日志）
├── mcp_server.py        # MCP Server：把案例库暴露给任何 MCP 客户端（独立进程）
├── tests/               # 单元测试（338 项）
│   ├── test_parser.py           # 解析器
│   ├── test_context.py          # 上下文聚合
│   ├── test_guard.py            # 鉴权 / 限流 / 缓存 / 度量
│   ├── test_prompts.py          # Prompt 结构校验
│   ├── test_eval_dataset.py     # 评测集结构 + 与 Few-shot 的防泄漏检查
│   ├── test_knowledge_base.py   # 知识库结构 + 与评测集 / 示例的隔离检查
│   ├── test_retriever.py        # 检索层（分词 / 打分 / Dense / Hybrid）
│   ├── test_embedding.py        # 向量化客户端
│   ├── test_diagnose.py         # 诊断引擎（结构校验 / 格式重试 / 超时配置）
│   ├── test_apikey.py           # API Key 占位符判定（含三处调用方的一致性）
│   ├── test_rag.py              # RAG 注入
│   ├── test_agent.py            # Agent 循环（含流式事件顺序）
│   ├── test_agent_langgraph.py  # LangGraph 对比实现（与手写版行为等价）
│   ├── test_session.py          # 多轮追问（TTL / LRU / 轮数上限 / 失败回滚）
│   ├── test_mcp_server.py       # MCP Server（工具注册 / 实际检索）
│   └── test_api.py              # HTTP 接口层（含 SSE 断连不写缓存）
├── k8s/                 # Deployment / Service / HPA
├── demo_parse.py        # 演示：日志解析
├── demo_diagnose.py     # 演示：调用 LLM 诊断
├── demo_context.py      # 演示：上下文聚合带来的差别（新旧对比）
├── eval_hdfs.py         # 解析器评测（Loghub HDFS_2k）
├── eval_diagnosis.py    # 诊断质量评测（准确率 / 幻觉 / 置信度校准 / 成本）
├── eval_retrieval.py    # 检索质量评测（Recall@k / MRR；--compare 三后端对比）
├── build_index.py       # 构建向量索引（报出实际花费；已缓存的条目不重复计费）
├── Dockerfile
├── docker-compose.yml
├── .dockerignore        # ⚠️ 判断标准是「运行时是否需要」，别把 data/ 排除了（见 PITFALLS.md）
├── requirements.txt
├── .github/workflows/   # CI：跑测试 → 构建镜像 → 容器冒烟测试
└── .env.example         # 环境变量模板（真实 Key 放在 .env，已被 gitignore）
```

## CI

`.github/workflows/ci.yml` 两个 job：

1. **test** —— 装依赖 + `pytest tests/ -v`
2. **build** —— 构建镜像，然后**真的把容器跑起来**做冒烟测试：
   轮询 `/health`；打 `/stats`（依赖知识库）；`docker exec` 确认
   `data/fault_cases.jsonl` 确实在镜像里；打 `/context` 与静态首页。

> 第 2 步不是多余的。实测踩过一次：单元测试全过、镜像也能构建，
> 但 `.dockerignore` 排除了 `data/`，容器里没有知识库，
> 于是容器内的 `/stats` 直接 500 —— **pytest 完全测不出来，只有真跑起来才会暴露**。
