# LogWise · LLM 智能日志诊断助手

面向「云原生 + AI 大模型应用开发」的个人项目：粘贴一段报错日志，由 LLM 自动解析并给出结构化诊断（错误类型 / 根因 / 影响 / 排查步骤），支持 SSE 流式输出。

## 技术栈

- **Python 3.11 + FastAPI + uvicorn**：API 服务与 SSE 流式输出
- **DeepSeek API（OpenAI 兼容 SDK）**：结构化 JSON 诊断
- **pytest**：解析器单元测试（6/6 通过）
- **Docker / docker-compose**：容器化与本地编排
- **Kubernetes（Docker Desktop 内置集群）**：Deployment + Service(LoadBalancer) + HPA + Secret 注入 + 探针

## 功能特性

- 日志解析器：通用格式（时间 / 级别 / 模块 / 消息 / 堆栈）与 HDFS 格式双支持
- LLM 结构化诊断：错误类型、根因分析、影响范围、排查步骤、置信度
- `/diagnose/stream` SSE 接口：逐 token 流式返回诊断内容
- 静态演示页：粘贴日志 → 流式渲染诊断结果
- 云原生工程化：Docker 多阶段缓存、K8s 声明式部署、HPA 弹性伸缩

## 一键启动（推荐）

在文件管理器里**双击 `app\run.py`**，或在终端运行：

```bash
python app\run.py
```

启动后浏览器会自动打开 <http://127.0.0.1:8000>。关掉窗口即停止服务。

`app/run.py` 会自动完成这些事：

- 切换到项目自带的 `.venv`（即使被系统全局 Python 启动也能正确切换）
- 检查 Python 版本、依赖库、`.env` 里的 API Key 并给出中文提示
- 端口被占用时（例如 Docker 也在跑）自动改用下一个可用端口
- 服务就绪后自动打开浏览器

可选参数：`--reload`（改代码后自动重启）、`--port 9000`、`--no-browser`

## 快速开始（本地）

```bash
cd LogWise
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # 填入 DEEPSEEK_API_KEY
uvicorn app.main:app --reload --port 8000
```

- API 文档：http://127.0.0.1:8000/docs
- 演示页：http://127.0.0.1:8000/

## 测试

```bash
pytest                # 单元测试（6/6 通过）
python eval_hdfs.py   # 真实数据评测（Loghub HDFS_2k）
```

## 真实数据评测结果

使用 **Loghub 开源真实日志数据集**（HDFS_2k.log，Apache Hadoop 分布式文件系统 2000 行真实运行日志，无人工标注）评测解析器：

| 指标 | 结果 |
| --- | --- |
| 解析覆盖率 | 100%（2000/2000 行） |
| 字段完整率 | 100%（时间/级别/模块/消息） |
| 格式异常 | 0（时间戳 / 级别枚举 / 模块模式全合法） |
| 解析吞吐 | 37.7 万行/秒（2000 行耗时 5.3ms） |
| 级别分布 | INFO 1920 / WARN 80 |
| 模块分布 | dfs.FSNamesystem 659、dfs.DataNode$PacketResponder 603、dfs.DataNode$DataXceiver 454… |

> 数据来源：https://github.com/logpai/loghub （Loghub：日志解析领域公认基准数据集）
>
> 说明：该数据集采集自 Apache Hadoop 生产集群，内容已做匿名化处理。日志中出现的
> `10.x.x.x` 地址为数据集自带的匿名内网地址，与任何个人或组织的真实网络无关。

## Docker 部署

```bash
docker compose up --build
```

> 注意：`.dockerignore` 已排除 `.env`、测试与示例数据；API Key 通过环境变量注入。

## Kubernetes 部署

```bash
# 1. 构建镜像并打唯一 tag（重要：见下方「已知坑」）
#    国内网络需指定依赖源，海外环境可省略 --build-arg
docker build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
             -t logwise-logwise:v2 .

# 2. 创建 Secret（从 .env 注入 API Key）
kubectl create secret generic logwise-secret --from-env-file=.env

# 3. 部署（deployment.yaml 已指向 logwise-logwise:v2）
kubectl apply -f k8s/

# 4. 访问（LoadBalancer 8080）
curl http://localhost:8080/health
curl.exe -N "http://localhost:8080/diagnose/stream?text=2025-09-14%2010:23:46,789%20ERROR%20main%20app.db%20-%20connect%20timeout"
```

- HPA：`kubectl get hpa`（CPU 超过 50% 自动扩容，min 1 / max 5）
- 滚动更新：`kubectl rollout restart deployment/logwise`

## API

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/health` | GET | 健康检查（K8s 探针使用） |
| `/parse` | POST | 解析日志，返回结构化字段列表 |
| `/diagnose` | POST | 解析 + 诊断，一次性返回 JSON |
| `/diagnose/stream` | GET | SSE 流式诊断，逐 token 返回 |

## 已知坑（排障记录）

1. **SSE 一次性输出（本地）**：在 async 生成器里用同步 OpenAI 客户端迭代会阻塞事件循环，导致 uvicorn 攒到结束才一次性发送。修复：改用 `AsyncOpenAI` + `async for chunk in resp` 逐 token yield。
2. **openai SDK 版本**：openai 3.x 改用 httpx2 传输栈，实测流式行为异常；`requirements.txt` 固定 `openai>=1.40,<2`。
3. **K8s 更新镜像后 pod 仍是旧镜像**：Docker Desktop 的 K8s 使用独立的 containerd 镜像缓存，`docker build` 新镜像不会同步；`imagePullPolicy: IfNotPresent` 下 `latest` tag 命中旧缓存。修复：每次构建打唯一 tag（如 `v2`），并让 deployment 指向该 tag。

## 项目结构

```
LogWise/
├── app/
│   ├── run.py           # 一键启动器（双击即可运行）
│   ├── main.py          # FastAPI 入口 + SSE 流式接口
│   ├── parser.py        # 日志解析器（通用 + HDFS 格式）
│   ├── diagnose.py      # DeepSeek 诊断引擎（同步 + 异步客户端）
│   └── prompts.py       # System Prompt + Few-shot 示例
├── data/                # 示例日志
│   ├── sample.log               # 通用格式样例
│   ├── sample_hdfs.log          # HDFS 格式样例
│   ├── HDFS_2k.log              # Loghub 公开数据集（2000 行，用于评测）
│   ├── demo_order_timeout.log   # 演示：数据库连接池耗尽
│   └── demo_disk_full.log       # 演示：磁盘写满
├── static/index.html    # 演示页
├── tests/               # 解析器单元测试
├── k8s/                 # Deployment / Service / HPA
├── demo_parse.py        # 演示：日志解析
├── demo_diagnose.py     # 演示：调用 LLM 诊断
├── eval_hdfs.py         # 真实数据评测脚本
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example         # 环境变量模板（真实 Key 放在 .env，已被 gitignore）
```
