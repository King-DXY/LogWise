# LogWise 容器化：Python 3.11 slim + 非 root 用户
FROM python:3.11-slim

WORKDIR /app

# 先复制依赖清单再装依赖：利用 Docker 层缓存，改代码不用重装依赖
# 依赖源默认用官方 PyPI —— GitHub Actions 等海外环境构建更快。
# 国内网络直连官方源容易超时，国内构建时指定镜像即可：
#   docker build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple .
#   （用 docker compose 构建时已自动带上，见 docker-compose.yml）
ARG PIP_INDEX=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir -i ${PIP_INDEX} -r requirements.txt

# 再复制代码（依赖层已缓存，只有这层会频繁变化）
COPY . .

# 安全最佳实践：不用 root 跑服务
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
