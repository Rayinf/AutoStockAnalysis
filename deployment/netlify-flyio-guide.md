# Netlify + Fly.io 部署指南

## 前端部署 (Netlify)

### 1. 构建配置
创建 `netlify.toml`:
```toml
[build]
  base = "frontend/"
  command = "npm run build"
  publish = "dist/"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/api/*"
  to = "https://your-app.fly.dev/api/:splat"
  status = 200
  force = true

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

### 2. 环境变量
```env
VITE_API_BASE_URL=https://your-app.fly.dev
```

## 后端部署 (Fly.io)

### 1. 安装 Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. 初始化应用
```bash
cd backend/
fly launch --no-deploy
```

### 3. Fly.io 配置
创建 `fly.toml`:
```toml
app = "your-stock-app"
primary_region = "nrt"  # 东京区域

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8000"

[[services]]
  internal_port = 8000
  protocol = "tcp"
  
  [[services.ports]]
    port = 80
    handlers = ["http"]
  
  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

[mounts]
  source = "data_volume"
  destination = "/app/data"
```

### 4. Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 创建数据目录
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. 创建持久化卷
```bash
fly volumes create data_volume --size 1 --region nrt
```

### 6. 部署
```bash
fly deploy
fly secrets set ANTHROPIC_AUTH_TOKEN=your_token
fly secrets set ANTHROPIC_BASE_URL=https://anyrouter.top
```
