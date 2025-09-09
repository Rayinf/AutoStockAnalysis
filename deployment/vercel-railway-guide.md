# Vercel + Railway 部署指南

## 前端部署 (Vercel)

### 1. 准备工作
```bash
cd frontend
npm run build
```

### 2. Vercel 配置文件
创建 `vercel.json`:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "dist"
      }
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "https://your-backend.railway.app/api/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

### 3. 环境变量
```env
VITE_API_BASE_URL=https://your-backend.railway.app
```

## 后端部署 (Railway)

### 1. 创建 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY *.db ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Railway 配置
创建 `railway.toml`:
```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn app:app --host 0.0.0.0 --port $PORT"
```

### 3. 环境变量设置
```env
ANTHROPIC_AUTH_TOKEN=your_token
ANTHROPIC_BASE_URL=https://anyrouter.top
PORT=8000
```
