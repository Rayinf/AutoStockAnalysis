# GitHub Pages + Koyeb 免费部署方案

## 前端部署 (GitHub Pages)

### 1. GitHub Actions 工作流
创建 `.github/workflows/deploy.yml`:
```yaml
name: Deploy Frontend

on:
  push:
    branches: [ main ]
    paths: [ 'frontend/**' ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install dependencies
      run: |
        cd frontend
        npm ci
    
    - name: Build
      run: |
        cd frontend
        npm run build
      env:
        VITE_API_BASE_URL: https://your-app.koyeb.app
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: frontend/dist
```

### 2. 前端API配置
修改前端代码，使用环境变量:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
```

## 后端部署 (Koyeb)

### 1. 准备 Dockerfile
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

### 2. Koyeb 配置
创建 `koyeb.yaml`:
```yaml
app:
  name: stock-analysis-api
  
services:
  - name: api
    type: web
    build:
      type: docker
      dockerfile: Dockerfile
    ports:
      - port: 8000
        protocol: http
    env:
      - name: ANTHROPIC_AUTH_TOKEN
        secret: anthropic-token
      - name: ANTHROPIC_BASE_URL
        value: https://anyrouter.top
    regions:
      - fra  # 法兰克福
    scaling:
      min: 1
      max: 1
```

### 3. 部署步骤

1. **GitHub 设置**:
   - 推送代码到 GitHub
   - 启用 GitHub Pages (Settings -> Pages -> GitHub Actions)

2. **Koyeb 设置**:
   - 注册 Koyeb 账号
   - 连接 GitHub 仓库
   - 设置环境变量
   - 部署应用

### 4. CORS 配置
修改后端 `app.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourusername.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
