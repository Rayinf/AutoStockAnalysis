# 云部署准备清单

## 🔧 代码修改需求

### 1. 环境变量配置
- [ ] 创建 `.env.production` 文件
- [ ] 配置 API 基础URL
- [ ] 设置数据库路径
- [ ] 配置 LLM API密钥

### 2. CORS 设置
```python
# backend/app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-domain.com",
        "https://your-app.vercel.app",
        "https://yourusername.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. 数据库路径修改
```python
# 使用相对路径或环境变量
DB_PATH = os.getenv("DB_PATH", "./data/")
```

### 4. 静态文件服务
```python
# 如果使用单一服务部署
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

## 📦 必需文件

### 通用文件
- [ ] `requirements.txt` (后端依赖)
- [ ] `Dockerfile` (容器化)
- [ ] `.dockerignore`
- [ ] `.gitignore`

### Vercel 部署
- [ ] `vercel.json`
- [ ] 前端环境变量配置

### Railway 部署  
- [ ] `railway.toml`
- [ ] `Procfile` (可选)

### Render 部署
- [ ] `render.yaml`
- [ ] `start.sh` 启动脚本

### Fly.io 部署
- [ ] `fly.toml`
- [ ] 持久化卷配置

## 🔐 环境变量清单

### 后端必需
```env
ANTHROPIC_AUTH_TOKEN=your_claude_api_token
ANTHROPIC_BASE_URL=https://anyrouter.top
PORT=8000
ENVIRONMENT=production
```

### 前端必需
```env
VITE_API_BASE_URL=https://your-backend-url.com
```

## 💰 成本对比

| 方案 | 前端成本 | 后端成本 | 数据库成本 | 总计/月 |
|------|----------|----------|------------|---------|
| Vercel + Railway | 免费 | $5 | 免费 | $5 |
| Render | 免费 | 免费 | 免费 | $0 |
| Netlify + Fly.io | 免费 | $1.94 | 免费 | $1.94 |
| GitHub + Koyeb | 免费 | 免费 | 免费 | $0 |

## ⚡ 性能对比

| 方案 | 部署难度 | 性能 | 稳定性 | 维护成本 |
|------|----------|------|--------|----------|
| Vercel + Railway | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Render | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Netlify + Fly.io | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| GitHub + Koyeb | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

## 🎯 推荐方案

### 个人学习项目
**GitHub Pages + Koyeb** - 完全免费

### 小型商业项目  
**Vercel + Railway** - 性能最佳

### 快速原型验证
**Render** - 部署最简单
