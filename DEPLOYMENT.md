# 部署指南

本项目支持前端部署到 GitHub Pages，后端部署到 Railway 的完整部署方案。

## 🚀 快速部署

### 前提条件

1. GitHub 账号
2. Railway 账号 (https://railway.app)
3. 获取必要的 API Keys：
   - KIMI API Key
   - OpenAI API Key (可选)
   - DeepSeek API Key (可选)

### 第一步：准备代码仓库

1. **Fork 或创建 GitHub 仓库**
   ```bash
   # 如果是本地项目，初始化 Git 仓库
   git init
   git add .
   git commit -m "Initial commit"
   
   # 添加远程仓库并推送
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

2. **配置 GitHub Pages**
   - 进入仓库 Settings → Pages
   - Source 选择 "GitHub Actions"
   - 系统会自动检测到 `.github/workflows/deploy.yml` 配置

### 第二步：部署后端到 Railway

1. **登录 Railway**
   - 访问 https://railway.app
   - 使用 GitHub 账号登录

2. **创建新项目**
   - 点击 "New Project"
   - 选择 "Deploy from GitHub repo"
   - 选择你的仓库

3. **配置环境变量**
   在 Railway 项目设置中添加以下环境变量：
   ```
   ENVIRONMENT=production
   PORT=$PORT
   KIMI_API_KEY=your_kimi_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ALLOWED_ORIGINS=https://YOUR_USERNAME.github.io
   DATABASE_URL=sqlite:///./stock_data.db
   LOG_LEVEL=INFO
   ```

4. **部署配置**
   - Railway 会自动检测到 `Dockerfile` 和 `railway.toml`
   - 系统会自动开始构建和部署
   - 部署完成后会获得一个 Railway 域名，如：`https://your-app-name.up.railway.app`

### 第三步：配置前端环境变量

1. **在 GitHub 仓库中设置 Secrets**
   - 进入仓库 Settings → Secrets and variables → Actions
   - 添加以下 Secret：
     - `VITE_API_BASE_URL`: Railway 后端域名 (如：`https://your-app-name.up.railway.app`)

2. **更新 CORS 配置**
   - 在 Railway 环境变量中更新 `ALLOWED_ORIGINS`
   - 添加你的 GitHub Pages 域名：`https://YOUR_USERNAME.github.io`

### 第四步：触发部署

1. **推送代码触发前端部署**
   ```bash
   git add .
   git commit -m "Configure deployment"
   git push origin main
   ```

2. **检查部署状态**
   - GitHub Actions: 仓库 → Actions 标签页
   - Railway: Railway 项目面板

本文档详细说明了如何部署 Auto-GPT-Stock 项目的前端和后端。

## 🚀 快速部署概览

### 前端部署（GitHub Pages）
- **平台**: GitHub Pages
- **自动化**: GitHub Actions
- **访问地址**: `https://yourusername.github.io/AutoStockAnalysis/`

### 后端部署（Railway）
- **平台**: Railway
- **容器化**: Docker
- **访问地址**: `https://your-app-name.railway.app`

## 📋 部署前准备

### 1. 环境变量配置

#### 前端环境变量
复制 `frontend/.env.example` 为 `frontend/.env.local`：
```bash
cp frontend/.env.example frontend/.env.local
```

编辑 `.env.local` 文件：
```env
# 生产环境API地址
VITE_API_BASE_URL=https://your-backend-domain.railway.app
```

#### 后端环境变量
复制 `backend/.env.example` 为 `backend/.env`：
```bash
cp backend/.env.example backend/.env
```

编辑 `.env` 文件：
```env
ENVIRONMENT=production
PORT=8000
ALLOWED_ORIGINS=https://yourusername.github.io
OPENAI_API_KEY=your_openai_api_key_here
KIMI_API_KEY=your_kimi_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 2. API Keys 获取

- **OpenAI API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Kimi API Key**: [Moonshot AI](https://platform.moonshot.cn/)
- **DeepSeek API Key**: [DeepSeek Platform](https://platform.deepseek.com/)

## 🖥️ 后端部署（Railway）

### 步骤 1: 准备 Railway 账户
1. 访问 [Railway](https://railway.app/)
2. 使用 GitHub 账户登录
3. 连接你的 GitHub 仓库

### 步骤 2: 创建新项目
1. 点击 "New Project"
2. 选择 "Deploy from GitHub repo"
3. 选择你的 `AutoStockAnalysis` 仓库
4. 选择 `backend` 目录作为根目录

### 步骤 3: 配置环境变量
在 Railway 项目设置中添加以下环境变量：
```
ENVIRONMENT=production
PORT=8000
ALLOWED_ORIGINS=https://yourusername.github.io
OPENAI_API_KEY=your_openai_api_key_here
KIMI_API_KEY=your_kimi_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 步骤 4: 部署
1. Railway 会自动检测 `Dockerfile` 并开始构建
2. 等待部署完成
3. 获取分配的域名（如：`https://your-app-name.railway.app`）

## 🌐 前端部署（GitHub Pages）

### 步骤 1: 更新前端配置
编辑 `frontend/.env.local`，设置后端 API 地址：
```env
VITE_API_BASE_URL=https://your-backend-domain.railway.app
```

### 步骤 2: 推送代码
```bash
git add .
git commit -m "Configure for production deployment"
git push origin main
```

### 步骤 3: 启用 GitHub Pages
1. 进入 GitHub 仓库设置
2. 找到 "Pages" 部分
3. Source 选择 "GitHub Actions"
4. GitHub Actions 工作流会自动运行

### 步骤 4: 访问应用
部署完成后，访问：`https://yourusername.github.io/AutoStockAnalysis/`

## 🔧 本地开发环境

### 后端启动
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 前端启动
```bash
cd frontend
npm install
npm run dev
```

## 📊 监控和维护

### 健康检查
- 后端健康检查：`https://your-backend-domain.railway.app/health`
- 前端访问检查：`https://yourusername.github.io/AutoStockAnalysis/`

### 日志查看
- **Railway**: 在 Railway 控制台查看后端日志
- **GitHub Actions**: 在 GitHub Actions 页面查看构建日志

### 常见问题

#### 1. CORS 错误
确保后端 `ALLOWED_ORIGINS` 环境变量包含前端域名。

#### 2. API 连接失败
检查前端 `VITE_API_BASE_URL` 是否正确设置为后端地址。

#### 3. 构建失败
检查 GitHub Actions 日志，通常是依赖安装或环境变量配置问题。

## 🔄 更新部署

### 后端更新
推送代码到 `main` 分支，Railway 会自动重新部署：
```bash
git push origin main
```

### 前端更新
推送代码到 `main` 分支，GitHub Actions 会自动重新构建和部署：
```bash
git push origin main
```

## 📞 技术支持

如果在部署过程中遇到问题，请：
1. 检查环境变量配置
2. 查看相关平台的日志
3. 确认 API Keys 有效性
4. 检查网络连接和域名解析

---

**注意**: 请确保妥善保管 API Keys，不要将其提交到公共仓库中。