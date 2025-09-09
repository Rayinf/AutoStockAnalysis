# GitHub Pages + Railway 部署指南

本指南将帮助您将股票分析项目部署到 GitHub Pages（前端）和 Railway（后端）。

## 部署架构

- **前端**: GitHub Pages（免费静态网站托管）
- **后端**: Railway（免费云平台，每月500小时免费额度）

## 前置要求

1. GitHub 账户
2. Railway 账户（使用 GitHub 登录）
3. 项目代码已推送到 GitHub 仓库

## 部署步骤

### 第一步：准备项目文件

确保项目根目录包含以下文件：
- `Dockerfile` - Railway 后端部署配置
- `railway.toml` - Railway 服务配置
- `requirements.txt` - Python 依赖
- `.github/workflows/deploy-frontend.yml` - GitHub Actions 工作流

### 第二步：部署后端到 Railway

1. 访问 [Railway](https://railway.app) 并使用 GitHub 登录
2. 点击 "New Project" → "Deploy from GitHub repo"
3. 选择您的项目仓库
4. Railway 会自动检测到 `Dockerfile` 并开始构建
5. 在项目设置中配置环境变量：
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_BASE_URL=https://api.openai.com/v1
   PORT=8000
   ```
6. 部署完成后，记录分配的域名（如：`https://your-app.railway.app`）

### 第三步：配置前端环境变量

1. 更新 `frontend/.env.production` 文件：
   ```
   VITE_API_BASE_URL=https://your-app.railway.app
   ```
   将 `your-app.railway.app` 替换为实际的 Railway 域名

### 第四步：部署前端到 GitHub Pages

1. 在 GitHub 仓库中，进入 Settings → Pages
2. 在 "Source" 下选择 "GitHub Actions"
3. 推送代码到 `main` 分支，GitHub Actions 会自动触发部署
4. 部署完成后，访问 `https://your-username.github.io/your-repo-name`

## 环境变量配置

### Railway 后端环境变量

在 Railway 项目设置中配置：

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
PORT=8000
PYTHON_VERSION=3.11
```

### GitHub Actions 环境变量

在 GitHub 仓库的 Settings → Secrets and variables → Actions 中添加：

```
VITE_API_BASE_URL=https://your-app.railway.app
```

## 验证部署

1. **后端验证**：访问 `https://your-app.railway.app/docs` 查看 API 文档
2. **前端验证**：访问 GitHub Pages 地址，确保页面正常加载
3. **功能验证**：在前端页面测试股票查询功能

## 故障排除

### 常见问题

1. **Railway 构建失败**
   - 检查 `Dockerfile` 语法
   - 确保 `requirements.txt` 包含所有依赖
   - 查看 Railway 构建日志

2. **GitHub Actions 失败**
   - 检查 `.github/workflows/deploy-frontend.yml` 配置
   - 确保环境变量正确设置
   - 查看 Actions 日志

3. **前端无法连接后端**
   - 确认 Railway 后端正常运行
   - 检查 CORS 配置
   - 验证 API 基础 URL 设置

### 日志查看

- **Railway 日志**：在 Railway 项目面板中查看部署和运行日志
- **GitHub Actions 日志**：在仓库的 Actions 标签页查看工作流日志

## 成本说明

- **GitHub Pages**: 完全免费
- **Railway**: 每月 500 小时免费额度，超出后按使用量计费

## 更新部署

- **后端更新**：推送代码到 GitHub，Railway 会自动重新部署
- **前端更新**：推送代码到 `main` 分支，GitHub Actions 会自动重新部署

## 支持

如遇到问题，请检查：
1. Railway 项目日志
2. GitHub Actions 工作流日志
3. 浏览器开发者工具的网络和控制台日志