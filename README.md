# 🚀 AutoStockAnalysis - 智能股票分析系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-19.1+-61DAFB.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-brightgreen.svg)](https://rayinf.github.io/AutoStockAnalysis/)

> 🎯 **让股票分析变得简单智能** - 基于大语言模型的全栈股票分析平台，集成Web界面、智能策略、实时数据分析于一体。

## ✨ 核心特性

### 🧠 智能分析引擎
- **多模型支持**: OpenAI GPT-4、DeepSeek、Qwen、Claude等主流LLM
- **ReAct智能推理**: 自然语言交互，自动选择最佳分析策略
- **实时数据获取**: 基于Akshare接口，覆盖股票、基金、期货全市场数据

### 📊 现代化Web界面
- **响应式设计**: 基于React + TypeScript + Tailwind CSS
- **交互式图表**: ECharts可视化，支持K线、技术指标、趋势分析
- **多策略模式**: 固定策略、LLM分析、量化回测一键切换

### 🎯 专业分析功能
- **技术指标**: MA、RSI、MACD、BOLL、OBV、ATR等20+指标
- **智能预测**: 开盘价预测、涨跌概率分析、价格区间预测
- **风险管理**: VaR计算、最大回撤、夏普比率等风险指标
- **投资组合**: 多股票组合管理、收益率分析、资产配置优化

### 🔧 技术架构
- **前端**: React 19 + TypeScript + Vite + Tailwind CSS
- **后端**: Python + FastAPI + LangChain
- **数据源**: Akshare金融数据接口
- **部署**: 支持GitHub Pages + Railway云部署

---

## 🎬 功能演示

### 💬 智能对话分析
```bash
🤖 我是智能金融分析助手，可以提供股票、基金、期货的信息，有什么可以帮您？
👨 分析一下中国银行的技术指标
🤖 中国银行(601988)当前价格5.47元，RSI指标显示超卖状态，MACD金叉信号...

👨 帮我预测明天开盘价
🤖 基于技术分析和市场情绪，预测明日开盘价区间为5.42-5.52元，上涨概率65%...
```

### 📈 Web界面功能
- **实时行情**: K线图表、技术指标叠加显示
- **策略回测**: 多种量化策略，可视化收益曲线
- **智能预测**: AI驱动的价格预测和风险评估
- **投资组合**: 多股票组合管理和优化建议

## 🚀 快速开始

### 📋 环境要求
- **Python**: 3.11+
- **Node.js**: 20.19+ 或 22.12+
- **操作系统**: Windows / macOS / Linux

### ⚡ 一键部署

#### 方式一：本地开发
```bash
# 1. 克隆项目
git clone https://github.com/Rayinf/AutoStockAnalysis.git
cd AutoStockAnalysis

# 2. 后端环境配置
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install poetry
poetry install

# 3. 前端环境配置
cd frontend
npm install

# 4. 环境变量配置
cp demo.env .env
# 编辑 .env 文件，设置你的 API Key
```

#### 方式二：在线体验
🌐 **[立即体验 →](https://rayinf.github.io/AutoStockAnalysis/)**

支持多种LLM提供商，在 `.env` 文件中配置：

```bash
# OpenAI (推荐)
OPENAI_API_KEY=sk-your-openai-key
MODEL_NAME=gpt-4-turbo

# DeepSeek (性价比高)
DEEPSEEK_API_KEY=sk-your-deepseek-key
MODEL_NAME=deepseek-v3

# 其他支持的模型
# MODEL_NAME=claude-3-haiku-20240307
# MODEL_NAME=qwen2.5-32b-instruct
```

### 🎮 运行方式

#### 命令行模式
```bash
python main.py
# 🤖 我是智能金融分析助手，可以提供股票、基金、期货的信息，有什么可以帮您？
```

#### Web界面模式
```bash
# 启动后端API
uvicorn web.api.main:app --reload --port 8000

# 启动前端界面
cd frontend
npm run dev
# 访问 http://localhost:5173
```

## 📊 功能模块

### 🔍 智能分析模式
| 模式 | 描述 | 适用场景 |
|------|------|----------|
| **固定策略** | 预设技术指标分析 | 快速技术面分析 |
| **LLM分析** | AI智能解读市场 | 深度基本面分析 |
| **量化回测** | 策略历史表现 | 策略验证优化 |

### 📈 技术指标支持
- **趋势指标**: MA5/20/60、EMA、MACD
- **震荡指标**: RSI、KDJ、威廉指标
- **成交量**: OBV、成交量比率
- **波动率**: ATR、布林带
- **实时数据**: 涨跌幅排行、热门股票

### 🎯 预测功能
- **开盘价预测**: 基于历史数据和技术指标
- **涨跌概率**: 机器学习模型预测
- **价格区间**: 支撑位和阻力位分析
- **风险评估**: VaR、最大回撤计算

## 🧪 模型性能对比

| 模型 | 股票分析 | 基金查询 | 期货数据 | 推荐指数 | 成本效益 |
|------|----------|----------|----------|----------|----------|
| **GPT-4 Turbo** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🔥🔥🔥🔥 | 💰💰💰 |
| **DeepSeek V3** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔥🔥🔥🔥🔥 | 💰💰💰💰💰 |
| **Claude Haiku** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 🔥🔥 | 💰💰💰💰 |
| **Qwen 2.5** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 🔥🔥🔥 | 💰💰💰💰 |
| **O1-Mini** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 🔥🔥🔥 | 💰💰 |

> 💡 **推荐**: DeepSeek V3 性价比最高，GPT-4 Turbo 功能最全面

## 🚀 部署指南

### GitHub Pages + Railway 部署

```bash
# 1. Fork 本项目到你的 GitHub
# 2. 在 Railway 创建新项目
# 3. 连接 GitHub 仓库
# 4. 配置环境变量
# 5. 自动部署完成
```

详细部署文档: [DEPLOYMENT_GITHUB_RAILWAY.md](./DEPLOYMENT_GITHUB_RAILWAY.md)

### Docker 部署

```bash
# 构建镜像
docker build -t autostock-analysis .

# 运行容器
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key autostock-analysis
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 🐛 问题反馈
- 发现Bug？[提交Issue](https://github.com/Rayinf/AutoStockAnalysis/issues)
- 功能建议？[讨论区交流](https://github.com/Rayinf/AutoStockAnalysis/discussions)

### 💻 代码贡献
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 📝 文档完善
- 改进README文档
- 添加代码注释
- 编写使用教程

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

- [Akshare](https://akshare.akfamily.xyz/) - 优秀的金融数据接口
- [LangChain](https://langchain.com/) - 强大的LLM应用框架
- [React](https://reactjs.org/) + [ECharts](https://echarts.apache.org/) - 现代化前端技术栈

## 📞 联系我们

- 📧 Email: [your-email@example.com](mailto:your-email@example.com)
- 💬 微信群: 扫码加入技术交流群
- 🐦 Twitter: [@YourTwitter](https://twitter.com/YourTwitter)

---

⭐ **如果这个项目对你有帮助，请给我们一个Star！** ⭐