import React, { useState } from 'react'

interface StrategyConfig {
  buy_threshold: number
  sell_threshold: number
  max_positions: number
  initial_capital: number
  transaction_cost: number
  position_size: number
}

interface PerformanceMetrics {
  total_return: number
  annualized_return: number
  max_drawdown: number
  sharpe_ratio: number
  win_rate: number
  profit_loss_ratio: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  avg_holding_period: number
  volatility: number
  calmar_ratio: number
}

interface BacktestResult {
  strategy_config: StrategyConfig
  performance_metrics: PerformanceMetrics
  total_trades: number
  final_portfolio_value: number
  portfolio_curve: Array<[string, number]>
  trades: Trade[]
  current_positions: Record<string, unknown>
  error?: string
  // 自动生成历史数据的相关字段
  auto_generated_data?: boolean
  generated_records?: number
  generated_symbols?: string[]
  message?: string
}

interface OptimizationResult {
  buy_threshold: number
  sell_threshold: number
  total_return: number
  sharpe_ratio: number
  score: number
}

interface Trade {
  date: string
  symbol: string
  action: 'buy' | 'sell'
  price: number
  quantity: number
  predicted_prob: number
  actual_return?: number
}



// 股票池管理移至外部StockPoolPanel组件

interface StrategyPanelProps {
  symbol?: string
  selectedSymbols: string
}

export const StrategyPanel: React.FC<StrategyPanelProps> = ({ selectedSymbols }) => {
  const [config, setConfig] = useState<StrategyConfig>({
    buy_threshold: 0.6,   // 文档标准参数
    sell_threshold: 0.4,  // 文档标准参数
    max_positions: 10,    // 文档标准参数
    initial_capital: 100000,
    transaction_cost: 0.002,
    position_size: 0.1    // 文档标准参数
  })
  
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [optimization, setOptimization] = useState<{
    best_config: StrategyConfig
    best_performance: PerformanceMetrics
    optimization_results: OptimizationResult[]
  } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'backtest' | 'optimize'>('backtest')
  const [useCustomDateRange, setUseCustomDateRange] = useState(false)
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [showTradeModal, setShowTradeModal] = useState(false)
  const [showChartModal, setShowChartModal] = useState(false)
  const [tooltip, setTooltip] = useState<{
    visible: boolean,
    x: number,
    y: number,
    content: {
      symbol: string,
      trade: {
        date: string,
        action: string,
        price: number,
        quantity: number,
        profit: number,
        cumulativeProfit: number,
        predicted_prob?: number
      },
      index: number
    } | null
  }>({ visible: false, x: 0, y: 0, content: null })
  const [tradeFilter, setTradeFilter] = useState<'all' | 'buy' | 'sell'>('all')
  const [tradeSortBy, setTradeSortBy] = useState<'date' | 'profit' | 'amount'>('date')
  
  // 策略模式选择 (集成到基础策略中)
  const [strategyType, setStrategyType] = useState<'original' | 'enhanced'>('original')
  
  // 计算回测天数的函数
  const calculateBacktestDays = (): number => {
    if (useCustomDateRange && startDate && endDate) {
      const start = new Date(startDate)
      const end = new Date(endDate)
      const diffTime = Math.abs(end.getTime() - start.getTime())
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
      return diffDays
    }
    // 如果不使用自定义范围，默认使用1年
    return 365
  }
  
  // 股票池管理移至外部StockPoolPanel组件

  const runBacktest = async () => {
    // 防止重复调用
    if (loading) {
      console.log('回测已在进行中，忽略重复调用')
      return
    }
    
    setLoading(true)
    setError(null)
    console.log(`开始${strategyType === 'enhanced' ? '增强策略' : '原始策略'}回测...`)
    
    try {
      // 统一使用基础策略API，通过参数区分模式
      const params = new URLSearchParams({
        buy_threshold: config.buy_threshold.toString(),
        sell_threshold: config.sell_threshold.toString(),
        max_positions: config.max_positions.toString(),
        initial_capital: config.initial_capital.toString(),
        transaction_cost: config.transaction_cost.toString(),
        position_size: config.position_size.toString(),
        strategy_mode: strategyType === 'enhanced' ? 'high_return' : 'standard'  // 转换策略类型为模式参数
      })
      
      // 添加股票池参数
      if (selectedSymbols && selectedSymbols.trim()) {
        params.append('symbols', selectedSymbols.trim())
      }
      
      // 添加日期范围参数
      if (useCustomDateRange && startDate && endDate) {
        params.append('start_date', startDate)
        params.append('end_date', endDate)
      }
      
      // 如果是增强策略，添加额外的时间参数
      if (strategyType === 'enhanced') {
        const days = calculateBacktestDays()
        params.append('days', days.toString())
      }
      
      const response = await fetch(`http://127.0.0.1:8000/api/strategy/run_backtest?${params}`, {
        method: 'POST'
      })
      
      if (!response.ok) {
        throw new Error('回测请求失败')
      }
      
      const data = await response.json()
      
      if (data.error || !data.success) {
        setError(data.error || '回测失败')
      } else {
        // 根据策略模式处理不同的数据格式
        if (strategyType === 'enhanced') {
          // 增强策略：转换数据格式以兼容前端显示
          const highReturnResult = data.results
          const convertedResult = {
            strategy_config: config, // 使用当前配置
            performance_metrics: {
              total_return: highReturnResult.total_return || 0,
              annualized_return: highReturnResult.annualized_return || 0,
              max_drawdown: highReturnResult.max_drawdown || 0,
              sharpe_ratio: highReturnResult.sharpe_ratio || 0,
              win_rate: highReturnResult.win_rate || 0,
              profit_loss_ratio: 1.0, // 默认值
              total_trades: highReturnResult.trade_count || 0,
              winning_trades: Math.floor((highReturnResult.trade_count || 0) * (highReturnResult.win_rate || 0)),
              losing_trades: Math.floor((highReturnResult.trade_count || 0) * (1 - (highReturnResult.win_rate || 0))),
              avg_holding_period: 10, // 默认值
              volatility: 0.2, // 默认值
              calmar_ratio: (highReturnResult.annualized_return || 0) / Math.max(highReturnResult.max_drawdown || 0.01, 0.01)
            },
            total_trades: highReturnResult.trade_count || 0,
            final_portfolio_value: highReturnResult.final_portfolio_value || 100000,
            portfolio_curve: highReturnResult.portfolio_values || [],
            trades: highReturnResult.trades || [],
            current_positions: {},
            strategy_name: '🚀 增强策略'
          }
          setResult(convertedResult)
        } else {
          // 原始策略：直接使用返回的数据
        setResult(data)
        }
        
        // 显示成功提示
        const mode = strategyType === 'enhanced' ? '增强策略' : '原始策略'
        console.log(`✅ ${mode}回测完成`)
        
        // 显示自动生成数据的提示
        if (data.auto_generated_data) {
          const message = `✅ ${data.message || `自动生成了 ${data.generated_records} 条历史数据并完成回测`}`
          console.log(message)
          
          // 显示临时提示
          const alertDiv = document.createElement('div')
          alertDiv.className = 'fixed top-4 right-4 p-4 bg-green-600/20 border border-green-600/50 rounded text-green-300 text-sm z-50 max-w-md'
          alertDiv.innerHTML = `
            <div class="flex items-start gap-2">
              <span>🤖</span>
              <div>
                <div class="font-medium">自动生成历史数据</div>
                <div class="text-xs mt-1">${message}</div>
                <div class="text-xs text-green-400 mt-1">
                  涉及股票: ${data.generated_symbols?.join(', ') || '未知'}
                </div>
              </div>
            </div>
          `
          document.body.appendChild(alertDiv)
          
          setTimeout(() => {
            if (document.body.contains(alertDiv)) {
              document.body.removeChild(alertDiv)
            }
          }, 8000) // 8秒后自动消失
        }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '回测失败')
    } finally {
      setLoading(false)
    }
  }

  const runOptimization = async () => {
    // 防止重复调用
    if (loading) {
      console.log('优化已在进行中，忽略重复调用')
      return
    }
    
    setLoading(true)
    setError(null)
    console.log('开始策略优化...')
    
    try {
      // 构建查询参数，与回测保持一致
      const params = new URLSearchParams({
        symbols: selectedSymbols || '600519,000001'
      })
      
      if (useCustomDateRange && startDate && endDate) {
        params.append('start_date', startDate)
        params.append('end_date', endDate)
      }
      
      const response = await fetch(`http://127.0.0.1:8000/api/strategy/optimize?${params}`, {
        method: 'POST'
      })
      
      if (!response.ok) {
        throw new Error('优化请求失败')
      }
      
      const data = await response.json()
      
      if (data.error) {
        setError(data.error)
      } else {
        setOptimization(data)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '优化失败')
    } finally {
      setLoading(false)
    }
  }

  // 股票池数据加载移至外部组件

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`
  const formatCurrency = (value: number) => `¥${value.toLocaleString()}`

  // 计算专业金融图表数据
  const calculateFinancialChartData = (trades: Trade[], initialCapital: number = 100000) => {
    console.log('🔍 开始计算金融图表数据 v2.1，交易总数:', trades.length)
    console.log('🔍 交易数据样本:', trades.slice(0, 2))  // 查看前两笔交易的数据结构
    
    // 按日期排序交易
    const sortedTrades = [...trades].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    
    if (sortedTrades.length === 0) {
      return { equityCurve: [], stockData: {}, finalEquity: initialCapital, totalReturn: 0 }
    }
    
    // 获取时间范围
    const startDate = new Date(sortedTrades[0].date)
    const endDate = new Date(sortedTrades[sortedTrades.length - 1].date)
    
    // 生成完整的日期序列
    const dateRange: string[] = []
    const currentDate = new Date(startDate)
    while (currentDate <= endDate) {
      dateRange.push(currentDate.toISOString().split('T')[0])
      currentDate.setDate(currentDate.getDate() + 1)
    }
    
    // 权益曲线数据
    const equityCurve: { date: string, equity: number, drawdown: number, maxEquity: number }[] = []
    let currentEquity = initialCapital
    let maxEquity = initialCapital
    const positions: { [symbol: string]: Array<{price: number, quantity: number, date: string}> } = {}
    
    // 每只股票的详细数据
    const stockData: { [symbol: string]: {
      trades: Array<{date: string, action: string, price: number, quantity: number, profit: number, cumulativeProfit: number, predicted_prob?: number}>,
      priceData: Array<{date: string, open: number, close: number, high: number, low: number, volume?: number}>,
      cumulativeProfit: number,
      maxProfit: number,
      maxDrawdown: number,
      winRate: number,
      totalTrades: number,
      priceRange: {min: number, max: number}
    }} = {}
    
    // 按日期分组交易
    const tradesByDate: { [date: string]: Trade[] } = {}
    sortedTrades.forEach(trade => {
      if (!tradesByDate[trade.date]) {
        tradesByDate[trade.date] = []
      }
      tradesByDate[trade.date].push(trade)
    })
    
    // 遍历每个日期
    dateRange.forEach(date => {
      const dayTrades = tradesByDate[date] || []
      
      // 处理当天的交易
      dayTrades.forEach(trade => {
        // 初始化股票数据
        if (!stockData[trade.symbol]) {
          stockData[trade.symbol] = {
            trades: [],
            priceData: [],
            cumulativeProfit: 0,
            maxProfit: 0,
            maxDrawdown: 0,
            winRate: 0,
            totalTrades: 0,
            priceRange: {min: Number.MAX_VALUE, max: Number.MIN_VALUE}
          }
          positions[trade.symbol] = []
        }
        
        const data = stockData[trade.symbol]
        let profit = 0
        
        if (trade.action === 'buy') {
          // 买入：记录持仓
          positions[trade.symbol].push({
            price: trade.price,
            quantity: trade.quantity,
            date: trade.date
          })
          currentEquity -= trade.price * trade.quantity // 减少现金
          profit = 0 // 买入时不计算盈亏
        } else if (trade.action === 'sell') {
          // 卖出：计算盈亏
          if (trade.actual_return !== undefined && positions[trade.symbol].length > 0) {
            const position = positions[trade.symbol].shift()
            if (position) {
              const buyValue = position.price * position.quantity
              profit = buyValue * trade.actual_return
              // 权益计算：返还买入成本 + 盈亏
              currentEquity += buyValue + profit
              data.totalTrades++
            }
          } else if (positions[trade.symbol].length > 0) {
            const position = positions[trade.symbol].shift()
            if (position) {
              const buyValue = position.price * position.quantity
              const sellValue = trade.price * trade.quantity
              profit = sellValue - buyValue
              // 权益计算：返还卖出收入
              currentEquity += sellValue
              data.totalTrades++
            }
          }
        }
        
        // 更新股票数据
        data.cumulativeProfit += profit
        data.maxProfit = Math.max(data.maxProfit, data.cumulativeProfit)
        data.maxDrawdown = Math.min(data.maxDrawdown, data.cumulativeProfit - data.maxProfit)
        
        data.trades.push({
          date: trade.date,
          action: trade.action,
          price: trade.price,
          quantity: trade.quantity,
          profit: profit,
          cumulativeProfit: data.cumulativeProfit,
          predicted_prob: trade.predicted_prob
        })
        
        // 更新价格范围
        data.priceRange.min = Math.min(data.priceRange.min, trade.price)
        data.priceRange.max = Math.max(data.priceRange.max, trade.price)
        
        data.priceData.push({
          date: trade.date,
          open: trade.price, // 简化处理，实际应该获取OHLC数据
          close: trade.price,
          high: trade.price,
          low: trade.price,
          volume: trade.quantity
        })
      })
      
      // 更新权益曲线（为每个日期都添加一个数据点）
      // 先更新历史最高权益
      maxEquity = Math.max(maxEquity, currentEquity)
      
      // 计算回撤百分比（当前权益低于历史最高时为负数）
      const drawdown = currentEquity < maxEquity ? ((currentEquity - maxEquity) / maxEquity) * 100 : 0
      
      // 🔍 调试信息：验证回撤计算
      if (Math.random() < 0.01) { // 1%概率输出调试信息
        console.log(`🔍 回撤计算调试: currentEquity=${currentEquity.toFixed(2)}, maxEquity=${maxEquity.toFixed(2)}, drawdown=${drawdown.toFixed(2)}%`)
      }
      
      equityCurve.push({
        date: date,
        equity: currentEquity,
        drawdown: drawdown,
        maxEquity: maxEquity
      })
    })
    
    // 计算胜率
    Object.keys(stockData).forEach(symbol => {
      const data = stockData[symbol]
      const winningTrades = data.trades.filter(t => t.profit > 0).length
      data.winRate = data.totalTrades > 0 ? (winningTrades / data.totalTrades) * 100 : 0
    })
    
    // 验证计算一致性
    const totalStockProfit = Object.values(stockData).reduce((sum, data) => sum + data.cumulativeProfit, 0)
    const totalPortfolioProfit = currentEquity - initialCapital
    
    console.log('📊 权益曲线数据点:', equityCurve.length)
    console.log('📈 股票数据:', Object.keys(stockData).length)
    console.log('📅 日期范围:', startDate.toISOString().split('T')[0], '至', endDate.toISOString().split('T')[0])
    console.log('💰 权益变化:', `${initialCapital.toFixed(2)} → ${currentEquity.toFixed(2)}`)
    console.log('📊 权益曲线样本:', equityCurve.slice(0, 3).map(p => ({ date: p.date, equity: p.equity.toFixed(2), drawdown: p.drawdown.toFixed(2) })))
    console.log('🔍 验证计算一致性:')
    console.log('   个股累计盈亏总和:', totalStockProfit.toFixed(2))
    console.log('   组合总盈亏:', totalPortfolioProfit.toFixed(2))
    console.log('   差异:', Math.abs(totalStockProfit - totalPortfolioProfit).toFixed(2))
    
    // 输出各股票盈亏明细
    Object.entries(stockData).forEach(([symbol, data]) => {
      console.log(`   ${symbol}: ¥${data.cumulativeProfit.toFixed(2)} (${data.totalTrades}笔交易)`)
    })
    
    return { equityCurve, stockData, finalEquity: currentEquity, totalReturn: ((currentEquity - initialCapital) / initialCapital) * 100 }
  }

  // 处理交易数据的筛选和排序
  const getFilteredAndSortedTrades = (trades: Trade[]) => {
    let filteredTrades = trades

    // 筛选
    if (tradeFilter !== 'all') {
      filteredTrades = trades.filter(trade => trade.action === tradeFilter)
    }

    // 排序
    filteredTrades.sort((a, b) => {
      switch (tradeSortBy) {
        case 'date':
          return new Date(b.date).getTime() - new Date(a.date).getTime() // 最新的在前
        case 'profit': {
          const aProfit = a.actual_return || 0
          const bProfit = b.actual_return || 0
          return bProfit - aProfit // 收益高的在前
        }
        case 'amount': {
          const aAmount = a.price * a.quantity
          const bAmount = b.price * b.quantity
          return bAmount - aAmount // 金额大的在前
        }
        default:
          return 0
      }
    })

    return filteredTrades
  }

  // 导出交易数据到CSV
  const exportTradesToCSV = (trades: Trade[]) => {
    try {
      // CSV头部
      const headers = [
        '序号',
        '交易日期', 
        '股票代码',
        '交易类型',
        '交易价格',
        '交易数量',
        '交易金额',
        '预测概率',
        '实际收益率',
        '收益金额'
      ]

      // 构建CSV内容
      const csvContent = [
        headers.join(','),
        ...trades.map((trade, index) => {
          const amount = (trade.price && trade.quantity) ? trade.price * trade.quantity : 0
          const profitAmount = trade.actual_return ? (amount * trade.actual_return) : 0
          
          return [
            index + 1,
            trade.date,
            trade.symbol,
            trade.action === 'buy' ? '买入' : '卖出',
            trade.price ? trade.price.toFixed(2) : 'N/A',
            trade.quantity || 'N/A',
            amount ? amount.toFixed(2) : 'N/A',
            trade.predicted_prob ? `${(trade.predicted_prob * 100).toFixed(2)}%` : 'N/A',
            trade.actual_return ? `${(trade.actual_return * 100).toFixed(2)}%` : '--',
            profitAmount ? profitAmount.toFixed(2) : '--'
          ].join(',')
        })
      ].join('\n')

      // 添加BOM以支持中文
      const BOM = '\uFEFF'
      const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' })
      
      // 创建下载链接
      const link = document.createElement('a')
      const url = URL.createObjectURL(blob)
      link.setAttribute('href', url)
      
      // 生成文件名（包含时间戳）
      const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
      const filename = `交易详情_${timestamp}.csv`
      link.setAttribute('download', filename)
      
      // 触发下载
      link.style.visibility = 'hidden'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      console.log(`成功导出 ${trades.length} 笔交易记录到 ${filename}`)
    } catch (error) {
      console.error('导出CSV失败:', error)
      alert('导出失败，请重试')
    }
  }

  // 导出完整策略报告
  const exportStrategyReport = (result: BacktestResult) => {
    try {
      const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
      
      // 构建完整报告内容
      const reportContent = [
        '# 量化交易策略回测报告',
        `生成时间: ${new Date().toLocaleString('zh-CN')}`,
        '',
        '## 策略配置',
        `买入阈值: ${result.strategy_config.buy_threshold}`,
        `卖出阈值: ${result.strategy_config.sell_threshold}`,
        `最大持仓数: ${result.strategy_config.max_positions}`,
        `初始资金: ¥${result.strategy_config.initial_capital.toLocaleString()}`,
        `单股仓位比例: ${(result.strategy_config.position_size * 100).toFixed(1)}%`,
        `交易成本: ${(result.strategy_config.transaction_cost * 100).toFixed(2)}%`,
        '',
        '## 策略表现',
        `总收益率: ${(result.performance_metrics.total_return * 100).toFixed(2)}%`,
        `年化收益率: ${(result.performance_metrics.annualized_return * 100).toFixed(2)}%`,
        `最大回撤: ${(result.performance_metrics.max_drawdown * 100).toFixed(2)}%`,
        `夏普比率: ${result.performance_metrics.sharpe_ratio.toFixed(3)}`,
        `卡尔玛比率: ${result.performance_metrics.calmar_ratio.toFixed(3)}`,
        `年化波动率: ${(result.performance_metrics.volatility * 100).toFixed(2)}%`,
        '',
        '## 交易统计',
        `总交易数: ${result.performance_metrics.total_trades}`,
        `盈利交易: ${result.performance_metrics.winning_trades}`,
        `亏损交易: ${result.performance_metrics.losing_trades}`,
        `胜率: ${(result.performance_metrics.win_rate * 100).toFixed(2)}%`,
        `盈亏比: ${result.performance_metrics.profit_loss_ratio.toFixed(2)}`,
        `平均持仓期: ${result.performance_metrics.avg_holding_period.toFixed(1)} 天`,
        '',
        '## 资金状况',
        `初始资金: ¥${result.strategy_config.initial_capital.toLocaleString()}`,
        `最终资产: ¥${result.final_portfolio_value.toLocaleString()}`,
        `净收益: ¥${(result.final_portfolio_value - result.strategy_config.initial_capital).toLocaleString()}`,
        '',
        '## 详细交易记录',
        '序号,交易日期,股票代码,交易类型,交易价格,交易数量,交易金额,预测概率,实际收益率,收益金额',
        ...result.trades.map((trade: Trade, index: number) => {
          const amount = (trade.price && trade.quantity) ? trade.price * trade.quantity : 0
          const profitAmount = trade.actual_return ? (amount * trade.actual_return) : 0
          
          return [
            index + 1,
            trade.date,
            trade.symbol,
            trade.action === 'buy' ? '买入' : '卖出',
            trade.price ? trade.price.toFixed(2) : 'N/A',
            trade.quantity || 'N/A',
            amount ? amount.toFixed(2) : 'N/A',
            trade.predicted_prob ? `${(trade.predicted_prob * 100).toFixed(2)}%` : 'N/A',
            trade.actual_return ? `${(trade.actual_return * 100).toFixed(2)}%` : '--',
            profitAmount ? profitAmount.toFixed(2) : '--'
          ].join(',')
        })
      ].join('\n')

      // 创建下载
      const BOM = '\uFEFF'
      const blob = new Blob([BOM + reportContent], { type: 'text/plain;charset=utf-8;' })
      const link = document.createElement('a')
      const url = URL.createObjectURL(blob)
      
      link.setAttribute('href', url)
      link.setAttribute('download', `策略回测报告_${timestamp}.txt`)
      link.style.visibility = 'hidden'
      
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      console.log(`成功导出完整策略报告`)
    } catch (error) {
      console.error('导出报告失败:', error)
      alert('导出报告失败，请重试')
    }
  }

  return (
    <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
      <div className="text-lg font-medium text-gray-300 mb-4">📈 策略回测系统</div>
      
      {/* 当前选中股票显示 */}
      {selectedSymbols && (
        <div className="mb-4 p-3 bg-blue-600/10 border border-blue-600/30 rounded">
          <div className="text-sm font-medium text-blue-300 mb-2">🎯 当前股票池</div>
          <div className="flex flex-wrap gap-1">
            {selectedSymbols.split(',').map(s => s.trim()).filter(s => s).map(symbol => (
              <span key={symbol} className="px-2 py-1 bg-blue-600/20 text-blue-300 rounded text-xs">
                {symbol}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* 标签页 */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setActiveTab('backtest')}
          className={`px-3 py-2 text-sm rounded border transition-colors ${
            activeTab === 'backtest'
              ? 'bg-slate-700 text-slate-100 border-slate-600'
              : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700'
          }`}
        >
          🎯 单次回测
        </button>
        <button
          onClick={() => setActiveTab('optimize')}
          className={`px-3 py-2 text-sm rounded border transition-colors ${
            activeTab === 'optimize'
              ? 'bg-slate-700 text-slate-100 border-slate-600'
              : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700'
          }`}
        >
          ⚡ 参数优化
        </button>

      </div>

      {activeTab === 'backtest' && (
        <>


          {/* 策略配置 */}
          <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-3">📊 策略配置</div>
            
            {/* 策略选择 */}
            <div className="mb-4">
              <label className="block text-gray-400 mb-2 text-xs">策略类型</label>
              <select
                value={strategyType}
                onChange={(e) => setStrategyType(e.target.value as 'original' | 'enhanced')}
                className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 text-gray-300 focus:border-blue-500 focus:outline-none"
              >
                <option value="original">📈 原始策略 - 经典阈值交易</option>
                <option value="enhanced">🚀 增强策略 - Kelly公式+风险调整</option>
              </select>
            </div>
            

            
            {/* 参数配置 */}
            {strategyType === 'original' && (
            
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-gray-400 mb-1">买入阈值</label>
                <input
                  type="number"
                  min="0.5"
                  max="1"
                  step="0.05"
                  value={config.buy_threshold}
                  onChange={(e) => setConfig({...config, buy_threshold: parseFloat(e.target.value)})}
                  className="w-full bg-[#121826] border border-gray-700 rounded px-2 py-1"
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-1">卖出阈值</label>
                <input
                  type="number"
                  min="0"
                  max="0.5"
                  step="0.05"
                  value={config.sell_threshold}
                  onChange={(e) => setConfig({...config, sell_threshold: parseFloat(e.target.value)})}
                  className="w-full bg-[#121826] border border-gray-700 rounded px-2 py-1"
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-1">最大持仓数</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={config.max_positions}
                  onChange={(e) => setConfig({...config, max_positions: parseInt(e.target.value)})}
                  className="w-full bg-[#121826] border border-gray-700 rounded px-2 py-1"
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-1">初始资金</label>
                <input
                  type="number"
                  min="10000"
                  step="10000"
                  value={config.initial_capital}
                  onChange={(e) => setConfig({...config, initial_capital: parseFloat(e.target.value)})}
                  className="w-full bg-[#121826] border border-gray-700 rounded px-2 py-1"
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-1">交易成本</label>
                <input
                  type="number"
                  min="0"
                  max="0.01"
                  step="0.001"
                  value={config.transaction_cost}
                  onChange={(e) => setConfig({...config, transaction_cost: parseFloat(e.target.value)})}
                  className="w-full bg-[#121826] border border-gray-700 rounded px-2 py-1"
                />
              </div>
              
              <div>
                <label className="block text-gray-400 mb-1">单股仓位比例</label>
                <input
                  type="number"
                  min="0.05"
                  max="0.5"
                  step="0.05"
                  value={config.position_size}
                  onChange={(e) => setConfig({...config, position_size: parseFloat(e.target.value)})}
                  className="w-full bg-[#121826] border border-gray-700 rounded px-2 py-1"
                />
              </div>
            </div>
            )}
          </div>

          {/* 回测时间范围 */}
          <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-3">
              📅 回测时间范围

            </div>
            
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => setUseCustomDateRange(false)}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  !useCustomDateRange 
                    ? 'bg-blue-600/30 text-blue-300 border border-blue-600/50' 
                    : 'bg-slate-700/30 text-slate-400 border border-slate-600/50'
                }`}
              >
                📊 全部数据
              </button>
              <button
                onClick={() => setUseCustomDateRange(true)}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  useCustomDateRange 
                    ? 'bg-blue-600/30 text-blue-300 border border-blue-600/50' 
                    : 'bg-slate-700/30 text-slate-400 border border-slate-600/50'
                }`}
              >
                📆 自定义范围
              </button>
            </div>
            
            {useCustomDateRange && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">开始日期</label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">结束日期</label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
              </div>
            )}
            
            {!useCustomDateRange && (
              <div className="text-xs text-gray-500">
                💡 使用股票池中所有可用的历史数据进行回测
              </div>
            )}
            
            {useCustomDateRange && startDate && endDate && (
              <div className="text-xs text-green-500">
                📊 将使用 {startDate} 至 {endDate} 的数据进行回测 
                (约 {calculateBacktestDays()} 天)
              </div>
            )}
          </div>

          {/* 回测按钮 */}
          <button
            onClick={runBacktest}
            disabled={loading}
            className={`w-full mb-4 px-4 py-2 disabled:opacity-50 disabled:cursor-not-allowed text-slate-100 text-sm rounded border transition-colors ${
              strategyType === 'enhanced'
                ? 'bg-purple-700 hover:bg-purple-600 border-purple-600'
                : 'bg-blue-700 hover:bg-blue-600 border-blue-600'
            }`}
          >
            {loading 
              ? '⏳ 回测中...' 
              : strategyType === 'enhanced' 
                ? '🚀 运行增强策略' 
                : '📈 运行原始策略'
            }
          </button>
        </>
      )}

      {activeTab === 'optimize' && (
        <>
          <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-2">⚡ 参数优化</div>
            <div className="text-xs text-gray-500">
              系统将自动测试不同的买入/卖出阈值组合，找到最优参数配置
            </div>
          </div>

          {/* 回测时间范围 */}
          <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-3">📅 回测时间范围</div>
            
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => setUseCustomDateRange(false)}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  !useCustomDateRange 
                    ? 'bg-blue-600/30 text-blue-300 border border-blue-600/50' 
                    : 'bg-slate-700/30 text-slate-400 border border-slate-600/50'
                }`}
              >
                📊 全部数据
              </button>
              <button
                onClick={() => setUseCustomDateRange(true)}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  useCustomDateRange 
                    ? 'bg-blue-600/30 text-blue-300 border border-blue-600/50' 
                    : 'bg-slate-700/30 text-slate-400 border border-slate-600/50'
                }`}
              >
                📆 自定义范围
              </button>
            </div>
            
            {useCustomDateRange && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">开始日期</label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 mb-1 block">结束日期</label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 text-sm"
                  />
                </div>
              </div>
            )}
            
            {!useCustomDateRange && (
              <div className="text-xs text-gray-500">
                💡 使用股票池中所有可用的历史数据进行回测
              </div>
            )}
          </div>

          <button
            onClick={runOptimization}
            disabled={loading}
            className="w-full mb-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-slate-100 text-sm rounded border border-slate-600 transition-colors"
          >
            {loading ? '⏳ 优化中...' : '⚡ 开始优化'}
          </button>
        </>
      )}



      {/* 错误显示 */}
      {error && (
        <div className="mb-4 p-3 bg-red-600/20 border border-red-600/50 rounded text-red-400 text-sm">
          ❌ {error}
        </div>
      )}

      {/* 回测结果 */}
      {result && !error && (
        <div className="space-y-4">
          {/* 自动生成数据提示 */}
          {result.auto_generated_data && (
            <div className="p-3 bg-green-600/10 rounded border border-green-600/30">
              <div className="flex items-start gap-2">
                <span className="text-green-400">🤖</span>
                <div className="flex-1">
                  <div className="text-sm font-medium text-green-300">自动生成历史数据</div>
                  <div className="text-xs text-green-400 mt-1">
                    {result.message || `系统自动生成了 ${result.generated_records} 条历史数据`}
                  </div>
                  {result.generated_symbols && result.generated_symbols.length > 0 && (
                    <div className="text-xs text-green-500 mt-1">
                      涉及股票: {result.generated_symbols.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* 核心指标 */}
          <div className="p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-3">📊 策略表现</div>
            
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">总收益率</div>
                <div className={`font-medium ${
                  result.performance_metrics.total_return >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatPercent(result.performance_metrics.total_return)}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">年化收益率</div>
                <div className={`font-medium ${
                  result.performance_metrics.annualized_return >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatPercent(result.performance_metrics.annualized_return)}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">最大回撤</div>
                <div className="text-red-400 font-medium">
                  -{formatPercent(result.performance_metrics.max_drawdown)}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">夏普比率</div>
                <div className={`font-medium ${
                  result.performance_metrics.sharpe_ratio >= 1 ? 'text-green-400' :
                  result.performance_metrics.sharpe_ratio >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {result.performance_metrics.sharpe_ratio ? result.performance_metrics.sharpe_ratio.toFixed(3) : 'N/A'}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">胜率</div>
                <div className={`font-medium ${
                  result.performance_metrics.win_rate >= 0.6 ? 'text-green-400' :
                  result.performance_metrics.win_rate >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {formatPercent(result.performance_metrics.win_rate)}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">盈亏比</div>
                <div className={`font-medium ${
                  result.performance_metrics.profit_loss_ratio >= 2 ? 'text-green-400' :
                  result.performance_metrics.profit_loss_ratio >= 1 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {result.performance_metrics.profit_loss_ratio ? result.performance_metrics.profit_loss_ratio.toFixed(2) : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* 交易统计 */}
          <div className="p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-3">📈 交易统计</div>
            
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="text-center">
                <div className="text-gray-400">总交易数</div>
                <div className="text-blue-400 font-medium text-lg">
                  {result.performance_metrics.total_trades}
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-gray-400">盈利交易</div>
                <div className="text-green-400 font-medium text-lg">
                  {result.performance_metrics.winning_trades}
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-gray-400">亏损交易</div>
                <div className="text-red-400 font-medium text-lg">
                  {result.performance_metrics.losing_trades}
                </div>
              </div>
            </div>
            
            <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">最终资产</div>
                <div className="text-blue-400 font-medium">
                  {formatCurrency(result.final_portfolio_value)}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">平均持仓期</div>
                <div className="text-gray-300 font-medium">
                  {result.performance_metrics.avg_holding_period ? result.performance_metrics.avg_holding_period.toFixed(1) : 'N/A'} 天
                </div>
              </div>
            </div>
            
            {/* 交易详情按钮 */}
            <div className="mt-3 space-y-2">
              <button
                onClick={() => setShowTradeModal(true)}
                className="w-full px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-600/50 text-blue-300 text-xs rounded transition-colors"
              >
                📋 查看交易详情 (共 {result.trades?.length || 0} 笔交易)
              </button>
              
              <button
                onClick={() => setShowChartModal(true)}
                className="w-full px-3 py-2 bg-green-600/20 hover:bg-green-600/30 border border-green-600/50 text-green-300 text-xs rounded transition-colors"
              >
                📊 查看各股盈利图表
              </button>
            </div>
          </div>


        </div>
      )}

      {/* 优化结果 */}
      {optimization && !error && (
        <div className="space-y-4">
            <div className="p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-3">⚡ 最优参数</div>
            
            <div className="grid grid-cols-2 gap-3 text-xs mb-3">
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">买入阈值</div>
                <div className="text-blue-400 font-medium">
                  {optimization.best_config.buy_threshold}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">卖出阈值</div>
                <div className="text-blue-400 font-medium">
                  {optimization.best_config.sell_threshold}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">最优收益率</div>
                <div className="text-green-400 font-medium">
                  {formatPercent(optimization.best_performance.total_return)}
                </div>
              </div>
              
              <div className="bg-gray-700/20 rounded px-2 py-2">
                <div className="text-gray-400">夏普比率</div>
                <div className="text-blue-400 font-medium">
                  {optimization.best_performance.sharpe_ratio ? optimization.best_performance.sharpe_ratio.toFixed(3) : 'N/A'}
                </div>
              </div>
            </div>
            
            <button
              onClick={() => {
                setConfig(optimization.best_config)
                setActiveTab('backtest')
              }}
              className="w-full mt-3 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm rounded border border-slate-600 transition-colors"
            >
              🎯 应用最优参数
            </button>
          </div>

          {/* 优化历史 */}
          <div className="p-3 bg-gray-800/30 rounded border border-gray-700/50">
            <div className="text-sm font-medium text-gray-400 mb-3">📊 参数测试结果</div>
            
            <div className="max-h-32 overflow-y-auto space-y-1">
              {optimization.optimization_results
                .sort((a, b) => b.score - a.score)
                .slice(0, 8)
                .map((opt, idx) => (
                <div key={idx} className="flex justify-between items-center text-xs bg-gray-700/20 rounded px-2 py-1">
                  <span className="text-blue-300 font-mono">
                    买:{opt.buy_threshold} 卖:{opt.sell_threshold}
                  </span>
                  <div className="flex gap-2 text-gray-400">
                    <span className={opt.total_return >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {formatPercent(opt.total_return)}
                    </span>
                    <span>SR:{opt.sharpe_ratio ? opt.sharpe_ratio.toFixed(2) : 'N/A'}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 说明 */}
      <div className="mt-4 text-xs text-gray-500 text-center">
        💡 基于历史预测数据的策略回测，用于评估交易策略的表现和风险
      </div>

      {/* 交易详情弹窗 */}
      {showTradeModal && result?.trades && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-lg border border-gray-700 w-[90vw] h-[90vh] max-w-6xl flex flex-col">
            {/* 弹窗头部 */}
            <div className="flex justify-between items-center p-4 border-b border-gray-700">
              <div>
                <h3 className="text-lg font-medium text-gray-200">📋 交易详情记录</h3>
                <p className="text-sm text-gray-400">共 {result.trades.length} 笔交易</p>
              </div>
              <button
                onClick={() => setShowTradeModal(false)}
                className="text-gray-400 hover:text-gray-200 text-2xl"
              >
                ✕
              </button>
            </div>

            {/* 弹窗内容 */}
            <div className="flex-1 p-4 overflow-hidden">
              {/* 筛选和排序控制 */}
              <div className="mb-4 flex flex-wrap gap-4 items-center">
                  <div className="flex gap-2 items-center">
                  <span className="text-sm text-gray-400">筛选:</span>
                    <select
                      value={tradeFilter}
                      onChange={(e) => setTradeFilter(e.target.value as 'all' | 'buy' | 'sell')}
                    className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-gray-300"
                    >
                      <option value="all">全部交易</option>
                      <option value="buy">仅买入</option>
                      <option value="sell">仅卖出</option>
                    </select>
                  </div>
                  
                  <div className="flex gap-2 items-center">
                  <span className="text-sm text-gray-400">排序:</span>
                    <select
                      value={tradeSortBy}
                      onChange={(e) => setTradeSortBy(e.target.value as 'date' | 'profit' | 'amount')}
                    className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-gray-300"
                    >
                      <option value="date">按日期</option>
                      <option value="profit">按收益</option>
                      <option value="amount">按金额</option>
                    </select>
                  </div>
                  
                <div className="text-sm text-gray-500">
                  显示 {getFilteredAndSortedTrades(result.trades).length} / {result.trades.length} 笔交易
                  {tradeFilter !== 'all' && (
                    <span className="ml-2 px-2 py-1 bg-blue-600/20 text-blue-300 rounded text-xs">
                      {tradeFilter === 'buy' ? '买入' : '卖出'}
                    </span>
                  )}
                </div>
              </div>
              
              {/* 交易列表 */}
              <div className="flex-1 overflow-y-auto">
                <div className="grid gap-3">
                {getFilteredAndSortedTrades(result.trades).map((trade, index) => (
                  <div
                    key={index}
                      className={`p-4 rounded-lg border ${
                      trade.action === 'buy'
                          ? 'bg-green-600/5 border-green-600/20'
                          : 'bg-red-600/5 border-red-600/20'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-3">
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          trade.action === 'buy'
                            ? 'bg-green-600/20 text-green-300'
                            : 'bg-red-600/20 text-red-300'
                        }`}>
                          {trade.action === 'buy' ? '📈 买入' : '📉 卖出'}
                        </span>
                          <span className="text-blue-300 font-medium text-lg">{trade.symbol}</span>
                      </div>
                      <div className="text-gray-400 text-right">
                          <div className="text-sm">{trade.date}</div>
                          <div className="text-xs">#{index + 1}</div>
                      </div>
                    </div>
                    
                      <div className="grid grid-cols-3 gap-6">
                      <div>
                          <div className="text-gray-500 mb-2 font-medium">交易信息</div>
                          <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-400">价格:</span>
                              <span className="text-gray-200 font-medium">¥{trade.price ? trade.price.toFixed(2) : 'N/A'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">数量:</span>
                              <span className="text-gray-200 font-medium">{trade.quantity ? trade.quantity.toLocaleString() : 'N/A'}股</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">金额:</span>
                              <span className="text-gray-200 font-medium">¥{(trade.price && trade.quantity) ? (trade.price * trade.quantity).toLocaleString() : 'N/A'}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                          <div className="text-gray-500 mb-2 font-medium">预测信息</div>
                          <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-400">预测概率:</span>
                            <span className={`font-medium ${
                              trade.predicted_prob > 0.6 ? 'text-green-400' :
                              trade.predicted_prob > 0.4 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                                {trade.predicted_prob ? (trade.predicted_prob * 100).toFixed(1) : 'N/A'}%
                            </span>
                          </div>
                          {trade.actual_return !== undefined && trade.actual_return !== 0 && (
                            <div className="flex justify-between">
                              <span className="text-gray-400">实际收益:</span>
                              <span className={`font-medium ${
                                trade.actual_return > 0 ? 'text-green-400' : 'text-red-400'
                              }`}>
                                  {trade.actual_return ? (trade.actual_return * 100).toFixed(2) : 'N/A'}%
                              </span>
                            </div>
                          )}
                        </div>
              </div>
              
                  <div>
                          <div className="text-gray-500 mb-2 font-medium">收益计算</div>
                          <div className="space-y-2">
                            {trade.actual_return !== undefined && trade.price && trade.quantity && (
                              <>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">收益金额:</span>
                                  <span className={`font-medium ${
                                    trade.actual_return > 0 ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                    {trade.actual_return > 0 ? '+' : ''}
                                    ¥{(trade.price * trade.quantity * trade.actual_return).toFixed(2)}
                                  </span>
                    </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">收益率:</span>
                                  <span className={`font-medium ${
                                    trade.actual_return > 0 ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                    {trade.actual_return > 0 ? '+' : ''}{(trade.actual_return * 100).toFixed(2)}%
                                  </span>
                  </div>
                              </>
                            )}
                    </div>
                  </div>
                    </div>
                  </div>
                  ))}
                    </div>
                  </div>
                </div>
                
            {/* 弹窗底部 */}
            <div className="p-4 border-t border-gray-700">
              <div className="flex justify-between items-center">
                <div className="flex gap-4 text-sm text-gray-400">
                  <span>买入: {result.trades.filter(t => t.action === 'buy').length}笔</span>
                  <span>卖出: {result.trades.filter(t => t.action === 'sell').length}笔</span>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => exportTradesToCSV(result.trades)}
                    className="px-4 py-2 bg-green-600/20 hover:bg-green-600/30 border border-green-600/50 text-green-300 text-sm rounded transition-colors"
                  >
                    📊 导出全部
                  </button>
                  <button
                    onClick={() => exportTradesToCSV(getFilteredAndSortedTrades(result.trades))}
                    className="px-4 py-2 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-600/50 text-blue-300 text-sm rounded transition-colors"
                  >
                    📋 导出筛选
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 盈利图表弹窗 */}
      {showChartModal && result?.trades && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-lg border border-gray-700 w-[90vw] h-[90vh] max-w-6xl flex flex-col">
            {/* 弹窗头部 */}
            <div className="flex justify-between items-center p-4 border-b border-gray-700">
              <div>
                <h3 className="text-lg font-medium text-gray-200">📊 各股盈利走势图</h3>
                <p className="text-sm text-gray-400">分析每只股票的盈利表现</p>
              </div>
                  <button
                onClick={() => setShowChartModal(false)}
                className="text-gray-400 hover:text-gray-200 text-2xl"
                  >
                ✕
                  </button>
                </div>

            {/* 弹窗内容 */}
            <div className="flex-1 p-4 overflow-y-auto">
              {(() => {
                const chartData = calculateFinancialChartData(result.trades, config.initial_capital)
                
                if (Object.keys(chartData.stockData).length === 0) {
                  return (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      暂无交易数据
              </div>
                  )
                }
                
                return (
                  <div className="space-y-6">
                    {/* 总体权益曲线 */}
                    <div className="bg-gray-800/30 rounded-lg p-6 border border-gray-700/50">
                      <div className="flex justify-between items-center mb-6">
                        <div>
                          <h3 className="text-xl font-bold text-blue-300">📈 组合权益曲线</h3>
                          <p className="text-gray-400 text-sm">Portfolio Equity Curve</p>
            </div>
                        <div className="text-right">
                          <div className={`text-2xl font-bold ${chartData.totalReturn >= 0 ? 'text-red-400' : 'text-green-500'}`}>
                            {chartData.totalReturn >= 0 ? '+' : ''}{chartData.totalReturn.toFixed(2)}%
        </div>
                          <div className="text-sm text-gray-400">总收益率</div>
                        </div>
                      </div>
                      
                      {/* 权益曲线图 */}
                      <div className="relative h-64 bg-gray-900/50 rounded-lg overflow-hidden mb-4">
                        {/* 网格线 */}
                        <div className="absolute inset-0">
                          {[0.2, 0.4, 0.6, 0.8].map(ratio => (
                            <div key={ratio} className="absolute w-full border-t border-gray-700/30" style={{ top: `${ratio * 100}%` }} />
                          ))}
                          {[0.2, 0.4, 0.6, 0.8].map(ratio => (
                            <div key={ratio} className="absolute h-full border-l border-gray-700/30" style={{ left: `${ratio * 100}%` }} />
                          ))}
                        </div>
                        
                        <svg className="absolute inset-0 w-full h-full">
                          <defs>
                            <linearGradient id="equityGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3"/>
                              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0"/>
                            </linearGradient>
                          </defs>
                          
                          {chartData.equityCurve.length > 1 && (
                            <>
                              {/* 权益曲线填充 */}
                              <polygon
                                points={`0,100 ${chartData.equityCurve.map((point, index) => {
                                  const x = (index / Math.max(chartData.equityCurve.length - 1, 1)) * 100
                                  // 计算权益在最高和最低值之间的相对位置
                                  const minEquity = Math.min(...chartData.equityCurve.map(p => p.equity))
                                  const maxEquity = Math.max(...chartData.equityCurve.map(p => p.equity))
                                  const equityRange = maxEquity - minEquity
                                  const y = equityRange > 0 ? 100 - ((point.equity - minEquity) / equityRange) * 80 : 50
                                  return `${x},${Math.max(10, Math.min(90, y))}`
                                }).join(' ')} 100,100`}
                                fill="url(#equityGradient)"
                              />
                              
                              {/* 权益曲线 */}
                              <polyline
                                points={chartData.equityCurve.map((point, index) => {
                                  const x = (index / Math.max(chartData.equityCurve.length - 1, 1)) * 100
                                  // 计算权益在最高和最低值之间的相对位置
                                  const minEquity = Math.min(...chartData.equityCurve.map(p => p.equity))
                                  const maxEquity = Math.max(...chartData.equityCurve.map(p => p.equity))
                                  const equityRange = maxEquity - minEquity
                                  const y = equityRange > 0 ? 100 - ((point.equity - minEquity) / equityRange) * 80 : 50
                                  return `${x},${Math.max(10, Math.min(90, y))}`
                                }).join(' ')}
                                fill="none"
                                stroke="#3b82f6"
                                strokeWidth="3"
                                className="drop-shadow-sm"
                              />
                              
                              {/* 基准线 */}
                              <line
                                x1="0"
                                y1="100"
                                x2="100%"
                                y2="100"
                                stroke="#6b7280"
                                strokeWidth="1"
                                strokeDasharray="5,5"
                              />
                            </>
                          )}
                        </svg>
                        
                        {/* Y轴标签 */}
                        <div className="absolute left-2 top-2 text-xs text-gray-400 bg-gray-800/80 px-1 rounded">
                          ¥{Math.max(...chartData.equityCurve.map(p => p.equity)).toLocaleString()}
                        </div>
                        <div className="absolute left-2 bottom-2 text-xs text-gray-400 bg-gray-800/80 px-1 rounded">
                          ¥{Math.min(...chartData.equityCurve.map(p => p.equity)).toLocaleString()}
                        </div>
                      </div>
                      
                      {/* 权益曲线统计 */}
                      <div className="grid grid-cols-4 gap-4 text-sm">
                        <div className="text-center">
                          <div className="text-gray-400">初始资金</div>
                          <div className="font-medium text-blue-400">¥{config.initial_capital.toLocaleString()}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-gray-400">最终资金</div>
                          <div className="font-medium text-blue-400">¥{chartData.finalEquity.toLocaleString()}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-gray-400">绝对收益</div>
                          <div className={`font-medium ${chartData.finalEquity >= config.initial_capital ? 'text-red-400' : 'text-green-500'}`}>
                            ¥{(chartData.finalEquity - config.initial_capital).toLocaleString()}
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-gray-400">最大回撤</div>
                          <div className="font-medium text-red-400">
                            {chartData.equityCurve.length > 0 ? Math.min(...chartData.equityCurve.map(p => p.drawdown)).toFixed(2) : '0.00'}%
                          </div>
                        </div>
                </div>
              </div>
              
                    {/* 回撤图 */}
                    <div className="bg-gray-800/30 rounded-lg p-6 border border-gray-700/50">
                      <div className="flex justify-between items-center mb-6">
                        <div>
                          <h3 className="text-xl font-bold text-red-300">📉 回撤分析 v2.1</h3>
                          <p className="text-gray-400 text-sm">Drawdown Analysis (Fixed)</p>
                </div>
                        <div className="text-right">
                          <div className="text-2xl font-bold text-red-400">
                            {chartData.equityCurve.length > 0 ? Math.min(...chartData.equityCurve.map(p => p.drawdown)).toFixed(2) : '0.00'}%
                          </div>
                          <div className="text-sm text-gray-400">最大回撤</div>
              </div>
            </div>

                      {/* 回撤图 */}
                      <div className="relative h-48 bg-gray-900/50 rounded-lg overflow-hidden">
                        {/* 网格线 */}
                        <div className="absolute inset-0">
                          {[0.25, 0.5, 0.75].map(ratio => (
                            <div key={ratio} className="absolute w-full border-t border-gray-700/30" style={{ top: `${ratio * 100}%` }} />
                          ))}
                        </div>
                        
                        <svg className="absolute inset-0 w-full h-full">
                          <defs>
                            <linearGradient id="drawdownGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                              <stop offset="0%" stopColor="#ef4444" stopOpacity="0"/>
                              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.3"/>
                            </linearGradient>
                          </defs>
                          
                          {chartData.equityCurve.length > 1 && (
                            <>
                              {/* 回撤填充 */}
                              <polygon
                                points={`0,0 ${chartData.equityCurve.map((point, index) => {
                                  const x = (index / Math.max(chartData.equityCurve.length - 1, 1)) * 100
                                  const maxDrawdown = Math.min(...chartData.equityCurve.map(p => p.drawdown))
                                  // 将负的回撤值转换为正的显示高度
                                  const drawdownPercent = Math.abs(point.drawdown)
                                  const maxDrawdownPercent = Math.abs(maxDrawdown)
                                  const y = maxDrawdownPercent > 0 ? (drawdownPercent / maxDrawdownPercent) * 90 : 0
                                  return `${x},${Math.max(0, Math.min(100, 100 - y))}`
                                }).join(' ')} 100,100`}
                                fill="url(#drawdownGradient)"
                              />
                              
                              {/* 回撤线 */}
                              <polyline
                                points={chartData.equityCurve.map((point, index) => {
                                  const x = (index / Math.max(chartData.equityCurve.length - 1, 1)) * 100
                                  const maxDrawdown = Math.min(...chartData.equityCurve.map(p => p.drawdown))
                                  // 将负的回撤值转换为正的显示高度
                                  const drawdownPercent = Math.abs(point.drawdown)
                                  const maxDrawdownPercent = Math.abs(maxDrawdown)
                                  const y = maxDrawdownPercent > 0 ? (drawdownPercent / maxDrawdownPercent) * 90 : 0
                                  return `${x},${Math.max(0, Math.min(100, 100 - y))}`
                                }).join(' ')}
                                fill="none"
                                stroke="#ef4444"
                                strokeWidth="2"
                              />
                            </>
                          )}
                        </svg>
                        
                        <div className="absolute left-2 bottom-2 text-xs text-gray-400">0%</div>
                        <div className="absolute left-2 top-2 text-xs text-gray-400">
                          {chartData.equityCurve.length > 0 ? Math.min(...chartData.equityCurve.map(p => p.drawdown)).toFixed(1) : '0.0'}%
                        </div>
                </div>
              </div>
              
                    {/* 个股详细分析 */}
                    <div className="space-y-4">
                      <h3 className="text-xl font-bold text-yellow-300">📊 个股分析</h3>
                      {Object.entries(chartData.stockData).map(([symbol, data]) => (
                        <div key={symbol} className="bg-gray-800/30 rounded-lg p-6 border border-gray-700/50">
                          <div className="flex justify-between items-center mb-4">
                            <div>
                              <h4 className="text-lg font-bold text-blue-300">{symbol}</h4>
                              <p className="text-gray-400 text-sm">交易次数: {data.totalTrades}笔</p>
                </div>
                            <div className="text-right">
                              <div className={`text-xl font-bold ${data.cumulativeProfit >= 0 ? 'text-red-400' : 'text-green-500'}`}>
                                {data.cumulativeProfit >= 0 ? '+' : ''}¥{data.cumulativeProfit.toFixed(2)}
                              </div>
                              <div className="text-sm text-gray-400">累计盈亏</div>
              </div>
            </div>
            
                          {/* 股价走势图 + 买卖点 */}
                          <div className="relative h-48 bg-gray-900/50 rounded-lg overflow-hidden mb-4 border border-gray-700">
                            {/* 网格线 */}
                            <div className="absolute inset-0">
                              {[0.2, 0.4, 0.6, 0.8].map(ratio => (
                                <div key={ratio} className="absolute w-full border-t border-gray-700/30" style={{ top: `${ratio * 100}%` }} />
                              ))}
                              {[0.2, 0.4, 0.6, 0.8].map(ratio => (
                                <div key={ratio} className="absolute h-full border-l border-gray-700/30" style={{ left: `${ratio * 100}%` }} />
                              ))}
                            </div>
                            
                            <svg className="absolute inset-0 w-full h-full">
                              <defs>
                                {/* 股价线渐变 */}
                                <linearGradient id={`priceGradient-${symbol}`} x1="0%" y1="0%" x2="0%" y2="100%">
                                  <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.3"/>
                                  <stop offset="100%" stopColor="#60a5fa" stopOpacity="0"/>
                                </linearGradient>
                                {/* 盈利渐变 */}
                                <linearGradient id={`profitGradient-${symbol}`} x1="0%" y1="0%" x2="0%" y2="100%">
                                  <stop offset="0%" stopColor={data.cumulativeProfit >= 0 ? '#ef4444' : '#22c55e'} stopOpacity="0.2"/>
                                  <stop offset="100%" stopColor={data.cumulativeProfit >= 0 ? '#ef4444' : '#22c55e'} stopOpacity="0"/>
                                </linearGradient>
                              </defs>
                              
                              {data.trades.length > 1 && (
                                <>
                                  {/* 股价线 (上半部分) */}
                                  <g>
                                    {/* 股价填充区域 */}
                                    <polygon
                                      points={`0,50 ${data.trades.length > 1 ? data.trades.map((trade, index) => {
                                        const x = (index / Math.max(data.trades.length - 1, 1)) * 100
                                        const priceRange = data.priceRange.max - data.priceRange.min
                                        const priceY = priceRange > 0 ? 50 - ((trade.price - data.priceRange.min) / priceRange) * 45 : 25
                                        return `${x},${Math.max(5, Math.min(50, priceY))}`
                                      }).join(' ') : '0,25 100,25'} 100,50`}
                                      fill={`url(#priceGradient-${symbol})`}
                                    />
                                    
                                    {/* 股价线 */}
                                    <polyline
                                      points={data.trades.length > 1 ? data.trades.map((trade, index) => {
                                        const x = (index / Math.max(data.trades.length - 1, 1)) * 100
                                        const priceRange = data.priceRange.max - data.priceRange.min
                                        const priceY = priceRange > 0 ? 50 - ((trade.price - data.priceRange.min) / priceRange) * 45 : 25
                                        return `${x},${Math.max(5, Math.min(50, priceY))}`
                                      }).join(' ') : '0,25 100,25'}
                                      fill="none"
                                      stroke="#60a5fa"
                                      strokeWidth="2"
                                      className="drop-shadow-sm"
                                    />
                                  </g>
                                  
                                  {/* 分隔线 */}
                                  <line
                                    x1="0"
                                    y1="50%"
                                    x2="100%"
                                    y2="50%"
                                    stroke="#6b7280"
                                    strokeWidth="1"
                                    strokeDasharray="5,5"
                                  />
                                  
                                  {/* 盈亏线 (下半部分) */}
                                  <g>
                                    {/* 盈亏填充区域 */}
                                    <polygon
                                      points={`0,100 ${data.trades.length > 0 ? data.trades.map((trade, index) => {
                                        const x = (index / Math.max(data.trades.length - 1, 1)) * 100
                                        const maxRange = Math.max(Math.abs(data.maxProfit), Math.abs(data.maxDrawdown), 100)
                                        const profitY = trade.cumulativeProfit >= 0 
                                          ? 100 - (Math.abs(trade.cumulativeProfit) / maxRange) * 45
                                          : 100
                                        return `${x},${Math.max(55, Math.min(100, profitY))}`
                                      }).join(' ') : '0,100 100,100'} 100,100`}
                                      fill={`url(#profitGradient-${symbol})`}
                                    />
                                    
                                    {/* 盈亏线 */}
                                    <polyline
                                      points={data.trades.length > 0 ? data.trades.map((trade, index) => {
                                        const x = (index / Math.max(data.trades.length - 1, 1)) * 100
                                        const maxRange = Math.max(Math.abs(data.maxProfit), Math.abs(data.maxDrawdown), 100)
                                        const profitY = trade.cumulativeProfit >= 0 
                                          ? 50 - (trade.cumulativeProfit / maxRange) * 45
                                          : 50 + (Math.abs(trade.cumulativeProfit) / maxRange) * 45
                                        return `${x},${Math.max(5, Math.min(95, profitY))}`
                                      }).join(' ') : '0,50 100,50'}
                                      fill="none"
                                      stroke={data.cumulativeProfit >= 0 ? '#ef4444' : '#22c55e'}
                                      strokeWidth="2"
                                      className="drop-shadow-sm"
                                    />
                                  </g>
                                  
                                  {/* 买卖点标记 */}
                                  {data.trades.map((trade, index) => {
                                    const x = (index / Math.max(data.trades.length - 1, 1)) * 100
                                    const priceRange = data.priceRange.max - data.priceRange.min
                                    const priceY = priceRange > 0 ? 50 - ((trade.price - data.priceRange.min) / priceRange) * 45 : 25
                                    
                                    return (
                                      <g key={index}>
                                        {/* 买卖点标记 */}
                                        <circle
                                          cx={`${x}%`}
                                          cy={Math.max(5, Math.min(50, priceY))}
                                          r="5"
                                          fill={trade.action === 'buy' ? '#fbbf24' : (trade.profit >= 0 ? '#ef4444' : '#22c55e')}
                                          stroke="white"
                                          strokeWidth="2"
                                          className="cursor-pointer drop-shadow-md"
                                          onMouseEnter={(e) => {
                                            const rect = e.currentTarget.getBoundingClientRect()
                                            setTooltip({
                                              visible: true,
                                              x: rect.left + window.scrollX,
                                              y: rect.top + window.scrollY - 10,
                                              content: {
                                                symbol,
                                                trade,
                                                index
                                              }
                                            })
                                          }}
                                          onMouseLeave={() => setTooltip(prev => ({ ...prev, visible: false }))}
                                        />
                                        
                                        {/* 买卖标记文字 */}
                                        <text
                                          x={`${x}%`}
                                          y={Math.max(5, Math.min(50, priceY)) - 8}
                                          textAnchor="middle"
                                          className="text-xs fill-white font-bold pointer-events-none"
                                          style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.8)' }}
                                        >
                                          {trade.action === 'buy' ? '买' : '卖'}
                                        </text>
                                      </g>
                                    )
                                  })}
                                </>
                              )}
                            </svg>
                            
                            {/* 价格轴标签 */}
                            <div className="absolute left-2 top-2 text-xs text-blue-400 bg-gray-800/80 px-2 py-1 rounded">
                              股价 ¥{data.priceRange.max.toFixed(2)}
                            </div>
                            <div className="absolute left-2 top-12 text-xs text-blue-400 bg-gray-800/80 px-2 py-1 rounded">
                              ¥{data.priceRange.min.toFixed(2)}
          </div>

                            {/* 盈亏轴标签 */}
                            <div className="absolute left-2 bottom-2 text-xs text-gray-400 bg-gray-800/80 px-2 py-1 rounded">
                              盈亏 ¥{data.maxProfit.toFixed(0)}
                            </div>
                            <div className="absolute left-2 bottom-12 text-xs text-gray-400 bg-gray-800/80 px-2 py-1 rounded">
                              ¥{data.maxDrawdown.toFixed(0)}
                            </div>
                            
                            {/* 图例 */}
                            <div className="absolute right-2 top-2 text-xs text-gray-400 bg-gray-800/90 px-3 py-2 rounded space-y-1">
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-0.5 bg-blue-400"></div>
                                <span>股价</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className={`w-3 h-0.5 ${data.cumulativeProfit >= 0 ? 'bg-red-400' : 'bg-green-500'}`}></div>
                                <span>盈亏</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                                <span>买入</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                                <span>卖出</span>
                              </div>
                            </div>
                          </div>
                          
                          {/* 个股统计 */}
                          <div className="grid grid-cols-4 gap-4 text-sm">
                            <div className="text-center">
                              <div className="text-gray-400">胜率</div>
                              <div className={`font-medium ${data.winRate >= 50 ? 'text-red-400' : 'text-green-500'}`}>
                                {data.winRate.toFixed(1)}%
                              </div>
                            </div>
                            <div className="text-center">
                              <div className="text-gray-400">最大盈利</div>
                              <div className="font-medium text-red-400">¥{data.maxProfit.toFixed(2)}</div>
                            </div>
                            <div className="text-center">
                              <div className="text-gray-400">最大回撤</div>
                              <div className="font-medium text-green-500">¥{Math.abs(data.maxDrawdown).toFixed(2)}</div>
                            </div>
                            <div className="text-center">
                              <div className="text-gray-400">总交易</div>
                              <div className="font-medium text-blue-400">{data.totalTrades}笔</div>
                            </div>
                  </div>
                </div>
              ))}
                    </div>
                  </div>
                )
              })()}
            </div>
          </div>
        </div>
      )}

      {/* 悬停提示框 */}
      {tooltip.visible && tooltip.content && (
        <div 
          className="fixed z-[60] bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl pointer-events-none"
          style={{ 
            left: tooltip.x - 100, 
            top: tooltip.y - 120,
            minWidth: '200px'
          }}
        >
          <div className="text-white text-sm space-y-2">
            <div className="font-bold text-blue-300 border-b border-gray-600 pb-1">
              {tooltip.content.symbol} - 第{tooltip.content.index + 1}笔交易
      </div>
            
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-400">日期:</span>
                <div className="text-white">{tooltip.content.trade.date}</div>
              </div>
              <div>
                <span className="text-gray-400">操作:</span>
                <div className={`font-bold ${tooltip.content.trade.action === 'buy' ? 'text-yellow-400' : 'text-orange-400'}`}>
                  {tooltip.content.trade.action === 'buy' ? '买入' : '卖出'}
                </div>
              </div>
              <div>
                <span className="text-gray-400">价格:</span>
                <div className="text-white">¥{tooltip.content.trade.price.toFixed(2)}</div>
              </div>
              <div>
                <span className="text-gray-400">数量:</span>
                <div className="text-white">{tooltip.content.trade.quantity}股</div>
              </div>
              <div>
                <span className="text-gray-400">金额:</span>
                <div className="text-white">¥{(tooltip.content.trade.price * tooltip.content.trade.quantity).toFixed(2)}</div>
              </div>
              {tooltip.content.trade.predicted_prob && (
                <div>
                  <span className="text-gray-400">预测概率:</span>
                  <div className={`font-medium ${tooltip.content.trade.predicted_prob > 0.6 ? 'text-red-400' : tooltip.content.trade.predicted_prob > 0.4 ? 'text-yellow-400' : 'text-green-500'}`}>
                    {(tooltip.content.trade.predicted_prob * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>
            
            {tooltip.content.trade.action === 'sell' && tooltip.content.trade.profit !== 0 && (
              <div className="border-t border-gray-600 pt-2">
                <div className="text-gray-400 text-xs mb-1">本次交易:</div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-xs">盈亏:</span>
                  <span className={`text-xs font-bold ${tooltip.content.trade.profit >= 0 ? 'text-red-400' : 'text-green-500'}`}>
                    {tooltip.content.trade.profit >= 0 ? '+' : ''}¥{tooltip.content.trade.profit.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-xs">累计:</span>
                  <span className={`text-xs font-bold ${tooltip.content.trade.cumulativeProfit >= 0 ? 'text-red-400' : 'text-green-500'}`}>
                    {tooltip.content.trade.cumulativeProfit >= 0 ? '+' : ''}¥{tooltip.content.trade.cumulativeProfit.toFixed(2)}
                  </span>
                </div>
              </div>
            )}
          </div>
          
          {/* 小箭头 */}
          <div className="absolute bottom-[-6px] left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-6 border-r-6 border-t-6 border-l-transparent border-r-transparent border-t-gray-800"></div>
        </div>
      )}
    </div>
  )
}

export default StrategyPanel
