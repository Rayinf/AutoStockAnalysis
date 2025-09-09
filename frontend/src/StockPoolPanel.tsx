import React, { useState, useEffect } from 'react'
import StockPoolManager from './StockPoolManager'

interface StockPool {
  name: string
  symbols: string[]
  description: string
}

interface AvailableStock {
  symbol: string
  name: string
  prediction_count: number
  first_date: string
  last_date: string
}

interface StockPoolPanelProps {
  selectedSymbols: string
  onSymbolsChange: (symbols: string) => void
  onGenerateData?: (symbols: string) => void
}

export const StockPoolPanel: React.FC<StockPoolPanelProps> = ({ 
  selectedSymbols, 
  onSymbolsChange, 
  onGenerateData 
}) => {
  const [stockPools, setStockPools] = useState<Record<string, StockPool>>({})
  const [availableStocks, setAvailableStocks] = useState<AvailableStock[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAllStocks, setShowAllStocks] = useState(false)
  const [dataGenDays, setDataGenDays] = useState(100)
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [useCustomDateRange, setUseCustomDateRange] = useState(false)
  const [showPoolManager, setShowPoolManager] = useState(false)

  // 加载股票池数据
  const loadStockPools = async () => {
    try {
      console.log('正在加载股票池数据...')
      
      const [poolsRes, stocksRes] = await Promise.all([
        fetch('http://127.0.0.1:8000/api/strategy/stock_pools'),
        fetch('http://127.0.0.1:8000/api/strategy/available_stocks')
      ])
      
      if (!poolsRes.ok || !stocksRes.ok) {
        throw new Error(`API请求失败: pools=${poolsRes.status}, stocks=${stocksRes.status}`)
      }
      
      const poolsData = await poolsRes.json()
      const stocksData = await stocksRes.json()
      
      console.log('股票池数据:', poolsData)
      console.log('可用股票数据:', stocksData)
      
      setStockPools(poolsData.stock_pools || {})
      setAvailableStocks(stocksData.available_stocks || [])
      
      console.log(`成功加载 ${stocksData.available_stocks?.length || 0} 只可用股票`)
    } catch (err) {
      console.error('加载股票池数据失败:', err)
      setError('无法加载股票池数据')
    }
  }

  // 生成历史数据
  const generateHistoricalData = async () => {
    if (!selectedSymbols.trim()) {
      setError('请先选择股票代码')
      return
    }

    // 验证日期范围
    if (useCustomDateRange) {
      if (!startDate || !endDate) {
        setError('请选择开始和结束日期')
        return
      }
      if (new Date(startDate) >= new Date(endDate)) {
        setError('开始日期必须早于结束日期')
        return
      }
    }

    setLoading(true)
    setError(null)
    
    try {
      let url = `http://127.0.0.1:8000/api/calibration/generate_real_data?symbols=${encodeURIComponent(selectedSymbols)}`
      
      if (useCustomDateRange) {
        url += `&start_date=${startDate}&end_date=${endDate}`
      } else {
        url += `&days=${dataGenDays}`
      }
      
      const response = await fetch(url, {
        method: 'POST'
      })
      
      if (!response.ok) {
        throw new Error('生成数据失败')
      }
      
      const data = await response.json()
      
      if (data.error) {
        setError(data.error)
      } else {
        // 通知父组件数据生成成功
        if (onGenerateData) {
          onGenerateData(selectedSymbols)
        }
        
        // 延迟重新加载可用股票列表，确保数据已写入数据库
        console.log('数据生成成功，正在刷新可用股票列表...')
        setTimeout(async () => {
          await loadStockPools()
          console.log('可用股票列表已刷新')
        }, 1000) // 延迟1秒
      }
    } catch (err: any) {
      setError(err.message || '生成历史数据失败')
    } finally {
      setLoading(false)
    }
  }

  // 添加股票到选择列表
  const addStock = (symbol: string) => {
    const symbols = selectedSymbols.split(',').map(s => s.trim()).filter(s => s)
    if (!symbols.includes(symbol)) {
      onSymbolsChange([...symbols, symbol].join(','))
    }
  }

  // 移除股票
  const removeStock = (symbol: string) => {
    const symbols = selectedSymbols.split(',').map(s => s.trim()).filter(s => s && s !== symbol)
    onSymbolsChange(symbols.join(','))
  }

  // 获取已选择的股票列表
  const getSelectedStocksList = () => {
    return selectedSymbols.split(',').map(s => s.trim()).filter(s => s)
  }

  useEffect(() => {
    loadStockPools()
  }, [])

  return (
    <div className="bg-[#0f1624] border border-gray-700 rounded p-4 h-full flex flex-col">
      <div className="text-lg font-medium text-gray-300 mb-4">🎯 股票池管理</div>
      
      {/* 已选择的股票 */}
      <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
        <div className="text-sm font-medium text-gray-400 mb-2">
          📋 已选择 ({getSelectedStocksList().length})
        </div>
        {getSelectedStocksList().length > 0 ? (
          <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
            {getSelectedStocksList().map(symbol => {
              const stock = availableStocks.find(s => s.symbol === symbol)
              return (
                <div key={symbol} className="flex items-center gap-1 bg-blue-600/20 text-blue-300 px-2 py-1 rounded text-xs">
                  <span>{stock?.name || symbol}</span>
                  <button
                    onClick={() => removeStock(symbol)}
                    className="hover:text-red-400 transition-colors"
                  >
                    ×
                  </button>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-xs text-gray-500">暂未选择股票</div>
        )}
      </div>

      {/* 预设股票池 */}
      <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
        <div className="flex justify-between items-center mb-3">
          <div className="text-sm font-medium text-gray-400">📊 预设股票池</div>
          <button
            onClick={() => setShowPoolManager(true)}
            className="px-2 py-1 bg-purple-600/30 hover:bg-purple-600/50 text-purple-300 text-xs rounded transition-colors"
            title="管理股票池"
          >
            ⚙️ 管理
          </button>
        </div>
        <div className="space-y-3">
          {Object.entries(stockPools).map(([key, pool]) => (
            <div key={key} className="p-2 bg-slate-700/20 rounded border border-slate-600/30">
              <div className="flex justify-between items-center mb-2">
                <div>
                  <div className="text-xs font-medium text-slate-300">{pool.name}</div>
                  <div className="text-xs text-slate-500">{pool.description}</div>
                </div>
                <button
                  onClick={() => onSymbolsChange(pool.symbols.join(','))}
                  className="px-2 py-1 bg-blue-600/30 hover:bg-blue-600/50 text-blue-300 text-xs rounded transition-colors"
                >
                  全选
                </button>
              </div>
              <div className="flex flex-wrap gap-1">
                {pool.symbols.map(symbol => {
                  const stock = availableStocks.find(s => s.symbol === symbol)
                  const name = stock?.name
                  return (
                    <button
                      key={symbol}
                      onClick={() => addStock(symbol)}
                      className="px-2 py-1 bg-slate-600/30 hover:bg-slate-500/50 text-slate-300 text-xs rounded transition-colors"
                      title={`添加 ${symbol}${name ? ` ${name}` : ''}`}
                    >
                      {name ? (
                        <span>
                          <span className="text-slate-200">{name}</span>
                          <span className="text-slate-500 ml-1">{symbol}</span>
                        </span>
                      ) : (
                        <span>{symbol}</span>
                      )}
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 自定义输入 */}
      <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
        <div className="flex justify-between items-center mb-3">
          <div className="text-sm font-medium text-gray-400">✏️ 自定义股票池</div>
          <button
            onClick={() => {
              // 批量导入示例数据
              const exampleData = "000501,000519,002182,600176,600585,002436,600710"
              onSymbolsChange(exampleData)
            }}
            className="px-2 py-1 bg-purple-600/30 hover:bg-purple-600/50 text-purple-300 text-xs rounded transition-colors"
            title="导入示例股票池"
          >
            📋 示例导入
          </button>
        </div>
        <textarea
          value={selectedSymbols}
          onChange={(e) => onSymbolsChange(e.target.value)}
          placeholder="批量导入股票代码，支持多种格式：&#10;• 逗号分隔：000501,000519,002182,600176,600585,002436,600710&#10;• 空格分隔：000501 000519 002182 600176&#10;• 换行分隔：每行一个代码&#10;• 混合格式：000501,000519 002182&#10;600176,600585"
          className="w-full h-20 bg-[#121826] border border-gray-700 rounded px-3 py-2 text-sm resize-none"
        />
        <div className="mt-2 space-y-2">
          <div className="flex justify-between items-center">
            <div className="text-xs text-gray-500">
              💡 支持格式：逗号、空格、换行分隔，自动识别6位股票代码
            </div>
            <button
              onClick={() => {
                // 格式化和验证股票代码
                const cleanedSymbols = selectedSymbols
                  .split(/[,，\s\n\r]+/)  // 支持中英文逗号、空格、换行
                  .map(s => s.trim())
                  .filter(s => s.length > 0)
                  .filter(s => /^\d{6}$/.test(s))  // 验证6位数字格式
                  .filter((s, index, arr) => arr.indexOf(s) === index)  // 去重
                  .sort()  // 排序
                  .join(',')
                
                onSymbolsChange(cleanedSymbols)
              }}
              className="px-2 py-1 bg-green-600/30 hover:bg-green-600/50 text-green-300 text-xs rounded transition-colors"
              title="自动格式化和去重"
            >
              🔧 格式化
            </button>
          </div>
          <div className="text-xs text-blue-400">
            🎯 示例：000501,000519,002182,600176,600585,002436,600710
          </div>
          <div className="text-xs text-gray-600">
            📊 当前已选择 {selectedSymbols.split(',').filter(s => s.trim().length === 6).length} 只股票
          </div>
        </div>
      </div>

      {/* 历史数据生成 */}
      <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700/50">
        <div className="text-sm font-medium text-gray-400 mb-3">📈 历史数据生成</div>
        
        {/* 时间模式选择 */}
        <div className="mb-3">
          <div className="flex gap-2 mb-2">
            <button
              onClick={() => setUseCustomDateRange(false)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                !useCustomDateRange 
                  ? 'bg-blue-600/30 text-blue-300 border border-blue-600/50' 
                  : 'bg-slate-700/30 text-slate-400 border border-slate-600/50'
              }`}
            >
              📅 按天数
            </button>
            <button
              onClick={() => setUseCustomDateRange(true)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                useCustomDateRange 
                  ? 'bg-blue-600/30 text-blue-300 border border-blue-600/50' 
                  : 'bg-slate-700/30 text-slate-400 border border-slate-600/50'
              }`}
            >
              📆 按日期范围
            </button>
          </div>
          
          {!useCustomDateRange ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400">生成天数:</span>
              <input
                type="number"
                value={dataGenDays}
                onChange={(e) => setDataGenDays(parseInt(e.target.value))}
                min="30"
                max="365"
                className="bg-[#121826] border border-gray-700 rounded px-2 py-1 text-xs w-20"
              />
              <span className="text-xs text-gray-400">天</span>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400 w-12">开始:</span>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="bg-[#121826] border border-gray-700 rounded px-2 py-1 text-xs flex-1"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400 w-12">结束:</span>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="bg-[#121826] border border-gray-700 rounded px-2 py-1 text-xs flex-1"
                />
              </div>
            </div>
          )}
        </div>
        
        <button
          onClick={generateHistoricalData}
          disabled={loading || !selectedSymbols.trim()}
          className="w-full px-3 py-2 bg-green-700 hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed text-green-100 text-sm rounded border border-green-600 transition-colors"
        >
          {loading ? '⏳ 生成中...' : '📊 生成历史数据'}
        </button>
        <div className="text-xs text-gray-500 mt-2">
          💡 为选中股票生成预测数据，用于策略回测
        </div>
      </div>

      {/* 可用股票列表 */}
      <div className="flex-1 p-3 bg-gray-800/30 rounded border border-gray-700/50 flex flex-col">
        <div className="flex justify-between items-center mb-3">
          <div className="text-sm font-medium text-gray-400">
            📋 可用股票 ({availableStocks.length})
          </div>
          <button
            onClick={() => setShowAllStocks(!showAllStocks)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {showAllStocks ? '收起' : '展开全部'}
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto space-y-1">
          {availableStocks.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <div className="text-2xl mb-2">📊</div>
              <div className="text-sm">暂无可用股票</div>
              <div className="text-xs mt-1">请先生成历史数据</div>
            </div>
          ) : (
            availableStocks
              .slice(0, showAllStocks ? availableStocks.length : 8)
              .map((stock) => (
              <div
                key={stock.symbol}
                className="flex justify-between items-center p-2 bg-slate-700/20 rounded text-xs cursor-pointer hover:bg-slate-600/30 transition-colors"
                onClick={() => addStock(stock.symbol)}
              >
                <div className="flex-1">
                  <div className="text-slate-300 font-medium">{stock.symbol}</div>
                  <div className="text-slate-500">{stock.name}</div>
                </div>
                <div className="text-slate-400 text-right">
                  <div>{stock.prediction_count} 条</div>
                  <div className="text-xs">{stock.first_date?.slice(5)}</div>
                </div>
              </div>
            ))
          )}
          
          {!showAllStocks && availableStocks.length > 8 && (
            <div className="text-xs text-gray-500 text-center py-2">
              ... 还有 {availableStocks.length - 8} 只股票
            </div>
          )}
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="mt-4 flex gap-2">
        <button
          onClick={() => onSymbolsChange('')}
          className="flex-1 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm rounded border border-slate-600 transition-colors"
        >
          🗑️ 清空
        </button>
        <button
          onClick={() => {
            console.log('手动刷新可用股票列表')
            loadStockPools()
          }}
          className="flex-1 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm rounded border border-slate-600 transition-colors"
        >
          🔄 刷新
        </button>
      </div>

      {/* 错误显示 */}
      {error && (
        <div className="mt-3 p-2 bg-red-600/20 border border-red-600/50 rounded text-red-400 text-xs">
          ❌ {error}
        </div>
      )}

      {/* 说明 */}
      <div className="mt-3 text-xs text-gray-500 text-center">
        💡 选择股票后可进行策略回测和数据生成
      </div>

      {/* 股票池管理器 */}
      {showPoolManager && (
        <StockPoolManager
          onClose={() => setShowPoolManager(false)}
          onSave={() => {
            loadStockPools() // 重新加载股票池数据
          }}
        />
      )}
    </div>
  )
}

export default StockPoolPanel
