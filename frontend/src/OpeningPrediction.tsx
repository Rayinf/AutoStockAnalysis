import React, { useState, useEffect } from 'react'

// 类型定义
interface TechnicalIndicators {
  MA5: number
  MA10: number
  MA20: number
  RSI: number
  MACD: number
  MACD_Signal: number
  BB_Upper: number
  BB_Lower: number
  BB_Middle: number
  current_price: number
  volume: number
  volume_ma: number
}

interface VolumeAnalysis {
  pattern: string
  volume_ratio: number
  recent_avg_volume: number
  historical_avg_volume: number
  price_volume_correlation: number
}

interface StockPrediction {
  symbol: string
  name: string
  current_price: number
  predicted_direction: 'up' | 'down' | 'hold'
  confidence: number
  target_price: number
  stop_loss: number
  recommendation: 'buy' | 'sell' | 'hold'
  reason: string
  risk_level: 'low' | 'medium' | 'high'
  volume_analysis: VolumeAnalysis
  technical_indicators: TechnicalIndicators
}

interface MarketOverview {
  market_sentiment: 'bullish' | 'bearish' | 'neutral'
  major_indices: Record<string, {
    current: number
    change: number
    volume: number
  }>
  market_news: string[]
  risk_factors: string[]
}

interface PortfolioRecommendation {
  action: string
  buy_list: Array<{
    symbol: string
    name: string
    action: string
    reason: string
    target_price?: number
    suggested_buy_price?: number
    confidence: number
    current_holding?: number
    suggested_quantity?: number
  }>
  sell_list: Array<{
    symbol: string
    name: string
    action: string
    reason: string
    stop_loss?: number
    suggested_sell_price?: number
    confidence: number
    current_holding?: number
    suggested_quantity?: number
  }>
  hold_list: Array<{
    symbol: string
    name: string
    action: string
    reason: string
    current_holding?: number
  }>
  reason: string
}

interface PredictionResult {
  prediction_time: string
  market_overview: MarketOverview
  stock_predictions: StockPrediction[]
  portfolio_recommendation: PortfolioRecommendation
  risk_assessment: {
    overall_risk: 'low' | 'medium' | 'high'
    confidence_level: number
    high_risk_stocks: number
    market_risk_factors: string[]
  }
}

interface PlannedTrade {
  action: 'buy' | 'sell'
  symbol: string
  name: string
  price: number
  quantity: number
  note?: string
}

const OpeningPrediction: React.FC = () => {
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [planned, setPlanned] = useState<PlannedTrade[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedPool, setSelectedPool] = useState<string>('我的自选')
  const [stockPools, setStockPools] = useState<Record<string, string[]>>({})
  const [poolsLoading, setPoolsLoading] = useState(true)

  // 获取股票池配置
  useEffect(() => {
    const fetchStockPools = async () => {
      try {
        setPoolsLoading(true)
        const response = await fetch('/api/strategy/stock_pools')
        const data = await response.json()
        
        if (data.stock_pools) {
          const pools: Record<string, string[]> = {}
          Object.entries(data.stock_pools).forEach(([key, value]: [string, any]) => {
            // 使用key作为显示名称，value.symbols作为股票代码
            pools[key] = value.symbols
          })
          setStockPools(pools)
          
          // 设置默认选择的股票池
          if (pools['我的自选']) {
            setSelectedPool('我的自选')
          } else {
            const firstPool = Object.keys(pools)[0]
            if (firstPool) {
              setSelectedPool(firstPool)
            }
          }
        }
      } catch (err) {
        console.error('获取股票池失败:', err)
        // 不再使用硬编码默认股票池，显示错误状态
        setStockPools({})
        setError('无法加载股票池数据，请检查后端服务')
      } finally {
        setPoolsLoading(false)
      }
    }

    fetchStockPools()
  }, [])

  const ensureName = async (symbol: string, fallback: string) => {
    if (!symbol) return fallback
    if (fallback && fallback !== symbol) return fallback
    try {
      const res = await fetch(`/api/stock_name/${symbol}`)
      const jd = await res.json()
      return (jd && jd.name) ? jd.name : fallback || symbol
    } catch {
      return fallback || symbol
    }
  }

  const generatePrediction = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/opening_prediction', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stock_pool: stockPools[selectedPool] || []
        }),
      })

      const data = await response.json()
      
      if (data.success) {
        const pr = data.data as PredictionResult
        const fixedStocks = await Promise.all((pr.stock_predictions || []).map(async (s: any) => ({
          ...s,
          name: await ensureName(s.symbol, s.name)
        })))
        const fixedPortfolio = {
          ...pr.portfolio_recommendation,
          buy_list: await Promise.all((pr.portfolio_recommendation?.buy_list || []).map(async (x: any) => ({...x, name: await ensureName(x.symbol, x.name)}))),
          sell_list: await Promise.all((pr.portfolio_recommendation?.sell_list || []).map(async (x: any) => ({...x, name: await ensureName(x.symbol, x.name)}))),
          hold_list: await Promise.all((pr.portfolio_recommendation?.hold_list || []).map(async (x: any) => ({...x, name: await ensureName(x.symbol, x.name)})))
        }
        setPredictionResult({
          ...pr,
          stock_predictions: fixedStocks,
          portfolio_recommendation: fixedPortfolio
        })
        if (Array.isArray(data.planned_trades)) {
          const fixedPlanned: PlannedTrade[] = await Promise.all((data.planned_trades as PlannedTrade[]).map(async (t: PlannedTrade) => ({
            ...t,
            name: await ensureName(t.symbol, t.name)
          })))
          setPlanned(fixedPlanned)
        } else {
          setPlanned([])
        }
      } else {
        setError(data.error || '预测失败')
      }
    } catch (err) {
      setError('网络请求失败')
      console.error('预测请求失败:', err)
    } finally {
      setLoading(false)
    }
  }


  // 格式化时间
  const formatTime = (isoString: string) => {
    return new Date(isoString).toLocaleString('zh-CN')
  }

  // 获取方向颜色
  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case 'up': return 'text-red-400'
      case 'down': return 'text-green-400'
      default: return 'text-gray-400'
    }
  }

  // 获取推荐颜色
  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'buy': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'sell': return 'bg-green-500/20 text-green-400 border-green-500/30'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  // 获取风险等级颜色
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-400'
      case 'medium': return 'text-yellow-400'
      case 'high': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  // 获取市场情绪颜色
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'text-red-400'
      case 'bearish': return 'text-green-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-900 min-h-screen">
      {/* 头部 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">📊 开盘预测系统</h1>
        <p className="text-gray-400">基于AKShare数据的实时股票分析与交易策略</p>
      </div>

      {/* 控制面板 */}
      <div className="bg-gray-800/50 rounded-lg p-6 mb-6 border border-gray-700/50">
        <div className="flex items-center gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">选择股票池</label>
            {poolsLoading ? (
              <div className="bg-gray-700 border border-gray-600 text-gray-400 rounded-lg px-3 py-2">
                加载中...
              </div>
            ) : (
              <select
                value={selectedPool}
                onChange={(e) => setSelectedPool(e.target.value)}
                className="bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500"
              >
                {Object.keys(stockPools).map(pool => (
                  <option key={pool} value={pool}>{pool}</option>
                ))}
              </select>
            )}
          </div>
          
          <div className="flex-1" />
          
          <button
            onClick={generatePrediction}
            disabled={loading || poolsLoading || !stockPools[selectedPool]}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              loading || poolsLoading || !stockPools[selectedPool]
                ? 'bg-gray-600 text-gray-300 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {loading ? '🔄 分析中...' : poolsLoading ? '⏳ 加载中...' : '🚀 生成预测'}
          </button>
        </div>

        {/* 选中股票池显示 */}
        <div className="text-sm text-gray-400">
          <span className="font-medium">当前股票池：</span>
          {poolsLoading ? '加载中...' : (stockPools[selectedPool]?.join(', ') || '无')}
        </div>
      </div>

      {/* 错误显示 */}
      {error && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 mb-6">
          <div className="text-red-400">❌ {error}</div>
        </div>
      )}

      {/* 预测结果 */}
      {predictionResult && (
        <div className="space-y-6">
          {/* 待执行交易 */}
          {planned.length > 0 && (
            <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700/50">
              <h2 className="text-xl font-bold text-white mb-4">📝 待执行交易（计划）</h2>
              <div className="grid gap-3">
                {planned.map((t, idx) => (
                  <div key={idx} className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/40 flex items-center justify-between">
                    <div className="text-sm text-gray-300">
                      <span className="text-white font-bold mr-2">{t.symbol}</span>
                      <span className="text-gray-400 mr-4">{t.name}</span>
                      <span className={t.action === 'buy' ? 'text-red-400' : 'text-green-400'}>
                        {t.action === 'buy' ? '买入' : '卖出'}
                      </span>
                      <span className="text-gray-400 ml-4">数量: {t.quantity}</span>
                      <span className="text-gray-400 ml-4">价格: ¥{t.price.toFixed(2)}</span>
                    </div>
                    {/* 这里预留执行按钮，后续可调用 /api/portfolio/transaction 落地 */}
                    <button
                      className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white text-sm"
                      onClick={async () => {
                        try {
                          const res = await fetch('/api/portfolio/transaction', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              symbol: t.symbol,
                              name: t.name,
                              action: t.action,
                              quantity: t.quantity,
                              price: t.price,
                              fee: 0,
                              note: t.note || 'execute_from_ui'
                            })
                          })
                          const jd = await res.json()
                          if (!jd.success) alert(jd.error || '执行失败')
                          else alert('已提交')
                        } catch {
                          alert('网络错误')
                        }
                      }}
                    >执行</button>
                  </div>
                ))}
              </div>
            </div>
          )}
          {/* 市场概览 */}
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700/50">
            <h2 className="text-xl font-bold text-white mb-4">🌐 市场概览</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="bg-gray-700/30 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">市场情绪</div>
                <div className={`text-lg font-bold ${getSentimentColor(predictionResult.market_overview.market_sentiment)}`}>
                  {predictionResult.market_overview.market_sentiment === 'bullish' ? '🐂 看涨' :
                   predictionResult.market_overview.market_sentiment === 'bearish' ? '🐻 看跌' : '😐 中性'}
                </div>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">整体风险</div>
                <div className={`text-lg font-bold ${getRiskColor(predictionResult.risk_assessment.overall_risk)}`}>
                  {predictionResult.risk_assessment.overall_risk === 'low' ? '🟢 低风险' :
                   predictionResult.risk_assessment.overall_risk === 'medium' ? '🟡 中等风险' : '🔴 高风险'}
                </div>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-1">预测置信度</div>
                <div className="text-lg font-bold text-blue-400">
                  {(predictionResult.risk_assessment.confidence_level * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* 主要指数 */}
            {Object.keys(predictionResult.market_overview.major_indices).length > 0 && (
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-white mb-2">📈 主要指数</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {Object.entries(predictionResult.market_overview.major_indices).map(([name, data]) => (
                    <div key={name} className="bg-gray-700/20 rounded-lg p-3">
                      <div className="text-sm font-medium text-gray-300">{name}</div>
                      <div className="text-lg font-bold text-white">{data.current.toFixed(2)}</div>
                      <div className={`text-sm ${data.change >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                        {data.change >= 0 ? '+' : ''}{data.change.toFixed(2)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 风险因素 */}
            {predictionResult.risk_assessment.market_risk_factors.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">⚠️ 风险因素</h3>
                <ul className="space-y-1">
                  {predictionResult.risk_assessment.market_risk_factors.map((factor, index) => (
                    <li key={index} className="text-yellow-400 text-sm">• {factor}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* 投资组合建议 */}
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700/50">
            <h2 className="text-xl font-bold text-white mb-4">💼 投资组合建议</h2>
            
            {/* 买入建议 */}
            {predictionResult.portfolio_recommendation.buy_list && predictionResult.portfolio_recommendation.buy_list.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-red-400 mb-3">🛒 建议买入</h3>
                <div className="grid gap-3">
                  {predictionResult.portfolio_recommendation.buy_list.map((rec, index) => (
                    <div key={index} className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <span className="font-bold text-white">{rec.symbol}</span>
                          <span className="text-gray-400 ml-2">{rec.name}</span>
                          {rec.current_holding ? (
                            <span className="text-blue-400 ml-2 text-sm">持仓: {rec.current_holding}股</span>
                          ) : null}
                        </div>
                        <div className="text-right">
                          {rec.suggested_buy_price && (
                            <div className="text-green-400 font-bold">建议买入价: ¥{rec.suggested_buy_price.toFixed(2)}</div>
                          )}
                          {rec.target_price && (
                            <div className="text-red-400 font-bold">目标价: ¥{rec.target_price.toFixed(2)}</div>
                          )}
                          <div className="text-sm text-gray-400">置信度: {(rec.confidence * 100).toFixed(1)}%</div>
                          {rec.suggested_quantity && (
                            <div className="text-sm text-yellow-400">建议: {rec.suggested_quantity}股</div>
                          )}
                        </div>
                      </div>
                      <div className="text-sm text-gray-300">{rec.reason}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 卖出建议 */}
            {predictionResult.portfolio_recommendation.sell_list && predictionResult.portfolio_recommendation.sell_list.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-green-400 mb-3">💰 建议卖出</h3>
                <div className="grid gap-3">
                  {predictionResult.portfolio_recommendation.sell_list.map((rec, index) => (
                    <div key={index} className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <span className="font-bold text-white">{rec.symbol}</span>
                          <span className="text-gray-400 ml-2">{rec.name}</span>
                          {rec.current_holding ? (
                            <span className="text-blue-400 ml-2 text-sm">持仓: {rec.current_holding}股</span>
                          ) : null}
                        </div>
                        <div className="text-right">
                          {rec.suggested_sell_price && (
                            <div className="text-orange-400 font-bold">建议卖出价: ¥{rec.suggested_sell_price.toFixed(2)}</div>
                          )}
                          {rec.stop_loss && (
                            <div className="text-red-400 font-bold">止损价: ¥{rec.stop_loss.toFixed(2)}</div>
                          )}
                          <div className="text-sm text-gray-400">置信度: {(rec.confidence * 100).toFixed(1)}%</div>
                          {rec.suggested_quantity && (
                            <div className="text-sm text-red-400">建议卖出: {rec.suggested_quantity}股</div>
                          )}
                        </div>
                      </div>
                      <div className="text-sm text-gray-300">{rec.reason}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 持有建议 */}
            {predictionResult.portfolio_recommendation.hold_list && predictionResult.portfolio_recommendation.hold_list.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-blue-400 mb-3">💎 建议持有</h3>
                <div className="grid gap-3">
                  {predictionResult.portfolio_recommendation.hold_list.map((rec, index) => (
                    <div key={index} className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <span className="font-bold text-white">{rec.symbol}</span>
                          <span className="text-gray-400 ml-2">{rec.name}</span>
                          {rec.current_holding ? (
                            <span className="text-blue-400 ml-2 text-sm">持仓: {rec.current_holding}股</span>
                          ) : null}
                        </div>
                        <div className="text-right">
                          <div className="text-sm text-blue-400">持有观望</div>
                        </div>
                      </div>
                      <div className="text-sm text-gray-300">{rec.reason}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 整体策略 */}
            <div className="bg-gray-700/30 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-2">🎯 整体策略</h3>
              <div className="text-blue-400 font-medium">
                {predictionResult.portfolio_recommendation.action === 'buy' ? '🚀 建议买入' :
                 predictionResult.portfolio_recommendation.action === 'sell' ? '🛡️ 建议卖出' :
                 predictionResult.portfolio_recommendation.action === 'hold' ? '⚖️ 持有观望' :
                 predictionResult.portfolio_recommendation.action === 'mixed' ? '🔀 混合策略' :
                 '⚖️ 中性持有'}
              </div>
              {predictionResult.portfolio_recommendation.reason && (
                <div className="text-sm text-gray-300 mt-2">
                  {predictionResult.portfolio_recommendation.reason}
                </div>
              )}
            </div>
          </div>

          {/* 个股预测详情 */}
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700/50">
            <h2 className="text-xl font-bold text-white mb-4">📊 个股分析详情</h2>
            <div className="grid gap-4">
              {predictionResult.stock_predictions.map((stock, index) => (
                <div key={index} className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/50">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="text-lg font-bold text-white">{stock.symbol}</h3>
                      <p className="text-gray-400">{stock.name}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-xl font-bold text-white">¥{stock.current_price.toFixed(2)}</div>
                      <div className={`inline-block px-2 py-1 rounded-full text-xs border ${getRecommendationColor(stock.recommendation)}`}>
                        {stock.recommendation === 'buy' ? '买入' : stock.recommendation === 'sell' ? '卖出' : '持有'}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                    <div>
                      <div className="text-xs text-gray-400">预测方向</div>
                      <div className={`font-bold ${getDirectionColor(stock.predicted_direction)}`}>
                        {stock.predicted_direction === 'up' ? '↗️ 上涨' : 
                         stock.predicted_direction === 'down' ? '↘️ 下跌' : '➡️ 横盘'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">置信度</div>
                      <div className="font-bold text-blue-400">{(stock.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">目标价</div>
                      <div className="font-bold text-yellow-400">¥{stock.target_price.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">止损价</div>
                      <div className="font-bold text-red-400">¥{stock.stop_loss.toFixed(2)}</div>
                    </div>
                  </div>

                  <div className="mb-3">
                    <div className="text-xs text-gray-400 mb-1">分析理由</div>
                    <div className="text-sm text-gray-300">{stock.reason}</div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                    <div>
                      <span className="text-gray-400">风险等级: </span>
                      <span className={getRiskColor(stock.risk_level)}>
                        {stock.risk_level === 'low' ? '低' : stock.risk_level === 'medium' ? '中' : '高'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">RSI: </span>
                      <span className="text-white">{stock.technical_indicators.RSI?.toFixed(1) || 'N/A'}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">MA5: </span>
                      <span className="text-white">¥{stock.technical_indicators.MA5?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">成交量: </span>
                      <span className={stock.volume_analysis.volume_ratio > 1.2 ? 'text-red-400' : 
                                     stock.volume_analysis.volume_ratio < 0.8 ? 'text-green-400' : 'text-gray-300'}>
                        {stock.volume_analysis.volume_ratio?.toFixed(1)}x
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 预测时间 */}
          <div className="text-center text-sm text-gray-500">
            预测生成时间: {formatTime(predictionResult.prediction_time)}
          </div>
        </div>
      )}
    </div>
  )
}

export default OpeningPrediction
