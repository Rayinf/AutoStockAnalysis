import React, { useState } from 'react'

interface QlibAnalysis {
  symbol: string
  data_info: {
    total_samples: number
    feature_count: number
    date_range: string
  }
  model_info: {
    status: string
    model_type: string
    n_features: number
    training_samples: number
    mse?: number
    r2_score?: number
  }
  prediction: {
    prediction_value: number
    direction: string
    confidence: number
    signal_strength: number
    model_type: string
  }
  backtest: {
    total_return?: number
    annualized_return?: number
    sharpe_ratio?: number
    max_drawdown?: number
    win_rate?: number
    total_trades?: number
    final_value?: number
    error?: string
  }
  feature_names: string[]
}

interface QlibStatus {
  qlib_available: boolean
  version: string
  features: string[]
}

interface QlibPanelProps {
  symbol?: string
}

export const QlibPanel: React.FC<QlibPanelProps> = ({ symbol }) => {
  const [analysis, setAnalysis] = useState<QlibAnalysis | null>(null)
  const [status, setStatus] = useState<QlibStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 检查Qlib状态
  const checkQlibStatus = async () => {
    try {
      const response = await fetch('/api/qlib/status')
      const data = await response.json()
      setStatus(data)
    } catch (err) {
      console.error('检查Qlib状态失败:', err)
      setError('无法连接到Qlib服务')
    }
  }

  // 运行Qlib分析
  const runQlibAnalysis = async (targetSymbol?: string) => {
    const analyzeSymbol = targetSymbol || symbol || '600519'
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/qlib/analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symbol: analyzeSymbol, 
          lookback_days: 120 
        })
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || '分析失败')
      }
      
      const data = await response.json()
      setAnalysis(data.qlib_analysis)
    } catch (err: any) {
      console.error('Qlib分析失败:', err)
      setError(err.message || 'Qlib分析失败')
    } finally {
      setLoading(false)
    }
  }

  // 初始化时检查状态
  React.useEffect(() => {
    checkQlibStatus()
  }, [])

  return (
    <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="text-lg font-medium text-gray-300">🤖 Qlib AI量化分析</div>
        {status && (
          <div className={`text-xs px-2 py-1 rounded ${
            status.qlib_available ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
          }`}>
            {status.qlib_available ? `可用 ${status.version}` : '不可用'}
          </div>
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-600/20 border border-red-600/50 rounded text-red-400 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* 功能列表 */}
      {status && status.qlib_available && (
        <div className="mb-4 p-3 bg-gray-800/50 rounded">
          <div className="text-xs font-medium text-gray-400 mb-2">📋 可用功能</div>
          <div className="flex flex-wrap gap-1">
            {status.features.map((feature, i) => (
              <span key={i} className="text-xs bg-blue-600/20 text-blue-300 px-2 py-1 rounded">
                {feature}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 分析结果 */}
      {analysis && (
        <div className="space-y-4">
          {/* 数据信息 */}
          <div className="p-3 bg-gray-800/50 rounded">
            <div className="text-sm font-medium text-gray-300 mb-2">📊 数据概览</div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <div className="text-gray-400">样本数</div>
                <div className="text-white font-mono">{analysis.data_info.total_samples}</div>
              </div>
              <div>
                <div className="text-gray-400">特征数</div>
                <div className="text-white font-mono">{analysis.data_info.feature_count}</div>
              </div>
              <div>
                <div className="text-gray-400">时间范围</div>
                <div className="text-white font-mono text-xs">{analysis.data_info.date_range.split(' to ')[0]}</div>
              </div>
            </div>
          </div>

          {/* 模型信息 */}
          <div className="p-3 bg-gray-800/50 rounded">
            <div className="text-sm font-medium text-gray-300 mb-2">🔧 模型信息</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-400">模型类型: </span>
                <span className="text-blue-300">{analysis.model_info.model_type}</span>
              </div>
              <div>
                <span className="text-gray-400">训练样本: </span>
                <span className="text-yellow-400">{analysis.model_info.training_samples}</span>
              </div>
              {analysis.model_info.r2_score && (
                <>
                  <div>
                    <span className="text-gray-400">R² 得分: </span>
                    <span className="text-green-400">{analysis.model_info.r2_score.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">MSE: </span>
                    <span className="text-purple-400">{analysis.model_info.mse?.toFixed(6)}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* AI预测结果 */}
          <div className="p-3 bg-gray-800/50 rounded">
            <div className="text-sm font-medium text-gray-300 mb-2">🎯 AI预测</div>
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center p-2 bg-gray-700/50 rounded">
                <div className={`text-lg font-bold ${
                  analysis.prediction.direction === '上涨' ? 'text-red-400' : 
                  analysis.prediction.direction === '下跌' ? 'text-green-400' : 'text-yellow-400'
                }`}>
                  {analysis.prediction.direction}
                </div>
                <div className="text-xs text-gray-400">预测方向</div>
              </div>
              <div className="text-center p-2 bg-gray-700/50 rounded">
                <div className="text-lg font-bold text-blue-400">
                  {(analysis.prediction.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-400">置信度</div>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500 text-center">
              信号强度: {(analysis.prediction.signal_strength * 100).toFixed(1)}% | 
              模型: {analysis.prediction.model_type}
            </div>
          </div>

          {/* 回测结果 */}
          {analysis.backtest && !analysis.backtest.error && (
            <div className="p-3 bg-gray-800/50 rounded">
              <div className="text-sm font-medium text-gray-300 mb-2">📈 策略回测</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-400">总收益率: </span>
                  <span className={analysis.backtest.total_return! > 0 ? 'text-red-400' : 'text-green-400'}>
                    {(analysis.backtest.total_return! * 100).toFixed(2)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">夏普比率: </span>
                  <span className="text-yellow-400">{analysis.backtest.sharpe_ratio?.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-gray-400">最大回撤: </span>
                  <span className="text-red-400">{(analysis.backtest.max_drawdown! * 100).toFixed(2)}%</span>
                </div>
                <div>
                  <span className="text-gray-400">胜率: </span>
                  <span className="text-blue-400">{(analysis.backtest.win_rate! * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="mt-2 text-xs text-gray-500 text-center">
                总交易: {analysis.backtest.total_trades} | 
                最终价值: ¥{analysis.backtest.final_value?.toFixed(0)}
              </div>
            </div>
          )}

          {/* 主要特征 */}
          <div className="p-3 bg-gray-800/50 rounded">
            <div className="text-sm font-medium text-gray-300 mb-2">🔍 主要特征</div>
            <div className="flex flex-wrap gap-1">
              {analysis.feature_names.slice(0, 8).map((feature, i) => (
                <span key={i} className="text-xs bg-purple-600/20 text-purple-300 px-2 py-1 rounded">
                  {feature}
                </span>
              ))}
              {analysis.feature_names.length > 8 && (
                <span className="text-xs text-gray-500">+{analysis.feature_names.length - 8}个特征</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 控制按钮 */}
      <div className="mt-4 flex gap-2">
        <button 
          onClick={() => runQlibAnalysis()}
          disabled={loading || !status?.qlib_available}
          className="flex-1 px-3 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-sm rounded hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? '🔄 分析中...' : '🚀 启动Qlib分析'}
        </button>
        
        <button 
          onClick={checkQlibStatus}
          className="px-3 py-2 bg-gray-700 text-gray-300 text-sm rounded hover:bg-gray-600"
        >
          🔍 检查状态
        </button>
      </div>

      {/* 说明 */}
      <div className="mt-3 text-xs text-gray-500 text-center">
        💡 Qlib是微软开源的AI量化投资平台，提供专业的量化分析能力
      </div>
    </div>
  )
}

export default QlibPanel
