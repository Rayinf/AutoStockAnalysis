import { useEffect, useMemo, useState, useRef } from 'react'
import './App.css'
import StrategyPanel from './StrategyPanel'
import StockPoolPanel from './StockPoolPanel'
import OpeningPrediction from './OpeningPrediction'
import PortfolioManager from './PortfolioManager'
import ReactECharts from 'echarts-for-react'

type Mode = 'auto' | 'quick' | 'llm'
type Tab = 'fixed' | 'llm' | 'strategy' | 'prediction' | 'portfolio'
type FixedType = 'MA5' | 'MA20' | 'MA60' | 'RSI14' | 'MACD' | 'BOLL' | 'OBV' | 'ATR' | 'REALTIME' | 'TOPUP' | 'TOPDOWN'

type SeriesMap = Record<string, { x: string[]; y: (number | null)[] }>
interface StructuredAnalysis {
  trend_signal: string
  confidence: number
  key_signals: string[]
  risk_points: string[]
  support_level?: number
  resistance_level?: number
  summary: string
}

interface PricePrediction {
  current_price: number
  prediction_days: string
  up_probability: number
  down_probability: number
  price_range_lower: number
  price_range_upper: number
  confidence_level: number
  key_factors: string[]
  model_details?: {
    feature_score: number
    volatility_regime: string
    signal_strength: string
  }
  error?: string
}

interface FixedResponse {
  meta: { symbol: string; start: string; end: string }
  series: SeriesMap
  indicators: Record<string, number | null>
  realtime: any
  board: Array<{ 代码: string; 名称: string; 最新价: number; 涨跌幅: number }> | null
  summary: string | null
  structured_analysis?: StructuredAnalysis
  price_prediction?: PricePrediction
  risk?: {
    max_drawdown: number
    volatility: number
    var_95: number
    win_rate: number
  }
  logs?: string[]
}

interface HistoryItem {
  query: string
  mode: Mode
  result: string
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('fixed')
  const [query, setQuery] = useState('根据MA20分析一下000501')
  const [mode, setMode] = useState<Mode>('llm')
  const [fixedTypes, setFixedTypes] = useState<FixedType[]>(['MA20'])
  const [stockCode, setStockCode] = useState('000501')
  const [lookbackDays, setLookbackDays] = useState(120)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState('')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [fixedRes, setFixedRes] = useState<FixedResponse | null>(null)
  const [nameMap, setNameMap] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  const [liveLogs, setLiveLogs] = useState<string[]>([])
  const [logsCollapsed, setLogsCollapsed] = useState(false)
  const [selectedSymbols, setSelectedSymbols] = useState<string>('')
  const timerRef = useRef<number | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const logContainerRef = useRef<HTMLDivElement>(null)

  const themeClass = useMemo(() => 'dark bg-[#0b0f17] text-gray-100 min-h-screen', [])

  // 自动滚动日志容器到底部
  useEffect(() => {
    if (logContainerRef.current && liveLogs.length > 0) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [liveLogs])

  const buildFixedQuery = (): string => {
    const code = (stockCode || '').trim()
    // 用于历史记录摘要；真实调用走 /api/fixed_analyse
    return `固定分析(${fixedTypes.join(',')}): ${code || '未指定'}`
  }

  const runSearch = async () => {
    const isFixed = activeTab === 'fixed'
    const finalQuery = isFixed ? buildFixedQuery() : query
    const finalMode: Mode | null = isFixed ? 'quick' : 'llm'
    if (isFixed) {
      if (!stockCode.trim() && !fixedTypes.some(t => t === 'TOPUP' || t === 'TOPDOWN')) return
    } else if (!finalQuery.trim()) return
    setLoading(true)
    setResult('')
    setFixedRes(null)
    setError(null)
    setLogsCollapsed(false) // 新分析开始时展开日志
    // 立即显示初始日志
    const initialLogs = [`[${new Date().toLocaleTimeString()}] 🚀 开始请求: ${isFixed ? '固定分析' : 'LLM分析'}`, `[${new Date().toLocaleTimeString()}] ⚙️ 准备参数…`]
    setLiveLogs(initialLogs)
    // 启动简单的等待计时日志
    let sec = 0
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = window.setInterval(() => {
      sec += 1
      setLiveLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ⏳ 执行中… 第 ${sec} 秒`])
    }, 1000)
    try {
      if (isFixed) {
        // 使用 SSE 流式接口
        // 关闭旧的事件源
        if (esRef.current) {
          esRef.current.close()
          esRef.current = null
        }
        const params = new URLSearchParams({
          symbol: stockCode.trim(),
          analyses: fixedTypes.join(','),
          lookback_days: String(lookbackDays)
        })
        const url = `http://127.0.0.1:8000/api/qlib/features_stream?${params.toString()}`
        console.log('SSE URL:', url)
        setLiveLogs(prev => [...prev, `🔗 连接SSE: ${url}`])
        await new Promise<void>((resolve, reject) => {
          try {
            const es = new EventSource(url)
            esRef.current = es
            console.log('EventSource created:', es)

            // 监听连接状态
            es.onopen = () => {
              console.log('SSE connection opened')
              setLiveLogs(prev => [...prev, `✅ SSE连接已打开`])
            }

            const onLog = (e: MessageEvent) => {
              console.log('SSE Log:', e.data) // 调试日志
              setLiveLogs(prev => {
                const newLogs = [...prev, `[${new Date().toLocaleTimeString()}] ${e.data}`]
                // 保持最多50条日志
                return newLogs.slice(-50)
              })
            }

            const onLLMInput = (e: MessageEvent) => {
              console.log('SSE LLM Input:', e.data)
              setLiveLogs(prev => {
                const newLogs = [...prev,
                  `[${new Date().toLocaleTimeString()}] 🔍 LLM输入:`,
                  `---`,
                  e.data.length > 500 ? `${e.data.substring(0, 500)}...` : e.data,
                  `---`
                ]
                // 保持最多50条日志
                return newLogs.slice(-50)
              })
            }
            const onDone = (e: MessageEvent) => {
              try {
                // 检查数据是否为空或不完整
                if (!e.data || e.data.trim() === '') {
                  console.warn('收到空的SSE数据')
                  return
                }
                
                const data = JSON.parse(e.data)
                if (data && data.series) {
                  const fixed: FixedResponse = {
                    meta: { symbol: stockCode.trim(), start: '', end: '' },
                    series: data.series as SeriesMap,
                    indicators: {},
                    realtime: null,
                    board: null,
                    summary: null,
                    price_prediction: undefined
                  }
                  setFixedRes(fixed)
                  setResult('Qlib因子已返回，正在生成预测与结构化结论…')

                  // 补发统一分析请求，获取预测/结构化/风险
                  fetch('http://127.0.0.1:8000/api/qlib/analyse', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      symbol: stockCode.trim(),
                      analyses: fixedTypes,
                      lookback_days: lookbackDays,
                      horizon_days: 5
                    })
                  }).then(async (res) => {
                    if (!res.ok) throw new Error('分析失败')
                    const full = await res.json()
                    setFixedRes(prev => prev ? {
                      ...prev,
                      structured_analysis: full.structured_analysis,
                      price_prediction: full.price_prediction,
                      risk: full.risk,
                    } : prev)
                    setResult('已生成预测与结构化结论')
                  }).catch(err => {
                    console.error('补发分析失败:', err)
                  })
                } else {
                  const msg = data?.error || 'Qlib因子构建失败'
                  setError(msg)
                  setResult(msg)
                }
              } catch (err) {
                console.error('JSON解析失败:', err)
                console.error('原始数据长度:', e.data?.length)
                console.error('原始数据前100字符:', e.data?.slice(0, 100))
                console.error('原始数据后100字符:', e.data?.slice(-100))
                setError('数据解析失败，请重试')
                setResult('数据解析失败，请重试')
              } finally {
                es.close()
                esRef.current = null
                resolve()
              }
            }
            const onError = (error: Event) => {
              console.error('SSE connection error:', error)
              setLiveLogs(prev => [...prev, `❌ SSE连接出错: ${error.type}`])
              try { es.close() } catch {}
              esRef.current = null
              resolve()
            }

            es.addEventListener('log', onLog)
            es.addEventListener('llm_input', onLLMInput)
            es.addEventListener('status', onLog)
            es.addEventListener('meta', onLog)
            es.addEventListener('cache_hit', onLog)
            es.addEventListener('built', onLog)
            es.addEventListener('done', onDone)
            es.onerror = onError
          } catch (err) {
            setLiveLogs(prev => [...prev, 'SSE初始化失败'])
            reject(err as any)
          }
        })
      } else {
        // LLM 模式：若输入是6位代码，先前置拉取技术指标以绘图
        const symbol = (stockCode || '').trim()
        if (/^\d{6}$/.test(symbol)) {
          if (esRef.current) { try { esRef.current.close() } catch {} esRef.current = null }
          const params = new URLSearchParams({
            symbol,
            analyses: ['MA20','MACD','BOLL','OBV','ATR','REALTIME','TOPUP','TOPDOWN'].join(','),
            lookback_days: String(lookbackDays)
          })
          const url = `http://127.0.0.1:8000/api/qlib/features_stream?${params.toString()}`
          setLiveLogs(prev => [...prev, `🔗 连接SSE(LLM前置特征): ${url}`])
          await new Promise<void>((resolve) => {
            try {
              const es = new EventSource(url)
              esRef.current = es
              const onLog = (e: MessageEvent) => setLiveLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${e.data}`].slice(-50))
              const onDone = (e: MessageEvent) => {
                try {
                  if (!e.data || e.data.trim() === '') return
                  const data = JSON.parse(e.data)
                  if (data && data.series) {
                    const fixed: FixedResponse = {
                      meta: { symbol, start: '', end: '' },
                      series: data.series as SeriesMap,
                      indicators: {},
                      realtime: null,
                      board: null,
                      summary: null,
                      price_prediction: undefined
                    }
                    setFixedRes(fixed)
                  }
                } catch {} finally {
                  try { es.close() } catch {}
                  esRef.current = null
                  resolve()
                }
              }
              es.addEventListener('log', onLog)
              es.addEventListener('status', onLog)
              es.addEventListener('built', onLog)
              es.addEventListener('done', onDone)
              es.onerror = () => { try { es.close() } catch {}; esRef.current = null; resolve() }
            } catch { resolve() }
          })
        }

        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: finalQuery, mode: finalMode }),
        })
        const data = await res.json()
        if (res.ok) {
          setResult(data.result || '')
          setLiveLogs(prev => [...prev, 'LLM返回成功'])
        } else {
          const msg = data?.detail || 'LLM 查询失败'
          setResult(msg)
          setError(msg)
          setLiveLogs(prev => [...prev, `错误: ${msg}`])
        }
      }
      // 刷新历史
      try {
        const hres = await fetch('/api/history')
        if (!hres.ok) {
          console.warn('历史记录API返回错误状态:', hres.status)
          return
        }
        const hdata = await hres.json()
        setHistory(hdata)
      } catch (historyErr) {
        console.error('获取历史记录失败:', historyErr)
        // 不影响主要功能，只是记录错误
      }
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : String(e)
      const msg = `请求失败: ${errorMsg}`
      setResult(msg)
      setError(errorMsg)
      setLiveLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ❌ 异常: ${msg}`])
    } finally {
      setLoading(false)
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
        timerRef.current = null
      }
      setLiveLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 🏁 流程结束`])
    }
  }

  // 持久化：记住固定查询配置与代码
  useEffect(() => {
    try {
      const saved = localStorage.getItem('agstock.fixed')
      if (saved) {
        const o = JSON.parse(saved)
        if (Array.isArray(o.fixedTypes)) setFixedTypes(o.fixedTypes)
        if (typeof o.stockCode === 'string') setStockCode(o.stockCode)
        if (typeof o.lookbackDays === 'number') setLookbackDays(o.lookbackDays)
      }
    } catch {}
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem('agstock.fixed', JSON.stringify({ fixedTypes, stockCode, lookbackDays }))
    } catch {}
  }, [fixedTypes, stockCode, lookbackDays])

  // 复制结论
  const copySummary = async () => {
    if (!fixedRes?.summary) return
    try {
      await navigator.clipboard.writeText(fixedRes.summary)
    } catch {}
  }

  useEffect(() => {
    ;(async () => {
      try {
        const hres = await fetch('/api/history')
        if (!hres.ok) {
          console.warn('初始历史记录API返回错误状态:', hres.status)
          return
        }
        const hdata = await hres.json()
        setHistory(hdata)
      } catch (err) {
        console.error('初始获取历史记录失败:', err)
      }
    })()
  }, [])

  // 载入可用股票名称映射（用于在图表/标题显示股票名称）
  useEffect(() => {
    ;(async () => {
      try {
        const res = await fetch('/api/strategy/available_stocks')
        if (!res.ok) return
        const data = await res.json()
        const map = (data?.stocks && typeof data.stocks === 'object') ? data.stocks : data
        if (map && typeof map === 'object') setNameMap(map)
      } catch {}
    })()
  }, [])

  return (
    <div className={themeClass}>
      <div className="max-w-[1600px] mx-auto p-6">
        <header className="mb-6">
          <h1 className="text-xl font-semibold mb-3">指标分析（AKShare + LLM）</h1>
          <div className="flex items-center gap-3 text-sm">
            <button
              onClick={() => setActiveTab('fixed')}
              className={`px-3 py-1 rounded border ${activeTab==='fixed' ? 'bg-blue-600' : 'bg-[#121826]'} border-gray-700`}
            >固定查询</button>
            <button
              onClick={() => setActiveTab('llm')}
              className={`px-3 py-1 rounded border ${activeTab==='llm' ? 'bg-blue-600' : 'bg-[#121826]'} border-gray-700`}
            >LLM 查询</button>
            <button
              onClick={() => setActiveTab('strategy')}
              className={`px-3 py-1 rounded border ${activeTab==='strategy' ? 'bg-green-600' : 'bg-[#121826]'} border-gray-700`}
            >📈 策略回测</button>
            <button
              onClick={() => setActiveTab('prediction')}
              className={`px-3 py-1 rounded border ${activeTab==='prediction' ? 'bg-orange-600' : 'bg-[#121826]'} border-gray-700`}
            >🔮 开盘预测</button>
            <button
              onClick={() => setActiveTab('portfolio')}
              className={`px-3 py-1 rounded border ${activeTab==='portfolio' ? 'bg-green-600' : 'bg-[#121826]'} border-gray-700`}
            >💼 持仓管理</button>
          </div>
        </header>

        <main className="grid md:grid-cols-12 gap-6 max-w-full mx-auto px-4">
          {/* 左侧：控制台 - 开盘预测和持仓管理时隐藏 */}
          {activeTab !== 'prediction' && activeTab !== 'portfolio' && (
          <aside className="md:col-span-3 space-y-3">
            <div className="bg-[#0f1624] border border-gray-700 rounded p-3 space-y-3">
              <div className="text-sm text-gray-400">查询设置</div>
              {activeTab === 'fixed' ? (
                <>
                  <div className="flex gap-2 items-center flex-wrap">
                    {(['MA5','MA20','MA60','RSI14','MACD','BOLL','OBV','ATR','REALTIME','TOPUP','TOPDOWN'] as FixedType[]).map(k => (
                      <label key={k} className="flex items-center gap-1 text-sm">
                        <input
                          type="checkbox"
                          checked={fixedTypes.includes(k)}
                          onChange={(e) => {
                            setFixedTypes(prev => e.target.checked ? Array.from(new Set([...prev, k])) : prev.filter(x => x!==k))
                          }}
                        />
                        {k}
                      </label>
                    ))}
      </div>
                  <input
                    className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 outline-none"
                    placeholder="股票代码或名称，如 000501 或 600519"
                    value={stockCode}
                    onChange={(e) => setStockCode(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && runSearch()}
                  />
                  <div className="flex gap-2 items-center">
                    <label className="text-sm text-gray-400 whitespace-nowrap">时间范围:</label>
                    <select
                      value={lookbackDays}
                      onChange={(e) => setLookbackDays(Number(e.target.value))}
                      className="bg-[#121826] border border-gray-700 rounded px-2 py-1 text-sm flex-1"
                    >
                      <option value={30}>1个月(30天)</option>
                      <option value={60}>2个月(60天)</option>
                      <option value={120}>4个月(120天)</option>
                      <option value={180}>6个月(180天)</option>
                      <option value={250}>1年(250天)</option>
                    </select>
                  </div>
                  <button
                    onClick={runSearch}
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded px-4 py-2"
                  >
                    {loading ? '分析中…' : '运行固定分析'}
                  </button>
                </>
              ) : activeTab === 'llm' ? (
                <>
                  <textarea
                    className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 outline-none min-h-[120px]"
                    placeholder="请输入分析问题，如：根据MA20分析一下000501"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <select
                    value={mode}
                    onChange={(e) => setMode(e.target.value as Mode)}
                    className="bg-[#121826] border border-gray-700 rounded px-2 py-2"
                  >
                    <option value="llm">LLM</option>
                    <option value="auto">自动</option>
                    <option value="quick">固定</option>
                  </select>
                  <button
                    onClick={runSearch}
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded px-4 py-2"
                  >
                    {loading ? '分析中…' : '运行LLM分析'}
        </button>
                </>
              ) : (
                <div className="text-sm text-gray-400">
                  策略回测面板已在右侧显示
                </div>
              )}
            </div>

            <div className="bg-[#0f1624] border border-gray-700 rounded p-3 text-sm">
              <div className="text-gray-400 mb-2">历史查询</div>
              {history.length === 0 ? (
                <div className="text-gray-500">暂无记录</div>
              ) : (
                <div className="space-y-2 max-h-[40vh] overflow-auto pr-1">
                  {history.map((h, idx) => (
                    <div key={idx} className="border border-gray-800 rounded p-2">
                      <div className="text-gray-300 mb-1">[{h.mode}] {h.query}</div>
                      <div className="text-gray-500 whitespace-pre-wrap line-clamp-3">{h.result}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 运行日志 - 可展开/收起 */}
            <div className="bg-[#0f1624] border border-gray-700 rounded p-3 text-xs">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-400">
                  运行日志
                  <span className="ml-2 text-xs text-blue-400">
                    (实时: {liveLogs.length} 条 | 服务端: {fixedRes?.logs?.length || 0} 条)
                  </span>
                </div>
                <button
                  onClick={() => setLogsCollapsed(!logsCollapsed)}
                  className="text-xs px-2 py-1 rounded bg-[#121826] border border-gray-700 hover:bg-[#1a202c] transition-colors"
                >
                  {logsCollapsed ? '📋 展开' : '📖 收起'}
                </button>
              </div>
              {!logsCollapsed && (
                <div className="max-h-[30vh] overflow-auto font-mono space-y-1" ref={logContainerRef}>
                  {liveLogs.length === 0 && (!fixedRes?.logs || fixedRes.logs.length === 0) && (
                    <div className="text-gray-500 italic">等待分析开始...</div>
                  )}
                  {liveLogs.map((l, i) => (
                    <div key={`live-${i}`} className="text-green-400">• {l}</div>
                  ))}
                  {fixedRes?.logs && fixedRes.logs.map((l, i) => (
                    <div key={`srv-${i}`} className="text-blue-400">• {l}</div>
                  ))}
                </div>
              )}
              {logsCollapsed && (
                <div className="flex justify-center py-2">
                  <button
                    onClick={() => setLogsCollapsed(false)}
                    className="text-xs px-3 py-1 rounded bg-[#121826] border border-gray-700 hover:bg-[#1a202c] transition-colors text-gray-400"
                  >
                    📋 日志 ({liveLogs.length + (fixedRes?.logs?.length || 0)} 条)
                  </button>
                </div>
              )}
            </div>

            {/* Qlib AI分析已合并到固定分析流程，移除独立面板 */}
          </aside>
          )}

          {/* 中间：结果与图表 */}
          <section className={`${activeTab === 'strategy' ? 'md:col-span-4' : (activeTab === 'prediction' || activeTab === 'portfolio') ? 'md:col-span-12' : 'md:col-span-6'} space-y-3`}>
            {activeTab === 'strategy' ? (
              <StrategyPanel symbol={stockCode} selectedSymbols={selectedSymbols} />
            ) : activeTab === 'prediction' ? (
              <div className="md:col-span-12">
                <OpeningPrediction />
              </div>
            ) : activeTab === 'portfolio' ? (
              <div className="md:col-span-12">
                <PortfolioManager />
              </div>
            ) : (
              <>
            {error && (
              <div className="bg-red-950/40 border border-red-700 text-red-200 rounded p-3 text-sm">
                {error}
              </div>
            )}
            <div className="bg-[#0f1624] border border-gray-700 rounded p-4 whitespace-pre-wrap">
              {result || '结果将显示在这里…'}
            </div>

            {fixedRes && (
              <div className="space-y-4">
                {/* 价格/均线/布林 */}
                {(fixedRes.series['close'] || fixedRes.series['MA5'] || fixedRes.series['MA20'] || fixedRes.series['MA60'] || fixedRes.series['BOLL_MID']) && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm text-gray-400">{nameMap[fixedRes.meta.symbol] || fixedRes.meta.symbol}｜价格与均线</h3>
                      <div className="text-xs text-gray-500">
                        ℹ️ 均线需要足够历史数据：MA5需5天，MA20需20天，MA60需60天
                      </div>
                    </div>
                    <ReactECharts option={{
                      backgroundColor: 'transparent',
                      textStyle: { color: '#cbd5e1' },
                      tooltip: { trigger: 'axis' },
                      legend: { textStyle: { color: '#94a3b8' } },
                      xAxis: { type: 'category', data: (fixedRes.series['close']?.x || fixedRes.series['MA20']?.x || fixedRes.series['MA5']?.x || fixedRes.series['MA60']?.x || fixedRes.series['BOLL_MID']?.x || []) },
                      yAxis: { type: 'value' },
                      series: [
                        fixedRes.series['close'] && { type: 'line', name: '收盘价', data: fixedRes.series['close'].y, smooth: true, lineStyle: { width: 2 } },
                        fixedRes.series['MA5'] && { type: 'line', name: 'MA5', data: fixedRes.series['MA5'].y, smooth: true, lineStyle: { color: '#ff6b6b' } },
                        fixedRes.series['MA20'] && { type: 'line', name: 'MA20', data: fixedRes.series['MA20'].y, smooth: true, lineStyle: { color: '#4ecdc4' } },
                        fixedRes.series['MA60'] && { type: 'line', name: 'MA60', data: fixedRes.series['MA60'].y, smooth: true, lineStyle: { color: '#45b7d1' } },
                        fixedRes.series['BOLL_UP'] && { type: 'line', name: 'BOLL上轨', data: fixedRes.series['BOLL_UP'].y, smooth: true, lineStyle: { type: 'dashed', color: '#ffd93d' } },
                        fixedRes.series['BOLL_MID'] && { type: 'line', name: 'BOLL中轨', data: fixedRes.series['BOLL_MID'].y, smooth: true, lineStyle: { color: '#ffd93d' } },
                        fixedRes.series['BOLL_LOW'] && { type: 'line', name: 'BOLL下轨', data: fixedRes.series['BOLL_LOW'].y, smooth: true, lineStyle: { type: 'dashed', color: '#ffd93d' } },
                      ].filter(Boolean) as any
                    }} />
                  </div>
                )}

                {/* RSI */}
                {fixedRes.series['RSI14'] && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
                    <h3 className="text-sm text-gray-400 mb-2">RSI相对强弱指标</h3>
                    <ReactECharts option={{
                      backgroundColor: 'transparent',
                      textStyle: { color: '#cbd5e1' },
                      tooltip: { trigger: 'axis' },
                      legend: { textStyle: { color: '#94a3b8' } },
                      xAxis: { type: 'category', data: fixedRes.series['RSI14'].x },
                      yAxis: { 
                        type: 'value',
                        min: 0,
                        max: 100,
                        axisLine: { lineStyle: { color: '#4a5568' } },
                        splitLine: { 
                          show: true,
                          lineStyle: { color: '#2d3748', type: 'dashed' }
                        }
                      },
                      series: [
                        { type: 'line', name: 'RSI14', data: fixedRes.series['RSI14'].y, smooth: true, lineStyle: { color: '#e53e3e' } },
                        { type: 'line', name: '超买线(70)', data: Array(fixedRes.series['RSI14'].x.length).fill(70), lineStyle: { type: 'dashed', color: '#ff6b6b', width: 1 }, symbol: 'none' },
                        { type: 'line', name: '超卖线(30)', data: Array(fixedRes.series['RSI14'].x.length).fill(30), lineStyle: { type: 'dashed', color: '#4ecdc4', width: 1 }, symbol: 'none' }
                      ]
                    }} />
                  </div>
                )}

                {/* MACD */}
                {(fixedRes.series['MACD'] || fixedRes.series['MACD_SIGNAL'] || fixedRes.series['MACD_HIST']) && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
                    <h3 className="text-sm text-gray-400 mb-2">MACD指标</h3>
                    <ReactECharts option={{
                      backgroundColor: 'transparent',
                      textStyle: { color: '#cbd5e1' },
                      tooltip: { trigger: 'axis' },
                      legend: { textStyle: { color: '#94a3b8' } },
                      xAxis: { type: 'category', data: (fixedRes.series['MACD']?.x || fixedRes.series['MACD_SIGNAL']?.x || fixedRes.series['MACD_HIST']?.x || []) },
                      yAxis: { 
                        type: 'value',
                        axisLine: { lineStyle: { color: '#4a5568' } },
                        splitLine: { 
                          show: true,
                          lineStyle: { color: '#2d3748', type: 'dashed' }
                        }
                      },
                      series: [
                        fixedRes.series['MACD_HIST'] && { type: 'bar', name: 'HIST', data: fixedRes.series['MACD_HIST'].y, itemStyle: { color: (params: any) => params.data >= 0 ? '#ef4444' : '#22c55e' } },
                        fixedRes.series['MACD'] && { type: 'line', name: 'MACD', data: fixedRes.series['MACD'].y, smooth: true, lineStyle: { color: '#ffd93d' } },
                        fixedRes.series['MACD_SIGNAL'] && { type: 'line', name: 'SIGNAL', data: fixedRes.series['MACD_SIGNAL'].y, smooth: true, lineStyle: { color: '#45b7d1' } },
                        { type: 'line', name: '零轴', data: Array((fixedRes.series['MACD']?.x || fixedRes.series['MACD_SIGNAL']?.x || fixedRes.series['MACD_HIST']?.x || []).length).fill(0), lineStyle: { type: 'solid', color: '#6c757d', width: 1 }, symbol: 'none' }
                      ].filter(Boolean) as any
                    }} />
                  </div>
                )}

                {/* OBV能量潮 */}
                {fixedRes.series['OBV'] && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
                    <h3 className="text-sm text-gray-400 mb-2">OBV能量潮</h3>
                    <ReactECharts option={{
                      backgroundColor: 'transparent',
                      textStyle: { color: '#cbd5e1' },
                      tooltip: { trigger: 'axis' },
                      legend: { textStyle: { color: '#94a3b8' } },
                      xAxis: { type: 'category', data: fixedRes.series['OBV'].x },
                      yAxis: { type: 'value' },
                      series: [{ type: 'line', name: 'OBV', data: fixedRes.series['OBV'].y, smooth: true, lineStyle: { color: '#ff9f43' } }]
                    }} />
                  </div>
                )}

                {/* ATR波动率 */}
                {fixedRes.series['ATR'] && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
                    <h3 className="text-sm text-gray-400 mb-2">ATR平均真实波幅</h3>
                    <ReactECharts option={{
                      backgroundColor: 'transparent',
                      textStyle: { color: '#cbd5e1' },
                      tooltip: { trigger: 'axis' },
                      legend: { textStyle: { color: '#94a3b8' } },
                      xAxis: { type: 'category', data: fixedRes.series['ATR'].x },
                      yAxis: { type: 'value' },
                      series: [{ type: 'line', name: 'ATR', data: fixedRes.series['ATR'].y, smooth: true, lineStyle: { color: '#fd79a8' } }]
                    }} />
                  </div>
                )}
              </div>
            )}
              </>
            )}
          </section>

          {/* 右侧：总结 / 实时 / 榜单 / 股票池 */}
          <aside className={`${activeTab === 'strategy' ? 'md:col-span-5' : 'md:col-span-3'} space-y-3`}>
            
            {/* 策略 / 固定 / LLM 页面显示股票池（右侧面板） */}
            {(activeTab === 'strategy' || activeTab === 'llm' || activeTab === 'fixed') && (
              <StockPoolPanel 
                selectedSymbols={selectedSymbols}
                onSymbolsChange={(val) => {
                  setSelectedSymbols(val)
                  const first = (val || '').split(',').map(s => s.trim()).filter(Boolean)[0]
                  if (first) {
                    setStockCode(first)
                    if (activeTab === 'llm') setQuery(`分析${first} 近况`)
                  }
                }}
                onGenerateData={() => {
                  // 数据生成成功后可以刷新相关组件
                  console.log('历史数据生成完成')
                }}
              />
            )}
            
            {/* 固定分析和LLM页面的原有内容 */}
            {(activeTab === 'fixed' || activeTab === 'llm') && fixedRes && (
              <>
                {/* 结构化分析卡片 */}
                {fixedRes.structured_analysis && (
                  <div className="space-y-3">
                    {/* 趋势信号卡片 */}
                    <div className="bg-[#0f1624] border border-gray-700 rounded p-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm font-medium text-gray-300">📊 趋势信号</div>
                        <div className={`px-2 py-1 rounded text-xs font-medium ${
                          fixedRes.structured_analysis.trend_signal === '多头' ? 'bg-red-600 text-white' :
                          fixedRes.structured_analysis.trend_signal === '空头' ? 'bg-green-600 text-white' :
                          'bg-yellow-600 text-black'
                        }`}>
                          {fixedRes.structured_analysis.trend_signal}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="text-sm text-gray-400">置信度:</div>
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${fixedRes.structured_analysis.confidence >= 0.7 ? 'bg-green-500' : fixedRes.structured_analysis.confidence >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                            style={{ width: `${(fixedRes.structured_analysis.confidence * 100)}%` }}
                          ></div>
                        </div>
                        <div className="text-sm text-gray-300">{(fixedRes.structured_analysis.confidence * 100).toFixed(1)}%</div>
                      </div>
                    </div>

                    {/* 关键信号 */}
                    {fixedRes.structured_analysis.key_signals?.length > 0 && (
                      <div className="bg-[#0f1624] border border-gray-700 rounded p-3">
                        <div className="text-sm font-medium text-gray-300 mb-2">🎯 关键信号</div>
                        {fixedRes.structured_analysis.key_signals.map((signal, i) => (
                          <div key={i} className="text-sm text-blue-300 mb-1">• {signal}</div>
                        ))}
                      </div>
                    )}

                    {/* 风险提示 */}
                    {fixedRes.structured_analysis.risk_points?.length > 0 && (
                      <div className="bg-[#0f1624] border border-gray-700 rounded p-3">
                        <div className="text-sm font-medium text-gray-300 mb-2">⚠️ 风险提示</div>
                        {fixedRes.structured_analysis.risk_points.map((risk, i) => (
                          <div key={i} className="text-sm text-red-300 mb-1">• {risk}</div>
                        ))}
                      </div>
                    )}

                    {/* 支撑阻力位 */}
                    {(fixedRes.structured_analysis.support_level || fixedRes.structured_analysis.resistance_level) && (
                      <div className="bg-[#0f1624] border border-gray-700 rounded p-3">
                        <div className="text-sm font-medium text-gray-300 mb-2">📏 关键点位</div>
                        {fixedRes.structured_analysis.support_level && (
                          <div className="text-sm text-green-300 mb-1">支撑位: {fixedRes.structured_analysis.support_level}</div>
                        )}
                        {fixedRes.structured_analysis.resistance_level && (
                          <div className="text-sm text-red-300">阻力位: {fixedRes.structured_analysis.resistance_level}</div>
                        )}
                      </div>
                    )}
                  </div>
                )}



                {/* 价格预测 */}
                {fixedRes.price_prediction && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
                    <div className="text-lg font-medium text-gray-300 mb-3">🔮 价格预测</div>
                    
                    {fixedRes.price_prediction.error ? (
                      <div className="text-red-400 text-center">预测失败: {fixedRes.price_prediction.error}</div>
                    ) : (
                      <>
                        {/* 当前价格和预测时间 */}
                        <div className="mb-3 text-center">
                          <div className="text-2xl font-bold text-white">¥{fixedRes.price_prediction.current_price}</div>
                          <div className="text-sm text-gray-400">{fixedRes.price_prediction.prediction_days}预测</div>
                        </div>

                    {/* 上涨下跌概率 */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="bg-red-600/20 border border-red-600/50 rounded p-3 text-center">
                        <div className="text-red-400 text-lg font-bold">{fixedRes.price_prediction.up_probability}%</div>
                        <div className="text-sm text-red-300">上涨概率</div>
                      </div>
                      <div className="bg-green-600/20 border border-green-600/50 rounded p-3 text-center">
                        <div className="text-green-400 text-lg font-bold">{fixedRes.price_prediction.down_probability}%</div>
                        <div className="text-sm text-green-300">下跌概率</div>
                      </div>
                    </div>

                    {/* 价格区间预测 */}
                    <div className="mb-4">
                      <div className="text-sm text-gray-400 mb-2">价格区间预测</div>
                      <div className="flex items-center justify-between">
                        <div className="text-green-400">¥{fixedRes.price_prediction.price_range_lower}</div>
                        <div className="flex-1 mx-3 h-2 bg-gray-700 rounded-full relative">
                          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-white rounded-full"></div>
                        </div>
                        <div className="text-red-400">¥{fixedRes.price_prediction.price_range_upper}</div>
                      </div>
                      <div className="text-center text-sm text-gray-400 mt-1">
                        置信度: {Math.round(fixedRes.price_prediction.confidence_level)}%
                      </div>
                    </div>

                    {/* 关键因素 */}
                    {fixedRes.price_prediction.key_factors && fixedRes.price_prediction.key_factors.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-medium text-gray-300 mb-2">🎯 关键因素</div>
                        {fixedRes.price_prediction.key_factors.map((factor, i) => (
                          <div key={i} className="text-sm text-blue-300 mb-1">• {factor}</div>
                        ))}
                      </div>
                    )}

                    {/* 模型详情 */}
                    {fixedRes.price_prediction.model_details && (
                      <div className="mb-4 bg-gray-800/50 rounded p-3">
                        <div className="text-xs font-medium text-gray-400 mb-2">📊 模型详情</div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div className="text-center">
                            <div className="text-yellow-400 font-mono">{fixedRes.price_prediction.model_details.feature_score}</div>
                            <div className="text-gray-500">特征得分</div>
                          </div>
                          <div className="text-center">
                            <div className={`font-medium ${
                              fixedRes.price_prediction.model_details.volatility_regime === '高' ? 'text-red-400' : 
                              fixedRes.price_prediction.model_details.volatility_regime === '低' ? 'text-green-400' : 'text-yellow-400'
                            }`}>
                              {fixedRes.price_prediction.model_details.volatility_regime}波动
                            </div>
                            <div className="text-gray-500">市场环境</div>
                          </div>
                          <div className="text-center">
                            <div className={`font-medium ${
                              fixedRes.price_prediction.model_details.signal_strength === '强' ? 'text-green-400' : 
                              fixedRes.price_prediction.model_details.signal_strength === '弱' ? 'text-red-400' : 'text-yellow-400'
                            }`}>
                              {fixedRes.price_prediction.model_details.signal_strength}信号
                            </div>
                            <div className="text-gray-500">信号强度</div>
                          </div>
                        </div>
                      </div>
                    )}

                        <div className="text-xs text-gray-500 mt-3 text-center">
                          ⚠️ 预测仅供参考，投资需谨慎 | 算法已优化升级
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* 风险指标 */}
                {fixedRes.risk && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4">
                    <div className="text-lg font-medium text-gray-300 mb-3">🛡️ 风险指标</div>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <div className="text-gray-400">最大回撤</div>
                        <div className="text-red-400 font-mono">{(fixedRes.risk.max_drawdown * 100).toFixed(2)}%</div>
                      </div>
                      <div>
                        <div className="text-gray-400">年化波动率</div>
                        <div className="text-yellow-400 font-mono">{(fixedRes.risk.volatility * 100).toFixed(2)}%</div>
                      </div>
                      <div>
                        <div className="text-gray-400">VaR(95%)</div>
                        <div className="text-purple-400 font-mono">{(fixedRes.risk.var_95 * 100).toFixed(2)}%</div>
                      </div>
                      <div>
                        <div className="text-gray-400">胜率</div>
                        <div className="text-blue-400 font-mono">{(fixedRes.risk.win_rate * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 综合结论 */}
                {fixedRes.summary && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-4 whitespace-pre-wrap max-h-[60vh] overflow-auto">
                    <div className="flex items-center justify-between mb-3">
                      <div className="text-lg font-medium text-gray-300">📋 综合分析</div>
                      <button onClick={copySummary} className="text-sm px-3 py-1 rounded bg-[#121826] border border-gray-700 hover:bg-[#1a202c] transition-colors">📋 复制</button>
                    </div>
                    <div className="text-sm leading-relaxed text-gray-200">
                      {fixedRes.summary}
                    </div>
                  </div>
                )}

                {fixedRes.realtime && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-3">
                    <div className="text-sm text-gray-400 mb-2">实时行情</div>
                    <div className="text-sm">{fixedRes.realtime['名称']} ({fixedRes.realtime['代码']})</div>
                    <div className="text-sm">
                      最新价：{fixedRes.realtime['最新价']} 涨跌幅：
                      <span className={fixedRes.realtime['涨跌幅'] >= 0 ? 'text-red-400' : 'text-green-400'}>
                        {fixedRes.realtime['涨跌幅']}%
                      </span>
                    </div>
                  </div>
                )}

                {fixedRes.board && (
                  <div className="bg-[#0f1624] border border-gray-700 rounded p-3 text-sm">
                    <div className="text-sm text-gray-400 mb-2">涨跌幅榜</div>
                    <ul className="space-y-1">
                      {fixedRes.board.map((r, i) => (
                        <li key={i} className="flex justify-between">
                          <span>{r['代码']} {r['名称']}</span>
                          <span>
                            <span className={r['涨跌幅'] >= 0 ? 'text-red-400' : 'text-green-400'}>
                              {r['涨跌幅']}%
                            </span>
                            {' '}{r['最新价']}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </aside>
        </main>

        {/* 图表主体已在左侧内容区按类型渲染 */}
      </div>
    </div>
  )
}

export default App
