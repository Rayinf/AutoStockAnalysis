import React, { useState, useEffect } from 'react'

interface StockPool {
  name: string
  symbols: string[]
  description: string
}

interface StockPoolManagerProps {
  onClose: () => void
  onSave: () => void
}

export const StockPoolManager: React.FC<StockPoolManagerProps> = ({ onClose, onSave }) => {
  const [stockPools, setStockPools] = useState<Record<string, StockPool>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [editingPool, setEditingPool] = useState<string | null>(null)
  const [newPoolName, setNewPoolName] = useState('')
  const [showAddNew, setShowAddNew] = useState(false)

  // 加载股票池配置
  const loadStockPools = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/strategy/stock_pools')
      const data = await response.json()
      setStockPools(data.stock_pools || {})
    } catch (err) {
      console.error('加载股票池失败:', err)
      setError('无法加载股票池配置')
    }
  }

  // 保存股票池配置
  const saveStockPools = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('http://127.0.0.1:8000/api/strategy/stock_pools', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ stock_pools: stockPools }),
      })
      
      const data = await response.json()
      
      if (data.success) {
        onSave()
        onClose()
      } else {
        setError(data.error || '保存失败')
      }
    } catch (err: any) {
      setError(err.message || '保存股票池配置失败')
    } finally {
      setLoading(false)
    }
  }

  // 删除股票池
  const deleteStockPool = async (poolName: string) => {
    if (!confirm(`确定要删除股票池 "${poolName}" 吗？`)) {
      return
    }
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/strategy/stock_pools/${encodeURIComponent(poolName)}`, {
        method: 'DELETE',
      })
      
      const data = await response.json()
      
      if (data.success) {
        const newPools = { ...stockPools }
        delete newPools[poolName]
        setStockPools(newPools)
      } else {
        setError(data.error || '删除失败')
      }
    } catch (err: any) {
      setError(err.message || '删除股票池失败')
    }
  }

  // 更新股票池
  const updateStockPool = (poolName: string, field: keyof StockPool, value: any) => {
    setStockPools(prev => ({
      ...prev,
      [poolName]: {
        ...prev[poolName],
        [field]: value
      }
    }))
  }

  // 添加新股票池
  const addNewStockPool = () => {
    if (!newPoolName.trim()) {
      setError('请输入股票池名称')
      return
    }
    
    if (stockPools[newPoolName]) {
      setError('股票池名称已存在')
      return
    }
    
    setStockPools(prev => ({
      ...prev,
      [newPoolName]: {
        name: newPoolName,
        symbols: [],
        description: '用户自定义股票池'
      }
    }))
    
    setNewPoolName('')
    setShowAddNew(false)
    setEditingPool(newPoolName)
  }

  // 添加股票到池中
  const addSymbolToPool = (poolName: string, symbol: string) => {
    if (!symbol.trim()) return
    
    const pool = stockPools[poolName]
    if (!pool.symbols.includes(symbol.trim())) {
      updateStockPool(poolName, 'symbols', [...pool.symbols, symbol.trim()])
    }
  }

  // 批量导入股票到池中
  const batchImportSymbols = (poolName: string, symbolsText: string) => {
    if (!symbolsText.trim()) return
    
    // 解析输入的股票代码：支持逗号、空格、换行分隔
    const symbols = symbolsText
      .split(/[,，\s\n\r]+/)  // 支持中英文逗号、空格、换行
      .map(s => s.trim())
      .filter(s => s.length > 0)
      .filter(s => /^\d{6}$/.test(s))  // 验证6位数字格式
    
    if (symbols.length === 0) {
      setError('未找到有效的股票代码，请确保格式为6位数字')
      return
    }
    
    const pool = stockPools[poolName]
    const existingSymbols = new Set(pool.symbols)
    const newSymbols = symbols.filter(s => !existingSymbols.has(s))
    
    if (newSymbols.length === 0) {
      setError('所有股票代码都已存在于股票池中')
      return
    }
    
    // 更新股票池
    updateStockPool(poolName, 'symbols', [...pool.symbols, ...newSymbols])
    
    // 显示导入结果
    const duplicateCount = symbols.length - newSymbols.length
    let message = `成功导入 ${newSymbols.length} 只股票`
    if (duplicateCount > 0) {
      message += `，跳过 ${duplicateCount} 只重复股票`
    }
    
    // 临时显示成功消息
    setError(null)
    const successDiv = document.createElement('div')
    successDiv.className = 'fixed top-4 right-4 p-3 bg-green-600/20 border border-green-600/50 rounded text-green-300 text-sm z-50'
    successDiv.textContent = `✅ ${message}`
    document.body.appendChild(successDiv)
    
    setTimeout(() => {
      document.body.removeChild(successDiv)
    }, 3000)
  }

  // 从池中移除股票
  const removeSymbolFromPool = (poolName: string, symbol: string) => {
    const pool = stockPools[poolName]
    updateStockPool(poolName, 'symbols', pool.symbols.filter(s => s !== symbol))
  }

  useEffect(() => {
    loadStockPools()
  }, [])

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-[#0f1624] border border-gray-700 rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold text-gray-200">🎯 股票池管理</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 text-xl"
          >
            ×
          </button>
        </div>

        {/* 添加新股票池 */}
        <div className="mb-6 p-4 bg-gray-800/30 rounded border border-gray-700/50">
          <div className="flex justify-between items-center mb-3">
            <div className="text-sm font-medium text-gray-300">➕ 添加新股票池</div>
            <button
              onClick={() => setShowAddNew(!showAddNew)}
              className="px-3 py-1 bg-blue-600/30 hover:bg-blue-600/50 text-blue-300 text-sm rounded transition-colors"
            >
              {showAddNew ? '取消' : '新建'}
            </button>
          </div>
          
          {showAddNew && (
            <div className="flex gap-2">
              <input
                type="text"
                value={newPoolName}
                onChange={(e) => setNewPoolName(e.target.value)}
                placeholder="输入股票池名称"
                className="flex-1 bg-[#121826] border border-gray-700 rounded px-3 py-2 text-sm"
                onKeyDown={(e) => e.key === 'Enter' && addNewStockPool()}
              />
              <button
                onClick={addNewStockPool}
                className="px-4 py-2 bg-green-600/30 hover:bg-green-600/50 text-green-300 text-sm rounded transition-colors"
              >
                创建
              </button>
            </div>
          )}
        </div>

        {/* 股票池列表 */}
        <div className="space-y-4">
          {Object.entries(stockPools).map(([poolName, pool]) => (
            <div key={poolName} className="p-4 bg-gray-800/30 rounded border border-gray-700/50">
              <div className="flex justify-between items-start mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <input
                      type="text"
                      value={pool.name}
                      onChange={(e) => updateStockPool(poolName, 'name', e.target.value)}
                      className="bg-transparent border-none text-lg font-medium text-gray-200 outline-none"
                      disabled={editingPool !== poolName}
                    />
                    <button
                      onClick={() => setEditingPool(editingPool === poolName ? null : poolName)}
                      className="text-blue-400 hover:text-blue-300 text-sm"
                    >
                      {editingPool === poolName ? '完成' : '编辑'}
                    </button>
                  </div>
                  <textarea
                    value={pool.description}
                    onChange={(e) => updateStockPool(poolName, 'description', e.target.value)}
                    className="w-full bg-transparent border-none text-sm text-gray-400 outline-none resize-none"
                    rows={2}
                    disabled={editingPool !== poolName}
                    placeholder="股票池描述"
                  />
                </div>
                <button
                  onClick={() => deleteStockPool(poolName)}
                  className="text-red-400 hover:text-red-300 text-sm ml-4"
                >
                  删除
                </button>
              </div>

              {/* 股票列表 */}
              <div className="mb-3">
                <div className="text-sm text-gray-400 mb-2">股票代码 ({pool.symbols.length})</div>
                <div className="flex flex-wrap gap-2 mb-2">
                  {pool.symbols.map((symbol, index) => (
                    <div key={index} className="flex items-center gap-1 bg-slate-600/30 text-slate-300 px-2 py-1 rounded text-sm">
                      <span>{symbol}</span>
                      {editingPool === poolName && (
                        <button
                          onClick={() => removeSymbolFromPool(poolName, symbol)}
                          className="text-red-400 hover:text-red-300"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                </div>
                
                {editingPool === poolName && (
                  <div className="space-y-3">
                    {/* 单个股票添加 */}
                    <div className="flex gap-2">
                      <input
                        type="text"
                        placeholder="输入股票代码，如：600519"
                        className="flex-1 bg-[#121826] border border-gray-700 rounded px-3 py-1 text-sm"
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            const input = e.target as HTMLInputElement
                            addSymbolToPool(poolName, input.value)
                            input.value = ''
                          }
                        }}
                      />
                      <button
                        onClick={(e) => {
                          const input = e.currentTarget.previousElementSibling as HTMLInputElement
                          addSymbolToPool(poolName, input.value)
                          input.value = ''
                        }}
                        className="px-3 py-1 bg-blue-600/30 hover:bg-blue-600/50 text-blue-300 text-sm rounded transition-colors"
                      >
                        添加
                      </button>
                    </div>
                    
                    {/* 批量导入 */}
                    <div className="p-3 bg-slate-800/50 rounded border border-slate-700/50">
                      <div className="text-xs font-medium text-slate-300 mb-2">📋 批量导入股票</div>
                      <textarea
                        placeholder="批量导入股票代码，支持多种格式：&#10;• 逗号分隔：000501,000519,002182,600176&#10;• 空格分隔：000501 000519 002182&#10;• 换行分隔：每行一个代码&#10;• 混合格式：000501,000519 002182&#10;600176,600585"
                        className="w-full bg-[#121826] border border-gray-700 rounded px-3 py-2 text-sm resize-none"
                        rows={4}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && e.ctrlKey) {
                            const textarea = e.target as HTMLTextAreaElement
                            batchImportSymbols(poolName, textarea.value)
                            textarea.value = ''
                          }
                        }}
                      />
                      <div className="flex justify-between items-center mt-2">
                        <div className="text-xs text-gray-500">
                          💡 支持格式：000501,000519,002182 或用空格、换行分隔
                        </div>
                        <button
                          onClick={(e) => {
                            const textarea = e.currentTarget.parentElement?.previousElementSibling as HTMLTextAreaElement
                            if (textarea) {
                              batchImportSymbols(poolName, textarea.value)
                              textarea.value = ''
                            }
                          }}
                          className="px-3 py-1 bg-green-600/30 hover:bg-green-600/50 text-green-300 text-xs rounded transition-colors"
                        >
                          📥 批量导入
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* 错误显示 */}
        {error && (
          <div className="mt-4 p-3 bg-red-600/20 border border-red-600/50 rounded text-red-400 text-sm">
            ❌ {error}
          </div>
        )}

        {/* 操作按钮 */}
        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600/30 hover:bg-gray-600/50 text-gray-300 rounded transition-colors"
          >
            取消
          </button>
          <button
            onClick={saveStockPools}
            disabled={loading}
            className="px-4 py-2 bg-blue-600/30 hover:bg-blue-600/50 disabled:opacity-50 text-blue-300 rounded transition-colors"
          >
            {loading ? '保存中...' : '保存配置'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default StockPoolManager

