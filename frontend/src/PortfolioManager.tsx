import React, { useState, useEffect } from 'react';

interface Position {
  symbol: string;
  name: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  profit_loss: number;
  profit_loss_pct: number;
  last_updated: string;
}

interface Portfolio {
  total_assets: number;
  available_cash: number;
  market_value: number;
  total_profit_loss: number;
  total_profit_loss_pct: number;
  positions: Position[];
  position_count: number;
}

interface Transaction {
  id: number;
  symbol: string;
  name: string;
  action: string;
  quantity: number;
  price: number;
  amount: number;
  fee: number;
  timestamp: string;
  note: string;
}

interface TransactionForm {
  symbol: string;
  name: string;
  action: 'buy' | 'sell';
  quantity: number;
  price: number;
  fee: number;
  note: string;
}

const PortfolioManager: React.FC = () => {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'positions' | 'transactions' | 'add' | 'analysis' | 'sync'>('overview');
  const [transactionForm, setTransactionForm] = useState<TransactionForm>({
    symbol: '',
    name: '',
    action: 'buy',
    quantity: 0,
    price: 0,
    fee: 0,
    note: ''
  });
  const [editingPosition, setEditingPosition] = useState<Position | null>(null);
  const [editForm, setEditForm] = useState({
    quantity: 0,
    avg_cost: 0,
    note: ''
  });
  const [analysis, setAnalysis] = useState<any>(null);
  const [syncResult, setSyncResult] = useState<any>(null);
  const [stockPools, setStockPools] = useState<any>({});
  const [editingCash, setEditingCash] = useState(false);
  const [cashAmount, setCashAmount] = useState(0);

  useEffect(() => {
    loadPortfolio();
    loadTransactions();
    loadStockPools();
  }, []);

  const loadPortfolio = async () => {
    try {
      const response = await fetch('/api/portfolio');
      const data = await response.json();
      if (data.success) {
        setPortfolio(data.data);
        setCashAmount(data.data.available_cash);
      }
    } catch (error) {
      console.error('加载投资组合失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadTransactions = async () => {
    try {
      const response = await fetch('/api/portfolio/transactions?limit=20');
      const data = await response.json();
      if (data.success) {
        setTransactions(data.data);
      }
    } catch (error) {
      console.error('加载交易记录失败:', error);
    }
  };

  const loadStockPools = async () => {
    try {
      const response = await fetch('/api/strategy/stock_pools');
      const data = await response.json();
      if (data.success) {
        setStockPools(data.data);
      }
    } catch (error) {
      console.error('加载股票池失败:', error);
    }
  };

  const loadAnalysis = async () => {
    try {
      const response = await fetch('/api/portfolio/analysis');
      const data = await response.json();
      if (data.success) {
        setAnalysis(data.data);
      }
    } catch (error) {
      console.error('加载持仓分析失败:', error);
    }
  };

  const handleAddTransaction = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/portfolio/transaction', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(transactionForm),
      });

      const data = await response.json();
      if (data.success) {
        alert('交易记录添加成功！');
        setTransactionForm({
          symbol: '',
          name: '',
          action: 'buy',
          quantity: 0,
          price: 0,
          fee: 0,
          note: ''
        });
        loadPortfolio();
        loadTransactions();
        setActiveTab('overview');
      } else {
        alert(`添加失败: ${data.error}`);
      }
    } catch (error) {
      console.error('添加交易记录失败:', error);
      alert('添加交易记录失败');
    }
  };

  const handleEditPosition = (position: Position) => {
    setEditingPosition(position);
    setEditForm({
      quantity: position.quantity,
      avg_cost: position.avg_cost,
      note: '手动调整持仓'
    });
  };

  const handleSaveEditPosition = async () => {
    if (!editingPosition) return;

    try {
      const response = await fetch('/api/portfolio/position', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: editingPosition.symbol,
          quantity: editForm.quantity,
          avg_cost: editForm.avg_cost,
          note: editForm.note
        }),
      });

      const data = await response.json();
      if (data.success) {
        alert('持仓编辑成功！');
        setEditingPosition(null);
        loadPortfolio();
      } else {
        alert(`编辑失败: ${data.error}`);
      }
    } catch (error) {
      console.error('编辑持仓失败:', error);
      alert('编辑持仓失败');
    }
  };

  const handleCancelEditPosition = () => {
    setEditingPosition(null);
    setEditForm({
      quantity: 0,
      avg_cost: 0,
      note: ''
    });
  };

  const handleDeletePosition = async (symbol: string, name: string) => {
    if (confirm(`确定要删除 ${symbol}(${name}) 的持仓吗？`)) {
      try {
        const response = await fetch(`/api/portfolio/position/${symbol}`, {
          method: 'DELETE',
        });

        const data = await response.json();
        if (data.success) {
          alert('持仓删除成功！');
          loadPortfolio();
        } else {
          alert(`删除失败: ${data.error}`);
        }
      } catch (error) {
        console.error('删除持仓失败:', error);
        alert('删除持仓失败');
      }
    }
  };

  const handleSyncWithStockPool = async (poolKey: string) => {
    try {
      const poolData = stockPools[poolKey];
      if (!poolData || !poolData.symbols) {
        alert('股票池数据无效');
        return;
      }

      const response = await fetch('/api/portfolio/sync_stock_pool', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(poolData.symbols),
      });

      const data = await response.json();
      if (data.success) {
        setSyncResult(data.data);
        setActiveTab('sync');
      } else {
        alert(`同步失败: ${data.error}`);
      }
    } catch (error) {
      console.error('股票池同步失败:', error);
      alert('股票池同步失败');
    }
  };

  const handleUpdateCash = async () => {
    try {
      const response = await fetch('/api/portfolio/cash', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount: cashAmount })
      });
      const data = await response.json();
      if (data.success) {
        setEditingCash(false);
        await loadPortfolio(); // 重新加载投资组合数据
        alert('资金更新成功');
      } else {
        alert('更新失败: ' + data.error);
      }
    } catch (error) {
      console.error('更新资金失败:', error);
      alert('更新资金失败');
    }
  };

  const handleCancelEditCash = () => {
    setEditingCash(false);
    setCashAmount(portfolio?.available_cash || 0);
  };

  const handleFixStockNames = async () => {
    try {
      const response = await fetch('/api/portfolio/fix_stock_names', {
        method: 'POST'
      });
      const data = await response.json();
      if (data.success) {
        alert(`股票名称修复成功！修复了 ${data.fixed_count} 个股票名称`);
        loadPortfolio(); // 重新加载数据
      } else {
        alert(`修复失败: ${data.error}`);
      }
    } catch (error) {
      console.error('修复股票名称失败:', error);
      alert('修复股票名称失败');
    }
  };

  const formatCurrency = (amount: number) => {
    return `¥${amount.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPercent = (pct: number) => {
    const sign = pct >= 0 ? '+' : '';
    return `${sign}${pct.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className="bg-gray-900 text-white min-h-screen p-6">
        <div className="flex justify-center items-center h-64">
          <div className="text-xl">加载持仓数据中...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 text-white min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">💼 持仓管理</h1>

        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2 mb-6">
          {[
            { key: 'overview', label: '📊 投资概览' },
            { key: 'positions', label: '📋 持仓明细' },
            { key: 'transactions', label: '📈 交易记录' },
            { key: 'add', label: '➕ 添加交易' },
            { key: 'analysis', label: '🔍 持仓分析' },
            { key: 'sync', label: '🔄 股票池同步' }
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => {
                setActiveTab(tab.key as any);
                if (tab.key === 'analysis') {
                  loadAnalysis();
                }
              }}
              className={`px-4 py-2 rounded-lg transition-colors text-sm ${
                activeTab === tab.key
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && portfolio && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="text-sm text-gray-400 mb-2">总资产</h3>
                <div className="text-2xl font-bold">{formatCurrency(portfolio.total_assets)}</div>
              </div>
              <div className="bg-gray-800 p-4 rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-sm text-gray-400">可用资金</h3>
                  {!editingCash && (
                    <button
                      onClick={() => {
                        setEditingCash(true);
                        setCashAmount(portfolio.available_cash);
                      }}
                      className="text-xs text-blue-400 hover:text-blue-300"
                    >
                      编辑
                    </button>
                  )}
                </div>
                {editingCash ? (
                  <div className="space-y-2">
                    <input
                      type="number"
                      value={cashAmount}
                      onChange={(e) => setCashAmount(parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-1 text-lg font-bold bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                      step="0.01"
                      min="0"
                    />
                    <div className="flex space-x-2">
                      <button
                        onClick={handleUpdateCash}
                        className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
                      >
                        保存
                      </button>
                      <button
                        onClick={handleCancelEditCash}
                        className="px-3 py-1 text-xs bg-gray-600 text-white rounded hover:bg-gray-700"
                      >
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="text-2xl font-bold">{formatCurrency(portfolio.available_cash)}</div>
                )}
              </div>
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="text-sm text-gray-400 mb-2">持仓市值</h3>
                <div className="text-2xl font-bold">{formatCurrency(portfolio.market_value)}</div>
              </div>
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="text-sm text-gray-400 mb-2">总盈亏</h3>
                <div className={`text-2xl font-bold ${portfolio.total_profit_loss >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                  {formatCurrency(portfolio.total_profit_loss)}
                  <span className="text-sm ml-2">
                    ({formatPercent(portfolio.total_profit_loss_pct)})
                  </span>
                </div>
              </div>
            </div>

            {/* Top Holdings */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">主要持仓 ({portfolio.position_count}只)</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700">
                      <th className="text-left py-2">股票</th>
                      <th className="text-right py-2">数量</th>
                      <th className="text-right py-2">成本价</th>
                      <th className="text-right py-2">现价</th>
                      <th className="text-right py-2">市值</th>
                      <th className="text-right py-2">盈亏</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.positions.slice(0, 5).map((position) => (
                      <tr key={position.symbol} className="border-b border-gray-700">
                        <td className="py-3">
                          <div>
                            <span className="font-medium">{position.symbol}</span>
                            <div className="text-sm text-gray-400">{position.name}</div>
                          </div>
                        </td>
                        <td className="text-right py-3">{position.quantity}股</td>
                        <td className="text-right py-3">¥{position.avg_cost.toFixed(2)}</td>
                        <td className="text-right py-3">¥{position.current_price.toFixed(2)}</td>
                        <td className="text-right py-3">{formatCurrency(position.market_value)}</td>
                        <td className={`text-right py-3 ${position.profit_loss >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                          {formatCurrency(position.profit_loss)}
                          <div className="text-sm">
                            {formatPercent(position.profit_loss_pct)}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Positions Tab */}
        {activeTab === 'positions' && portfolio && (
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold">持仓明细</h3>
              <button
                onClick={handleFixStockNames}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors"
              >
                🔧 修复股票名称
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-700">
                    <th className="text-left py-2">股票代码</th>
                    <th className="text-left py-2">股票名称</th>
                    <th className="text-right py-2">持有数量</th>
                    <th className="text-right py-2">平均成本</th>
                    <th className="text-right py-2">当前价格</th>
                    <th className="text-right py-2">市值</th>
                    <th className="text-right py-2">盈亏金额</th>
                    <th className="text-right py-2">盈亏比例</th>
                    <th className="text-right py-2">更新时间</th>
                    <th className="text-center py-2">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.positions.map((position) => (
                    <tr key={position.symbol} className="border-b border-gray-700 hover:bg-gray-750">
                      <td className="py-3 font-medium">{position.symbol}</td>
                      <td className="py-3">{position.name}</td>
                      <td className="text-right py-3">{position.quantity}股</td>
                      <td className="text-right py-3">¥{position.avg_cost.toFixed(2)}</td>
                      <td className="text-right py-3">¥{position.current_price.toFixed(2)}</td>
                      <td className="text-right py-3">{formatCurrency(position.market_value)}</td>
                      <td className={`text-right py-3 ${position.profit_loss >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                        {formatCurrency(position.profit_loss)}
                      </td>
                      <td className={`text-right py-3 ${position.profit_loss_pct >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                        {formatPercent(position.profit_loss_pct)}
                      </td>
                      <td className="text-right py-3 text-sm text-gray-400">
                        {new Date(position.last_updated).toLocaleString('zh-CN')}
                      </td>
                      <td className="text-center py-3">
                        <div className="flex space-x-2 justify-center">
                          <button
                            onClick={() => handleEditPosition(position)}
                            className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                          >
                            编辑
                          </button>
                          <button
                            onClick={() => handleDeletePosition(position.symbol, position.name)}
                            className="px-2 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                          >
                            删除
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Transactions Tab */}
        {activeTab === 'transactions' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">交易记录</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-700">
                    <th className="text-left py-2">时间</th>
                    <th className="text-left py-2">股票</th>
                    <th className="text-center py-2">操作</th>
                    <th className="text-right py-2">数量</th>
                    <th className="text-right py-2">价格</th>
                    <th className="text-right py-2">金额</th>
                    <th className="text-right py-2">手续费</th>
                    <th className="text-left py-2">备注</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.map((transaction) => (
                    <tr key={transaction.id} className="border-b border-gray-700 hover:bg-gray-750">
                      <td className="py-3 text-sm">
                        {new Date(transaction.timestamp).toLocaleString('zh-CN')}
                      </td>
                      <td className="py-3">
                        <div>
                          <span className="font-medium">{transaction.symbol}</span>
                          <div className="text-sm text-gray-400">{transaction.name}</div>
                        </div>
                      </td>
                      <td className="text-center py-3">
                        <span className={`px-2 py-1 rounded text-sm ${
                          transaction.action === 'buy' 
                            ? 'bg-red-600 text-white' 
                            : 'bg-green-600 text-white'
                        }`}>
                          {transaction.action === 'buy' ? '买入' : '卖出'}
                        </span>
                      </td>
                      <td className="text-right py-3">{transaction.quantity}股</td>
                      <td className="text-right py-3">¥{transaction.price.toFixed(2)}</td>
                      <td className="text-right py-3">{formatCurrency(transaction.amount)}</td>
                      <td className="text-right py-3">¥{transaction.fee.toFixed(2)}</td>
                      <td className="py-3 text-sm text-gray-400">{transaction.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Add Transaction Tab */}
        {activeTab === 'add' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">添加交易记录</h3>
            <form onSubmit={handleAddTransaction} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">股票代码</label>
                  <input
                    type="text"
                    value={transactionForm.symbol}
                    onChange={(e) => setTransactionForm({...transactionForm, symbol: e.target.value})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500"
                    placeholder="如: 000001"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">股票名称</label>
                  <input
                    type="text"
                    value={transactionForm.name}
                    onChange={(e) => setTransactionForm({...transactionForm, name: e.target.value})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500"
                    placeholder="如: 平安银行"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">操作类型</label>
                  <select
                    value={transactionForm.action}
                    onChange={(e) => setTransactionForm({...transactionForm, action: e.target.value as 'buy' | 'sell'})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500"
                  >
                    <option value="buy">买入</option>
                    <option value="sell">卖出</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">数量（股）</label>
                  <input
                    type="number"
                    value={transactionForm.quantity}
                    onChange={(e) => setTransactionForm({...transactionForm, quantity: parseInt(e.target.value) || 0})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500"
                    min="1"
                    step="100"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">价格（元）</label>
                  <input
                    type="number"
                    value={transactionForm.price}
                    onChange={(e) => setTransactionForm({...transactionForm, price: parseFloat(e.target.value) || 0})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500"
                    min="0"
                    step="0.01"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">手续费（元）</label>
                  <input
                    type="number"
                    value={transactionForm.fee}
                    onChange={(e) => setTransactionForm({...transactionForm, fee: parseFloat(e.target.value) || 0})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500"
                    min="0"
                    step="0.01"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">备注</label>
                <input
                  type="text"
                  value={transactionForm.note}
                  onChange={(e) => setTransactionForm({...transactionForm, note: e.target.value})}
                  className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500"
                  placeholder="可选"
                />
              </div>
              <div className="flex space-x-4">
                <button
                  type="submit"
                  className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                >
                  添加交易记录
                </button>
                <button
                  type="button"
                  onClick={() => setTransactionForm({
                    symbol: '',
                    name: '',
                    action: 'buy',
                    quantity: 0,
                    price: 0,
                    fee: 0,
                    note: ''
                  })}
                  className="px-6 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
                >
                  重置
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Analysis Tab */}
        {activeTab === 'analysis' && (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">📊 持仓分析报告</h3>
              {analysis ? (
                <div className="space-y-4">
                  {/* 基础统计 */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-700 p-4 rounded">
                      <h4 className="text-sm text-gray-400 mb-2">持仓股票数</h4>
                      <div className="text-2xl font-bold">{analysis.total_positions}只</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded">
                      <h4 className="text-sm text-gray-400 mb-2">胜率</h4>
                      <div className="text-2xl font-bold text-green-400">{(analysis.win_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded">
                      <h4 className="text-sm text-gray-400 mb-2">持仓集中度</h4>
                      <div className="text-2xl font-bold text-yellow-400">{(analysis.concentration_ratio * 100).toFixed(1)}%</div>
                    </div>
                  </div>

                  {/* 盈亏分析 */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-700 p-4 rounded">
                      <h4 className="text-sm text-gray-400 mb-2">盈利股票</h4>
                      <div className="text-lg font-bold text-red-400">{analysis.profit_stocks_count}只</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded">
                      <h4 className="text-sm text-gray-400 mb-2">亏损股票</h4>
                      <div className="text-lg font-bold text-green-400">{analysis.loss_stocks_count}只</div>
                    </div>
                  </div>

                  {/* 最大盈亏股票 */}
                  {analysis.max_profit_stock && (
                    <div className="bg-gray-700 p-4 rounded">
                      <h4 className="text-sm text-gray-400 mb-2">🏆 最大盈利股票</h4>
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{analysis.max_profit_stock.symbol} ({analysis.max_profit_stock.name})</span>
                        <span className="text-red-400 font-bold">
                          {formatCurrency(analysis.max_profit_stock.profit_loss)} ({formatPercent(analysis.max_profit_stock.profit_loss_pct)})
                        </span>
                      </div>
                    </div>
                  )}

                  {analysis.max_loss_stock && (
                    <div className="bg-gray-700 p-4 rounded">
                      <h4 className="text-sm text-gray-400 mb-2">📉 最大亏损股票</h4>
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{analysis.max_loss_stock.symbol} ({analysis.max_loss_stock.name})</span>
                        <span className="text-green-400 font-bold">
                          {formatCurrency(analysis.max_loss_stock.profit_loss)} ({formatPercent(analysis.max_loss_stock.profit_loss_pct)})
                        </span>
                      </div>
                    </div>
                  )}

                  {/* 风险警告 */}
                  {analysis.risk_warnings && analysis.risk_warnings.length > 0 && (
                    <div className="bg-red-900/30 border border-red-700 rounded p-4">
                      <h4 className="text-red-400 font-bold mb-2">⚠️ 风险提醒</h4>
                      <ul className="space-y-1">
                        {analysis.risk_warnings.map((warning: string, index: number) => (
                          <li key={index} className="text-red-300 text-sm">• {warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-gray-400">点击"持仓分析"按钮加载分析数据</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Sync Tab */}
        {activeTab === 'sync' && (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">🔄 股票池同步</h3>
              
              {/* 股票池选择 */}
              <div className="mb-6">
                <h4 className="text-lg font-medium mb-3">选择股票池进行同步分析：</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {Object.entries(stockPools).map(([key, pool]: [string, any]) => (
                    <button
                      key={key}
                      onClick={() => handleSyncWithStockPool(key)}
                      className="p-4 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors text-left"
                    >
                      <div className="font-medium">{pool.name || key}</div>
                      <div className="text-sm text-gray-400">{pool.symbols?.length || 0}只股票</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* 同步结果 */}
              {syncResult && (
                <div className="space-y-4">
                  <h4 className="text-lg font-medium">同步分析结果：</h4>
                  
                  {/* 统计概览 */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-700 p-4 rounded">
                      <div className="text-sm text-gray-400">当前持仓</div>
                      <div className="text-xl font-bold">{syncResult.current_positions_count}只</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded">
                      <div className="text-sm text-gray-400">股票池股票</div>
                      <div className="text-xl font-bold">{syncResult.stock_pool_count}只</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded">
                      <div className="text-sm text-gray-400">重叠股票</div>
                      <div className="text-xl font-bold text-green-400">{syncResult.in_both.length}只</div>
                    </div>
                  </div>

                  {/* 同步建议 */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* 考虑买入 */}
                    <div className="bg-blue-900/30 border border-blue-700 rounded p-4">
                      <h5 className="text-blue-400 font-bold mb-2">🛒 考虑买入 ({syncResult.sync_suggestions.consider_buying.length}只)</h5>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {syncResult.sync_suggestions.consider_buying.map((symbol: string) => (
                          <div key={symbol} className="text-sm text-blue-300">{symbol}</div>
                        ))}
                      </div>
                    </div>

                    {/* 考虑卖出 */}
                    <div className="bg-red-900/30 border border-red-700 rounded p-4">
                      <h5 className="text-red-400 font-bold mb-2">💸 考虑卖出 ({syncResult.sync_suggestions.consider_selling.length}只)</h5>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {syncResult.sync_suggestions.consider_selling.map((symbol: string) => (
                          <div key={symbol} className="text-sm text-red-300">{symbol}</div>
                        ))}
                      </div>
                    </div>

                    {/* 继续持有 */}
                    <div className="bg-green-900/30 border border-green-700 rounded p-4">
                      <h5 className="text-green-400 font-bold mb-2">✅ 继续持有 ({syncResult.sync_suggestions.keep_holding.length}只)</h5>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {syncResult.sync_suggestions.keep_holding.map((symbol: string) => (
                          <div key={symbol} className="text-sm text-green-300">{symbol}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Edit Position Modal */}
        {editingPosition && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4">
              <h3 className="text-xl font-bold mb-4">编辑持仓</h3>
              <div className="mb-4">
                <div className="text-sm text-gray-400 mb-2">
                  {editingPosition.symbol} - {editingPosition.name}
                </div>
                <div className="text-xs text-gray-500">
                  当前: {editingPosition.quantity}股 @ ¥{editingPosition.avg_cost.toFixed(2)}
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">持仓数量（股）</label>
                  <input
                    type="number"
                    value={editForm.quantity}
                    onChange={(e) => setEditForm({...editForm, quantity: parseInt(e.target.value) || 0})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                    min="0"
                    step="100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">平均成本（元）</label>
                  <input
                    type="number"
                    value={editForm.avg_cost}
                    onChange={(e) => setEditForm({...editForm, avg_cost: parseFloat(e.target.value) || 0})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                    min="0"
                    step="0.01"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">备注</label>
                  <input
                    type="text"
                    value={editForm.note}
                    onChange={(e) => setEditForm({...editForm, note: e.target.value})}
                    className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                    placeholder="编辑原因..."
                  />
                </div>
              </div>

              <div className="flex space-x-3 mt-6">
                <button
                  onClick={handleSaveEditPosition}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                >
                  保存
                </button>
                <button
                  onClick={handleCancelEditPosition}
                  className="flex-1 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
                >
                  取消
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PortfolioManager;
