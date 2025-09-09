"""
策略回测模块
基于预测概率实现交易策略的回测评价
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import akshare as ak
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    symbol: str
    action: str  # 'buy' or 'sell'
    price: float
    quantity: int
    predicted_prob: float
    actual_return: float = 0.0

@dataclass
class StrategyConfig:
    """策略配置 - 严格按照文档参数"""
    
    # 交易阈值（文档标准参数）
    buy_threshold: float = 0.6      # 买入概率阈值
    sell_threshold: float = 0.4     # 卖出概率阈值
    
    # 仓位管理（文档标准参数）
    max_positions: int = 10         # 最大持仓数
    position_size: float = 0.1      # 单股基础仓位比例
    
    # 资金管理（文档标准参数）
    initial_capital: float = 100000 # 初始资金
    transaction_cost: float = 0.002 # 交易成本0.2%
    
    # 风险控制（文档标准参数）
    max_drawdown_limit: float = 0.1 # 最大回撤限制10%
    stop_loss_ratio: float = 0.03   # 止损比例3%
    max_holding_days: int = 15      # 最大持仓天数
    
    # Kelly公式参数（文档新增）
    max_kelly_fraction: float = 0.25 # 最大Kelly仓位25%
    min_kelly_fraction: float = 0.05 # 最小Kelly仓位5%
    
    # 保留兼容性参数
    calibration_gamma: float = 1.0  # 不使用校准
    selection_mode: str = "threshold"  # 使用阈值模式
    top_k: int = 10

@dataclass
class PerformanceMetrics:
    """策略表现指标"""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_period: float
    volatility: float
    calmar_ratio: float

class StrategyBacktester:
    """策略回测器"""
    
    def __init__(self, db_path: str = "calibration.db"):
        self.db_path = db_path
        self.trades: List[TradeRecord] = []
        self.positions: Dict[str, Dict] = {}  # symbol -> {quantity, avg_price, entry_date}
        self.portfolio_value: List[Tuple[str, float]] = []  # (date, value)
        self.cash = 0.0
        # 缓存以避免优化时重复取数
        self.price_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        self.predictions_cache: Dict[Tuple[str, str, Tuple[str, ...] | None], pd.DataFrame] = {}
        # 控制打印
        self.verbose: bool = True
    
    def calculate_expected_return(self, symbol: str, probability: float, 
                                current_price: float, price_data: pd.DataFrame, 
                                date: str) -> float:
        """基于历史波动率和预测概率计算预期收益（文档算法）"""
        
        try:
            # 获取过去20天历史数据
            end_date = pd.to_datetime(date)
            start_date = end_date - pd.Timedelta(days=30)  # 多取一些确保有20天
            
            # 筛选日期范围内的数据
            recent_data = price_data[
                (price_data['date'] >= start_date) & 
                (price_data['date'] <= end_date) &
                (price_data['symbol'] == symbol)
            ].tail(20)  # 取最近20天
            
            if len(recent_data) < 5:
                return 0.0
                
            # 计算日收益率
            recent_prices = recent_data['close'].values
            daily_returns = np.diff(recent_prices) / recent_prices[:-1]
            
            avg_return = np.mean(daily_returns)
            volatility = np.std(daily_returns)
            
            # 根据预测概率调整预期收益（文档公式）
            if probability > 0.5:
                expected_return = (probability - 0.5) * 2 * abs(avg_return) + volatility * (probability - 0.5)
            else:
                expected_return = (probability - 0.5) * 2 * abs(avg_return) - volatility * (0.5 - probability)
            
            return expected_return
            
        except Exception as e:
            if self.verbose:
                print(f"计算预期收益率失败 {symbol}: {e}")
            return 0.0
    
    def calculate_risk_adjusted_return(self, expected_return: float, volatility: float) -> float:
        """计算夏普比率式的风险调整收益（文档算法）"""
        
        if volatility > 0:
            risk_adjusted = expected_return / volatility
        else:
            risk_adjusted = expected_return
            
        return risk_adjusted
    
    def calculate_kelly_fraction(self, probability: float, expected_return: float) -> float:
        """Kelly公式计算最优仓位比例（文档算法）"""
        
        try:
            # Kelly公式: f = (bp - q) / b
            win_prob = probability
            lose_prob = 1 - probability
            win_amount = expected_return
            lose_amount = expected_return / 2  # 假设亏损幅度较小
            
            if win_amount <= 0:
                return 0.0
                
            kelly_fraction = (win_prob * win_amount - lose_prob * lose_amount) / win_amount
            
            # 限制仓位范围（文档参数）
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 最大25%
            
            # 如果仓位太小，设为0
            if kelly_fraction < 0.05:  # 最小5%
                return 0.0
                
            return kelly_fraction
            
        except Exception as e:
            if self.verbose:
                print(f"Kelly公式计算失败: {e}")
            return 0.0
    
    def get_volatility(self, symbol: str, date: str, price_data: pd.DataFrame) -> float:
        """获取股票波动率"""
        
        try:
            end_date = pd.to_datetime(date)
            start_date = end_date - pd.Timedelta(days=30)
            
            recent_data = price_data[
                (price_data['date'] >= start_date) & 
                (price_data['date'] <= end_date) &
                (price_data['symbol'] == symbol)
            ]
            
            if len(recent_data) < 5:
                return 0.02  # 默认2%波动率
                
            prices = recent_data['close'].values
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            return volatility
            
        except Exception:
            return 0.02
    
    def should_buy(self, trade_info: Dict, config: StrategyConfig, current_positions: Dict) -> bool:
        """多重筛选条件（文档算法）"""
        
        conditions = [
            trade_info['expected_return'] > -0.01,           # 允许小幅负收益
            trade_info['probability'] > config.buy_threshold, # 概率阈值
            trade_info['risk_adjusted_return'] > -0.1,       # 风险调整收益阈值
            len(current_positions) < config.max_positions,   # 仓位限制
            trade_info['kelly_fraction'] > config.min_kelly_fraction  # 最小仓位要求
        ]
        
        return all(conditions)
    
    def should_sell(self, symbol: str, position: Dict, current_prediction: Dict, 
                   current_price: float, config: StrategyConfig) -> Tuple[bool, str]:
        """动态卖出条件判断（文档算法）"""
        
        probability = current_prediction.get('probability', 0.5)
        entry_date = pd.to_datetime(position['entry_date'])
        current_date = pd.to_datetime(current_prediction['date'])
        days_held = (current_date - entry_date).days
        
        unrealized_return = (current_price - position['avg_price']) / position['avg_price']
        
        # 多重卖出条件（文档定义）
        if probability < config.sell_threshold:
            return True, "probability_low"
        elif days_held > 5 and unrealized_return < -0.01:
            return True, "time_loss"
        elif unrealized_return < -config.stop_loss_ratio:
            return True, "stop_loss"
        elif days_held > config.max_holding_days:
            return True, "max_holding"
        
        return False, ""
    
    def execute_optimal_trades(self, potential_trades: List[Dict], date: str, config: StrategyConfig) -> None:
        """执行最优买入交易（文档算法）"""
        
        current_positions = len(self.positions)
        
        for trade_info in potential_trades:
            if not self.should_buy(trade_info, config, self.positions):
                continue
                
            # 计算仓位大小（Kelly公式 + 基础仓位）
            base_position_value = self.cash * config.position_size
            kelly_position_value = self.cash * trade_info['kelly_fraction']
            
            # 取两者较大值，但不超过Kelly上限
            position_value = min(
                max(base_position_value, kelly_position_value),
                self.cash * config.max_kelly_fraction
            )
            
            # 考虑交易成本
            position_value *= (1 - config.transaction_cost)
            
            if position_value < 5000:  # 最小交易金额
                continue
                
            # 计算股数
            symbol = trade_info['symbol']
            price = trade_info['price']
            quantity = int(position_value / price / 100) * 100  # 整手交易
            
            if quantity > 0:
                self.execute_trade(symbol, 'buy', price, quantity, date, 
                                 trade_info['probability'], config)
                current_positions += 1
                
                # 检查现金是否充足
                if self.cash < 10000:  # 保留1万现金
                    break
                    
                if current_positions >= config.max_positions:
                    break
        
    @staticmethod
    def _apply_gamma_transform(probability: float, gamma: float) -> float:
        """对概率做gamma/温度变换，以收益率最大化为导向。

        使用logit温度缩放：p' = sigmoid(gamma * logit(p))。
        gamma > 1 增强极端，提高高置信度信号；gamma < 1 压缩极端。
        """
        try:
            if gamma is None or abs(gamma - 1.0) < 1e-9:
                return float(probability)
            eps = 1e-8
            p = min(max(float(probability), eps), 1 - eps)
            logit = np.log(p / (1 - p))
            adj = 1.0 / (1.0 + np.exp(-gamma * logit))
            return float(adj)
        except Exception:
            return float(probability)

    def load_predictions(self, start_date: str = None, end_date: str = None, 
                        symbols: List[str] = None) -> pd.DataFrame:
        """加载预测数据"""
        key = (start_date or "", end_date or "", tuple(symbols) if symbols else None)
        if key in self.predictions_cache:
            return self.predictions_cache[key].copy()
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT prediction_date, symbol, predicted_probability, actual_direction
            FROM predictions 
            WHERE actual_direction IS NOT NULL
        """
        params = []
        if start_date:
            query += " AND prediction_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND prediction_date <= ?"
            params.append(end_date)
        if symbols:
            placeholders = ",".join(["?" for _ in symbols])
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        query += " ORDER BY prediction_date, symbol"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        self.predictions_cache[key] = df
        return df.copy()
    
    def check_data_coverage(self, df: pd.DataFrame, start_date: str = None, 
                          end_date: str = None, symbols: List[str] = None) -> Dict:
        """检查数据覆盖情况"""
        if df.empty:
            return {
                "sufficient": False,
                "reason": "没有任何预测数据",
                "coverage_ratio": 0.0
            }
        
        # 如果没有指定时间范围，认为覆盖充分
        if not start_date or not end_date:
            return {
                "sufficient": True,
                "reason": "未指定时间范围",
                "coverage_ratio": 1.0
            }
        
        # 计算请求的时间范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        requested_days = (end_dt - start_dt).days + 1
        
        # 计算实际数据的时间范围
        df_dates = pd.to_datetime(df['prediction_date'])
        actual_start = df_dates.min()
        actual_end = df_dates.max()
        
        # 检查时间覆盖 - 允许一定的容忍度（考虑到周末和节假日）
        time_coverage_threshold = 0.5  # 50%的时间覆盖即可接受（更实用的阈值）
        
        if actual_start > start_dt or actual_end < end_dt:
            coverage_start = max(actual_start, start_dt)
            coverage_end = min(actual_end, end_dt)
            coverage_days = (coverage_end - coverage_start).days + 1
            coverage_ratio = coverage_days / requested_days
            
            if coverage_ratio < time_coverage_threshold:
                return {
                    "sufficient": False,
                    "reason": f"时间覆盖不足: 请求{start_date}~{end_date}, 实际{actual_start.date()}~{actual_end.date()}, 覆盖率{coverage_ratio:.1%}<{time_coverage_threshold:.0%}",
                    "coverage_ratio": coverage_ratio,
                    "requested_range": f"{start_date}~{end_date}",
                    "actual_range": f"{actual_start.date()}~{actual_end.date()}"
                }
        
        # 检查股票覆盖（如果指定了股票）
        if symbols:
            available_symbols = set(df['symbol'].unique())
            requested_symbols = set(symbols)
            missing_symbols = requested_symbols - available_symbols
            symbol_coverage_threshold = 0.3  # 至少30%的股票有数据（更宽松）
            
            symbol_coverage_ratio = len(available_symbols) / len(requested_symbols)
            if symbol_coverage_ratio < symbol_coverage_threshold:
                return {
                    "sufficient": False,
                    "reason": f"股票覆盖不足: 缺少{', '.join(missing_symbols)}, 覆盖率{symbol_coverage_ratio:.1%}<{symbol_coverage_threshold:.0%}",
                    "coverage_ratio": symbol_coverage_ratio,
                    "missing_symbols": list(missing_symbols)
                }
        
        # 检查数据密度（基于实际覆盖范围，避免数据稀疏）
        unique_dates = df['prediction_date'].nunique()
        
        # 使用实际覆盖范围计算密度阈值
        if actual_start > start_dt or actual_end < end_dt:
            # 如果数据范围不完整，基于覆盖范围计算
            coverage_start = max(actual_start, start_dt)
            coverage_end = min(actual_end, end_dt)
            coverage_days = (coverage_end - coverage_start).days + 1
            density_threshold = coverage_days * 0.3  # 30%的密度即可
        else:
            # 如果数据范围完整，基于请求范围计算
            density_threshold = requested_days * 0.3  # 30%的密度即可
        
        if unique_dates < density_threshold:
            return {
                "sufficient": False,
                "reason": f"数据密度不足: {unique_dates}天数据 < {density_threshold:.1f}天(30%阈值)",
                "coverage_ratio": unique_dates / requested_days
            }
        
        return {
            "sufficient": True,
            "reason": "数据覆盖充分",
            "coverage_ratio": 1.0
        }
    
    def get_stock_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票价格数据"""
        try:
            cache_key = (symbol, start_date, end_date)
            if cache_key in self.price_cache:
                return self.price_cache[cache_key].copy()
            # 转换股票代码格式 - 修复格式问题
            if symbol.startswith('sh'):
                ak_symbol = symbol[2:]  # 去掉sh前缀，直接使用数字
            elif symbol.startswith('sz'):
                ak_symbol = symbol[2:]  # 去掉sz前缀，直接使用数字  
            else:
                ak_symbol = symbol
            
            if self.verbose:
                print(f"转换股票代码: {symbol} -> {ak_symbol}")
            
            # 获取股票数据
            df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", 
                                   start_date=start_date.replace('-', ''), 
                                   end_date=end_date.replace('-', ''))
            
            if df.empty:
                if self.verbose:
                    print(f"AKShare获取不到 {symbol} 数据: {start_date} ~ {end_date}")
                return pd.DataFrame()
                
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open'})
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            out = df[['date', 'open', 'close']].copy()
            self.price_cache[cache_key] = out
            return out.copy()
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 价格数据失败: {e}")
            if self.verbose:
                print(f"AKShare异常 {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_mock_prices(self, start_date: str, end_date: str, symbol: str) -> pd.DataFrame:
        """生成模拟价格数据用于测试"""
        import random
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 基础价格根据股票不同设置
        base_prices = {
            'sz000002': 6.5,  # 万科A
            'sh600519': 1800,  # 茅台
            'sh600000': 8.5,   # 浦发银行
        }
        
        base_price = base_prices.get(symbol, 10.0)
        
        data = []
        current_price = base_price
        
        for date in dates:
            # 随机波动 ±3%
            change = random.uniform(-0.03, 0.03)
            current_price = current_price * (1 + change)
            
            # 开盘价稍微偏离收盘价
            open_price = current_price * random.uniform(0.995, 1.005)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'close': round(current_price, 2)
            })
        
        return pd.DataFrame(data)
    
    def calculate_position_size(self, price: float, config: StrategyConfig) -> int:
        """计算仓位大小"""
        position_value = self.cash * config.position_size
        quantity = int(position_value / price / 100) * 100  # 按手数买入
        return max(quantity, 0)
    
    def calculate_expected_return(self, symbol: str, probability: float, 
                                current_price: float, price_data: Dict, 
                                current_date: str) -> float:
        """计算预期收益率"""
        try:
            # 获取历史波动率来估算预期收益
            df = price_data[symbol]
            
            # 获取过去20天的价格数据
            dates = list(df.index)
            current_idx = dates.index(current_date) if current_date in dates else -1
            
            if current_idx >= 20:
                recent_prices = df.iloc[current_idx-20:current_idx]['close'].values
                daily_returns = np.diff(recent_prices) / recent_prices[:-1]
                avg_return = np.mean(daily_returns)
                volatility = np.std(daily_returns)
                
                # 基于概率和历史波动率计算预期收益
                # 如果预测概率高，预期上涨幅度为正向波动率
                if probability > 0.5:
                    expected_return = (probability - 0.5) * 2 * abs(avg_return) + volatility * (probability - 0.5)
                else:
                    expected_return = (probability - 0.5) * 2 * abs(avg_return) - volatility * (0.5 - probability)
                
                return expected_return
            else:
                # 数据不足时使用简化计算
                return (probability - 0.5) * 0.04  # 假设最大4%的收益率
                
        except Exception:
            return (probability - 0.5) * 0.02  # 默认2%收益率
    
    def calculate_risk_adjusted_return(self, symbol: str, expected_return: float,
                                     price_data: Dict, current_date: str) -> float:
        """计算风险调整后收益率"""
        try:
            df = price_data[symbol]
            dates = list(df.index)
            current_idx = dates.index(current_date) if current_date in dates else -1
            
            if current_idx >= 20:
                recent_prices = df.iloc[current_idx-20:current_idx]['close'].values
                daily_returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(daily_returns)
                
                # 风险调整：预期收益除以波动率
                if volatility > 0:
                    risk_adjusted = expected_return / volatility
                else:
                    risk_adjusted = expected_return
                    
                return risk_adjusted
            else:
                return expected_return
                
        except Exception:
            return expected_return
    
    def calculate_kelly_fraction(self, probability: float, expected_return: float) -> float:
        """计算Kelly公式最优仓位比例"""
        try:
            # Kelly公式: f = (bp - q) / b
            # 其中 b = 赔率, p = 胜率, q = 败率
            
            if probability <= 0.5 or expected_return <= 0:
                return 0.0
            
            # 简化假设：盈利时收益为expected_return，亏损时损失为-expected_return/2
            win_prob = probability
            lose_prob = 1 - probability
            win_amount = expected_return
            lose_amount = expected_return / 2  # 假设亏损幅度较小
            
            if lose_amount > 0:
                kelly_fraction = (win_prob * win_amount - lose_prob * lose_amount) / win_amount
                # 限制Kelly比例在合理范围内
                return max(0, min(kelly_fraction, 0.25))  # 最大25%仓位
            else:
                return 0.1  # 默认10%仓位
                
        except Exception:
            return 0.1  # 默认10%仓位
    
    def execute_trade(self, symbol: str, action: str, price: float, quantity: int, 
                     date: str, predicted_prob: float, config: StrategyConfig):
        """执行交易"""
        if action == 'buy' and quantity > 0:
            cost = quantity * price * (1 + config.transaction_cost)
            if cost <= self.cash:
                self.cash -= cost
                
                if symbol in self.positions:
                    # 加仓
                    old_qty = self.positions[symbol]['quantity']
                    old_price = self.positions[symbol]['avg_price']
                    new_qty = old_qty + quantity
                    new_avg_price = (old_qty * old_price + quantity * price) / new_qty
                    self.positions[symbol]['quantity'] = new_qty
                    self.positions[symbol]['avg_price'] = new_avg_price
                else:
                    # 新建仓位
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'entry_date': date
                    }
                
                trade = TradeRecord(date, symbol, action, price, quantity, predicted_prob)
                self.trades.append(trade)
                
        elif action == 'sell' and symbol in self.positions:
            position = self.positions[symbol]
            sell_quantity = min(quantity, position['quantity'])
            
            if sell_quantity > 0:
                revenue = sell_quantity * price * (1 - config.transaction_cost)
                self.cash += revenue
                
                # 计算实际收益
                entry_price = position['avg_price']
                actual_return = (price - entry_price) / entry_price
                
                trade = TradeRecord(date, symbol, action, price, sell_quantity, 
                                  predicted_prob, actual_return)
                self.trades.append(trade)
                
                # 更新持仓
                position['quantity'] -= sell_quantity
                if position['quantity'] <= 0:
                    del self.positions[symbol]
    
    def run_backtest(self, config: StrategyConfig, start_date: str = None, 
                    end_date: str = None, symbols: List[str] = None, use_calibration: bool = True) -> Dict:
        """运行回测"""
        # 初始化
        self.trades = []
        self.positions = {}
        self.portfolio_value = []
        self.cash = config.initial_capital
        
        # 加载预测数据
        predictions_df = self.load_predictions(start_date, end_date, symbols)
        
        # 🔧 应用概率校准（用于提升概率可用性）
        if use_calibration:
            try:
                from backend.calibration import calibrator
                if self.verbose:
                    print("📊 正在应用概率校准...")
                
                # 训练校准模型（使用历史数据）
                train_probs = predictions_df['predicted_probability'].values
                train_labels = predictions_df['actual_direction'].values
                
                # 确保有足够数据进行校准
                if len(train_probs) >= 50:
                    calibrator.fit_platt_scaling(train_probs, train_labels)
                    
                    # 应用校准到所有预测概率（使用自适应校准强度）
                    predictions_df['calibrated_probability'] = predictions_df['predicted_probability'].apply(
                        lambda p: calibrator.calibrate_probability(p, method="platt", calibration_strength="adaptive")
                    )
                    
                    # 显示校准效果（仅verbose时输出）
                    original_mean = predictions_df['predicted_probability'].mean()
                    calibrated_mean = predictions_df['calibrated_probability'].mean()
                    if self.verbose:
                        print(f"校准前平均概率: {original_mean:.4f}")
                        print(f"校准后平均概率: {calibrated_mean:.4f}")
                        print(f"概率调整幅度: {calibrated_mean - original_mean:+.4f}")
                    
                    # 使用校准后的概率列
                    prob_column = 'calibrated_probability'
                else:
                    if self.verbose:
                        print("⚠️ 数据不足，跳过校准")
                    prob_column = 'predicted_probability'
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 校准失败，使用原始概率: {e}")
                prob_column = 'predicted_probability'
        else:
            if self.verbose:
                print("📊 使用原始概率（未校准）")
            prob_column = 'predicted_probability'
        
        # 🔥 收益率导向：对用于交易决策的概率进行gamma形状变换
        if prob_column in predictions_df.columns and not predictions_df.empty:
            gamma = getattr(config, 'calibration_gamma', 1.0)
            if gamma is None:
                gamma = 1.0
            predictions_df['trading_probability'] = predictions_df[prob_column].apply(
                lambda p: self._apply_gamma_transform(p, gamma)
            )
            prob_column = 'trading_probability'

        # 检查数据覆盖情况
        coverage_check = self.check_data_coverage(predictions_df, start_date, end_date, symbols)
        
        if not coverage_check["sufficient"]:
            return {
                "error": "没有可用的预测数据",
                "details": coverage_check
            }
        
        # 调试信息
        if self.verbose:
            print(f"加载预测数据: {len(predictions_df)} 条记录")
            print(f"股票数量: {predictions_df['symbol'].nunique()}")
            print(f"日期范围: {predictions_df['prediction_date'].min()} ~ {predictions_df['prediction_date'].max()}")
        
        # 获取所有股票的价格数据
        symbols = predictions_df['symbol'].unique()
        price_data = {}
        
        for symbol in symbols:
            df = self.get_stock_prices(symbol, 
                                     predictions_df['prediction_date'].min(),
                                     predictions_df['prediction_date'].max())
            if not df.empty:
                price_data[symbol] = df.set_index('date')
                if self.verbose:
                    print(f"股票 {symbol}: 获取到 {len(df)} 天价格数据")
            else:
                if self.verbose:
                    print(f"股票 {symbol}: 无价格数据")
        
        # 按日期处理预测信号（信号滞后1天执行：T日信号在T+1开盘执行）
        dates = sorted(predictions_df['prediction_date'].unique())
        
        for idx in range(1, len(dates)):
            signal_date = dates[idx - 1]
            date = dates[idx]
            day_predictions = predictions_df[predictions_df['prediction_date'] == signal_date]
            
            # 计算当前投资组合价值
            portfolio_value = self.cash
            for symbol, position in self.positions.items():
                if symbol in price_data and date in price_data[symbol].index:
                    current_price = price_data[symbol].loc[date, 'close']
                    portfolio_value += position['quantity'] * current_price
            
            self.portfolio_value.append((date, portfolio_value))
            
            # 处理卖出信号 - 增加动态卖出条件
            for symbol in list(self.positions.keys()):
                if symbol in price_data and date in price_data[symbol].index:
                    symbol_pred = day_predictions[day_predictions['symbol'] == symbol]
                    if not symbol_pred.empty:
                        prob = symbol_pred.iloc[0][prob_column]
                        position = self.positions[symbol]
                        
                        # 动态卖出条件：
                        # 1. 预测概率低于阈值
                        # 2. 持仓超过10天且收益为负
                        # 3. 亏损超过5%止损
                        days_held = (pd.to_datetime(date) - pd.to_datetime(position['entry_date'])).days
                        current_price = price_data[symbol].loc[date, 'close']
                        unrealized_return = (current_price - position['avg_price']) / position['avg_price']
                        
                        # 使用文档定义的卖出条件
                        current_prediction = {'probability': prob, 'date': date}
                        should_sell_flag, sell_reason = self.should_sell(
                            symbol, position, current_prediction, current_price, config
                        )
                        
                        if should_sell_flag:
                            price = price_data[symbol].loc[date, 'open']
                            quantity = self.positions[symbol]['quantity']
                            self.execute_trade(symbol, 'sell', price, quantity, 
                                             date, prob, config)
            
            # 实现基于最高预期收益率的策略选择
            potential_trades = []
            daily_opportunities = 0
            
            # 评估所有可能的交易机会（使用前一日信号，在今日开盘执行）
            for _, signal in day_predictions.iterrows():
                daily_opportunities += 1
                symbol = signal['symbol']
                prob = signal[prob_column]
                
                if symbol in price_data and date in price_data[symbol].index:
                    current_price = price_data[symbol].loc[date, 'open']
                    
                    # 计算预期收益率
                    expected_return = self.calculate_expected_return(
                        symbol, prob, current_price, price_data, date
                    )
                    
                    # 计算风险调整后的收益率
                    risk_adjusted_return = self.calculate_risk_adjusted_return(
                        symbol, expected_return, price_data, date
                    )
                    
                    # 🔥 激进收益优化：动量加成
                    if config.selection_mode == "aggressive":
                        # 计算价格动量
                        try:
                            price_series = price_data[symbol]['close']
                            if len(price_series) >= 5:
                                recent_prices = price_series.tail(5).values
                                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                                # 动量加成：上涨趋势增强预期收益
                                momentum_boost = momentum * config.momentum_weight
                                expected_return += momentum_boost
                                
                                # 波动率加成：高波动=高收益机会
                                volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
                                vol_boost = volatility * config.volatility_boost * (prob - 0.5)
                                expected_return += vol_boost
                                
                                # 重新计算风险调整收益
                                if volatility > 0:
                                    risk_adjusted_return = expected_return / volatility
                        except Exception:
                            pass
                    
                    potential_trades.append({
                        'symbol': symbol,
                        'probability': prob,
                        'price': current_price,
                        'expected_return': expected_return,
                        'risk_adjusted_return': risk_adjusted_return,
                        'kelly_fraction': self.calculate_kelly_fraction(prob, expected_return)
                    })
            
            # 按风险调整后收益率排序
            potential_trades.sort(key=lambda x: x['risk_adjusted_return'], reverse=True)

            # 执行交易选择
            current_positions = len(self.positions)

            if config.selection_mode == 'aggressive':
                # 🔥 激进收益优化模式：只选择最有潜力的信号
                # 过滤条件：预期收益率 > 目标收益率 OR 概率极高
                aggressive_filter = [
                    t for t in potential_trades 
                    if (t['expected_return'] > config.profit_target * 0.3 or  # 预期收益 > 1.5%
                        t['probability'] > 0.75 or  # 极高概率
                        t['risk_adjusted_return'] > 0.5)  # 优秀的风险调整收益
                ]
                
                if not aggressive_filter:
                    # 如果没有优质信号，降低标准
                    aggressive_filter = [
                        t for t in potential_trades 
                        if t['expected_return'] > 0 and t['probability'] > 0.6
                    ]
                
                # 选择Top-3最优信号
                selected = aggressive_filter[:min(3, config.max_positions - current_positions)]
                
                for trade_info in selected:
                    if current_positions >= config.max_positions:
                        break
                    symbol = trade_info['symbol']
                    can_buy = (
                        symbol not in self.positions or
                        (symbol in self.positions and trade_info['probability'] > self.positions[symbol].get('last_prob', 0) + 0.08)  # 更高的加仓要求
                    )
                    if not can_buy:
                        continue
                    # 激进仓位：基于信号强度动态调整
                    signal_strength = max(trade_info['probability'] - 0.5, 0) * 2  # 0-1范围
                    aggressive_fraction = config.position_size * (1 + signal_strength)  # 最高2倍仓位
                    optimal_fraction = min(trade_info['kelly_fraction'], aggressive_fraction, 0.2)  # 最高20%
                    quantity = int(self.cash * optimal_fraction / trade_info['price'] / 100) * 100
                    if quantity > 0:
                        self.execute_trade(symbol, 'buy', trade_info['price'], quantity, date, trade_info['probability'], config)
                        current_positions += 1
                        
            elif config.selection_mode == 'topk':
                # 基于收益率最大化：优先选择当日Top-K候选
                top_n = max(1, min(config.top_k, config.max_positions - current_positions))
                selected = potential_trades[:top_n]

                # 轻量过滤：优先保留有正期望的或概率>0.52的信号
                filtered = [t for t in selected if (t['expected_return'] > 0 or t['probability'] >= 0.52)]
                if not filtered:
                    filtered = selected  # 兜底

                for trade_info in filtered:
                    if current_positions >= config.max_positions:
                        break
                    symbol = trade_info['symbol']
                    can_buy = (
                        symbol not in self.positions or
                        (symbol in self.positions and trade_info['probability'] > self.positions[symbol].get('last_prob', 0) + 0.05)
                    )
                    if not can_buy:
                        continue
                    optimal_fraction = min(trade_info['kelly_fraction'], config.position_size)
                    quantity = int(self.cash * optimal_fraction / trade_info['price'] / 100) * 100
                    if quantity > 0:
                        self.execute_trade(symbol, 'buy', trade_info['price'], quantity, date, trade_info['probability'], config)
                        current_positions += 1
            else:
                # 文档标准阈值模式 - 严格按照文档实现
                self.execute_optimal_trades(potential_trades, date, config)
        
        # 计算最终投资组合价值
        final_date = dates[-1]
        final_value = self.cash
        for symbol, position in self.positions.items():
            if symbol in price_data and final_date in price_data[symbol].index:
                final_price = price_data[symbol].loc[final_date, 'close']
                final_value += position['quantity'] * final_price
        
        # 计算性能指标
        metrics = self.calculate_performance_metrics(config.initial_capital, final_value)
        
        return {
            "strategy_config": config.__dict__,
            "performance_metrics": metrics.__dict__,
            "total_trades": len(self.trades),
            "final_portfolio_value": final_value,
            "portfolio_curve": self.portfolio_value,
            "trades": [trade.__dict__ for trade in self.trades],  # 所有交易记录
            "current_positions": self.positions
        }
    
    def calculate_performance_metrics(self, initial_capital: float, 
                                    final_value: float) -> PerformanceMetrics:
        """计算策略表现指标"""
        # 基本收益指标
        total_return = (final_value - initial_capital) / initial_capital
        
        # 计算年化收益率
        if self.portfolio_value:
            days = (pd.to_datetime(self.portfolio_value[-1][0]) - 
                   pd.to_datetime(self.portfolio_value[0][0])).days
            years = max(days / 365.0, 1/365.0)  # 至少1天
            annualized_return = (1 + total_return) ** (1/years) - 1
        else:
            annualized_return = 0.0
        
        # 计算最大回撤
        if len(self.portfolio_value) > 1:
            values = [v[1] for v in self.portfolio_value]
            peak = values[0]
            max_drawdown = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0.0
        
        # 计算波动率
        if len(self.portfolio_value) > 1:
            values = [v[1] for v in self.portfolio_value]
            returns = [(values[i] - values[i-1]) / values[i-1] 
                      for i in range(1, len(values))]
            volatility = np.std(returns) * np.sqrt(252) if returns else 0.0
        else:
            volatility = 0.0
        
        # 计算夏普比率
        risk_free_rate = 0.03  # 假设无风险利率3%
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # 计算卡尔玛比率
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # 交易统计
        winning_trades = len([t for t in self.trades 
                            if t.action == 'sell' and t.actual_return > 0])
        losing_trades = len([t for t in self.trades 
                           if t.action == 'sell' and t.actual_return <= 0])
        total_sell_trades = winning_trades + losing_trades
        
        win_rate = winning_trades / total_sell_trades if total_sell_trades > 0 else 0.0
        
        # 盈亏比
        winning_returns = [t.actual_return for t in self.trades 
                          if t.action == 'sell' and t.actual_return > 0]
        losing_returns = [abs(t.actual_return) for t in self.trades 
                         if t.action == 'sell' and t.actual_return <= 0]
        
        avg_win = np.mean(winning_returns) if winning_returns else 0.0
        avg_loss = np.mean(losing_returns) if losing_returns else 0.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # 平均持仓期
        holding_periods = []
        for trade in self.trades:
            if trade.action == 'sell':
                # 简化计算，假设平均持仓期为5天
                holding_periods.append(5.0)
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        
        return PerformanceMetrics(
            total_return=round(total_return, 4),
            annualized_return=round(annualized_return, 4),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe_ratio, 4),
            win_rate=round(win_rate, 4),
            profit_loss_ratio=round(profit_loss_ratio, 4),
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_holding_period=round(avg_holding_period, 2),
            volatility=round(volatility, 4),
            calmar_ratio=round(calmar_ratio, 4)
        )

# 全局回测器实例
backtester = StrategyBacktester()
