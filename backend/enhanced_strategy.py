#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强策略实现 - 基于量化交易策略详细说明.md
实现文档中描述的完整最高收益率策略算法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedStrategyConfig:
    """增强策略配置 - 基于文档最优参数"""
    
    # 交易阈值（调整为更适合短期回测）
    buy_threshold: float = 0.55      # 买入概率阈值（从0.65降低到0.55）
    sell_threshold: float = 0.45     # 卖出概率阈值（从0.35提高到0.45）
    
    # 仓位管理
    max_positions: int = 10          # 最大持仓数
    position_size: float = 0.12      # 单股基础仓位比例（文档最优）
    
    # 资金管理
    initial_capital: float = 100000  # 初始资金
    transaction_cost: float = 0.002  # 交易成本0.2%
    
    # 风险控制
    max_drawdown_limit: float = 0.1  # 最大回撤限制10%
    stop_loss_ratio: float = 0.03    # 止损比例3%
    max_holding_days: int = 15       # 最大持仓天数
    
    # Kelly公式参数
    max_kelly_fraction: float = 0.25 # 最大Kelly仓位25%
    min_kelly_fraction: float = 0.05 # 最小Kelly仓位5%

class EnhancedStrategy:
    """增强策略 - 实现文档中的完整算法"""
    
    def __init__(self, config: EnhancedStrategyConfig = None):
        self.config = config or EnhancedStrategyConfig()
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.cash = self.config.initial_capital
        
    def calculate_expected_return(self, symbol: str, probability: float, 
                                current_price: float, price_data: pd.DataFrame, 
                                date: str) -> float:
        """基于历史波动率和预测概率计算预期收益"""
        
        try:
            # 获取过去20天历史数据
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=30)  # 多取一些确保有20天
            
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
            logger.warning(f"计算预期收益率失败 {symbol}: {e}")
            return 0.0
    
    def calculate_risk_adjusted_return(self, expected_return: float, volatility: float) -> float:
        """计算夏普比率式的风险调整收益"""
        
        if volatility > 0:
            risk_adjusted = expected_return / volatility
        else:
            risk_adjusted = expected_return
            
        return risk_adjusted
    
    def calculate_kelly_fraction(self, probability: float, expected_return: float) -> float:
        """Kelly公式计算最优仓位比例"""
        
        try:
            # Kelly公式: f = (bp - q) / b
            win_prob = probability
            lose_prob = 1 - probability
            win_amount = expected_return
            lose_amount = expected_return / 2  # 假设亏损幅度较小
            
            if win_amount <= 0:
                return 0.0
                
            kelly_fraction = (win_prob * win_amount - lose_prob * lose_amount) / win_amount
            
            # 限制仓位范围
            kelly_fraction = max(0, min(kelly_fraction, self.config.max_kelly_fraction))
            
            # 如果仓位太小，设为0
            if kelly_fraction < self.config.min_kelly_fraction:
                return 0.0
                
            return kelly_fraction
            
        except Exception as e:
            logger.warning(f"Kelly公式计算失败: {e}")
            return 0.0
    
    def get_volatility(self, symbol: str, date: str, price_data: pd.DataFrame) -> float:
        """获取股票波动率"""
        
        try:
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=30)
            
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
    
    def should_buy(self, trade_info: Dict, current_positions: Dict) -> bool:
        """多重筛选条件（文档算法）"""
        
        conditions = [
            trade_info['expected_return'] > -0.01,           # 允许小幅负收益
            trade_info['probability'] > self.config.buy_threshold, # 概率阈值
            trade_info['risk_adjusted_return'] > -0.1,       # 风险调整收益阈值
            len(current_positions) < self.config.max_positions,   # 仓位限制
            trade_info['kelly_fraction'] > self.config.min_kelly_fraction  # 最小仓位要求
        ]
        
        return all(conditions)
    
    def should_sell(self, symbol: str, position: Dict, current_prediction: Dict, 
                   current_price: float) -> Tuple[bool, str]:
        """动态卖出条件判断（文档算法）"""
        
        probability = current_prediction.get('probability', 0.5)
        entry_date = pd.to_datetime(position['entry_date'])
        current_date = pd.to_datetime(current_prediction['date'])
        days_held = (current_date - entry_date).days
        
        unrealized_return = (current_price - position['avg_price']) / position['avg_price']
        
        # 多重卖出条件
        if probability < self.config.sell_threshold:
            return True, "probability_low"
        elif days_held > 5 and unrealized_return < -0.01:
            return True, "time_loss"
        elif unrealized_return < -self.config.stop_loss_ratio:
            return True, "stop_loss"
        elif days_held > self.config.max_holding_days:
            return True, "max_holding"
        
        return False, ""
    
    def daily_trading_process(self, date: str, predictions: List[Dict], 
                            price_data: pd.DataFrame) -> None:
        """每日交易决策流程（文档核心算法）"""
        
        logger.info(f"🔍 {date} 开始每日交易决策，预测数量: {len(predictions)}")
        
        # 步骤1: 评估所有交易机会
        potential_trades = []
        
        for prediction in predictions:
            symbol = prediction['symbol']
            probability = prediction['probability']
            
            # 获取当前价格
            current_price_data = price_data[
                (price_data['symbol'] == symbol) & 
                (price_data['date'] == pd.to_datetime(date))
            ]
            
            if current_price_data.empty:
                continue
                
            current_price = current_price_data.iloc[0]['close']
            
            # 计算预期收益率
            expected_return = self.calculate_expected_return(
                symbol, probability, current_price, price_data, date
            )
            
            # 计算风险调整收益率
            volatility = self.get_volatility(symbol, date, price_data)
            risk_adjusted_return = self.calculate_risk_adjusted_return(expected_return, volatility)
            
            # 计算Kelly最优仓位
            kelly_fraction = self.calculate_kelly_fraction(probability, expected_return)
            
            trade_info = {
                'symbol': symbol,
                'probability': probability,
                'expected_return': expected_return,
                'risk_adjusted_return': risk_adjusted_return,
                'kelly_fraction': kelly_fraction,
                'price': current_price,
                'volatility': volatility
            }
            
            potential_trades.append(trade_info)
            
            # 调试信息：记录每个交易机会的详细信息
            logger.debug(f"  📊 {symbol}: 概率={probability:.3f}, 预期收益={expected_return:.3f}, Kelly={kelly_fraction:.3f}")
        
        # 步骤2: 按风险调整收益率排序
        potential_trades.sort(key=lambda x: x['risk_adjusted_return'], reverse=True)
        
        # 步骤3: 处理卖出信号
        self.process_sell_signals(date, predictions, price_data)
        
        # 步骤4: 执行最优买入交易
        self.execute_optimal_trades(potential_trades, date)
    
    def process_sell_signals(self, date: str, predictions: List[Dict], 
                           price_data: pd.DataFrame) -> None:
        """处理卖出信号"""
        
        symbols_to_sell = []
        
        for symbol in list(self.positions.keys()):
            # 获取当前预测
            current_prediction = None
            for pred in predictions:
                if pred['symbol'] == symbol:
                    current_prediction = pred
                    break
            
            if current_prediction is None:
                continue
                
            # 获取当前价格
            current_price_data = price_data[
                (price_data['symbol'] == symbol) & 
                (price_data['date'] == pd.to_datetime(date))
            ]
            
            if current_price_data.empty:
                continue
                
            current_price = current_price_data.iloc[0]['close']
            
            # 判断是否卖出
            should_sell_flag, reason = self.should_sell(
                symbol, self.positions[symbol], current_prediction, current_price
            )
            
            if should_sell_flag:
                symbols_to_sell.append((symbol, current_price, reason))
        
        # 执行卖出
        for symbol, price, reason in symbols_to_sell:
            self.execute_sell(symbol, price, date, reason)
    
    def execute_optimal_trades(self, potential_trades: List[Dict], date: str) -> None:
        """执行最优买入交易"""
        
        logger.info(f"🎯 {date} 评估 {len(potential_trades)} 个交易机会")
        
        buy_candidates = 0
        for trade_info in potential_trades:
            # 详细记录买入决策过程
            should_buy_result = self.should_buy(trade_info, self.positions)
            
            if should_buy_result:
                buy_candidates += 1
                
            logger.info(f"  📈 {trade_info['symbol']}: 概率={trade_info['probability']:.3f} "
                       f"(阈值={self.config.buy_threshold}), 预期收益={trade_info['expected_return']:.3f}, "
                       f"Kelly={trade_info['kelly_fraction']:.3f}, 买入={should_buy_result}")
            
            if not should_buy_result:
                continue
                
            # 计算仓位大小（Kelly公式 + 基础仓位）
            base_position_value = self.cash * self.config.position_size
            kelly_position_value = self.cash * trade_info['kelly_fraction']
            
            # 取两者较大值，但不超过Kelly上限
            position_value = min(
                max(base_position_value, kelly_position_value),
                self.cash * self.config.max_kelly_fraction
            )
            
            # 考虑交易成本
            position_value *= (1 - self.config.transaction_cost)
            
            if position_value < 5000:  # 最小交易金额
                continue
                
            # 执行买入
            self.execute_buy(trade_info, position_value, date)
            
            # 检查现金是否充足
            if self.cash < 10000:  # 保留1万现金
                break
    
    def execute_buy(self, trade_info: Dict, position_value: float, date: str) -> None:
        """执行买入交易"""
        
        symbol = trade_info['symbol']
        price = trade_info['price']
        shares = position_value / price
        
        # 更新仓位
        self.positions[symbol] = {
            'shares': shares,
            'avg_price': price,
            'entry_date': date,
            'entry_value': position_value
        }
        
        # 更新现金
        self.cash -= position_value
        
        # 记录交易
        self.trades.append({
            'symbol': symbol,
            'action': 'buy',
            'price': price,
            'shares': shares,
            'value': position_value,
            'date': date,
            'probability': trade_info['probability'],
            'expected_return': trade_info['expected_return'],
            'kelly_fraction': trade_info['kelly_fraction']
        })
        
        logger.info(f"买入 {symbol}: {shares:.0f}股 @ {price:.2f}, 仓位价值: {position_value:.0f}")
    
    def execute_sell(self, symbol: str, price: float, date: str, reason: str) -> None:
        """执行卖出交易"""
        
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        shares = position['shares']
        sell_value = shares * price
        
        # 计算收益
        profit = sell_value - position['entry_value']
        return_rate = profit / position['entry_value']
        
        # 更新现金
        self.cash += sell_value * (1 - self.config.transaction_cost)
        
        # 记录交易
        self.trades.append({
            'symbol': symbol,
            'action': 'sell',
            'price': price,
            'shares': shares,
            'value': sell_value,
            'date': date,
            'reason': reason,
            'profit': profit,
            'return_rate': return_rate
        })
        
        # 删除仓位
        del self.positions[symbol]
        
        logger.info(f"卖出 {symbol}: {shares:.0f}股 @ {price:.2f}, 收益率: {return_rate:.2%}, 原因: {reason}")
    
    def update_portfolio_value(self, date: str, price_data: pd.DataFrame) -> None:
        """更新投资组合价值"""
        
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price_data = price_data[
                (price_data['symbol'] == symbol) & 
                (price_data['date'] == pd.to_datetime(date))
            ]
            
            if not current_price_data.empty:
                current_price = current_price_data.iloc[0]['close']
                position_value = position['shares'] * current_price
                portfolio_value += position_value
        
        self.portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions)
        })
    
    def run_backtest(self, predictions: pd.DataFrame, price_data: pd.DataFrame,
                    start_date: str = None, end_date: str = None) -> Dict:
        """运行增强策略回测"""
        
        logger.info("🚀 开始运行增强策略回测（基于文档完整算法）")
        
        # 初始化
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.cash = self.config.initial_capital
        
        # 筛选日期范围
        if start_date:
            predictions = predictions[predictions['prediction_date'] >= start_date]
            price_data = price_data[price_data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            predictions = predictions[predictions['prediction_date'] <= end_date]
            price_data = price_data[price_data['date'] <= pd.to_datetime(end_date)]
        
        # 按日期执行策略
        dates = sorted(predictions['prediction_date'].unique())
        
        for date in dates:
            # 获取当日预测
            daily_predictions = predictions[predictions['prediction_date'] == date]
            daily_pred_list = daily_predictions.to_dict('records')
            
            # 执行日交易决策
            self.daily_trading_process(date, daily_pred_list, price_data)
            
            # 更新投资组合价值
            self.update_portfolio_value(date, price_data)
        
        # 计算性能指标
        performance = self.calculate_performance_metrics()
        
        logger.info(f"📊 增强策略回测完成:")
        logger.info(f"   总收益率: {performance['total_return']:.2%}")
        logger.info(f"   年化收益率: {performance['annualized_return']:.2%}")
        logger.info(f"   夏普比率: {performance['sharpe_ratio']:.3f}")
        logger.info(f"   最大回撤: {performance['max_drawdown']:.2%}")
        logger.info(f"   交易次数: {len(self.trades)}")
        
        return {
            'performance_metrics': performance,
            'trades': self.trades,
            'portfolio_curve': self.portfolio_values,
            'final_positions': self.positions,
            'strategy_name': '📈 增强策略（文档完整版）'
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """计算性能指标"""
        
        if not self.portfolio_values:
            return {}
            
        initial_value = self.config.initial_capital
        final_value = self.portfolio_values[-1]['value']
        
        # 总收益率
        total_return = (final_value - initial_value) / initial_value
        
        # 年化收益率
        days = len(self.portfolio_values)
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0
        
        # 计算日收益率
        values = [pv['value'] for pv in self.portfolio_values]
        daily_returns = []
        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            daily_returns.append(daily_return)
        
        # 夏普比率
        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            daily_volatility = np.std(daily_returns)
            sharpe_ratio = avg_daily_return / daily_volatility * np.sqrt(252) if daily_volatility > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 交易统计
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        
        winning_trades = [t for t in sell_trades if t.get('return_rate', 0) > 0]
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': daily_volatility * np.sqrt(252) if daily_returns else 0,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'final_value': final_value,
            'days': days
        }

def main():
    """测试增强策略"""
    print("🧪 增强策略测试")
    
    # 这里需要真实的预测数据和价格数据来测试
    # 暂时只是框架展示
    
if __name__ == "__main__":
    main()
