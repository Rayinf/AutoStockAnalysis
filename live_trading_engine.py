"""
实盘交易引擎
基于回测算法的实时交易系统
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import akshare as ak
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    action: str  # 'buy' or 'sell'
    probability: float
    expected_return: float
    position_size: float
    current_price: float
    timestamp: datetime
    reason: str

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    entry_date: datetime
    last_update: datetime
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0

@dataclass
class OrderResult:
    """订单结果"""
    success: bool
    order_id: Optional[str]
    message: str
    timestamp: datetime

@dataclass
class RiskAlert:
    """风险警报"""
    type: str
    symbol: str
    message: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    timestamp: datetime

class LiveTradingConfig:
    """实盘交易配置"""
    def __init__(self):
        # 策略参数（从回测系统移植）
        self.buy_threshold = 0.6
        self.sell_threshold = 0.4
        self.max_positions = 10
        self.position_size = 0.1
        self.initial_capital = 100000.0
        self.transaction_cost = 0.002
        
        # 风险控制参数
        self.stop_loss_threshold = 0.03  # 3%止损
        self.max_single_position = 0.15  # 单股最大15%
        self.max_sector_concentration = 0.3  # 单行业最大30%
        self.max_holding_days = 15
        self.max_daily_loss = 0.02  # 日最大亏损2%
        
        # 系统参数
        self.data_update_interval = 60  # 数据更新间隔（秒）
        self.signal_check_interval = 300  # 信号检查间隔（秒）
        self.order_timeout = 300  # 订单超时时间（秒）
        self.market_open_time = "09:30"
        self.market_close_time = "15:00"

class RealTimeDataProvider:
    """实时数据提供器"""
    
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        self.update_interval = 60
    
    def get_real_time_price(self, symbol: str) -> Optional[Dict]:
        """获取实时价格"""
        try:
            # 使用AKShare获取实时行情
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == symbol]
            
            if not stock_data.empty:
                data = {
                    'symbol': symbol,
                    'price': float(stock_data['最新价'].iloc[0]),
                    'change_pct': float(stock_data['涨跌幅'].iloc[0]) / 100,
                    'volume': int(stock_data['成交量'].iloc[0]),
                    'turnover': float(stock_data['成交额'].iloc[0]),
                    'timestamp': datetime.now()
                }
                
                # 缓存数据
                self.cache[symbol] = data
                self.last_update[symbol] = datetime.now()
                
                return data
            
        except Exception as e:
            logger.error(f"获取实时数据失败 {symbol}: {e}")
            # 返回缓存数据（如果有）
            if symbol in self.cache:
                cached_data = self.cache[symbol].copy()
                cached_data['is_cached'] = True
                return cached_data
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """获取历史数据"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                   start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            logger.error(f"获取历史数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def is_data_fresh(self, symbol: str, max_age_seconds: int = 300) -> bool:
        """检查数据是否新鲜"""
        if symbol not in self.last_update:
            return False
        
        age = (datetime.now() - self.last_update[symbol]).total_seconds()
        return age <= max_age_seconds

class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def calculate_ma(prices: pd.Series, window: int) -> pd.Series:
        """计算移动平均线"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_volume_ratio(volumes: pd.Series, window: int = 5) -> pd.Series:
        """计算成交量比率"""
        avg_volume = volumes.rolling(window=window).mean()
        return volumes / avg_volume
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """计算所有技术指标"""
        if df.empty or len(df) < 20:
            return {}
        
        close_prices = df['收盘']
        volumes = df['成交量']
        
        indicators = {
            'ma5': self.calculate_ma(close_prices, 5).iloc[-1],
            'ma20': self.calculate_ma(close_prices, 20).iloc[-1],
            'rsi': self.calculate_rsi(close_prices).iloc[-1],
            'volume_ratio': self.calculate_volume_ratio(volumes).iloc[-1],
            'current_price': close_prices.iloc[-1],
            'price_change_pct': (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]
        }
        
        return indicators

class PredictionEngine:
    """预测引擎（移植自回测系统）"""
    
    def __init__(self):
        self.indicators_calc = TechnicalIndicators()
        self.data_provider = RealTimeDataProvider()
    
    def generate_prediction(self, symbol: str) -> Optional[Dict]:
        """生成实时预测"""
        try:
            # 获取历史数据计算技术指标
            hist_data = self.data_provider.get_historical_data(symbol, 30)
            if hist_data.empty:
                return None
            
            # 获取实时价格
            real_time_data = self.data_provider.get_real_time_price(symbol)
            if not real_time_data:
                return None
            
            # 计算技术指标
            indicators = self.indicators_calc.calculate_all_indicators(hist_data)
            if not indicators:
                return None
            
            # 生成预测概率（使用回测系统的逻辑）
            probability = self._calculate_probability(indicators, real_time_data)
            
            # 计算预期收益
            expected_return = self._calculate_expected_return(probability, indicators)
            
            return {
                'symbol': symbol,
                'probability': probability,
                'expected_return': expected_return,
                'current_price': real_time_data['price'],
                'indicators': indicators,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"预测生成失败 {symbol}: {e}")
            return None
    
    def _calculate_probability(self, indicators: Dict, real_time_data: Dict) -> float:
        """计算预测概率（移植自回测系统）"""
        try:
            current_price = real_time_data['price']
            ma5 = indicators.get('ma5', current_price)
            ma20 = indicators.get('ma20', current_price)
            rsi = indicators.get('rsi', 50)
            volume_ratio = indicators.get('volume_ratio', 1)
            
            # 综合评分
            score = 0
            
            # 均线信号
            if current_price > ma5:
                score += 0.2
            if current_price > ma20:
                score += 0.3
            
            # RSI信号
            if 30 < rsi < 70:
                score += 0.1
            elif rsi < 30:  # 超卖
                score += 0.15
            
            # 成交量信号
            if volume_ratio > 1.2:
                score += 0.15
            
            # 价格相对位置
            if ma20 > 0:
                price_position = (current_price - ma20) / ma20
                if price_position > 0:
                    score += 0.25
            
            # 转换为概率（0.1-0.9区间）
            probability = max(0.1, min(0.9, 0.5 + score - 0.5))
            
            return probability
            
        except Exception as e:
            logger.error(f"概率计算失败: {e}")
            return 0.5  # 默认概率
    
    def _calculate_expected_return(self, probability: float, indicators: Dict) -> float:
        """计算预期收益（移植自回测系统）"""
        try:
            # 基于历史波动率和预测概率
            price_change_pct = indicators.get('price_change_pct', 0)
            volatility = abs(price_change_pct)  # 简化的波动率
            
            if probability > 0.5:
                expected_return = (probability - 0.5) * 2 * 0.02 + volatility * (probability - 0.5)
            else:
                expected_return = (probability - 0.5) * 2 * 0.02 - volatility * (0.5 - probability)
            
            return expected_return
            
        except Exception as e:
            logger.error(f"预期收益计算失败: {e}")
            return 0.0

class PositionManager:
    """持仓管理器"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.data_provider = RealTimeDataProvider()
    
    def add_position(self, symbol: str, quantity: int, price: float):
        """添加持仓"""
        if symbol in self.positions:
            # 加仓
            old_position = self.positions[symbol]
            total_quantity = old_position.quantity + quantity
            total_cost = old_position.quantity * old_position.avg_price + quantity * price
            new_avg_price = total_cost / total_quantity
            
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                avg_price=new_avg_price,
                current_price=price,
                entry_date=old_position.entry_date,
                last_update=datetime.now()
            )
        else:
            # 新建持仓
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                entry_date=datetime.now(),
                last_update=datetime.now()
            )
    
    def reduce_position(self, symbol: str, quantity: int) -> Optional[Position]:
        """减少持仓"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        if quantity >= position.quantity:
            # 全部卖出
            return self.positions.pop(symbol)
        else:
            # 部分卖出
            position.quantity -= quantity
            position.last_update = datetime.now()
            return position
    
    def update_positions(self):
        """更新所有持仓的市值"""
        for symbol, position in self.positions.items():
            real_time_data = self.data_provider.get_real_time_price(symbol)
            if real_time_data:
                position.current_price = real_time_data['price']
                position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
                position.unrealized_pnl_percent = (position.current_price - position.avg_price) / position.avg_price
                position.last_update = datetime.now()
    
    def get_total_market_value(self) -> float:
        """获取总市值"""
        self.update_positions()
        return sum(pos.current_price * pos.quantity for pos in self.positions.values())
    
    def get_total_unrealized_pnl(self) -> float:
        """获取总未实现盈亏"""
        self.update_positions()
        return sum(pos.unrealized_pnl for pos in self.positions.values())

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.position_manager = PositionManager()
        self.daily_trades = []
        self.daily_pnl = 0.0
    
    def validate_order(self, signal: TradingSignal, available_cash: float) -> Tuple[bool, str]:
        """验证订单风险"""
        
        # 1. 资金检查
        if signal.action == 'buy':
            required_capital = signal.current_price * signal.position_size * available_cash / signal.current_price
            if required_capital > available_cash:
                return False, f"资金不足: 需要{required_capital:.2f}, 可用{available_cash:.2f}"
        
        # 2. 持仓数量检查
        if signal.action == 'buy' and len(self.position_manager.positions) >= self.config.max_positions:
            if signal.symbol not in self.position_manager.positions:  # 不是加仓
                return False, f"超过最大持仓数: {len(self.position_manager.positions)}/{self.config.max_positions}"
        
        # 3. 单股仓位检查
        if signal.action == 'buy':
            position_value = signal.current_price * signal.position_size * available_cash / signal.current_price
            total_capital = available_cash + self.position_manager.get_total_market_value()
            position_weight = position_value / total_capital
            
            if position_weight > self.config.max_single_position:
                return False, f"单股仓位过大: {position_weight:.2%} > {self.config.max_single_position:.2%}"
        
        # 4. 日亏损检查
        if self.daily_pnl < -self.config.max_daily_loss * self.config.initial_capital:
            return False, f"超过日最大亏损限制: {self.daily_pnl:.2f}"
        
        return True, "风险检查通过"
    
    def monitor_positions(self) -> List[RiskAlert]:
        """监控持仓风险"""
        alerts = []
        self.position_manager.update_positions()
        
        for symbol, position in self.position_manager.positions.items():
            # 止损检查
            if position.unrealized_pnl_percent < -self.config.stop_loss_threshold:
                alerts.append(RiskAlert(
                    type='STOP_LOSS',
                    symbol=symbol,
                    message=f"触发止损: {position.unrealized_pnl_percent:.2%}",
                    severity='HIGH',
                    timestamp=datetime.now()
                ))
            
            # 持仓时间检查
            holding_days = (datetime.now() - position.entry_date).days
            if holding_days > self.config.max_holding_days:
                alerts.append(RiskAlert(
                    type='LONG_HOLDING',
                    symbol=symbol,
                    message=f"持仓时间过长: {holding_days}天",
                    severity='MEDIUM',
                    timestamp=datetime.now()
                ))
        
        return alerts
    
    def calculate_kelly_fraction(self, probability: float, expected_return: float) -> float:
        """计算Kelly公式仓位（移植自回测系统）"""
        try:
            if probability <= 0.5 or expected_return <= 0:
                return 0.0
            
            win_prob = probability
            lose_prob = 1 - probability
            win_amount = expected_return
            lose_amount = expected_return / 2
            
            if lose_amount > 0:
                kelly_fraction = (win_prob * win_amount - lose_prob * lose_amount) / win_amount
                return max(0, min(kelly_fraction, 0.25))  # 最大25%仓位
            else:
                return 0.1
                
        except Exception:
            return 0.1

class MockBrokerInterface:
    """模拟券商接口（用于测试）"""
    
    def __init__(self):
        self.orders = {}
        self.order_counter = 1
        self.account_cash = 100000.0
    
    def submit_order(self, symbol: str, action: str, quantity: int, price: float) -> OrderResult:
        """提交订单"""
        try:
            order_id = f"ORDER_{self.order_counter:06d}"
            self.order_counter += 1
            
            # 模拟订单处理
            if action == 'buy':
                cost = quantity * price * 1.002  # 包含手续费
                if cost > self.account_cash:
                    return OrderResult(
                        success=False,
                        order_id=None,
                        message="资金不足",
                        timestamp=datetime.now()
                    )
                self.account_cash -= cost
            else:  # sell
                revenue = quantity * price * 0.998  # 扣除手续费
                self.account_cash += revenue
            
            # 记录订单
            self.orders[order_id] = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'status': 'FILLED',
                'timestamp': datetime.now()
            }
            
            return OrderResult(
                success=True,
                order_id=order_id,
                message="订单成交",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                message=str(e),
                timestamp=datetime.now()
            )
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        return {
            'cash': self.account_cash,
            'total_value': self.account_cash,  # 简化处理
            'available_cash': self.account_cash
        }

class LiveTradingEngine:
    """实盘交易引擎"""
    
    def __init__(self, config: LiveTradingConfig, broker_interface=None):
        self.config = config
        self.broker = broker_interface or MockBrokerInterface()
        self.prediction_engine = PredictionEngine()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(config)
        
        self.running = False
        self.watch_list = []  # 监控股票列表
        self.trade_log = []
        
        # 性能统计
        self.daily_returns = []
        self.total_trades = 0
        self.winning_trades = 0
    
    def add_to_watchlist(self, symbols: List[str]):
        """添加到监控列表"""
        self.watch_list.extend(symbols)
        self.watch_list = list(set(self.watch_list))  # 去重
        logger.info(f"监控列表更新: {self.watch_list}")
    
    def is_trading_hours(self) -> bool:
        """检查是否在交易时间"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # 简化处理，仅检查时间，不检查交易日
        return self.config.market_open_time <= current_time <= self.config.market_close_time
    
    async def generate_signals(self) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []
        
        if not self.is_trading_hours():
            return signals
        
        for symbol in self.watch_list:
            try:
                # 生成预测
                prediction = self.prediction_engine.generate_prediction(symbol)
                if not prediction:
                    continue
                
                # 检查买入信号
                buy_signal = self._check_buy_signal(symbol, prediction)
                if buy_signal:
                    signals.append(buy_signal)
                
                # 检查卖出信号（已持仓股票）
                if symbol in self.position_manager.positions:
                    sell_signal = self._check_sell_signal(symbol, prediction)
                    if sell_signal:
                        signals.append(sell_signal)
                        
            except Exception as e:
                logger.error(f"信号生成失败 {symbol}: {e}")
        
        return signals
    
    def _check_buy_signal(self, symbol: str, prediction: Dict) -> Optional[TradingSignal]:
        """检查买入信号（移植自回测系统）"""
        probability = prediction['probability']
        expected_return = prediction['expected_return']
        current_price = prediction['current_price']
        
        # 买入条件（与回测系统一致）
        can_buy = (
            symbol not in self.position_manager.positions or  # 新股票
            (symbol in self.position_manager.positions and 
             probability > 0.65)  # 概率提升可加仓
        )
        
        if (expected_return > -0.01 and 
            probability > self.config.buy_threshold and
            can_buy):
            
            # 计算Kelly仓位
            kelly_fraction = self.risk_manager.calculate_kelly_fraction(probability, expected_return)
            optimal_fraction = min(kelly_fraction, self.config.position_size)
            
            return TradingSignal(
                symbol=symbol,
                action='buy',
                probability=probability,
                expected_return=expected_return,
                position_size=optimal_fraction,
                current_price=current_price,
                timestamp=datetime.now(),
                reason=f"概率:{probability:.3f}, 预期收益:{expected_return:.3f}"
            )
        
        return None
    
    def _check_sell_signal(self, symbol: str, prediction: Dict) -> Optional[TradingSignal]:
        """检查卖出信号（移植自回测系统）"""
        probability = prediction['probability']
        current_price = prediction['current_price']
        
        position = self.position_manager.positions.get(symbol)
        if not position:
            return None
        
        # 动态卖出条件
        holding_days = (datetime.now() - position.entry_date).days
        unrealized_return = position.unrealized_pnl_percent
        
        sell_conditions = [
            probability < self.config.sell_threshold,  # 概率低于卖出阈值
            (holding_days > 5 and unrealized_return < -0.01),  # 持有超过5天且亏损1%
            unrealized_return < -0.03,  # 亏损超过3%（止损）
            holding_days > 15  # 持有超过15天（时间止损）
        ]
        
        if any(sell_conditions):
            return TradingSignal(
                symbol=symbol,
                action='sell',
                probability=probability,
                expected_return=0.0,
                position_size=1.0,  # 全部卖出
                current_price=current_price,
                timestamp=datetime.now(),
                reason=f"卖出条件触发: 概率{probability:.3f}, 持仓{holding_days}天, 收益{unrealized_return:.2%}"
            )
        
        return None
    
    async def execute_signal(self, signal: TradingSignal) -> bool:
        """执行交易信号"""
        try:
            # 获取账户信息
            account_info = self.broker.get_account_info()
            available_cash = account_info['available_cash']
            
            # 风险检查
            risk_passed, risk_message = self.risk_manager.validate_order(signal, available_cash)
            if not risk_passed:
                logger.warning(f"风险检查失败 {signal.symbol}: {risk_message}")
                return False
            
            # 计算交易数量
            if signal.action == 'buy':
                trade_amount = available_cash * signal.position_size
                quantity = int(trade_amount / signal.current_price / 100) * 100  # 整手
            else:  # sell
                position = self.position_manager.positions.get(signal.symbol)
                quantity = position.quantity if position else 0
            
            if quantity <= 0:
                logger.warning(f"交易数量为0 {signal.symbol}")
                return False
            
            # 提交订单
            order_result = self.broker.submit_order(
                symbol=signal.symbol,
                action=signal.action,
                quantity=quantity,
                price=signal.current_price
            )
            
            if order_result.success:
                # 更新持仓
                if signal.action == 'buy':
                    self.position_manager.add_position(signal.symbol, quantity, signal.current_price)
                else:
                    self.position_manager.reduce_position(signal.symbol, quantity)
                
                # 记录交易
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': quantity,
                    'price': signal.current_price,
                    'probability': signal.probability,
                    'reason': signal.reason,
                    'order_id': order_result.order_id
                }
                self.trade_log.append(trade_record)
                self.total_trades += 1
                
                logger.info(f"交易执行成功: {signal.action} {signal.symbol} {quantity}股 @{signal.current_price}")
                return True
            else:
                logger.error(f"订单提交失败 {signal.symbol}: {order_result.message}")
                return False
                
        except Exception as e:
            logger.error(f"执行交易失败 {signal.symbol}: {e}")
            return False
    
    async def run_trading_loop(self):
        """运行交易循环"""
        logger.info("开始实盘交易...")
        self.running = True
        
        try:
            while self.running:
                try:
                    # 生成交易信号
                    signals = await self.generate_signals()
                    
                    # 执行信号
                    for signal in signals:
                        await self.execute_signal(signal)
                    
                    # 风险监控
                    risk_alerts = self.risk_manager.monitor_positions()
                    for alert in risk_alerts:
                        logger.warning(f"风险警报: {alert.type} {alert.symbol} - {alert.message}")
                    
                    # 等待下一次检查
                    await asyncio.sleep(self.config.signal_check_interval)
                    
                except Exception as e:
                    logger.error(f"交易循环异常: {e}")
                    await asyncio.sleep(60)  # 异常时等待1分钟
                    
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭...")
        finally:
            self.running = False
            logger.info("交易引擎已停止")
    
    def stop_trading(self):
        """停止交易"""
        self.running = False
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        total_market_value = self.position_manager.get_total_market_value()
        total_unrealized_pnl = self.position_manager.get_total_unrealized_pnl()
        account_info = self.broker.get_account_info()
        
        return {
            'total_capital': account_info['cash'] + total_market_value,
            'cash': account_info['cash'],
            'market_value': total_market_value,
            'unrealized_pnl': total_unrealized_pnl,
            'total_trades': self.total_trades,
            'positions_count': len(self.position_manager.positions),
            'watch_list_size': len(self.watch_list)
        }

# 使用示例
async def main():
    """主函数示例"""
    # 创建配置
    config = LiveTradingConfig()
    
    # 创建交易引擎
    engine = LiveTradingEngine(config)
    
    # 添加监控股票
    engine.add_to_watchlist(['000001', '000002', '600519', '000858'])
    
    # 运行交易引擎
    try:
        await engine.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("手动停止交易")
    finally:
        # 打印性能摘要
        summary = engine.get_performance_summary()
        logger.info(f"交易摘要: {summary}")

if __name__ == "__main__":
    asyncio.run(main())
