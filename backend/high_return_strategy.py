"""
高收益策略实现 - 集成到系统中的35%+年化收益策略
基于机器学习和多信号融合的激进量化策略
"""

import sys
import os
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
import akshare as ak
import yaml
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from backend.strategy_backtest import StrategyConfig, StrategyBacktester, TradeRecord

logger = logging.getLogger(__name__)

class HighReturnStrategy:
    """高收益策略 - 35%+年化收益率策略实现"""
    
    def __init__(self, config_path: str = None):
        """初始化高收益策略"""
        self.config_path = config_path or os.path.join(project_root, "high_return_strategy_config.yaml")
        self.config = self.load_config()
        
        # 策略参数
        self.strategy_params = self.config['strategy']
        self.risk_params = self.config['risk_management']
        self.ml_params = self.config['machine_learning']
        self.tech_params = self.config['technical_indicators']
        self.signal_params = self.config['signal_fusion']
        
        # ML模型存储
        self.ml_models = {}
        
        # 交易记录
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        
        # 数据缓存
        self.data_cache = {}
        
        logger.info("🚀 高收益策略初始化完成")
        logger.info(f"📊 目标年化收益率: {self.config['performance_targets']['annual_return']*100:.1f}%")
        logger.info(f"⭐ 目标夏普比率: {self.config['performance_targets']['sharpe_ratio']}")
    
    def load_config(self) -> Dict:
        """加载策略配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def get_strategy_config(self) -> StrategyConfig:
        """获取策略配置对象"""
        return StrategyConfig(
            buy_threshold=self.strategy_params['buy_threshold'],
            sell_threshold=self.strategy_params['sell_threshold'],
            max_positions=self.strategy_params['max_positions'],
            initial_capital=self.strategy_params['initial_capital'],
            transaction_cost=self.strategy_params['transaction_cost'],
            position_size=self.strategy_params['position_size'],
            calibration_gamma=1.0,  # 使用默认值
            selection_mode="aggressive",  # 使用激进模式
            top_k=self.strategy_params['max_positions'],
            momentum_weight=0.5,
            volatility_boost=1.2,
            profit_target=self.risk_params['take_profit_threshold'],
            loss_tolerance=self.risk_params['stop_loss_threshold']
        )
    
    def load_stock_data(self, symbols: List[str], days: int = 365, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """加载股票数据"""
        data_dict = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"📥 加载股票数据 ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})...")
        if force_refresh:
            logger.info("🔄 强制刷新模式：跳过所有缓存")
        
        for symbol in symbols:
            try:
                # 检查缓存（除非强制刷新）
                cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                if not force_refresh and cache_key in self.data_cache:
                    data_dict[symbol] = self.data_cache[cache_key]
                    logger.debug(f"   📋 {symbol}: 使用缓存数据")
                    continue
                
                # 获取数据
                df = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"
                )
                
                if df is not None and len(df) > 0:
                    # 标准化列名
                    cols = list(df.columns)
                    if len(cols) >= 7:
                        column_mapping = {
                            cols[0]: 'date',
                            cols[2]: 'open',
                            cols[3]: 'close',
                            cols[4]: 'high',
                            cols[5]: 'low',
                            cols[6]: 'volume'
                        }
                        
                        df = df.rename(columns=column_mapping)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').reset_index(drop=True)
                        df = df[df['volume'] > 0].reset_index(drop=True)
                        
                        # 缓存数据
                        self.data_cache[cache_key] = df
                        data_dict[symbol] = df
                        logger.debug(f"   ✅ {symbol}: {len(df)} 条数据")
                
            except Exception as e:
                logger.error(f"   ❌ {symbol}: 数据加载失败 - {e}")
        
        logger.info(f"📊 成功加载 {len(data_dict)} 只股票数据")
        return data_dict
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 基础收益率
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 多周期动量
        for period in [1, 3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # RSI
        rsi_period = self.tech_params['rsi_period']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 移动平均
        df['ma_short'] = df['close'].rolling(window=self.tech_params['ma_short']).mean()
        df['ma_medium'] = df['close'].rolling(window=self.tech_params['ma_medium']).mean()
        df['ma_long'] = df['close'].rolling(window=self.tech_params['ma_long']).mean()
        
        # 布林带
        bb_period = self.tech_params['bb_period']
        bb_std_mult = self.tech_params['bb_std_multiplier']
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_mult * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_mult * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # MACD
        exp1 = df['close'].ewm(span=self.tech_params['macd_fast']).mean()
        exp2 = df['close'].ewm(span=self.tech_params['macd_slow']).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.tech_params['macd_signal']).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 波动率
        vol_period = self.tech_params['volatility_period']
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # 成交量
        df['volume_sma'] = df['volume'].rolling(window=self.tech_params['volume_ma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_momentum'] = df['volume'] / df['volume'].shift(1) - 1
        
        # 价格位置
        price_period = self.tech_params['price_position_period']
        df['price_position_20'] = (df['close'] - df['close'].rolling(price_period).min()) / (
            df['close'].rolling(price_period).max() - df['close'].rolling(price_period).min())
        
        # 趋势强度
        df['trend_strength'] = abs(df['ma_short'] - df['ma_long']) / df['close']
        df['trend_direction'] = np.where(df['ma_short'] > df['ma_medium'], 1, -1)
        
        # 支撑阻力
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['resistance_distance'] = (df['resistance_20'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['support_20']) / df['close']
        
        return df
    
    def train_ml_models(self, data_dict: Dict[str, pd.DataFrame]):
        """训练机器学习模型"""
        if not self.ml_params['enabled']:
            return
        
        logger.info("🤖 开始训练机器学习模型...")
        
        for symbol, data in data_dict.items():
            try:
                df = self.calculate_technical_indicators(data)
                
                # 准备特征
                feature_cols = self.ml_params['feature_columns']
                
                # 创建目标变量
                horizon = self.ml_params['prediction_horizon']
                df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
                
                # 清理数据
                valid_data = df[feature_cols + ['future_return']].dropna()
                
                if len(valid_data) > 50:
                    X = valid_data[feature_cols].values
                    y = valid_data['future_return'].values
                    
                    # 训练模型
                    model = RandomForestRegressor(
                        n_estimators=self.ml_params['n_estimators'],
                        max_depth=self.ml_params['max_depth'],
                        min_samples_split=self.ml_params['min_samples_split'],
                        random_state=42
                    )
                    model.fit(X, y)
                    
                    self.ml_models[symbol] = {
                        'model': model,
                        'feature_cols': feature_cols,
                        'train_samples': len(valid_data)
                    }
                    
                    logger.info(f"   ✅ {symbol}: 模型训练完成 ({len(valid_data)}样本)")
                else:
                    logger.warning(f"   ⚠️ {symbol}: 训练样本不足 ({len(valid_data)}<50)")
                    
            except Exception as e:
                logger.error(f"   ❌ {symbol}: 模型训练失败 - {e}")
        
        logger.info(f"🤖 ML模型训练完成，共训练 {len(self.ml_models)} 个模型")
    
    def detect_market_regime(self, data_dict: Dict[str, pd.DataFrame]) -> str:
        """检测市场环境"""
        if not self.config['market_regime']['enabled']:
            return 'neutral'
        
        # 计算市场整体特征
        all_returns = []
        all_volatilities = []
        
        for symbol, data in data_dict.items():
            if len(data) > 20:
                returns = data['close'].pct_change().dropna()
                volatility = returns.rolling(window=20).std().iloc[-1]
                
                all_returns.extend(returns.tail(20).tolist())
                all_volatilities.append(volatility)
        
        if not all_returns:
            return 'neutral'
        
        # 市场特征
        market_return = np.mean(all_returns)
        market_volatility = np.mean(all_volatilities)
        
        trend_threshold = self.config['market_regime']['trend_threshold']
        vol_threshold = self.config['market_regime']['volatility_threshold']
        
        # 环境分类
        if market_volatility > vol_threshold:
            if market_return > trend_threshold:
                return 'volatile_bullish'
            elif market_return < -trend_threshold:
                return 'volatile_bearish'
            else:
                return 'high_volatility'
        else:
            if market_return > trend_threshold:
                return 'trending_up'
            elif market_return < -trend_threshold:
                return 'trending_down'
            else:
                return 'low_volatility'
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, market_regime: str) -> pd.DataFrame:
        """生成交易信号"""
        df = data.copy()
        df['signal'] = 0
        df['signal_strength'] = 0.0
        df['signal_reason'] = ''
        
        # 获取市场环境调整参数
        regime_config = self.config['market_regime']['regimes'].get(market_regime, {})
        momentum_weight = regime_config.get('momentum_weight', 1.0)
        reversal_weight = regime_config.get('reversal_weight', 1.0)
        
        for i in range(len(df)):
            if i < 30:  # 需要足够历史数据
                continue
            
            signals = []
            reasons = []
            weights = []
            current_row = df.iloc[i]
            
            # 1. 强动量信号
            momentum_threshold = self.strategy_params['momentum_threshold']
            if current_row['momentum_5'] > momentum_threshold * 2:  # 强动量
                signal_val = min(current_row['momentum_5'] * 10, 1.0) * momentum_weight
                signals.append(signal_val)
                reasons.append('strong_momentum_up')
                weights.append(self.signal_params['signal_weights']['strong_momentum'])
            elif current_row['momentum_5'] < -momentum_threshold * 2:
                signal_val = max(current_row['momentum_5'] * 10, -1.0) * momentum_weight
                signals.append(signal_val)
                reasons.append('strong_momentum_down')
                weights.append(self.signal_params['signal_weights']['strong_momentum'])
            
            # 2. 突破信号
            breakout_threshold = self.tech_params['breakout_threshold']
            if current_row['price_position_20'] > breakout_threshold:
                signals.append(0.8)
                reasons.append('breakout_high')
                weights.append(self.signal_params['signal_weights']['breakout_signal'])
            elif current_row['price_position_20'] < (1 - breakout_threshold):
                signals.append(-0.8)
                reasons.append('breakout_low')
                weights.append(self.signal_params['signal_weights']['breakout_signal'])
            
            # 3. RSI极值
            rsi_extreme_oversold = self.tech_params['rsi_extreme_oversold']
            rsi_extreme_overbought = self.tech_params['rsi_extreme_overbought']
            if current_row['rsi'] < rsi_extreme_oversold:
                signals.append(0.7 * reversal_weight)
                reasons.append('rsi_extreme_oversold')
                weights.append(self.signal_params['signal_weights']['extreme_rsi'])
            elif current_row['rsi'] > rsi_extreme_overbought:
                signals.append(-0.7 * reversal_weight)
                reasons.append('rsi_extreme_overbought')
                weights.append(self.signal_params['signal_weights']['extreme_rsi'])
            
            # 4. 布林带挤压突破
            if (current_row['bb_squeeze'] < self.tech_params['bb_squeeze_threshold'] and
                current_row['volume_ratio'] > 1.5):
                direction = 0.6 if current_row['close'] > current_row['bb_middle'] else -0.6
                signals.append(direction)
                reasons.append('bb_squeeze_breakout')
                weights.append(self.signal_params['signal_weights']['bb_squeeze'])
            
            # 5. MACD信号
            if (current_row['macd'] > current_row['macd_signal'] and
                df.iloc[i-1]['macd'] <= df.iloc[i-1]['macd_signal']):
                signals.append(0.5)
                reasons.append('macd_golden_cross')
                weights.append(self.signal_params['signal_weights']['macd_cross'])
            elif (current_row['macd'] < current_row['macd_signal'] and
                  df.iloc[i-1]['macd'] >= df.iloc[i-1]['macd_signal']):
                signals.append(-0.5)
                reasons.append('macd_death_cross')
                weights.append(self.signal_params['signal_weights']['macd_cross'])
            
            # 6. 成交量异常
            volume_spike_threshold = self.tech_params['volume_spike_threshold']
            if current_row['volume_ratio'] > volume_spike_threshold:
                direction = 0.4 if current_row['returns'] > 0 else -0.4
                signals.append(direction)
                reasons.append('volume_spike')
                weights.append(self.signal_params['signal_weights']['volume_spike'])
            
            # 7. 机器学习预测
            if symbol in self.ml_models:
                try:
                    model_info = self.ml_models[symbol]
                    model = model_info['model']
                    feature_cols = model_info['feature_cols']
                    
                    features = []
                    for col in feature_cols:
                        val = current_row.get(col, 0)
                        features.append(val if not pd.isna(val) else 0)
                    
                    prediction = model.predict([features])[0]
                    confidence_threshold = self.ml_params['confidence_threshold']
                    
                    if abs(prediction) > confidence_threshold:
                        amplification = self.ml_params['signal_amplification']
                        ml_signal = np.clip(prediction * amplification, -0.8, 0.8)
                        signals.append(ml_signal)
                        reasons.append('ml_prediction')
                        weights.append(self.signal_params['signal_weights']['ml_prediction'])
                        
                except Exception:
                    pass  # 忽略ML预测错误
            
            # 8. 趋势跟踪
            if (current_row['trend_direction'] == 1 and 
                current_row['trend_strength'] > 0.02):
                signals.append(0.3)
                reasons.append('trend_following_up')
                weights.append(self.signal_params['signal_weights']['trend_following'])
            elif (current_row['trend_direction'] == -1 and 
                  current_row['trend_strength'] > 0.02):
                signals.append(-0.3)
                reasons.append('trend_following_down')
                weights.append(self.signal_params['signal_weights']['trend_following'])
            
            # 综合信号
            if signals:
                # 使用权重平均
                if len(weights) == len(signals):
                    combined_signal = np.average(signals, weights=weights)
                else:
                    combined_signal = np.mean(signals)
                
                df.loc[df.index[i], 'signal_strength'] = combined_signal
                df.loc[df.index[i], 'signal_reason'] = '|'.join(reasons)
                
                # 应用信号阈值
                signal_threshold = self.strategy_params['signal_threshold']
                if abs(combined_signal) > signal_threshold:
                    df.loc[df.index[i], 'signal'] = 1 if combined_signal > 0 else -1
        
        return df
    
    def run_backtest(self, symbols: List[str] = None, days: int = 365, force_refresh: bool = False) -> Dict:
        """运行回测"""
        if symbols is None:
            symbols = self.config['watchlist']['primary_symbols']
        
        logger.info(f"🚀 开始运行高收益策略回测")
        logger.info(f"📊 股票池: {symbols}")
        logger.info(f"📅 回测期间: {days}天")
        
        # 加载数据
        data_dict = self.load_stock_data(symbols, days, force_refresh=force_refresh)
        if not data_dict:
            raise ValueError("未能加载任何股票数据")
        
        # 训练ML模型
        self.train_ml_models(data_dict)
        
        # 检测市场环境
        market_regime = self.detect_market_regime(data_dict)
        logger.info(f"🌍 检测到市场环境: {market_regime}")
        
        # 生成信号
        processed_data = {}
        for symbol, data in data_dict.items():
            df_with_indicators = self.calculate_technical_indicators(data)
            df_with_signals = self.generate_signals(df_with_indicators, symbol, market_regime)
            processed_data[symbol] = df_with_signals
        
        # 执行回测
        results = self._execute_backtest(processed_data)
        results['market_regime'] = market_regime
        results['symbols'] = symbols
        results['backtest_days'] = days
        
        # 记录结果
        logger.info(f"🏆 回测完成!")
        logger.info(f"   总收益率: {results['total_return']:.4f} ({results['total_return']*100:.2f}%)")
        logger.info(f"   年化收益率: {results.get('annualized_return', 0):.4f} ({results.get('annualized_return', 0)*100:.2f}%)")
        logger.info(f"   夏普比率: {results['sharpe_ratio']:.4f}")
        logger.info(f"   最大回撤: {results['max_drawdown']:.4f} ({results['max_drawdown']*100:.2f}%)")
        logger.info(f"   交易次数: {results['trade_count']}")
        logger.info(f"   胜率: {results['win_rate']:.2%}")
        
        return results
    
    def _execute_backtest(self, processed_data: Dict[str, pd.DataFrame]) -> Dict:
        """执行回测逻辑"""
        # 获取交易日期
        all_dates = set()
        for data in processed_data.values():
            all_dates.update(data['date'].dt.date)
        trading_dates = sorted(list(all_dates))
        
        # 初始化组合
        initial_capital = self.strategy_params['initial_capital']
        cash = initial_capital
        positions = {}
        portfolio_values = []
        trades = []
        
        # 策略参数
        max_positions = self.strategy_params['max_positions']
        max_position_size = self.strategy_params['max_position_size']
        cash_reserve = self.strategy_params['cash_reserve']
        stop_loss = self.risk_params['stop_loss_threshold']
        take_profit = self.risk_params['take_profit_threshold']
        max_holding_days = self.risk_params['max_holding_days']
        
        for date in trading_dates:
            daily_signals = {}
            daily_prices = {}
            
            # 收集当日信号和价格
            for symbol, data in processed_data.items():
                day_data = data[data['date'].dt.date == date]
                if not day_data.empty:
                    row = day_data.iloc[0]
                    daily_signals[symbol] = {
                        'signal': row['signal'],
                        'signal_strength': row['signal_strength'],
                        'reason': row['signal_reason']
                    }
                    daily_prices[symbol] = row['close']
            
            # 计算当前组合价值
            portfolio_value = cash
            for symbol, position in positions.items():
                if symbol in daily_prices:
                    position_value = position['size'] * daily_prices[symbol]
                    portfolio_value += position_value
            
            # 平仓检查
            for symbol in list(positions.keys()):
                if symbol in daily_prices:
                    position = positions[symbol]
                    current_price = daily_prices[symbol]
                    
                    return_rate = (current_price - position['entry_price']) / position['entry_price']
                    if position['direction'] == 'short':
                        return_rate = -return_rate
                    
                    should_close = False
                    close_reason = ''
                    
                    # 止盈止损
                    if return_rate >= take_profit:
                        should_close = True
                        close_reason = 'take_profit'
                    elif return_rate <= -stop_loss:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif (date - position['entry_date']).days >= max_holding_days:
                        should_close = True
                        close_reason = 'time_stop'
                    elif symbol in daily_signals:
                        # 强信号反转
                        signal_strength = daily_signals[symbol]['signal_strength']
                        if ((position['direction'] == 'long' and signal_strength < -0.5) or 
                            (position['direction'] == 'short' and signal_strength > 0.5)):
                            should_close = True
                            close_reason = 'signal_reversal'
                    
                    if should_close:
                        trade_value = position['size'] * current_price
                        cash += trade_value
                        
                        trades.append({
                            'symbol': symbol,
                            'direction': position['direction'],
                            'entry_date': position['entry_date'].strftime('%Y-%m-%d') if hasattr(position['entry_date'], 'strftime') else str(position['entry_date']),
                            'exit_date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'size': position['size'],
                            'return': return_rate,
                            'reason': close_reason
                        })
                        
                        del positions[symbol]
            
            # 开仓检查
            available_cash = cash * (1 - cash_reserve)
            
            if len(positions) < max_positions and available_cash > 5000:
                # 按信号强度排序
                signals_sorted = []
                for symbol, sig in daily_signals.items():
                    if (sig['signal'] != 0 and 
                        symbol not in positions and 
                        symbol in daily_prices):
                        
                        score = abs(sig['signal_strength'])
                        # 给强信号加权
                        if 'strong_momentum' in sig['reason'] or 'breakout' in sig['reason']:
                            score *= 1.5
                        if 'extreme' in sig['reason'] or 'ml_prediction' in sig['reason']:
                            score *= 1.3
                        
                        signals_sorted.append((symbol, sig, score))
                
                signals_sorted.sort(key=lambda x: x[2], reverse=True)
                
                # 开仓
                for symbol, signal_data, score in signals_sorted[:max_positions - len(positions)]:
                    signal_strength = signal_data['signal_strength']
                    
                    # 动态仓位
                    signal_multiplier = min(abs(signal_strength) * 2, 1.5)
                    position_size_ratio = max_position_size * signal_multiplier
                    
                    position_value = min(
                        available_cash * position_size_ratio,
                        available_cash / max(1, max_positions - len(positions))
                    )
                    
                    if position_value > 5000:
                        price = daily_prices[symbol]
                        size = position_value / price
                        
                        positions[symbol] = {
                            'direction': 'long' if signal_strength > 0 else 'short',
                            'entry_date': date,  # 保持日期对象用于计算
                            'entry_price': price,
                            'size': size,
                            'signal_strength': signal_strength
                        }
                        
                        cash -= position_value
                        available_cash = cash * (1 - cash_reserve)
            
            # 记录组合价值
            portfolio_value = cash
            for symbol, position in positions.items():
                if symbol in daily_prices:
                    position_value = position['size'] * daily_prices[symbol]
                    portfolio_value += position_value
            
            portfolio_values.append(portfolio_value)
        
        # 计算绩效指标
        return self._calculate_performance_metrics(
            initial_capital, portfolio_values, trades, len(trading_dates)
        )
    
    def _calculate_performance_metrics(self, initial_capital: float, 
                                     portfolio_values: List[float], 
                                     trades: List[Dict],
                                     trading_days: int) -> Dict:
        """计算绩效指标"""
        if not portfolio_values:
            return {}
        
        # 基础指标
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # 日收益率
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        # 年化收益率
        if trading_days > 0:
            annualized_return = (final_value / initial_capital) ** (252 / trading_days) - 1
        else:
            annualized_return = 0
        
        # 夏普比率
        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            daily_volatility = np.std(daily_returns)
            sharpe_ratio = avg_daily_return / daily_volatility * np.sqrt(252) if daily_volatility > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 交易统计
        winning_trades = [t for t in trades if t['return'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_return_per_trade = np.mean([t['return'] for t in trades]) if trades else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': len(trades),
            'avg_return_per_trade': avg_return_per_trade,
            'final_portfolio_value': final_value,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns
        }

def main():
    """测试高收益策略"""
    try:
        # 创建策略实例
        strategy = HighReturnStrategy()
        
        # 运行回测
        results = strategy.run_backtest()
        
        # 显示结果
        print("\n" + "="*60)
        print("🎉 高收益策略系统集成测试完成!")
        print("="*60)
        
        annual_return = results.get('annualized_return', 0)
        if annual_return > 0.30:
            print("🌟🌟🌟🌟🌟 卓越表现!")
        elif annual_return > 0.20:
            print("🌟🌟🌟🌟 优秀表现!")
        elif annual_return > 0.10:
            print("🌟🌟🌟 良好表现!")
        else:
            print("📈 仍有改进空间")
        
    except Exception as e:
        logger.error(f"❌ 策略测试失败: {e}")
        raise

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
