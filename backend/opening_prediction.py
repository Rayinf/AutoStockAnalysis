"""
开盘预测模块 - 基于AKShare数据的当日交易策略预测
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import os
import warnings

# 禁用tqdm进度条，避免AttributeError
os.environ['AKSHARE_DISABLE_TQDM'] = '1'
# 忽略pandas和其他库的警告
warnings.filterwarnings('ignore')
import json

# 导入全局股票名称缓存（延迟导入避免循环依赖）
def get_stock_name_cache():
    """获取股票名称缓存实例"""
    try:
        from backend.portfolio_manager import stock_name_cache
        return stock_name_cache
    except ImportError:
        return None

logger = logging.getLogger(__name__)

@dataclass
class StockPrediction:
    """单只股票预测结果"""
    symbol: str
    name: str
    current_price: float
    predicted_direction: str  # 'up', 'down', 'hold'
    confidence: float  # 0-1
    target_price: float
    stop_loss: float
    recommendation: str  # 'buy', 'sell', 'hold'
    reason: str
    risk_level: str  # 'low', 'medium', 'high'
    volume_analysis: Dict
    technical_indicators: Dict

@dataclass
class MarketOverview:
    """市场概览"""
    market_sentiment: str  # 'bullish', 'bearish', 'neutral'
    major_indices: Dict
    market_news: List[str]
    risk_factors: List[str]

@dataclass
class PortfolioRecommendation:
    """投资组合推荐"""
    action: str  # 'buy', 'sell', 'hold', 'mixed'
    buy_list: List[Dict]
    sell_list: List[Dict]
    hold_list: List[Dict]
    reason: str

@dataclass
class OpeningPredictionResult:
    """开盘预测完整结果"""
    prediction_time: str
    market_overview: MarketOverview
    stock_predictions: List[StockPrediction]
    portfolio_recommendation: PortfolioRecommendation
    risk_assessment: Dict

class OpeningPredictor:
    """开盘预测器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 预定义股票名称映射
        self.stock_names = {
            '000001': '平安银行',
            '000002': '万科A', 
            '000501': '武商集团',
            '000519': '中兵红箭',
            '002182': '宝武镁业',
            '600176': '中国巨石',
            '600585': '海螺水泥',
            '002436': '兴森科技',
            '600710': '苏美达',
            '600519': '贵州茅台',
            '600000': '浦发银行',
            '600036': '招商银行',
            '601398': '工商银行'
        }
    
    def get_stock_name(self, symbol: str, from_api_data: str = None) -> str:
        """获取股票名称，优先使用API数据，然后使用预定义映射"""
        # 如果API提供了名称且不是N/A，优先使用
        if from_api_data and from_api_data != 'N/A' and from_api_data != symbol:
            return from_api_data
        
        # 否则使用预定义映射
        return self.stock_names.get(symbol, symbol)
        
    def get_current_market_data(self) -> Dict:
        """获取当前市场数据"""
        try:
            # 获取上证指数
            sh_index = ak.stock_zh_index_spot_em(symbol="000001")
            sz_index = ak.stock_zh_index_spot_em(symbol="399001")
            
            # 获取市场资金流向
            money_flow = ak.stock_market_fund_flow()
            
            return {
                'sh_index': sh_index,
                'sz_index': sz_index,
                'money_flow': money_flow,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return {}
    
    def get_stock_realtime_data(self, symbol: str, force_refresh_history: bool = False) -> Dict:
        """获取股票实时数据，使用正确的AKShare API接口"""
        try:
            self.logger.info(f"正在获取股票 {symbol} 的实时数据...")
            
            stock_info = None
            
            # 优先从缓存获取股票名称（避免慢速API调用）
            cache = get_stock_name_cache()
            stock_name_from_cache = cache.get(symbol) if cache else None
            
            # 方法1: 只有在缓存未命中时才使用stock_individual_info_em获取股票信息
            stock_name_from_api = stock_name_from_cache
            if not stock_name_from_cache:
                try:
                    self.logger.info(f"缓存未命中，使用API获取股票 {symbol} 信息")
                    individual_info = ak.stock_individual_info_em(symbol=symbol)
                    if not individual_info.empty:
                        # 解析个股信息数据
                        info_dict = {}
                        for _, row in individual_info.iterrows():
                            info_dict[row['item']] = row['value']
                        
                        # 提取股票名称并缓存
                        stock_name_from_api = info_dict.get('股票简称', symbol)
                        if cache and stock_name_from_api != symbol:
                            cache.set(symbol, stock_name_from_api)
                            self.logger.info(f"缓存股票名称: {symbol} -> {stock_name_from_api}")
                        
                        # 检查是否获取到有效的价格信息
                        today_open = info_dict.get('今开', 'N/A')
                        yesterday_close = info_dict.get('昨收', 'N/A')
                        latest_price = info_dict.get('最新价', 'N/A')
                        
                        # 尝试获取有效的当前价格
                        current_price = 0
                        if latest_price != 'N/A' and str(latest_price).replace('.', '').isdigit():
                            current_price = float(latest_price)
                        elif today_open != 'N/A' and str(today_open).replace('.', '').isdigit():
                            current_price = float(today_open)
                        elif yesterday_close != 'N/A' and str(yesterday_close).replace('.', '').isdigit():
                            current_price = float(yesterday_close)
                        
                        # 如果获取到有效价格，直接使用
                        if current_price > 0:
                            stock_name = self.get_stock_name(symbol, stock_name_from_api)
                            stock_info = {
                                '代码': symbol,
                                '名称': stock_name,
                                '最新价': current_price,
                                '涨跌幅': float(info_dict.get('涨跌幅', 0)) if info_dict.get('涨跌幅', 'N/A') != 'N/A' else 0,
                                '成交量': float(info_dict.get('成交量', 1000000)) if info_dict.get('成交量', 'N/A') != 'N/A' else 1000000,
                                '总市值': info_dict.get('总市值', 'N/A'),
                                '流通市值': info_dict.get('流通市值', 'N/A')
                            }
                            self.logger.info(f"API获取股票 {symbol} 实时数据: 价格={stock_info['最新价']}")
                        else:
                            self.logger.debug(f"API价格数据无效，但获取到名称: {stock_name_from_api}")
                            
                except Exception as e:
                    self.logger.debug(f"API方法失败: {e}")
                    if not stock_name_from_api:
                        stock_name_from_api = symbol
            else:
                self.logger.debug(f"使用缓存的股票名称: {symbol} -> {stock_name_from_cache}")
            
            # 方法2: 如果方法1失败，直接使用历史数据获取最新价格（更可靠）
            if stock_info is None:
                try:
                    self.logger.info("实时数据获取失败，使用最新历史数据作为当前价格")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5)  # 获取最近5天数据
                    
                    recent_data = ak.stock_zh_a_hist(
                        symbol=symbol, 
                        period="daily", 
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d'),
                        adjust="qfq"
                    )
                    
                    if not recent_data.empty:
                        latest_data = recent_data.iloc[-1]  # 最新一天的数据
                        # 使用从API获取的股票名称，如果没有则使用预定义映射
                        stock_name = self.get_stock_name(symbol, stock_name_from_api)
                        stock_info = {
                            '代码': symbol,
                            '名称': stock_name,
                            '最新价': float(latest_data['收盘']),
                            '涨跌幅': float(latest_data.get('涨跌幅', 0)) if '涨跌幅' in latest_data else 0.0,
                            '成交量': float(latest_data.get('成交量', 1000000)) if '成交量' in latest_data else 1000000
                        }
                        self.logger.info(f"使用最新历史数据: {symbol} - ¥{stock_info['最新价']:.2f}")
                    else:
                        raise Exception("历史数据也为空")
                        
                except Exception as e:
                    self.logger.error(f"获取历史数据也失败: {e}")
                    # 最后使用基于hash的确定性模拟数据
                    base_price = 10.0 + (hash(symbol) % 100) / 10
                    # 使用从API获取的股票名称，如果没有则使用预定义映射
                    stock_name = self.get_stock_name(symbol, stock_name_from_api)
                    stock_info = {
                        '代码': symbol,
                        '名称': stock_name,
                        '最新价': base_price,
                        '涨跌幅': (hash(symbol) % 200 - 100) / 1000,  # 更小的涨跌幅
                        '成交量': 1000000 + (hash(symbol) % 9000000)
                    }
                    self.logger.warning(f"使用确定性模拟数据: 价格={stock_info['最新价']}")
                
            # 获取历史数据用于技术分析
            try:
                self.logger.info(f"正在获取股票 {symbol} 的历史数据...")
                if force_refresh_history:
                    self.logger.info("🔄 强制刷新模式：重新获取历史数据")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)  # 增加到60天以获得更多数据
                
                hist_data = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period="daily", 
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    adjust="qfq"  # 前复权
                )
                
                if hist_data.empty:
                    self.logger.warning(f"股票 {symbol} 历史数据为空，生成模拟数据")
                    # 生成30天的模拟历史数据
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    base_price = stock_info['最新价']
                    
                    hist_data = pd.DataFrame({
                        '日期': dates[:30],
                        '开盘': [base_price * (1 + (i % 10 - 5) / 100) for i in range(30)],
                        '收盘': [base_price * (1 + (i % 8 - 4) / 100) for i in range(30)],
                        '最高': [base_price * (1 + (i % 12 - 3) / 100) for i in range(30)],
                        '最低': [base_price * (1 + (i % 6 - 6) / 100) for i in range(30)],
                        '成交量': [stock_info['成交量'] * (0.5 + (i % 10) / 20) for i in range(30)],
                        '成交额': [0] * 30,
                        '振幅': [0] * 30,
                        '涨跌幅': [(i % 20 - 10) / 100 for i in range(30)],
                        '涨跌额': [0] * 30,
                        '换手率': [0] * 30
                    })
                else:
                    self.logger.info(f"成功获取股票 {symbol} 历史数据，共 {len(hist_data)} 条记录")
                    
            except Exception as e:
                self.logger.error(f"获取历史数据失败: {e}")
                # 生成模拟历史数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                base_price = stock_info['最新价']
                
                hist_data = pd.DataFrame({
                    '日期': dates[:30],
                    '开盘': [base_price * (1 + (i % 10 - 5) / 100) for i in range(30)],
                    '收盘': [base_price * (1 + (i % 8 - 4) / 100) for i in range(30)],
                    '最高': [base_price * (1 + (i % 12 - 3) / 100) for i in range(30)],
                    '最低': [base_price * (1 + (i % 6 - 6) / 100) for i in range(30)],
                    '成交量': [stock_info['成交量'] * (0.5 + (i % 10) / 20) for i in range(30)],
                    '成交额': [0] * 30,
                    '振幅': [0] * 30,
                    '涨跌幅': [(i % 20 - 10) / 100 for i in range(30)],
                    '涨跌额': [0] * 30,
                    '换手率': [0] * 30
                })
                self.logger.info(f"使用模拟历史数据，共 {len(hist_data)} 条记录")
            
            return {
                'realtime': stock_info,
                'history': hist_data,
                'symbol': symbol
            }
            
        except Exception as e:
            self.logger.error(f"获取股票 {symbol} 数据完全失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {}
    
    def calculate_technical_indicators(self, hist_data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if hist_data.empty or len(hist_data) < 20:
            return {}
            
        try:
            # 移动平均线（增加min_periods避免前段NaN）
            hist_data['MA5'] = hist_data['收盘'].rolling(window=5, min_periods=5).mean()
            hist_data['MA10'] = hist_data['收盘'].rolling(window=10, min_periods=10).mean()
            hist_data['MA20'] = hist_data['收盘'].rolling(window=20, min_periods=20).mean()
            
            # RSI
            delta = hist_data['收盘'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = hist_data['收盘'].ewm(span=12).mean()
            exp2 = hist_data['收盘'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            # 布林带
            bb_period = 20
            bb_std = 2
            bb_ma = hist_data['收盘'].rolling(window=bb_period, min_periods=bb_period).mean()
            bb_std_val = hist_data['收盘'].rolling(window=bb_period, min_periods=bb_period).std()
            bb_upper = bb_ma + (bb_std_val * bb_std)
            bb_lower = bb_ma - (bb_std_val * bb_std)
            
            latest_data = hist_data.iloc[-1]
            
            return {
                'MA5': float(latest_data['MA5']) if pd.notna(latest_data['MA5']) else 0,
                'MA10': float(latest_data['MA10']) if pd.notna(latest_data['MA10']) else 0,
                'MA20': float(latest_data['MA20']) if pd.notna(latest_data['MA20']) else 0,
                'RSI': float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50,
                'MACD': float(macd.iloc[-1]) if pd.notna(macd.iloc[-1]) else 0,
                'MACD_Signal': float(signal.iloc[-1]) if pd.notna(signal.iloc[-1]) else 0,
                'BB_Upper': float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else 0,
                'BB_Lower': float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else 0,
                'BB_Middle': float(bb_ma.iloc[-1]) if pd.notna(bb_ma.iloc[-1]) else 0,
                'current_price': float(latest_data['收盘']),
                'volume': float(latest_data['成交量']),
                'volume_ma': float(hist_data['成交量'].rolling(window=5, min_periods=5).mean().iloc[-1]) if len(hist_data) >= 5 else 0
            }
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            return {}
    
    def analyze_volume_pattern(self, hist_data: pd.DataFrame) -> Dict:
        """分析成交量模式"""
        if hist_data.empty or len(hist_data) < 5:
            return {'pattern': 'insufficient_data', 'strength': 0}
            
        try:
            recent_volume = hist_data['成交量'].tail(3).mean()
            avg_volume = hist_data['成交量'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # 价量关系分析
            price_change = (hist_data['收盘'].iloc[-1] - hist_data['收盘'].iloc[-2]) / hist_data['收盘'].iloc[-2]
            volume_change = (hist_data['成交量'].iloc[-1] - hist_data['成交量'].iloc[-2]) / hist_data['成交量'].iloc[-2]
            
            pattern = 'normal'
            if volume_ratio > 1.5 and price_change > 0:
                pattern = 'volume_breakout_up'
            elif volume_ratio > 1.5 and price_change < 0:
                pattern = 'volume_breakout_down'
            elif volume_ratio < 0.7:
                pattern = 'low_volume'
            
            return {
                'pattern': pattern,
                'volume_ratio': volume_ratio,
                'recent_avg_volume': recent_volume,
                'historical_avg_volume': avg_volume,
                'price_volume_correlation': price_change * volume_change
            }
        except Exception as e:
            self.logger.error(f"分析成交量失败: {e}")
            return {'pattern': 'error', 'strength': 0}
    
    def calculate_dynamic_prices(self, current_price: float, direction: str, 
                               indicators: Dict, volume_analysis: Dict, 
                               hist_data: pd.DataFrame) -> tuple:
        """基于技术指标动态计算目标价和止损价"""
        try:
            # 获取关键技术指标
            rsi = indicators.get('RSI', 50)
            bb_upper = indicators.get('BB_Upper', current_price * 1.05)
            bb_lower = indicators.get('BB_Lower', current_price * 0.95)
            bb_middle = indicators.get('BB_Middle', current_price)
            ma5 = indicators.get('MA5', current_price)
            ma20 = indicators.get('MA20', current_price)
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_Signal', 0)
            
            # 计算ATR（平均真实波动幅度）
            atr = self.calculate_atr(hist_data)
            
            # 计算支撑阻力位
            support_resistance = self.calculate_support_resistance(hist_data, current_price)
            
            if direction == 'up':
                # 看涨目标价计算
                target_candidates = []
                
                # 1. 基于布林带上轨（只有当上轨高于当前价时才考虑）
                if bb_upper > current_price:
                    target_candidates.append(bb_upper)
                
                # 2. 基于ATR的目标价（2-3倍ATR）
                atr_target = current_price + (atr * 2.5)
                target_candidates.append(atr_target)
                
                # 3. 基于阻力位（只考虑高于当前价的阻力位）
                if support_resistance['resistance'] and support_resistance['resistance'] > current_price:
                    target_candidates.append(support_resistance['resistance'])
                
                # 4. 确保至少有一个合理的上涨目标
                if not target_candidates:
                    target_candidates.append(current_price * 1.1)  # 最少10%上涨
                
                # 4. 基于RSI调整（更温和的调整，避免目标价低于现价）
                rsi_multiplier = 1.0
                if rsi > 80:  # 极度超买才调整
                    rsi_multiplier = 0.95  # 轻微保守，避免过度调整
                elif rsi < 20:  # 极度超卖才调整
                    rsi_multiplier = 1.05  # 轻微激进
                
                # 选择最合理的目标价（看涨时选择适中的上涨目标）
                if target_candidates:
                    # 过滤掉低于当前价格的不合理目标
                    valid_targets = [t for t in target_candidates if t > current_price]
                    if valid_targets:
                        # 选择适中的目标价（不要过于激进）
                        raw_target = min(valid_targets)
                        target_price = raw_target * rsi_multiplier
                        
                        # 关键修复：确保目标价始终高于当前价格
                        if target_price <= current_price:
                            target_price = current_price * 1.02  # 至少2%上涨
                    else:
                        # 如果没有有效目标，使用ATR目标
                        target_price = (current_price + (atr * 2.5)) * rsi_multiplier
                        if target_price <= current_price:
                            target_price = current_price * 1.02
                else:
                    target_price = (current_price + (atr * 2.5)) * rsi_multiplier
                    if target_price <= current_price:
                        target_price = current_price * 1.02
                
                # 止损价：基于支撑位或ATR
                stop_candidates = []
                if support_resistance['support'] and support_resistance['support'] < current_price:
                    stop_candidates.append(support_resistance['support'])
                
                # ATR止损（1.5倍ATR）
                atr_stop = current_price - (atr * 1.5)
                stop_candidates.append(atr_stop)
                
                # 基于MA20的止损
                if ma20 < current_price:
                    stop_candidates.append(ma20 * 0.98)  # MA20下方2%
                
                stop_loss = max(stop_candidates) if stop_candidates else current_price * 0.95
                
            elif direction == 'down':
                # 不允许做空：当信号为看跌时，输出为“观望”，返回略保守的保护参数
                target_price = bb_middle if bb_middle else current_price
                stop_loss = current_price * 0.98
            else:
                # 横盘：目标价接近当前价，保护性止损
                target_price = bb_middle if bb_middle else current_price
                stop_loss = current_price - (atr * 1.0)
            
            # 确保价格合理性
            target_price = max(target_price, 0.01)  # 最小价格
            stop_loss = max(stop_loss, 0.01)
            
            # 确保止损价始终低于当前价
            if stop_loss >= current_price:
                stop_loss = current_price * 0.98
            
            return target_price, stop_loss
            
        except Exception as e:
            self.logger.error(f"动态价格计算失败: {e}")
            # 回退到简单计算
            volatility = 0.1
            if direction == 'up':
                return current_price * 1.1, current_price * 0.95
            elif direction == 'down':
                return current_price * 0.9, current_price * 0.95
            else:
                return current_price, current_price * 0.95
    
    def calculate_atr(self, hist_data: pd.DataFrame, period: int = 14) -> float:
        """计算平均真实波动幅度(ATR)"""
        try:
            if len(hist_data) < period + 1:
                return hist_data['收盘'].std() if len(hist_data) > 1 else 0.1
            
            high = hist_data['最高']
            low = hist_data['最低']
            close = hist_data['收盘']
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if pd.notna(atr) else 0.1
        except Exception as e:
            self.logger.error(f"ATR计算失败: {e}")
            return 0.1
    
    def calculate_support_resistance(self, hist_data: pd.DataFrame, current_price: float) -> Dict:
        """计算支撑阻力位"""
        try:
            if len(hist_data) < 20:
                return {'support': None, 'resistance': None}
            
            # 获取最近20天的高低点
            recent_data = hist_data.tail(20)
            highs = recent_data['最高']
            lows = recent_data['最低']
            
            # 寻找局部高点和低点
            resistance_levels = []
            support_levels = []
            
            for i in range(2, len(recent_data) - 2):
                high_val = highs.iloc[i]
                low_val = lows.iloc[i]
                
                # 局部高点（阻力位）
                if (high_val > highs.iloc[i-1] and high_val > highs.iloc[i-2] and
                    high_val > highs.iloc[i+1] and high_val > highs.iloc[i+2]):
                    resistance_levels.append(high_val)
                
                # 局部低点（支撑位）
                if (low_val < lows.iloc[i-1] and low_val < lows.iloc[i-2] and
                    low_val < lows.iloc[i+1] and low_val < lows.iloc[i+2]):
                    support_levels.append(low_val)
            
            # 找到最近的支撑阻力位
            resistance = None
            support = None
            
            if resistance_levels:
                # 找到最接近当前价格且在其上方的阻力位
                upper_resistance = [r for r in resistance_levels if r > current_price]
                if upper_resistance:
                    resistance = min(upper_resistance)
            
            if support_levels:
                # 找到最接近当前价格且在其下方的支撑位
                lower_support = [s for s in support_levels if s < current_price]
                if lower_support:
                    support = max(lower_support)
            
            return {'support': support, 'resistance': resistance}
            
        except Exception as e:
            self.logger.error(f"支撑阻力位计算失败: {e}")
            return {'support': None, 'resistance': None}
    
    def predict_stock_direction(self, symbol: str, stock_data: Dict) -> StockPrediction:
        """预测单只股票方向"""
        try:
            if not stock_data or 'realtime' not in stock_data:
                return self._create_default_prediction(symbol, "数据获取失败")
            
            realtime = stock_data['realtime']
            hist_data = stock_data['history']
            
            # 计算技术指标
            indicators = self.calculate_technical_indicators(hist_data)
            volume_analysis = self.analyze_volume_pattern(hist_data)
            
            if not indicators:
                return self._create_default_prediction(symbol, "技术指标计算失败")
            
            current_price = float(realtime.get('最新价', 0))
            stock_name = realtime.get('名称', symbol)
            
            # 综合分析逻辑
            signals = []
            confidence_factors = []
            
            # 1. 移动平均线信号
            if current_price > indicators['MA5'] > indicators['MA10']:
                signals.append('bullish_ma')
                confidence_factors.append(0.2)
            elif current_price < indicators['MA5'] < indicators['MA10']:
                signals.append('bearish_ma')
                confidence_factors.append(0.2)
            
            # 2. RSI信号
            rsi = indicators['RSI']
            if rsi < 30:
                signals.append('oversold')
                confidence_factors.append(0.15)
            elif rsi > 70:
                signals.append('overbought')
                confidence_factors.append(0.15)
            
            # 3. MACD信号
            if indicators['MACD'] > indicators['MACD_Signal']:
                signals.append('macd_bullish')
                confidence_factors.append(0.15)
            else:
                signals.append('macd_bearish')
                confidence_factors.append(0.15)
            
            # 4. 布林带信号
            if current_price < indicators['BB_Lower']:
                signals.append('bb_oversold')
                confidence_factors.append(0.1)
            elif current_price > indicators['BB_Upper']:
                signals.append('bb_overbought')
                confidence_factors.append(0.1)
            
            # 5. 成交量信号
            if volume_analysis['pattern'] == 'volume_breakout_up':
                signals.append('volume_bullish')
                confidence_factors.append(0.2)
            elif volume_analysis['pattern'] == 'volume_breakout_down':
                signals.append('volume_bearish')
                confidence_factors.append(0.2)
            
            # 综合判断
            bullish_signals = ['bullish_ma', 'oversold', 'macd_bullish', 'bb_oversold', 'volume_bullish']
            bearish_signals = ['bearish_ma', 'overbought', 'macd_bearish', 'bb_overbought', 'volume_bearish']
            
            bullish_count = sum(1 for s in signals if s in bullish_signals)
            bearish_count = sum(1 for s in signals if s in bearish_signals)
            
            # 仅做多：预先计算多头与观望两套价格
            target_up, stop_up = self.calculate_dynamic_prices(
                current_price, 'up', indicators, volume_analysis, hist_data
            )
            target_hold, stop_hold = self.calculate_dynamic_prices(
                current_price, 'hold', indicators, volume_analysis, hist_data
            )

            # 信号差异
            delta_signals = bullish_count - bearish_count
            total_signals = max(1, len(signals))

            # 基于多头价格计算期望与风险（用于弱买/强买判断）
            expected_return_up = (target_up - current_price) / current_price if current_price > 0 else 0.0
            potential_loss_up = abs((current_price - stop_up) / current_price) if (current_price > 0 and stop_up > 0) else 0.0
            risk_adjusted_return_up = expected_return_up - potential_loss_up

            if delta_signals > 0:
                # 强买：多头信号占优
                direction = 'up'
                target_price, stop_loss = target_up, stop_up
                confidence = max(0.55, min(0.75, 0.55 + 0.2 * (delta_signals / total_signals)))
                recommendation = 'buy' if expected_return_up > 0 else 'hold'
            elif delta_signals == 0 and expected_return_up > 0 and risk_adjusted_return_up > 0:
                # 弱买：信号持平但期望与风险调整均为正
                direction = 'up'
                target_price, stop_loss = target_up, stop_up
                confidence = 0.56
                recommendation = 'buy'
            else:
                # 观望
                direction = 'hold'
                target_price, stop_loss = target_hold, stop_hold
                confidence = 0.5
                recommendation = 'hold'
            
            # 基于ATR和RSI的风险评估
            atr = self.calculate_atr(hist_data)
            rsi = indicators.get('RSI', 50)
            
            risk_level = 'low'
            if atr > current_price * 0.05 or rsi > 80 or rsi < 20:  # ATR超过5%或RSI极端
                risk_level = 'high'
            elif atr > current_price * 0.03 or rsi > 70 or rsi < 30:  # ATR超过3%或RSI偏极端
                risk_level = 'medium'
            
            # 生成分析理由
            reason_parts = []
            if 'bullish_ma' in signals:
                reason_parts.append("均线多头排列")
            if 'oversold' in signals:
                reason_parts.append(f"RSI超卖({rsi:.1f})")
            if 'volume_bullish' in signals:
                reason_parts.append("放量上涨")
            if 'bearish_ma' in signals:
                reason_parts.append("均线空头排列")
            if 'overbought' in signals:
                reason_parts.append(f"RSI超买({rsi:.1f})")
            if 'volume_bearish' in signals:
                reason_parts.append("放量下跌")
            
            reason = "；".join(reason_parts) if reason_parts else "技术指标中性"
            
            return StockPrediction(
                symbol=symbol,
                name=stock_name,
                current_price=current_price,
                predicted_direction=direction,
                confidence=confidence,
                target_price=target_price,
                stop_loss=stop_loss,
                recommendation=recommendation,
                reason=reason,
                risk_level=risk_level,
                volume_analysis=volume_analysis,
                technical_indicators=indicators
            )
            
        except Exception as e:
            self.logger.error(f"预测股票 {symbol} 失败: {e}")
            return self._create_default_prediction(symbol, f"预测失败: {str(e)}")
    
    def _create_default_prediction(self, symbol: str, reason: str) -> StockPrediction:
        """创建默认预测结果"""
        return StockPrediction(
            symbol=symbol,
            name=symbol,
            current_price=0.0,
            predicted_direction='hold',
            confidence=0.0,
            target_price=0.0,
            stop_loss=0.0,
            recommendation='hold',
            reason=reason,
            risk_level='high',
            volume_analysis={},
            technical_indicators={}
        )
    
    def get_market_overview(self) -> MarketOverview:
        """获取市场概览"""
        try:
            # 获取主要指数
            indices_data = {}
            
            # 上证指数
            try:
                sh_data = ak.stock_zh_index_spot_em(symbol="000001")
                if not sh_data.empty:
                    sh_info = sh_data.iloc[0]
                    indices_data['上证指数'] = {
                        'current': float(sh_info['最新价']),
                        'change': float(sh_info['涨跌幅']),
                        'volume': float(sh_info['成交量'])
                    }
            except:
                pass
            
            # 深证成指
            try:
                sz_data = ak.stock_zh_index_spot_em(symbol="399001")
                if not sz_data.empty:
                    sz_info = sz_data.iloc[0]
                    indices_data['深证成指'] = {
                        'current': float(sz_info['最新价']),
                        'change': float(sz_info['涨跌幅']),
                        'volume': float(sz_info['成交量'])
                    }
            except:
                pass
            
            # 创业板指
            try:
                cyb_data = ak.stock_zh_index_spot_em(symbol="399006")
                if not cyb_data.empty:
                    cyb_info = cyb_data.iloc[0]
                    indices_data['创业板指'] = {
                        'current': float(cyb_info['最新价']),
                        'change': float(cyb_info['涨跌幅']),
                        'volume': float(cyb_info['成交量'])
                    }
            except:
                pass
            
            # 判断市场情绪
            avg_change = 0
            if indices_data:
                changes = [data['change'] for data in indices_data.values()]
                avg_change = sum(changes) / len(changes)
            
            if avg_change > 1:
                sentiment = 'bullish'
            elif avg_change < -1:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            # 风险因素（简化版）
            risk_factors = []
            if avg_change < -2:
                risk_factors.append("市场大幅下跌")
            if any(data.get('volume', 0) > 1000000000 for data in indices_data.values()):
                risk_factors.append("成交量异常放大")
            
            return MarketOverview(
                market_sentiment=sentiment,
                major_indices=indices_data,
                market_news=[],  # 可以后续添加新闻API
                risk_factors=risk_factors
            )
            
        except Exception as e:
            self.logger.error(f"获取市场概览失败: {e}")
            return MarketOverview(
                market_sentiment='neutral',
                major_indices={},
                market_news=[],
                risk_factors=['数据获取失败']
            )
    
    def generate_portfolio_recommendation(self, predictions: List[StockPrediction]) -> Dict:
        """生成投资组合建议"""
        try:
            buy_stocks = [p for p in predictions if p.recommendation == 'buy' and p.confidence > 0.6]
            sell_stocks = [p for p in predictions if p.recommendation == 'sell' and p.confidence > 0.6]
            hold_stocks = [p for p in predictions if p.recommendation == 'hold']
            
            # 按置信度排序
            buy_stocks.sort(key=lambda x: x.confidence, reverse=True)
            sell_stocks.sort(key=lambda x: x.confidence, reverse=True)
            
            # 计算建议仓位
            total_confidence = sum(p.confidence for p in buy_stocks)
            position_suggestions = []
            
            for stock in buy_stocks[:5]:  # 最多推荐5只
                if total_confidence > 0:
                    suggested_weight = (stock.confidence / total_confidence) * 0.8  # 最大80%仓位
                    position_suggestions.append({
                        'symbol': stock.symbol,
                        'name': stock.name,
                        'action': 'buy',
                        'weight': min(suggested_weight, 0.2),  # 单只股票最大20%
                        'confidence': stock.confidence,
                        'reason': stock.reason
                    })
            
            return {
                'buy_recommendations': position_suggestions,
                'sell_recommendations': [
                    {
                        'symbol': s.symbol,
                        'name': s.name,
                        'action': 'sell',
                        'confidence': s.confidence,
                        'reason': s.reason
                    } for s in sell_stocks[:3]
                ],
                'hold_recommendations': len(hold_stocks),
                'overall_strategy': self._determine_overall_strategy(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"生成投资组合建议失败: {e}")
            return {'error': str(e)}
    
    def _determine_overall_strategy(self, predictions: List[StockPrediction]) -> str:
        """确定整体策略"""
        if not predictions:
            return 'neutral'
            
        buy_count = sum(1 for p in predictions if p.recommendation == 'buy')
        sell_count = sum(1 for p in predictions if p.recommendation == 'sell')
        
        if buy_count > sell_count * 2:
            return 'aggressive_buy'
        elif sell_count > buy_count * 2:
            return 'defensive_sell'
        elif buy_count > sell_count:
            return 'moderate_buy'
        elif sell_count > buy_count:
            return 'moderate_sell'
        else:
            return 'neutral'
    
    def predict_opening_strategy(self, stock_pool: List[str], current_positions: List = None, force_refresh_history: bool = False) -> OpeningPredictionResult:
        """生成开盘预测策略（集成持仓信息）"""
        self.logger.info(f"开始预测 {len(stock_pool)} 只股票的开盘策略")
        
        # 获取市场概览
        market_overview = self.get_market_overview()
        
        # 预测各只股票
        stock_predictions = []
        for symbol in stock_pool:
            self.logger.info(f"正在分析股票: {symbol}")
            stock_data = self.get_stock_realtime_data(symbol, force_refresh_history=force_refresh_history)
            prediction = self.predict_stock_direction(symbol, stock_data)
            
            # 持仓信息将在投资组合建议中使用，这里不修改prediction对象
            
            stock_predictions.append(prediction)
        
        # 生成投资组合建议（考虑持仓）
        portfolio_recommendation = self.generate_portfolio_recommendation_with_holdings(
            stock_predictions, current_positions
        )
        
        # 风险评估
        high_risk_count = sum(1 for p in stock_predictions if p.risk_level == 'high')
        total_confidence = sum(p.confidence for p in stock_predictions) / len(stock_predictions) if stock_predictions else 0
        
        risk_assessment = {
            'overall_risk': 'high' if high_risk_count > len(stock_predictions) * 0.5 else 'medium' if high_risk_count > 0 else 'low',
            'confidence_level': total_confidence,
            'high_risk_stocks': high_risk_count,
            'market_risk_factors': market_overview.risk_factors
        }
        
        return OpeningPredictionResult(
            prediction_time=datetime.now().isoformat(),
            market_overview=market_overview,
            stock_predictions=stock_predictions,
            portfolio_recommendation=portfolio_recommendation,
            risk_assessment=risk_assessment
        )
    
    def _get_holding_info(self, symbol: str, current_positions: List) -> Dict:
        """获取股票的持仓信息"""
        try:
            # 优先从传入的current_positions匹配
            if current_positions:
                for position in current_positions:
                    if position.get('symbol') == symbol:
                        return {
                            'quantity': position.get('quantity', 0),
                            'avg_cost': position.get('avg_cost', 0),
                            'current_price': position.get('current_price', 0),
                            'market_value': position.get('market_value', 0),
                            'profit_loss': position.get('profit_loss', 0),
                            'profit_loss_pct': position.get('profit_loss_pct', 0),
                            'is_holding': True
                        }
            # 回退到组合管理器，确保联动
            try:
                from backend.portfolio_manager import portfolio_manager
                pos = portfolio_manager.get_position_by_symbol(symbol)
                if pos:
                    return {
                        'quantity': pos.quantity,
                        'avg_cost': pos.avg_cost,
                        'current_price': pos.current_price,
                        'market_value': pos.market_value,
                        'profit_loss': pos.profit_loss,
                        'profit_loss_pct': pos.profit_loss_pct,
                        'is_holding': True
                    }
            except Exception:
                pass
            return {'quantity': 0, 'avg_cost': 0, 'is_holding': False}
        except Exception:
            return {'quantity': 0, 'avg_cost': 0, 'is_holding': False}
    
    def _calculate_backtest_metrics(self, prediction: StockPrediction) -> Dict:
        """计算回测系统使用的指标"""
        # 将预测信心度转换为概率
        probability = prediction.confidence
        
        # 估算预期收益（基于目标价和当前价）
        if prediction.current_price > 0:
            expected_return = (prediction.target_price - prediction.current_price) / prediction.current_price
        else:
            expected_return = 0
        
        # 估算风险调整收益（考虑止损）
        if prediction.current_price > 0 and prediction.stop_loss > 0:
            potential_loss = abs((prediction.current_price - prediction.stop_loss) / prediction.current_price)
            risk_adjusted_return = expected_return - potential_loss
        else:
            potential_loss = 0.05  # 默认5%风险
            risk_adjusted_return = expected_return
        
        # 简化的Kelly公式计算
        win_prob = probability
        loss_prob = 1 - probability
        
        # 确保有合理的收益和损失值
        if abs(expected_return) < 0.001:
            expected_return = 0.01 if expected_return >= 0 else -0.01
        if abs(potential_loss) < 0.001:
            potential_loss = 0.02
            
        if loss_prob > 0 and abs(expected_return) > 0:
            kelly_numerator = win_prob * abs(expected_return) - loss_prob * abs(potential_loss)
            kelly_fraction = min(0.25, max(0, kelly_numerator / abs(expected_return)))
            
        else:
            kelly_fraction = 0.05  # 默认5%仓位
        
        return {
            'symbol': prediction.symbol,
            'probability': probability,
            'expected_return': expected_return,
            'risk_adjusted_return': risk_adjusted_return,
            'kelly_fraction': kelly_fraction
        }
    
    def _should_buy_backtest_style(self, trade_info: Dict, current_positions: List) -> bool:
        """使用回测系统的买入条件"""
        # 回测系统的买入阈值（调整为更宽松的条件）
        buy_threshold = 0.60  # 概率阈值
        min_kelly_fraction = 0.01  # 最小Kelly仓位
        max_positions = 10  # 最大持仓数量
        
        conditions = [
            trade_info['expected_return'] > -0.01,           # 预期收益 > -1%
            trade_info['probability'] > buy_threshold,       # 概率 > 60%
            trade_info['risk_adjusted_return'] > -0.1,       # 风险调整收益 > -10%
            len(current_positions) < max_positions,          # 仓位限制
            trade_info['kelly_fraction'] > min_kelly_fraction # 最小Kelly仓位要求
        ]
        
        return all(conditions)
    
    def _should_sell_backtest_style(self, prediction: StockPrediction, holding_info: Dict) -> bool:
        """使用回测系统的卖出条件"""
        if not holding_info.get('is_holding', False):
            return False
        
        # 卖出阈值（仅做多：保守风控 + 止盈/移动止损）
        sell_threshold = 0.4       # 模型置信度过低
        stop_loss_ratio = 0.03     # 固定止损3%
        take_profit_ratio = 0.10   # 触发止盈的最低盈利
        trail_activate = 0.06      # 激活移动止损的最低盈利
        
        probability = prediction.confidence
        
        # 计算未实现收益
        if holding_info.get('avg_cost', 0) > 0:
            unrealized_return = (prediction.current_price - holding_info['avg_cost']) / holding_info['avg_cost']
        else:
            unrealized_return = 0
        
        # 指标辅助（从预测技术指标中获取）
        ind = prediction.technical_indicators or {}
        ma5 = ind.get('MA5', prediction.current_price)
        ma20 = ind.get('MA20', prediction.current_price)
        rsi = ind.get('RSI', 50)
        macd = ind.get('MACD', 0)
        macd_signal = ind.get('MACD_Signal', 0)
        
        # 触发规则
        cond_prob_low = probability < sell_threshold
        cond_stop_loss = unrealized_return <= -stop_loss_ratio
        cond_take_profit = (unrealized_return >= take_profit_ratio) and ((macd < macd_signal) or (rsi >= 75) or (prediction.current_price < ma20))
        cond_trailing = (unrealized_return >= trail_activate) and (prediction.current_price < ma5)
        
        sell_conditions = [
            cond_prob_low,
            cond_stop_loss,
            cond_take_profit,
            cond_trailing,
        ]
        
        return any(sell_conditions)
    
    def _calculate_buy_strategies(self, prediction: StockPrediction, trade_info: Dict) -> Dict:
        """计算多种买入策略价格"""
        current_price = prediction.current_price
        confidence = prediction.confidence
        
        strategies = {
            # 策略1：市价买入（适合强烈看好的股票）
            'market_buy': {
                'price': current_price,
                'description': '市价买入',
                'suitable_for': '强烈看好，立即执行'
            },
            
            # 策略2：小幅回调买入（等待1-3%回调）
            'pullback_buy': {
                'price': current_price * (1 - 0.01 - 0.02 * confidence),
                'description': f'回调买入 (-{(0.01 + 0.02 * confidence)*100:.1f}%)',
                'suitable_for': '等待小幅回调，降低成本'
            },
            
            # 策略3：分批买入（当前价格买入部分，回调时加仓）
            'batch_buy': {
                'price': current_price,
                'price_2': current_price * (1 - 0.02 - 0.01 * confidence),
                'description': f'分批买入 (50%现价 + 50%回调{(0.02 + 0.01 * confidence)*100:.1f}%)',
                'suitable_for': '分散风险，逐步建仓'
            },
            
            # 策略4：技术位买入（基于支撑位）
            'support_buy': {
                'price': max(prediction.stop_loss * 1.02, current_price * 0.98),
                'description': '支撑位买入',
                'suitable_for': '等待技术支撑位，安全边际较高'
            }
        }
        
        return strategies
    
    def _calculate_sell_strategies(self, prediction: StockPrediction) -> Dict:
        """计算卖出策略价格"""
        current_price = prediction.current_price
        confidence = prediction.confidence
        
        strategies = {
            'market_sell': {
                'price': current_price,
                'description': '市价卖出'
            },
            'limit_sell': {
                'price': current_price * (1 + 0.01 + 0.02 * confidence),
                'description': f'限价卖出 (+{(0.01 + 0.02 * confidence)*100:.1f}%)'
            }
        }
        
        return strategies
    
    def _get_recommended_buy_strategy(self, buy_strategies: Dict, trade_info: Dict) -> Dict:
        """根据股票特征推荐最适合的买入策略"""
        confidence = trade_info['probability']
        expected_return = trade_info['expected_return']
        
        # 高信心度 + 高收益 -> 市价买入
        if confidence > 0.75 and expected_return > 0.02:
            return buy_strategies['market_buy']
        
        # 中等信心度 -> 分批买入
        elif confidence > 0.65:
            return buy_strategies['batch_buy']
        
        # 较低信心度但有收益 -> 等待回调
        elif expected_return > 0.01:
            return buy_strategies['pullback_buy']
        
        # 保守策略 -> 支撑位买入
        else:
            return buy_strategies['support_buy']
    
    def _add_backtest_analysis_to_predictions(self, predictions: List[StockPrediction], current_positions: List):
        """为所有股票预测添加回测分析信息"""
        for prediction in predictions:
            # 计算回测指标
            trade_info = self._calculate_backtest_metrics(prediction)
            holding_info = self._get_holding_info(prediction.symbol, current_positions)
            
            # 判断买入卖出条件
            should_buy = self._should_buy_backtest_style(trade_info, current_positions)
            should_sell = self._should_sell_backtest_style(prediction, holding_info)
            
            # 计算多种买入策略价格
            buy_strategies = self._calculate_buy_strategies(prediction, trade_info)
            sell_strategies = self._calculate_sell_strategies(prediction)
            
            # 计算建议数量（非买入/观望时强制Kelly为0）
            kelly_fraction_effective = trade_info['kelly_fraction'] if prediction.recommendation == 'buy' else 0.0
            kelly_quantity = int(kelly_fraction_effective * 100000 / prediction.current_price) if prediction.current_price > 0 else 100
            suggested_buy_quantity = min(2000, max(100, kelly_quantity))
            
            # 计算未实现收益（如果持有）
            unrealized_return = 0
            if holding_info.get('is_holding') and holding_info.get('avg_cost', 0) > 0:
                unrealized_return = (prediction.current_price - holding_info['avg_cost']) / holding_info['avg_cost']
            
            # 生成详细的分析原因
            analysis_parts = [
                f"技术分析: {prediction.reason}",
                f"回测指标: 概率{trade_info['probability']:.1%}, 预期收益{trade_info['expected_return']:.1%}, Kelly仓位{trade_info['kelly_fraction']:.2%}",
            ]
            
            # 添加买入卖出建议
            if should_buy:
                action = '加仓' if holding_info.get('is_holding') else '买入'
                # 推荐最适合的买入策略
                recommended_strategy = self._get_recommended_buy_strategy(buy_strategies, trade_info)
                analysis_parts.append(f"✅ 回测策略建议: {action}")
                analysis_parts.append(f"📈 推荐策略: {recommended_strategy['description']} ¥{recommended_strategy['price']:.2f}")
                analysis_parts.append(f"💰 建议数量: {suggested_buy_quantity}股")
                
                # 显示所有可选策略
                strategy_options = []
                for key, strategy in buy_strategies.items():
                    if key != 'batch_buy':  # 分批买入单独处理
                        strategy_options.append(f"{strategy['description']}¥{strategy['price']:.2f}")
                    else:
                        strategy_options.append(f"{strategy['description']} ¥{strategy['price']:.2f}/¥{strategy['price_2']:.2f}")
                analysis_parts.append(f"🎯 买入选项: {' | '.join(strategy_options)}")
                
            elif should_sell and holding_info.get('is_holding'):
                # 仅做多：允许卖出（止损/风控/止盈），不做空
                analysis_parts.append(f"⚠️ 回测策略建议: 卖出")
                analysis_parts.append(f"📉 卖出策略: {sell_strategies['limit_sell']['description']} ¥{sell_strategies['limit_sell']['price']:.2f}")
                analysis_parts.append(f"💰 建议数量: 全部{holding_info.get('quantity', 0)}股")
                # 原因细化：止损 / 触发止盈 / 移动止损
                reason_flags = []
                if unrealized_return <= -0.03:
                    reason_flags.append(f"止损 {unrealized_return:.1%}")
                if unrealized_return >= 0.10:
                    reason_flags.append(f"止盈 {unrealized_return:.1%}")
                # 简化移动止损文案：价格跌破MA5
                ind = prediction.technical_indicators or {}
                ma5 = ind.get('MA5', prediction.current_price)
                if prediction.current_price < ma5:
                    reason_flags.append("移动止损(MA5)触发")
                if reason_flags:
                    analysis_parts.append("⚠️ 卖出原因: " + " / ".join(reason_flags))
            else:
                # 即使不满足条件，也显示如果执行的预测结果
                if prediction.predicted_direction == 'up' and prediction.recommendation == 'buy':
                    analysis_parts.append(f"💡 如果买入: 目标价¥{prediction.target_price:.2f} (潜在收益{trade_info['expected_return']:.1%}), 止损价¥{prediction.stop_loss:.2f}")
                    
                    # 显示买入选项
                    strategy_options = []
                    for key, strategy in buy_strategies.items():
                        if key != 'batch_buy':
                            strategy_options.append(f"{strategy['description']}¥{strategy['price']:.2f}")
                        else:
                            strategy_options.append(f"{strategy['description']} ¥{strategy['price']:.2f}/¥{strategy['price_2']:.2f}")
                    analysis_parts.append(f"🎯 买入选项: {' | '.join(strategy_options)}")
                    
                    if not should_buy:
                        reasons = []
                        if trade_info['expected_return'] <= -0.01:
                            reasons.append("预期收益不足")
                        if trade_info['probability'] <= 0.60:
                            reasons.append(f"概率过低({trade_info['probability']:.1%})")
                        if trade_info['kelly_fraction'] <= 0.01:
                            reasons.append("Kelly仓位过小")
                        analysis_parts.append(f"❌ 未买入原因: {', '.join(reasons)}")
                elif prediction.predicted_direction == 'down':
                    # 仅做多：看跌仅提示观望
                    analysis_parts.append("💡 预测方向: 看跌，策略不做空，建议观望")
                else:
                    analysis_parts.append("💡 预测方向: 横盘整理，建议观望")
            
            # 更新预测的原因说明，包含完整分析
            prediction.reason = " | ".join(analysis_parts)
    
    def generate_portfolio_recommendation_with_holdings(self, predictions: List[StockPrediction], 
                                                       current_positions: List = None) -> PortfolioRecommendation:
        """生成考虑持仓的投资组合建议"""
        buy_recommendations = []
        sell_recommendations = []
        hold_recommendations = []
        
        # 为所有股票添加回测分析信息
        self._add_backtest_analysis_to_predictions(predictions, current_positions or [])
        
        for prediction in predictions:
            holding_info = self._get_holding_info(prediction.symbol, current_positions or [])
            is_holding = holding_info.get('is_holding', False)
            
            # 计算回测系统使用的指标
            trade_info = self._calculate_backtest_metrics(prediction)
            should_buy_flag = self._should_buy_backtest_style(trade_info, current_positions or [])
            should_sell_flag = self._should_sell_backtest_style(prediction, holding_info)
            
            # 使用回测系统的买入条件
            if should_buy_flag:
                # 计算多种买入策略
                buy_strategies = self._calculate_buy_strategies(prediction, trade_info)
                recommended_strategy = self._get_recommended_buy_strategy(buy_strategies, trade_info)
                
                # 计算建议买入数量（基于Kelly公式）
                kelly_quantity = int(trade_info['kelly_fraction'] * 100000 / prediction.current_price)  # 假设10万资金
                suggested_quantity = min(2000, max(100, kelly_quantity))  # 限制在100-2000股之间
                
                action = '加仓' if is_holding else '买入'
                reason_parts = [
                    f"概率{trade_info['probability']:.1%}",
                    f"预期收益{trade_info['expected_return']:.1%}",
                    f"推荐{recommended_strategy['description']}"
                ]
                reason = f"回测策略信号: {', '.join(reason_parts)}"
                
                buy_recommendations.append({
                    'symbol': prediction.symbol,
                    'name': prediction.name,
                    'action': action,
                    'reason': reason,
                    'target_price': prediction.target_price,
                    'suggested_buy_price': recommended_strategy['price'],
                    'confidence': prediction.confidence,
                    'current_holding': holding_info.get('quantity', 0),
                    'suggested_quantity': suggested_quantity,
                    'kelly_fraction': trade_info['kelly_fraction'],
                    'expected_return': trade_info['expected_return'],
                    'buy_strategies': buy_strategies  # 包含所有买入选项
                })
                    
            elif should_sell_flag:
                # 计算建议卖出价（当前价格上调1-3%作为卖出机会）
                suggested_sell_price = prediction.current_price * (1 + 0.01 + 0.02 * prediction.confidence)
                
                # 计算未实现收益
                unrealized_return = 0
                if holding_info.get('avg_cost', 0) > 0:
                    unrealized_return = (prediction.current_price - holding_info['avg_cost']) / holding_info['avg_cost']
                
                # 确定卖出原因
                if prediction.confidence < 0.4:
                    sell_reason = f"回测策略: 概率过低({prediction.confidence:.1%})"
                elif unrealized_return < -0.03:
                    sell_reason = f"回测策略: 触发止损(亏损{unrealized_return:.1%})"
                else:
                    sell_reason = f"回测策略: 卖出信号"
                
                sell_recommendations.append({
                    'symbol': prediction.symbol,
                    'name': prediction.name,
                    'action': '卖出',
                    'reason': sell_reason,
                    'stop_loss': prediction.stop_loss,
                    'suggested_sell_price': suggested_sell_price,
                    'confidence': prediction.confidence,
                    'current_holding': holding_info.get('quantity', 0),
                    'suggested_quantity': holding_info.get('quantity', 0),  # 全部卖出
                    'unrealized_return': unrealized_return
                })
                    
            else:
                # 不满足买入或卖出条件
                if is_holding:
                    hold_recommendations.append({
                        'symbol': prediction.symbol,
                        'name': prediction.name,
                        'action': '持有',
                        'reason': f"回测策略: 暂无明确信号，持有观望（当前{holding_info.get('quantity', 0)}股）",
                        'current_holding': holding_info.get('quantity', 0)
                    })
        
        # 按信心度排序
        buy_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        sell_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return PortfolioRecommendation(
            action='mixed',
            buy_list=buy_recommendations[:5],  # 最多推荐5只买入
            sell_list=sell_recommendations[:5],  # 最多推荐5只卖出
            hold_list=hold_recommendations,
            reason=f"基于当前持仓分析，推荐买入{len(buy_recommendations)}只，卖出{len(sell_recommendations)}只，持有{len(hold_recommendations)}只"
        )

# 工具函数
def format_prediction_result(result: OpeningPredictionResult) -> Dict:
    """格式化预测结果为JSON可序列化格式"""
    return {
        'prediction_time': result.prediction_time,
        'market_overview': {
            'market_sentiment': result.market_overview.market_sentiment,
            'major_indices': result.market_overview.major_indices,
            'market_news': result.market_overview.market_news,
            'risk_factors': result.market_overview.risk_factors
        },
        'stock_predictions': [
            {
                'symbol': p.symbol,
                'name': p.name,
                'current_price': p.current_price,
                'predicted_direction': p.predicted_direction,
                'confidence': p.confidence,
                'target_price': p.target_price,
                'stop_loss': p.stop_loss,
                'recommendation': p.recommendation,
                'reason': p.reason,
                'risk_level': p.risk_level,
                'volume_analysis': p.volume_analysis,
                'technical_indicators': p.technical_indicators
            } for p in result.stock_predictions
        ],
        'portfolio_recommendation': {
            'action': result.portfolio_recommendation.action,
            'buy_list': result.portfolio_recommendation.buy_list,
            'sell_list': result.portfolio_recommendation.sell_list,
            'hold_list': result.portfolio_recommendation.hold_list,
            'reason': result.portfolio_recommendation.reason
        },
        'risk_assessment': result.risk_assessment
    }
