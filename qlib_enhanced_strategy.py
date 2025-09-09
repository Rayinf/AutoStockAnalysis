#!/usr/bin/env python3
"""
基于Qlib高级因子的收益优化策略
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict
from backend.qlib_demo import QlibDataAdapter, QlibPredictor
from backend.strategy_backtest import StrategyBacktester, StrategyConfig
from backend.calibration import calibrator, PredictionRecord
import akshare as ak

class QlibEnhancedStrategy:
    """基于Qlib高级因子的增强策略"""
    
    def __init__(self):
        self.data_adapter = QlibDataAdapter()
        self.predictor = QlibPredictor()
        
    def generate_enhanced_predictions(self, symbols: str, start_date: str, end_date: str) -> Dict:
        """使用Qlib高级因子生成增强预测"""
        print("🔥 使用Qlib高级因子生成增强预测")
        print("=" * 60)
        
        symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        total_records = 0
        success_symbols = []
        
        for symbol in symbol_list:
            try:
                print(f"📊 处理股票: {symbol}")
                
                # 获取历史数据
                ak_symbol = symbol[2:] if symbol.startswith(('sh', 'sz')) else symbol
                df = ak.stock_zh_a_hist(
                    symbol=ak_symbol,
                    period="daily",
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
                
                if df.empty or len(df) < 60:
                    print(f"⚠️ {symbol} 数据不足")
                    continue
                
                # 使用Qlib数据适配器计算高级因子
                features_result = self.data_adapter.get_features(symbol, start_date, end_date, df)
                
                if "error" in features_result:
                    print(f"❌ {symbol} 因子计算失败")
                    continue
                
                features = features_result["features"]
                
                # 生成增强预测
                count = self._generate_predictions_with_qlib_factors(symbol, df, features)
                
                if count > 0:
                    total_records += count
                    success_symbols.append(symbol)
                    print(f"✅ {symbol}: 生成 {count} 条增强预测")
                
            except Exception as e:
                print(f"❌ {symbol} 处理失败: {e}")
                continue
        
        return {
            "total_records": total_records,
            "success_symbols": success_symbols,
            "method": "Qlib Enhanced Factors"
        }
    
    def _generate_predictions_with_qlib_factors(self, symbol: str, df: pd.DataFrame, features: Dict) -> int:
        """基于Qlib因子生成预测"""
        count = 0
        
        # 获取因子数据
        momentum_5 = features.get("momentum_5", [])
        momentum_20 = features.get("momentum_20", [])
        reversal_1 = features.get("reversal_1", [])
        volatility_20 = features.get("volatility_20", [])
        volume_ratio = features.get("volume_ratio", [])
        rsi_14 = features.get("rsi_14", [])
        ma_ratio_5 = features.get("ma_ratio_5", [])
        ma_ratio_20 = features.get("ma_ratio_20", [])
        price_position = features.get("price_position", [])
        
        # 确保所有因子长度一致
        min_length = min(len(f) for f in [momentum_5, momentum_20, volatility_20, rsi_14] if len(f) > 0)
        
        if min_length < 20:
            return 0
        
        for i in range(20, min_length):
            try:
                # 🔥 Qlib增强因子评分系统
                score = 0.5  # 基础概率
                
                # 动量因子组合 (35%权重)
                if i < len(momentum_5):
                    mom5 = momentum_5[i] if not np.isnan(momentum_5[i]) else 0
                    score += mom5 * 0.2  # 短期动量
                
                if i < len(momentum_20):
                    mom20 = momentum_20[i] if not np.isnan(momentum_20[i]) else 0
                    score += mom20 * 0.15  # 长期动量
                
                # 反转因子 (15%权重)
                if i < len(reversal_1):
                    rev1 = reversal_1[i] if not np.isnan(reversal_1[i]) else 0
                    score += rev1 * 0.15  # 短期反转
                
                # 技术指标因子 (25%权重)
                if i < len(rsi_14):
                    rsi = rsi_14[i] if not np.isnan(rsi_14[i]) else 50
                    rsi_signal = (rsi - 50) / 50  # 标准化到-1到1
                    if rsi < 30:  # 超卖
                        score += 0.1
                    elif rsi > 70:  # 超买
                        score -= 0.1
                    score += rsi_signal * 0.1
                
                if i < len(ma_ratio_5):
                    ma5_ratio = ma_ratio_5[i] if not np.isnan(ma_ratio_5[i]) else 0
                    score += ma5_ratio * 0.05
                
                # 成交量因子 (15%权重)
                if i < len(volume_ratio):
                    vol_ratio = volume_ratio[i] if not np.isnan(volume_ratio[i]) else 1
                    if vol_ratio > 1.5:  # 放量
                        score += 0.1
                    score += min(0.05, (vol_ratio - 1) * 0.05)
                
                # 波动率因子 (10%权重)
                if i < len(volatility_20):
                    vol = volatility_20[i] if not np.isnan(volatility_20[i]) else 0.02
                    # 高波动率增加不确定性，但也增加机会
                    vol_factor = min(0.1, vol * 10)
                    score += vol_factor * np.random.uniform(-0.5, 1.0)  # 偏向正面
                
                # 价格位置因子 (补充)
                if i < len(price_position):
                    price_pos = price_position[i] if not np.isnan(price_position[i]) else 0.5
                    score += (price_pos - 0.5) * 0.1  # 价格位置偏离中位数的影响
                
                # 添加适量随机性
                noise = np.random.normal(0, 0.05)
                predicted_prob = np.clip(score + noise, 0.02, 0.98)
                
                # 计算实际方向
                if i + 1 < len(df):
                    current_close = df.iloc[i]['收盘']
                    next_close = df.iloc[i + 1]['收盘']
                    actual_direction = 1 if next_close > current_close else 0
                    
                    # 保存增强预测记录
                    record = PredictionRecord(
                        symbol=symbol,
                        prediction_date=df.iloc[i]['日期'].strftime('%Y-%m-%d'),
                        predicted_probability=predicted_prob,
                        actual_direction=actual_direction,
                        features={
                            "qlib_enhanced": True,
                            "momentum_5": float(mom5) if 'mom5' in locals() else 0,
                            "momentum_20": float(mom20) if 'mom20' in locals() else 0,
                            "rsi_14": float(rsi) if 'rsi' in locals() else 50,
                            "volume_ratio": float(vol_ratio) if 'vol_ratio' in locals() else 1,
                            "volatility_20": float(vol) if 'vol' in locals() else 0.02,
                            "price_position": float(price_pos) if 'price_pos' in locals() else 0.5
                        }
                    )
                    
                    calibrator.save_prediction(record)
                    count += 1
                    
            except Exception as e:
                continue
        
        return count

def test_qlib_enhanced_strategy():
    """测试Qlib增强策略"""
    print("🚀 测试Qlib增强收益策略")
    print("=" * 60)
    
    # 清除旧数据
    import sqlite3
    conn = sqlite3.connect("calibration.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE symbol IN ('600519', '000001', '600000')")
    conn.commit()
    conn.close()
    
    symbols = "600519,000001,600000"
    start_date = "2025-04-01"
    end_date = "2025-08-01"
    
    # 生成Qlib增强预测数据
    enhancer = QlibEnhancedStrategy()
    result = enhancer.generate_enhanced_predictions(symbols, start_date, end_date)
    
    print(f"📊 Qlib增强预测生成结果:")
    print(f"  总记录数: {result['total_records']}")
    print(f"  成功股票: {result['success_symbols']}")
    
    if result['total_records'] > 0:
        # 测试不同策略配置
        strategies = [
            {
                "name": "Qlib增强 + 传统阈值",
                "config": StrategyConfig(
                    buy_threshold=0.6,
                    sell_threshold=0.4,
                    selection_mode="threshold",
                    position_size=0.1
                )
            },
            {
                "name": "Qlib增强 + 激进配置",
                "config": StrategyConfig(
                    buy_threshold=0.52,
                    sell_threshold=0.32,
                    selection_mode="threshold",
                    calibration_gamma=1.8,
                    position_size=0.08,
                    max_positions=15
                )
            },
            {
                "name": "Qlib增强 + 激进模式",
                "config": StrategyConfig(
                    buy_threshold=0.5,
                    sell_threshold=0.3,
                    selection_mode="aggressive",
                    calibration_gamma=2.0,
                    momentum_weight=0.4,
                    profit_target=0.04,
                    loss_tolerance=0.015,
                    position_size=0.12
                )
            }
        ]
        
        results = []
        
        for strategy in strategies:
            print(f"\n🧪 测试: {strategy['name']}")
            print("-" * 40)
            
            backtester = StrategyBacktester()
            backtester.verbose = True
            
            backtest_result = backtester.run_backtest(
                strategy['config'],
                start_date,
                end_date,
                symbols.split(','),
                use_calibration=False
            )
            
            if 'error' not in backtest_result:
                metrics = backtest_result['performance_metrics']
                print(f"✅ 回测成功:")
                print(f"  总收益率: {metrics['total_return']:.2%}")
                print(f"  年化收益: {metrics['annualized_return']:.2%}")
                print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
                print(f"  交易次数: {metrics['total_trades']}")
                print(f"  胜率: {metrics['win_rate']:.1%}")
                
                results.append({
                    'strategy': strategy['name'],
                    'total_return': metrics['total_return'],
                    'annualized_return': metrics['annualized_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_trades': metrics['total_trades'],
                    'win_rate': metrics['win_rate']
                })
            else:
                print(f"❌ 回测失败: {backtest_result['error']}")
        
        # 对比结果
        if results:
            print(f"\n📊 Qlib增强策略对比结果")
            print("=" * 70)
            print(f"{'策略名称':<20} {'总收益率':<8} {'年化收益':<8} {'夏普比率':<8} {'交易次数':<6}")
            print("-" * 70)
            
            results.sort(key=lambda x: x['total_return'], reverse=True)
            
            for r in results:
                print(f"{r['strategy']:<20} {r['total_return']:<8.2%} {r['annualized_return']:<8.2%} "
                      f"{r['sharpe_ratio']:<8.3f} {r['total_trades']:<6}")
            
            best = results[0]
            print(f"\n🏆 最佳Qlib增强策略: {best['strategy']}")
            print(f"   收益率: {best['total_return']:.2%}")
            print(f"   年化收益: {best['annualized_return']:.2%}")
            
            # 与原策略对比
            print(f"\n📈 与原策略对比:")
            original_return = 0.0258  # 原高频小仓位策略
            improvement = (best['total_return'] - original_return) / original_return * 100
            print(f"  原策略收益率: 2.58%")
            print(f"  Qlib增强收益率: {best['total_return']:.2%}")
            print(f"  提升幅度: {improvement:+.1f}%")
            
            if best['total_return'] > original_return:
                print("✅ Qlib增强策略显著提升收益率！")
            else:
                print("⚠️ Qlib增强策略需要进一步调优")

def analyze_qlib_factors():
    """分析Qlib因子的预测能力"""
    print(f"\n🔍 Qlib因子预测能力分析")
    print("-" * 40)
    
    print("📊 Qlib提供的高级因子:")
    print("1. 动量因子:")
    print("   - momentum_5: 5日价格动量")
    print("   - momentum_20: 20日价格动量")
    
    print("\n2. 反转因子:")
    print("   - reversal_1: 1日价格反转")
    print("   - reversal_5: 5日价格反转")
    
    print("\n3. 波动率因子:")
    print("   - volatility_20: 20日波动率")
    print("   - volatility_60: 60日波动率")
    
    print("\n4. 成交量因子:")
    print("   - volume_ratio: 成交量比率")
    print("   - turnover: 换手率代理")
    
    print("\n5. 技术指标因子:")
    print("   - rsi_14: 14日RSI")
    print("   - ma_ratio_5: 与5日均线偏离度")
    print("   - ma_ratio_20: 与20日均线偏离度")
    print("   - price_position: 价格在高低点中的位置")

def suggest_qlib_improvements():
    """建议Qlib改进方案"""
    print(f"\n💡 Qlib收益优化建议")
    print("-" * 40)
    
    print("🎯 高收益因子组合:")
    print("1. 强动量策略:")
    print("   - 权重: momentum_5(30%) + momentum_20(20%)")
    print("   - 逻辑: 追涨杀跌，捕捉趋势")
    
    print("\n2. 反转+波动策略:")
    print("   - 权重: reversal_1(20%) + volatility_20(15%)")
    print("   - 逻辑: 短期反转+高波动机会")
    
    print("\n3. 技术指标确认:")
    print("   - 权重: rsi_14(10%) + ma_ratio_5(5%)")
    print("   - 逻辑: 技术指标确认信号")
    
    print("\n🔥 激进优化方案:")
    print("- 提高动量因子权重到50%")
    print("- 增加波动率因子权重到20%")
    print("- 降低买入阈值到0.48")
    print("- 设置动态止盈：2-8%区间")

if __name__ == "__main__":
    test_qlib_enhanced_strategy()
    analyze_qlib_factors()
    suggest_qlib_improvements()
