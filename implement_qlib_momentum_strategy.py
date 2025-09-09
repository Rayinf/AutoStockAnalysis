#!/usr/bin/env python3
"""
实施基于Qlib的强动量高收益策略
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict
from backend.qlib_demo import QlibDataAdapter
from backend.strategy_backtest import StrategyBacktester, StrategyConfig
from backend.calibration import calibrator, PredictionRecord
import akshare as ak
import sqlite3

def create_qlib_momentum_predictions(symbols: str, start_date: str, end_date: str) -> Dict:
    """创建基于Qlib动量因子的高质量预测数据"""
    print("🚀 创建Qlib动量因子高质量预测")
    print("=" * 60)
    
    symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
    total_records = 0
    success_symbols = []
    
    # 清除旧数据
    conn = sqlite3.connect("calibration.db")
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in symbol_list])
    cursor.execute(f"DELETE FROM predictions WHERE symbol IN ({placeholders})", symbol_list)
    conn.commit()
    conn.close()
    
    data_adapter = QlibDataAdapter()
    
    for symbol in symbol_list:
        try:
            print(f"📊 处理股票: {symbol}")
            
            # 获取更长时间的历史数据以计算因子
            extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=120)
            
            ak_symbol = symbol[2:] if symbol.startswith(('sh', 'sz')) else symbol
            df = ak.stock_zh_a_hist(
                symbol=ak_symbol,
                period="daily",
                start_date=extended_start.strftime('%Y%m%d'),
                end_date=end_date.replace('-', '')
            )
            
            if df.empty or len(df) < 100:
                print(f"⚠️ {symbol} 数据不足: {len(df)}")
                continue
            
            # 使用Qlib计算高级因子
            features_result = data_adapter.get_features(
                symbol, 
                extended_start.strftime('%Y-%m-%d'), 
                end_date, 
                df
            )
            
            if "error" in features_result:
                print(f"❌ {symbol} 因子计算失败")
                continue
            
            features = features_result["features"]
            
            # 生成基于强动量的预测
            count = generate_momentum_predictions(symbol, df, features, start_date, end_date)
            
            if count > 0:
                total_records += count
                success_symbols.append(symbol)
                print(f"✅ {symbol}: 生成 {count} 条动量预测")
            
        except Exception as e:
            print(f"❌ {symbol} 处理失败: {e}")
            continue
    
    return {
        "total_records": total_records,
        "success_symbols": success_symbols,
        "method": "Qlib Momentum Enhanced"
    }

def generate_momentum_predictions(symbol: str, df: pd.DataFrame, features: Dict, 
                                start_date: str, end_date: str) -> int:
    """基于强动量因子生成预测"""
    count = 0
    
    # 获取因子数据
    momentum_5 = np.array(features.get("momentum_5", []))
    momentum_20 = np.array(features.get("momentum_20", []))
    reversal_1 = np.array(features.get("reversal_1", []))
    volatility_20 = np.array(features.get("volatility_20", []))
    volume_ratio = np.array(features.get("volume_ratio", []))
    rsi_14 = np.array(features.get("rsi_14", []))
    
    # 筛选目标时间范围
    df['日期'] = pd.to_datetime(df['日期'])
    target_start = pd.to_datetime(start_date)
    target_end = pd.to_datetime(end_date)
    
    target_mask = (df['日期'] >= target_start) & (df['日期'] <= target_end)
    target_indices = df.index[target_mask].tolist()
    
    for idx in target_indices:
        try:
            # 确保有下一天数据计算actual_direction
            if idx + 1 >= len(df):
                continue
                
            # 获取因子在当前时点的值
            factor_idx = idx
            
            if (factor_idx >= len(momentum_5) or factor_idx >= len(momentum_20) or 
                factor_idx >= len(volatility_20) or factor_idx >= len(rsi_14)):
                continue
            
            # 🔥 强动量评分系统（高收益导向）
            score = 0.5  # 基础概率
            
            # 主导因子：动量 (60%权重)
            mom5 = momentum_5[factor_idx] if not np.isnan(momentum_5[factor_idx]) else 0
            mom20 = momentum_20[factor_idx] if not np.isnan(momentum_20[factor_idx]) else 0
            
            # 强动量信号：同向动量叠加
            if mom5 > 0 and mom20 > 0:  # 双重上涨动量
                momentum_score = (mom5 * 0.4 + mom20 * 0.2) * 2  # 放大同向动量
            elif mom5 < 0 and mom20 < 0:  # 双重下跌动量
                momentum_score = (mom5 * 0.4 + mom20 * 0.2) * 2
            else:  # 混合动量
                momentum_score = mom5 * 0.3 + mom20 * 0.1
            
            score += momentum_score
            
            # 波动率机会因子 (20%权重)
            vol = volatility_20[factor_idx] if not np.isnan(volatility_20[factor_idx]) else 0.02
            # 高波动率 = 高收益机会
            vol_opportunity = min(0.2, vol * 10) * np.sign(momentum_score)  # 与动量同向
            score += vol_opportunity
            
            # 反转修正因子 (10%权重)
            if factor_idx < len(reversal_1):
                rev1 = reversal_1[factor_idx] if not np.isnan(reversal_1[factor_idx]) else 0
                # 适度反转修正，避免过度追涨
                score += rev1 * 0.1
            
            # RSI确认因子 (10%权重)
            rsi = rsi_14[factor_idx] if not np.isnan(rsi_14[factor_idx]) else 50
            if rsi < 20:  # 极度超卖，强烈买入信号
                score += 0.15
            elif rsi > 80:  # 极度超买，强烈卖出信号
                score -= 0.15
            elif rsi < 30:  # 超卖
                score += 0.05
            elif rsi > 70:  # 超买
                score -= 0.05
            
            # 成交量确认 (补充)
            if factor_idx < len(volume_ratio):
                vol_ratio = volume_ratio[factor_idx] if not np.isnan(volume_ratio[factor_idx]) else 1
                if vol_ratio > 2.0:  # 大幅放量
                    score += 0.05 * np.sign(momentum_score)  # 与动量同向
            
            # 限制概率范围，保持合理分布
            predicted_prob = np.clip(score, 0.05, 0.95)
            
            # 计算实际方向
            current_close = df.iloc[idx]['收盘']
            next_close = df.iloc[idx + 1]['收盘']
            actual_direction = 1 if next_close > current_close else 0
            
            # 保存增强预测记录
            record = PredictionRecord(
                symbol=symbol,
                prediction_date=df.iloc[idx]['日期'].strftime('%Y-%m-%d'),
                predicted_probability=predicted_prob,
                actual_direction=actual_direction,
                features={
                    "qlib_momentum_enhanced": True,
                    "momentum_5": float(mom5),
                    "momentum_20": float(mom20),
                    "volatility_20": float(vol),
                    "rsi_14": float(rsi),
                    "momentum_score": float(momentum_score),
                    "vol_opportunity": float(vol_opportunity)
                }
            )
            
            calibrator.save_prediction(record)
            count += 1
            
        except Exception as e:
            continue
    
    return count

def test_qlib_momentum_strategy():
    """测试Qlib动量策略"""
    symbols = "600519,000001,600000,600036,601398"
    start_date = "2025-04-01"
    end_date = "2025-08-01"
    
    # 生成Qlib动量增强预测
    result = create_qlib_momentum_predictions(symbols, start_date, end_date)
    
    if result['total_records'] > 100:
        print(f"\n🧪 测试Qlib动量增强策略")
        print("-" * 40)
        
        # 强动量策略配置
        momentum_config = StrategyConfig(
            buy_threshold=0.48,  # 更低阈值，捕捉更多动量信号
            sell_threshold=0.28,  # 更低卖出阈值
            selection_mode="threshold",
            calibration_gamma=2.5,  # 强放大动量信号
            position_size=0.12,  # 适中仓位
            max_positions=12,
            profit_target=0.06,  # 6%止盈
            loss_tolerance=0.015  # 1.5%止损
        )
        
        backtester = StrategyBacktester()
        backtester.verbose = True
        
        backtest_result = backtester.run_backtest(
            momentum_config,
            start_date,
            end_date,
            symbols.split(','),
            use_calibration=False
        )
        
        if 'error' not in backtest_result:
            metrics = backtest_result['performance_metrics']
            print(f"🔥 Qlib动量策略结果:")
            print(f"  总收益率: {metrics['total_return']:.2%}")
            print(f"  年化收益: {metrics['annualized_return']:.2%}")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"  交易次数: {metrics['total_trades']}")
            print(f"  胜率: {metrics['win_rate']:.1%}")
            
            # 与基准对比
            baseline_return = 0.0258  # 原策略
            improvement = (metrics['total_return'] - baseline_return) / baseline_return * 100
            
            print(f"\n📈 与基准策略对比:")
            print(f"  基准收益率: 2.58%")
            print(f"  Qlib动量收益率: {metrics['total_return']:.2%}")
            print(f"  提升幅度: {improvement:+.1f}%")
            
            if metrics['total_return'] > 0.04:
                print("🎉 Qlib动量策略实现高收益！")
                return True
            else:
                print("⚠️ 收益率仍需进一步优化")
                return False
        else:
            print(f"❌ Qlib动量策略回测失败: {backtest_result['error']}")
            return False
    else:
        print(f"❌ 数据生成不足，无法测试")
        return False

def implement_qlib_enhancements():
    """实施Qlib增强方案"""
    print(f"\n🔧 实施Qlib收益增强方案")
    print("-" * 40)
    
    print("📋 实施计划:")
    print("1. 将Qlib因子集成到历史数据生成")
    print("2. 更新预测算法使用强动量因子")
    print("3. 调整策略参数适配动量信号")
    print("4. 实施动态止盈止损机制")
    
    # 更新calibration.py中的预测生成逻辑
    print(f"\n🔄 建议的代码更新:")
    print("```python")
    print("# 在calibration.py中替换简单评分为Qlib因子评分")
    print("# 动量因子权重: 50%")
    print("# 波动率因子权重: 20%")
    print("# 技术指标权重: 20%")
    print("# 成交量因子权重: 10%")
    print("```")

if __name__ == "__main__":
    # 生成更多数据用于测试
    symbols = "600519,000001,600000,600036,601398,000002,002594,300750"
    start_date = "2025-03-01"  # 扩大时间范围
    end_date = "2025-08-01"
    
    success = test_qlib_momentum_strategy()
    implement_qlib_enhancements()
    
    if success:
        print(f"\n🎉 Qlib动量策略验证成功！")
        print("建议将此策略设为系统默认配置。")
    else:
        print(f"\n🔧 需要进一步优化Qlib因子权重和策略参数")



