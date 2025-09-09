#!/usr/bin/env python3
"""
测试激进收益优化策略
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.strategy_backtest import StrategyBacktester, StrategyConfig
from backend.calibration import calibrator
import pandas as pd

def test_aggressive_strategy():
    """测试激进收益优化策略"""
    print("🔥 测试激进收益优化策略")
    print("=" * 60)
    
    # 先生成测试数据
    symbols = "600519,000001,600000,600036,601398"
    start_date = "2025-04-01"
    end_date = "2025-08-01"
    
    print(f"📊 生成测试数据: {symbols} ({start_date} ~ {end_date})")
    try:
        result = calibrator.generate_historical_backtest_data_by_date_range(symbols, start_date, end_date)
        print(f"生成结果: {result}")
    except Exception as e:
        print(f"生成失败: {e}")
    
    # 测试不同策略模式
    strategies = [
        {
            "name": "传统阈值策略",
            "config": StrategyConfig(
                buy_threshold=0.6,
                sell_threshold=0.4,
                selection_mode="threshold",
                calibration_gamma=1.0
            )
        },
        {
            "name": "Top-K策略",
            "config": StrategyConfig(
                buy_threshold=0.55,
                sell_threshold=0.35,
                selection_mode="topk",
                top_k=3,
                calibration_gamma=1.2
            )
        },
        {
            "name": "激进收益优化策略",
            "config": StrategyConfig(
                buy_threshold=0.5,
                sell_threshold=0.3,
                selection_mode="aggressive",
                calibration_gamma=2.0,
                momentum_weight=0.3,
                profit_target=0.05,
                loss_tolerance=0.02
            )
        }
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n🧪 测试: {strategy['name']}")
        print("-" * 40)
        
        backtester = StrategyBacktester()
        backtester.verbose = True  # 显示详细信息
        
        result = backtester.run_backtest(
            strategy['config'], 
            start_date, 
            end_date, 
            symbols.split(','),
            use_calibration=False
        )
        
        if 'error' not in result:
            metrics = result['performance_metrics']
            print(f"✅ 回测成功:")
            print(f"  总收益率: {metrics['total_return']:.2%}")
            print(f"  年化收益: {metrics['annualized_return']:.2%}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"  交易次数: {metrics['total_trades']}")
            print(f"  胜率: {metrics['win_rate']:.1%}")
            
            results.append({
                'strategy': strategy['name'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate']
            })
        else:
            print(f"❌ 回测失败: {result['error']}")
    
    # 对比结果
    print(f"\n📊 策略对比结果")
    print("=" * 60)
    print(f"{'策略名称':<15} {'总收益率':<10} {'夏普比率':<10} {'交易次数':<8} {'胜率':<8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['strategy']:<15} {r['total_return']:<10.2%} {r['sharpe_ratio']:<10.3f} {r['total_trades']:<8} {r['win_rate']:<8.1%}")
    
    # 找出最佳策略
    if results:
        best_strategy = max(results, key=lambda x: x['total_return'])
        print(f"\n🏆 最佳策略: {best_strategy['strategy']}")
        print(f"   收益率: {best_strategy['total_return']:.2%}")
        print(f"   夏普比率: {best_strategy['sharpe_ratio']:.3f}")

def analyze_why_low_returns():
    """分析为什么收益率低"""
    print(f"\n🔍 分析收益率低的原因")
    print("-" * 40)
    
    # 检查预测概率分布
    import sqlite3
    conn = sqlite3.connect("calibration.db")
    query = """
        SELECT 
            AVG(predicted_probability) as avg_prob,
            MIN(predicted_probability) as min_prob,
            MAX(predicted_probability) as max_prob,
            AVG(CASE WHEN actual_direction = 1 THEN 1.0 ELSE 0.0 END) as actual_up_rate,
            COUNT(*) as total_records
        FROM predictions 
        WHERE actual_direction IS NOT NULL
        AND prediction_date >= '2025-04-01'
        AND prediction_date <= '2025-08-01'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) > 0:
        row = df.iloc[0]
        print(f"📊 预测数据分析:")
        print(f"  平均概率: {row['avg_prob']:.3f}")
        print(f"  概率范围: {row['min_prob']:.3f} ~ {row['max_prob']:.3f}")
        print(f"  实际上涨率: {row['actual_up_rate']:.1%}")
        print(f"  总记录数: {row['total_records']}")
        
        # 分析问题
        if row['avg_prob'] < 0.55:
            print("⚠️ 问题1: 平均概率偏低，缺乏强信号")
        if abs(row['actual_up_rate'] - 0.5) < 0.05:
            print("⚠️ 问题2: 实际上涨率接近50%，预测能力有限")
        if row['max_prob'] - row['min_prob'] < 0.3:
            print("⚠️ 问题3: 概率分布范围窄，缺乏区分度")

def suggest_improvements():
    """建议改进方案"""
    print(f"\n💡 收益率提升建议")
    print("-" * 40)
    
    print("1. 🎯 提高信号质量:")
    print("   - 改进预测算法，提高概率准确性")
    print("   - 增加更多技术指标（RSI、MACD、布林带）")
    print("   - 考虑基本面因素（PE、PB、ROE）")
    
    print("\n2. 🔥 优化交易策略:")
    print("   - 实施动量策略：追涨杀跌")
    print("   - 增加波动率过滤：高波动=高收益机会")
    print("   - 实施配对交易：多空对冲")
    
    print("\n3. ⚡ 激进参数调整:")
    print("   - 降低买入阈值到0.52")
    print("   - 提高仓位上限到15%")
    print("   - 缩短持仓期到5-7天")
    print("   - 设置更激进的止盈止损")

if __name__ == "__main__":
    test_aggressive_strategy()
    analyze_why_low_returns()
    suggest_improvements()



