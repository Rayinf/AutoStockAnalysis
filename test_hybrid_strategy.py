#!/usr/bin/env python3
"""
测试混合收益优化策略
结合各策略优点
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.strategy_backtest import StrategyBacktester, StrategyConfig
from backend.calibration import calibrator

def test_hybrid_strategies():
    """测试混合收益优化策略"""
    print("🔥 测试混合收益优化策略")
    print("=" * 60)
    
    symbols = "600519,000001,600000,600036,601398"
    start_date = "2025-04-01"
    end_date = "2025-08-01"
    
    # 测试不同的混合策略
    strategies = [
        {
            "name": "最佳传统策略（基准）",
            "config": StrategyConfig(
                buy_threshold=0.6,
                sell_threshold=0.4,
                selection_mode="threshold",
                calibration_gamma=1.0,
                position_size=0.1
            )
        },
        {
            "name": "混合策略1：激进阈值+传统仓位",
            "config": StrategyConfig(
                buy_threshold=0.5,  # 激进阈值
                sell_threshold=0.3,  # 激进阈值
                selection_mode="threshold",  # 传统模式
                calibration_gamma=1.5,  # 中等放大
                position_size=0.1  # 传统仓位
            )
        },
        {
            "name": "混合策略2：传统阈值+激进仓位",
            "config": StrategyConfig(
                buy_threshold=0.6,  # 传统阈值
                sell_threshold=0.4,  # 传统阈值
                selection_mode="threshold",
                calibration_gamma=2.0,  # 激进放大
                position_size=0.15  # 激进仓位
            )
        },
        {
            "name": "混合策略3：优化Top-K",
            "config": StrategyConfig(
                buy_threshold=0.55,
                sell_threshold=0.35,
                selection_mode="topk",
                top_k=2,  # 减少选择数量
                calibration_gamma=1.2,
                position_size=0.12
            )
        },
        {
            "name": "混合策略4：保守激进模式",
            "config": StrategyConfig(
                buy_threshold=0.55,  # 稍微保守的阈值
                sell_threshold=0.35,
                selection_mode="aggressive",
                calibration_gamma=1.5,
                momentum_weight=0.2,  # 降低动量权重
                profit_target=0.03,  # 降低止盈目标
                loss_tolerance=0.015,  # 更严格止损
                position_size=0.12
            )
        },
        {
            "name": "混合策略5：高频小仓位",
            "config": StrategyConfig(
                buy_threshold=0.52,  # 更低阈值
                sell_threshold=0.32,
                selection_mode="threshold",
                calibration_gamma=1.8,
                position_size=0.08,  # 更小仓位
                max_positions=15  # 更多持仓
            )
        }
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n🧪 测试: {strategy['name']}")
        print("-" * 50)
        
        backtester = StrategyBacktester()
        backtester.verbose = False  # 静默模式
        
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
            
            # 综合评分：收益率为主，夏普为辅
            composite_score = metrics['total_return'] * 0.7 + metrics['sharpe_ratio'] * 0.3
            
            results.append({
                'strategy': strategy['name'],
                'total_return': metrics['total_return'],
                'annualized_return': metrics['annualized_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'composite_score': composite_score
            })
        else:
            print(f"❌ 回测失败: {result['error']}")
    
    # 对比结果
    print(f"\n📊 混合策略对比结果")
    print("=" * 80)
    print(f"{'策略名称':<25} {'总收益率':<8} {'年化收益':<8} {'夏普比率':<8} {'最大回撤':<8} {'交易次数':<6}")
    print("-" * 80)
    
    # 按综合评分排序
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    for r in results:
        print(f"{r['strategy']:<25} {r['total_return']:<8.2%} {r['annualized_return']:<8.2%} "
              f"{r['sharpe_ratio']:<8.3f} {r['max_drawdown']:<8.2%} {r['total_trades']:<6}")
    
    # 推荐最佳策略
    if results:
        best_strategy = results[0]
        print(f"\n🏆 最佳混合策略: {best_strategy['strategy']}")
        print(f"   总收益率: {best_strategy['total_return']:.2%}")
        print(f"   年化收益: {best_strategy['annualized_return']:.2%}")
        print(f"   夏普比率: {best_strategy['sharpe_ratio']:.3f}")
        print(f"   综合评分: {best_strategy['composite_score']:.3f}")
        
        # 分析优势
        print(f"\n📈 优势分析:")
        if best_strategy['total_return'] > 0.025:
            print("✅ 收益率优秀（>2.5%）")
        if best_strategy['sharpe_ratio'] > 0.5:
            print("✅ 风险调整收益优秀（>0.5）")
        if best_strategy['max_drawdown'] < 0.05:
            print("✅ 风险控制良好（<5%）")
        if best_strategy['win_rate'] > 0.6:
            print("✅ 胜率优秀（>60%）")

def update_system_defaults():
    """更新系统默认配置为最佳策略"""
    print(f"\n🔧 更新系统默认配置")
    print("-" * 40)
    
    print("基于测试结果，建议的系统默认配置:")
    print("```python")
    print("StrategyConfig(")
    print("    buy_threshold=0.5,")
    print("    sell_threshold=0.3,") 
    print("    selection_mode='threshold',")
    print("    calibration_gamma=1.5,")
    print("    position_size=0.12,")
    print("    max_positions=12")
    print(")")
    print("```")
    
    print("\n这个配置结合了:")
    print("- 激进的买卖阈值（更多交易机会）")
    print("- 传统的阈值模式（稳定可靠）")
    print("- 适度的概率放大（增强信号）")
    print("- 平衡的仓位配置（风险可控）")

if __name__ == "__main__":
    test_hybrid_strategies()
    update_system_defaults()

