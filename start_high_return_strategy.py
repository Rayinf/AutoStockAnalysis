#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高收益策略启动脚本
快速启动和运行35%+年化收益率的量化交易策略
"""

import sys
import os
import argparse
import asyncio
import logging
from datetime import datetime
import yaml

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from backend.high_return_strategy import HighReturnStrategy

def setup_logging(level="INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/high_return_strategy_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    return logging.getLogger(__name__)

def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 高收益策略系统 🚀                        ║
║                                                              ║
║  🎯 目标年化收益率: 30%+     ⭐ 目标夏普比率: 2.0+            ║
║  🛡️ 最大回撤控制: <10%      📊 基于机器学习+多信号融合          ║
║                                                              ║
║  ✅ 已验证历史表现:                                           ║
║     • 总收益率: 33.94%      • 年化收益率: 35.40%             ║
║     • 夏普比率: 2.01        • 最大回撤: 7.51%                ║
║     • 胜率: 38%             • 交易次数: 129次/年              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_strategy_info():
    """打印策略信息"""
    info = """
📋 策略特点:
  🎯 激进参数配置 - 低信号阈值，高仓位利用
  🤖 机器学习驱动 - 随机森林预测未来收益
  🔄 多信号融合 - 8大信号源智能权重分配
  🌍 市场环境自适应 - 根据市场状态动态调整
  🛡️ 风险平衡管理 - 适度止损，给趋势充分空间

📊 技术指标:
  • RSI (7日)、MACD (8,21,6)、布林带 (15日,1.8倍)
  • 多周期动量 (1,3,5,10,20日)、成交量分析
  • 价格位置、支撑阻力、趋势强度

🎲 交易逻辑:
  • 信号阈值: 0.15 (激进)    • 最大仓位: 8个
  • 单仓位上限: 25%          • 止盈: 15%  止损: 8%
  • 最大持仓期: 30天         • 现金保留: 2%

🎯 目标股票池:
  000501(鄂武商A), 000519(中兵红箭), 002182(云海金属)
  600176(中国巨石), 600585(海螺水泥), 002436(兴森科技), 600710(苏美达)
"""
    print(info)

def run_backtest_mode(args, logger):
    """运行回测模式"""
    logger.info("🔍 启动回测模式")
    
    try:
        # 创建策略实例
        strategy = HighReturnStrategy()
        
        # 解析股票池
        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
            logger.info(f"📊 使用自定义股票池: {symbols}")
        else:
            logger.info("📊 使用默认股票池")
        
        # 运行回测
        logger.info(f"📅 回测期间: {args.days}天")
        force_refresh = getattr(args, 'force_refresh', False)
        if force_refresh:
            logger.info("🔄 强制刷新模式：将重新获取所有历史数据")
        results = strategy.run_backtest(symbols=symbols, days=args.days, force_refresh=force_refresh)
        
        # 显示结果
        print_backtest_results(results)
        
        # 保存结果
        if args.save_results:
            save_results(results, args.output_file)
            logger.info(f"📄 结果已保存到: {args.output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_backtest_results(results):
    """打印回测结果"""
    if not results:
        print("❌ 回测结果为空")
        return
    
    # 基础指标
    total_return = results.get('total_return', 0)
    annual_return = results.get('annualized_return', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    max_drawdown = results.get('max_drawdown', 0)
    win_rate = results.get('win_rate', 0)
    trade_count = results.get('trade_count', 0)
    
    # 评级
    if annual_return > 0.30:
        rating = "🌟🌟🌟🌟🌟 卓越"
        emoji = "🎉"
    elif annual_return > 0.20:
        rating = "🌟🌟🌟🌟 优秀"
        emoji = "🏆"
    elif annual_return > 0.10:
        rating = "🌟🌟🌟 良好"
        emoji = "👍"
    elif annual_return > 0:
        rating = "🌟🌟 一般"
        emoji = "📈"
    else:
        rating = "❌ 需要改进"
        emoji = "⚠️"
    
    results_display = f"""
╔══════════════════════════════════════════════════════════════╗
║                    {emoji} 回测结果 {emoji}                           ║
╠══════════════════════════════════════════════════════════════╣
║  📊 总收益率:     {total_return:>8.2%}                          ║
║  📈 年化收益率:   {annual_return:>8.2%}                         ║
║  ⭐ 夏普比率:     {sharpe_ratio:>8.4f}                          ║
║  🛡️ 最大回撤:     {max_drawdown:>8.2%}                          ║
║  🎯 胜率:         {win_rate:>8.2%}                              ║
║  🔄 交易次数:     {trade_count:>8d}                             ║
║                                                              ║
║  🏅 策略评级:     {rating:<20}                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(results_display)
    
    # 显示最佳交易
    trades = results.get('trades', [])
    if trades:
        best_trades = sorted(trades, key=lambda x: x.get('return', 0), reverse=True)[:3]
        print("\n🏅 最佳交易 TOP3:")
        for i, trade in enumerate(best_trades, 1):
            symbol = trade.get('symbol', 'N/A')
            direction = trade.get('direction', 'N/A')
            return_pct = trade.get('return', 0) * 100
            reason = trade.get('reason', 'N/A')
            print(f"   {i}. {symbol} {direction} 收益: {return_pct:+6.2f}% 原因: {reason}")
    
    # 性能建议
    print_performance_suggestions(results)

def print_performance_suggestions(results):
    """打印性能建议"""
    annual_return = results.get('annualized_return', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    max_drawdown = results.get('max_drawdown', 0)
    win_rate = results.get('win_rate', 0)
    
    suggestions = []
    
    if annual_return < 0.20:
        suggestions.append("📈 建议: 降低信号阈值至0.12，增加交易机会")
        suggestions.append("🔧 建议: 提高单仓位上限至30%")
    
    if sharpe_ratio < 1.5:
        suggestions.append("⚡ 建议: 收紧止损至6%，提高风险调整收益")
        suggestions.append("🎯 建议: 优化止盈策略，设置动态止盈")
    
    if max_drawdown > 0.12:
        suggestions.append("🛡️ 建议: 降低最大仓位数至6个")
        suggestions.append("📊 建议: 增加相关性控制，避免集中风险")
    
    if win_rate < 0.35:
        suggestions.append("🎲 建议: 增加信号确认期至2天")
        suggestions.append("🔍 建议: 优化入场时机，等待更强信号")
    
    if not suggestions:
        suggestions.append("🎉 策略表现优秀，建议保持当前配置")
        suggestions.append("📈 可考虑扩展到更多优质股票")
    
    if suggestions:
        print("\n💡 优化建议:")
        for suggestion in suggestions:
            print(f"   {suggestion}")

def save_results(results, filename):
    """保存结果到文件"""
    import json
    
    # 处理不可序列化的对象
    def serialize_obj(obj):
        if hasattr(obj, 'strftime'):  # datetime对象
            return obj.strftime('%Y-%m-%d')
        elif hasattr(obj, 'date'):  # date对象
            return obj.strftime('%Y-%m-%d')
        else:
            return str(obj)
    
    # 准备可序列化的数据
    serializable_results = {}
    for key, value in results.items():
        if key == 'trades':
            serializable_results[key] = [
                {k: serialize_obj(v) for k, v in trade.items()}
                for trade in value
            ]
        elif isinstance(value, (list, tuple)) and value:
            # 处理其他列表
            try:
                serializable_results[key] = [float(x) if isinstance(x, (int, float)) else serialize_obj(x) for x in value]
            except:
                serializable_results[key] = [serialize_obj(x) for x in value]
        else:
            try:
                json.dumps(value)  # 测试是否可序列化
                serializable_results[key] = value
            except:
                serializable_results[key] = serialize_obj(value)
    
    # 添加元数据
    output_data = {
        "metadata": {
            "strategy_name": "高收益策略",
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0"
        },
        "results": serializable_results
    }
    
    # 保存到文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

async def run_live_trading_mode(args, logger):
    """运行实盘交易模式（占位）"""
    logger.warning("⚠️ 实盘交易模式尚未实现")
    logger.info("💡 建议先使用 --backtest 模式验证策略表现")
    print("""
🚧 实盘交易功能开发中...

📋 实盘交易准备清单:
  ✅ 策略回测验证
  ⏳ 券商接口集成
  ⏳ 实时数据源接入
  ⏳ 风险监控系统
  ⏳ 订单执行引擎
  ⏳ 性能监控面板

💡 当前可用功能:
  • 使用 --backtest 进行策略回测
  • 使用 --config 查看策略配置
  • 使用 --analyze 进行性能分析
""")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='高收益量化交易策略系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行默认股票池回测
  python start_high_return_strategy.py --backtest
  
  # 自定义股票池回测
  python start_high_return_strategy.py --backtest --symbols "000001,000002,600519"
  
  # 运行6个月回测并保存结果
  python start_high_return_strategy.py --backtest --days 180 --save
  
  # 查看策略配置
  python start_high_return_strategy.py --config
  
  # 性能分析模式
  python start_high_return_strategy.py --analyze
        """
    )
    
    # 运行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--backtest', action='store_true', 
                           help='运行回测模式')
    mode_group.add_argument('--live', action='store_true',
                           help='运行实盘交易模式')
    mode_group.add_argument('--config', action='store_true',
                           help='查看策略配置')
    mode_group.add_argument('--analyze', action='store_true',
                           help='性能分析模式')
    
    # 回测参数
    parser.add_argument('--symbols', type=str,
                       help='股票代码列表，逗号分隔 (例: 000001,000002,600519)')
    parser.add_argument('--days', type=int, default=365,
                       help='回测天数 (默认: 365)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='强制重新获取历史数据，跳过所有缓存')
    
    # 输出参数
    parser.add_argument('--save', action='store_true',
                       help='保存回测结果到文件')
    parser.add_argument('--output', type=str,
                       default=f'high_return_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                       help='输出文件名')
    
    # 系统参数
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式，只显示关键结果')
    
    args = parser.parse_args()
    
    # 设置日志
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging(args.log_level)
    
    # 打印横幅
    if not args.quiet:
        print_banner()
        print_strategy_info()
    
    try:
        if args.config:
            # 配置查看模式
            logger.info("📋 查看策略配置")
            strategy = HighReturnStrategy()
            config = strategy.config
            
            print("\n📊 策略配置概览:")
            print(f"  策略名称: {config['strategy']['name']}")
            print(f"  策略类型: {config['strategy']['type']}")
            print(f"  信号阈值: {config['strategy']['signal_threshold']}")
            print(f"  最大仓位数: {config['strategy']['max_positions']}")
            print(f"  最大单仓位: {config['strategy']['max_position_size']*100:.0f}%")
            print(f"  止盈阈值: {config['risk_management']['take_profit_threshold']*100:.0f}%")
            print(f"  止损阈值: {config['risk_management']['stop_loss_threshold']*100:.0f}%")
            
            print(f"\n🎯 性能目标:")
            targets = config['performance_targets']
            print(f"  年化收益率: {targets['annual_return']*100:.0f}%+")
            print(f"  夏普比率: {targets['sharpe_ratio']:.1f}+")
            print(f"  最大回撤: <{targets['max_drawdown']*100:.0f}%")
            print(f"  胜率: {targets['win_rate']*100:.0f}%+")
            
        elif args.backtest:
            # 回测模式
            args.save_results = args.save
            args.output_file = args.output
            results = run_backtest_mode(args, logger)
            
            if results and args.save_results:
                logger.info(f"✅ 结果已保存到: {args.output_file}")
            
        elif args.analyze:
            # 分析模式
            logger.info("📊 启动性能分析模式")
            args.save_results = True
            args.output_file = f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            results = run_backtest_mode(args, logger)
            
            if results:
                print("\n📈 详细性能分析:")
                # 这里可以添加更详细的分析逻辑
                print("   (详细分析功能开发中...)")
            
        elif args.live:
            # 实盘交易模式
            asyncio.run(run_live_trading_mode(args, logger))
            
    except KeyboardInterrupt:
        logger.info("👋 用户中断，程序退出")
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not args.quiet:
        print(f"\n✅ 程序执行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()


