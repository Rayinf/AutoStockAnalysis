#!/usr/bin/env python3
"""
实盘交易启动脚本
"""

import asyncio
import yaml
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from live_trading_engine import LiveTradingEngine, LiveTradingConfig

def setup_logging(config):
    """设置日志"""
    log_config = config.get('logging', {})
    
    # 创建日志目录
    log_file = log_config.get('file_path', 'logs/trading.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger

def load_config(config_file='trading_config.yaml'):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_file} 不存在")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"配置文件格式错误: {e}")
        sys.exit(1)

def create_trading_config(config_dict):
    """创建交易配置对象"""
    trading_config = LiveTradingConfig()
    
    # 策略参数
    strategy = config_dict.get('strategy', {})
    trading_config.buy_threshold = strategy.get('buy_threshold', 0.6)
    trading_config.sell_threshold = strategy.get('sell_threshold', 0.4)
    trading_config.max_positions = strategy.get('max_positions', 10)
    trading_config.position_size = strategy.get('position_size', 0.1)
    trading_config.initial_capital = strategy.get('initial_capital', 100000)
    trading_config.transaction_cost = strategy.get('transaction_cost', 0.002)
    
    # 风险控制参数
    risk = config_dict.get('risk_management', {})
    trading_config.stop_loss_threshold = risk.get('stop_loss_threshold', 0.03)
    trading_config.max_single_position = risk.get('max_single_position', 0.15)
    trading_config.max_sector_concentration = risk.get('max_sector_concentration', 0.3)
    trading_config.max_holding_days = risk.get('max_holding_days', 15)
    trading_config.max_daily_loss = risk.get('max_daily_loss', 0.02)
    
    # 系统参数
    system = config_dict.get('system', {})
    trading_config.data_update_interval = system.get('data_update_interval', 60)
    trading_config.signal_check_interval = system.get('signal_check_interval', 300)
    trading_config.order_timeout = system.get('order_timeout', 300)
    trading_config.market_open_time = system.get('market_open_time', "09:30")
    trading_config.market_close_time = system.get('market_close_time', "15:00")
    
    return trading_config

def print_startup_info(config, trading_config):
    """打印启动信息"""
    print("=" * 60)
    print("🚀 实盘交易系统启动")
    print("=" * 60)
    print(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 初始资金: ¥{trading_config.initial_capital:,.2f}")
    print(f"📊 买入阈值: {trading_config.buy_threshold}")
    print(f"📉 卖出阈值: {trading_config.sell_threshold}")
    print(f"📈 最大持仓: {trading_config.max_positions} 只")
    print(f"⚠️  止损阈值: {trading_config.stop_loss_threshold:.1%}")
    print(f"🕘 交易时间: {trading_config.market_open_time} - {trading_config.market_close_time}")
    
    # 监控股票列表
    watchlist = config.get('watchlist', {}).get('default_symbols', [])
    print(f"👀 监控股票: {', '.join(watchlist)}")
    
    # 风险控制
    print("\n🛡️  风险控制:")
    print(f"   • 单股最大仓位: {trading_config.max_single_position:.1%}")
    print(f"   • 日最大亏损: {trading_config.max_daily_loss:.1%}")
    print(f"   • 最大持仓天数: {trading_config.max_holding_days} 天")
    
    # 运行模式
    paper_trading = config.get('paper_trading', {}).get('enabled', False)
    auto_trading = config.get('automation', {}).get('auto_trading', False)
    
    print(f"\n🎮 运行模式:")
    print(f"   • 纸面交易: {'✅ 开启' if paper_trading else '❌ 关闭'}")
    print(f"   • 自动交易: {'✅ 开启' if auto_trading else '❌ 关闭'}")
    
    print("=" * 60)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实盘交易系统')
    parser.add_argument('--config', '-c', default='trading_config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--paper', '-p', action='store_true',
                       help='启用纸面交易模式')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='干运行模式（不执行实际交易）')
    parser.add_argument('--watchlist', '-w', nargs='+',
                       help='自定义监控股票列表')
    
    args = parser.parse_args()
    
    # 加载配置
    config_dict = load_config(args.config)
    
    # 设置日志
    logger = setup_logging(config_dict)
    
    # 创建交易配置
    trading_config = create_trading_config(config_dict)
    
    # 打印启动信息
    print_startup_info(config_dict, trading_config)
    
    # 创建交易引擎
    engine = LiveTradingEngine(trading_config)
    
    # 添加监控股票
    if args.watchlist:
        watchlist = args.watchlist
    else:
        watchlist = config_dict.get('watchlist', {}).get('default_symbols', [])
    
    engine.add_to_watchlist(watchlist)
    
    # 运行前检查
    if args.dry_run:
        logger.info("🔍 干运行模式：将生成信号但不执行实际交易")
    elif args.paper:
        logger.info("📝 纸面交易模式：使用虚拟资金进行交易")
    else:
        logger.info("💰 实盘交易模式：将使用真实资金进行交易")
        
        # 实盘交易确认
        if not config_dict.get('automation', {}).get('auto_trading', False):
            confirm = input("\n⚠️  即将开始实盘交易，请确认 (输入 'YES' 继续): ")
            if confirm != 'YES':
                logger.info("用户取消交易")
                return
    
    # 启动交易引擎
    try:
        logger.info("🎯 交易引擎启动中...")
        await engine.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("👋 收到停止信号，正在安全关闭...")
    except Exception as e:
        logger.error(f"❌ 交易引擎异常: {e}")
        raise
    finally:
        # 打印最终摘要
        summary = engine.get_performance_summary()
        logger.info("📊 交易摘要:")
        logger.info(f"   总资产: ¥{summary['total_capital']:,.2f}")
        logger.info(f"   现金: ¥{summary['cash']:,.2f}")
        logger.info(f"   持仓市值: ¥{summary['market_value']:,.2f}")
        logger.info(f"   未实现盈亏: ¥{summary['unrealized_pnl']:,.2f}")
        logger.info(f"   总交易次数: {summary['total_trades']}")
        logger.info(f"   当前持仓数: {summary['positions_count']}")
        
        print("\n" + "=" * 60)
        print("🏁 交易引擎已安全关闭")
        print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        sys.exit(1)
