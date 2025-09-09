#!/usr/bin/env python3
"""
测试自动生成历史数据功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import requests
from backend.calibration import calibrator
from backend.strategy_backtest import StrategyBacktester

def clear_test_data():
    """清除测试股票的数据"""
    print("🧹 清除测试数据")
    
    test_symbols = ['600519', '000001']
    
    conn = sqlite3.connect("calibration.db")
    cursor = conn.cursor()
    
    for symbol in test_symbols:
        cursor.execute("DELETE FROM predictions WHERE symbol = ?", (symbol,))
        print(f"  清除股票 {symbol} 的历史数据")
    
    conn.commit()
    conn.close()
    
    print("✅ 测试数据清除完成")

def verify_no_data():
    """验证确实没有数据"""
    print("\n📊 验证数据状态")
    
    test_start = "2025-07-01"
    test_end = "2025-08-01"
    
    backtester = StrategyBacktester()
    predictions_df = backtester.load_predictions(test_start, test_end, ['600519'])
    
    print(f"  时间范围: {test_start} ~ {test_end}")
    print(f"  股票: 600519")
    print(f"  预测数据量: {len(predictions_df)} 条")
    
    if len(predictions_df) == 0:
        print("✅ 确认没有预测数据，应该触发自动生成")
        return True
    else:
        print("❌ 仍有预测数据，清除可能不完全")
        return False

def test_direct_generation():
    """测试直接调用生成函数"""
    print("\n🔧 测试直接生成历史数据")
    
    test_symbols = "600519"
    test_start = "2025-07-01"
    test_end = "2025-08-01"
    
    try:
        print(f"  调用: generate_historical_backtest_data_by_date_range")
        print(f"  参数: symbols={test_symbols}, start={test_start}, end={test_end}")
        
        result = calibrator.generate_historical_backtest_data_by_date_range(
            test_symbols, test_start, test_end
        )
        
        print(f"  生成结果: {result}")
        
        if isinstance(result, dict):
            success_count = len(result.get("success_symbols", []))
            total_records = result.get("total_records", 0)
            
            if success_count > 0:
                print(f"✅ 直接生成成功: {success_count}只股票, {total_records}条记录")
                return True
            else:
                print(f"❌ 直接生成失败: 没有成功的股票")
                return False
        else:
            print(f"❌ 直接生成失败: 返回格式异常")
            return False
            
    except Exception as e:
        print(f"❌ 直接生成异常: {e}")
        return False

def test_api_auto_generation():
    """测试API自动生成功能"""
    print("\n🌐 测试API自动生成功能")
    
    url = "http://127.0.0.1:8000/api/strategy/run_backtest"
    
    params = {
        "symbols": "600519",
        "start_date": "2025-07-01",
        "end_date": "2025-08-01",
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "use_calibration": False  # 先关闭校准，专注测试数据生成
    }
    
    try:
        print(f"  发送回测请求...")
        print(f"  参数: {params}")
        
        response = requests.post(url, params=params, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'error' in result:
                print(f"❌ API回测失败: {result['error']}")
                return False
            else:
                # 检查是否有自动生成标记
                auto_generated = result.get('auto_generated_data', False)
                generated_records = result.get('generated_records', 0)
                
                if auto_generated:
                    print(f"✅ API自动生成成功: {generated_records}条记录")
                    print(f"  回测结果: {result['performance_metrics']['total_trades']}次交易")
                    return True
                else:
                    print(f"⚠️ API回测成功但未触发自动生成")
                    print(f"  可能原因: 数据已存在或其他逻辑问题")
                    return False
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"  响应: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️ API调用超时（5分钟）")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 无法连接到后端服务")
        return False
    except Exception as e:
        print(f"❌ API测试异常: {e}")
        return False

def verify_generated_data():
    """验证生成的数据"""
    print("\n📈 验证生成的数据")
    
    test_start = "2025-07-01"
    test_end = "2025-08-01"
    
    conn = sqlite3.connect("calibration.db")
    query = """
        SELECT 
            symbol,
            COUNT(*) as records,
            MIN(prediction_date) as start_date,
            MAX(prediction_date) as end_date
        FROM predictions 
        WHERE symbol = '600519'
        AND prediction_date >= ?
        AND prediction_date <= ?
        GROUP BY symbol
    """
    
    import pandas as pd
    df = pd.read_sql_query(query, conn, params=[test_start, test_end])
    conn.close()
    
    if len(df) > 0:
        for _, row in df.iterrows():
            print(f"  股票 {row['symbol']}: {row['records']}条记录")
            print(f"    日期范围: {row['start_date']} ~ {row['end_date']}")
        print("✅ 数据生成验证通过")
        return True
    else:
        print("❌ 没有找到生成的数据")
        return False

def test_backtest_after_generation():
    """测试生成数据后的回测"""
    print("\n🚀 测试生成数据后的回测")
    
    from backend.strategy_backtest import StrategyBacktester, StrategyConfig
    
    config = StrategyConfig(
        initial_capital=100000,
        buy_threshold=0.6,
        sell_threshold=0.4
    )
    
    backtester = StrategyBacktester()
    result = backtester.run_backtest(
        config, "2025-07-01", "2025-08-01", ["600519"], use_calibration=False
    )
    
    if 'error' not in result:
        metrics = result['performance_metrics']
        print(f"✅ 回测成功:")
        print(f"  总收益率: {metrics['total_return']:.2%}")
        print(f"  交易次数: {metrics['total_trades']}")
        print(f"  胜率: {metrics['win_rate']:.1%}")
        return True
    else:
        print(f"❌ 回测失败: {result['error']}")
        return False

def main():
    """主测试流程"""
    print("🧪 自动生成历史数据功能测试")
    print("=" * 60)
    
    # 1. 清除测试数据
    clear_test_data()
    
    # 2. 验证没有数据
    no_data = verify_no_data()
    
    if not no_data:
        print("⚠️ 数据清除不完全，但继续测试")
    
    # 3. 测试直接生成
    direct_success = test_direct_generation()
    
    # 4. 验证生成的数据
    if direct_success:
        data_verified = verify_generated_data()
    else:
        print("⚠️ 直接生成失败，跳过数据验证")
        data_verified = False
    
    # 5. 测试生成后的回测
    if data_verified:
        backtest_success = test_backtest_after_generation()
    else:
        print("⚠️ 数据验证失败，跳过回测测试")
        backtest_success = False
    
    # 6. 清除数据后测试API自动生成
    print("\n" + "="*60)
    print("测试API自动生成功能")
    clear_test_data()  # 重新清除数据
    api_success = test_api_auto_generation()
    
    # 7. 总结
    print(f"\n🎯 测试总结:")
    print(f"  直接生成: {'✅' if direct_success else '❌'}")
    print(f"  数据验证: {'✅' if data_verified else '❌'}")
    print(f"  回测功能: {'✅' if backtest_success else '❌'}")
    print(f"  API自动生成: {'✅' if api_success else '❌'}")
    
    if direct_success and api_success:
        print("🎉 自动生成功能正常工作！")
    else:
        print("⚠️ 自动生成功能存在问题，需要进一步调试")
        
        if not direct_success:
            print("  - 直接生成函数可能有问题")
        if not api_success:
            print("  - API自动生成逻辑可能有问题")

if __name__ == "__main__":
    main()

测试自动生成历史数据功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import requests
from backend.calibration import calibrator
from backend.strategy_backtest import StrategyBacktester

def clear_test_data():
    """清除测试股票的数据"""
    print("🧹 清除测试数据")
    
    test_symbols = ['600519', '000001']
    
    conn = sqlite3.connect("calibration.db")
    cursor = conn.cursor()
    
    for symbol in test_symbols:
        cursor.execute("DELETE FROM predictions WHERE symbol = ?", (symbol,))
        print(f"  清除股票 {symbol} 的历史数据")
    
    conn.commit()
    conn.close()
    
    print("✅ 测试数据清除完成")

def verify_no_data():
    """验证确实没有数据"""
    print("\n📊 验证数据状态")
    
    test_start = "2025-07-01"
    test_end = "2025-08-01"
    
    backtester = StrategyBacktester()
    predictions_df = backtester.load_predictions(test_start, test_end, ['600519'])
    
    print(f"  时间范围: {test_start} ~ {test_end}")
    print(f"  股票: 600519")
    print(f"  预测数据量: {len(predictions_df)} 条")
    
    if len(predictions_df) == 0:
        print("✅ 确认没有预测数据，应该触发自动生成")
        return True
    else:
        print("❌ 仍有预测数据，清除可能不完全")
        return False

def test_direct_generation():
    """测试直接调用生成函数"""
    print("\n🔧 测试直接生成历史数据")
    
    test_symbols = "600519"
    test_start = "2025-07-01"
    test_end = "2025-08-01"
    
    try:
        print(f"  调用: generate_historical_backtest_data_by_date_range")
        print(f"  参数: symbols={test_symbols}, start={test_start}, end={test_end}")
        
        result = calibrator.generate_historical_backtest_data_by_date_range(
            test_symbols, test_start, test_end
        )
        
        print(f"  生成结果: {result}")
        
        if isinstance(result, dict):
            success_count = len(result.get("success_symbols", []))
            total_records = result.get("total_records", 0)
            
            if success_count > 0:
                print(f"✅ 直接生成成功: {success_count}只股票, {total_records}条记录")
                return True
            else:
                print(f"❌ 直接生成失败: 没有成功的股票")
                return False
        else:
            print(f"❌ 直接生成失败: 返回格式异常")
            return False
            
    except Exception as e:
        print(f"❌ 直接生成异常: {e}")
        return False

def test_api_auto_generation():
    """测试API自动生成功能"""
    print("\n🌐 测试API自动生成功能")
    
    url = "http://127.0.0.1:8000/api/strategy/run_backtest"
    
    params = {
        "symbols": "600519",
        "start_date": "2025-07-01",
        "end_date": "2025-08-01",
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "use_calibration": False  # 先关闭校准，专注测试数据生成
    }
    
    try:
        print(f"  发送回测请求...")
        print(f"  参数: {params}")
        
        response = requests.post(url, params=params, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'error' in result:
                print(f"❌ API回测失败: {result['error']}")
                return False
            else:
                # 检查是否有自动生成标记
                auto_generated = result.get('auto_generated_data', False)
                generated_records = result.get('generated_records', 0)
                
                if auto_generated:
                    print(f"✅ API自动生成成功: {generated_records}条记录")
                    print(f"  回测结果: {result['performance_metrics']['total_trades']}次交易")
                    return True
                else:
                    print(f"⚠️ API回测成功但未触发自动生成")
                    print(f"  可能原因: 数据已存在或其他逻辑问题")
                    return False
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"  响应: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️ API调用超时（5分钟）")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 无法连接到后端服务")
        return False
    except Exception as e:
        print(f"❌ API测试异常: {e}")
        return False

def verify_generated_data():
    """验证生成的数据"""
    print("\n📈 验证生成的数据")
    
    test_start = "2025-07-01"
    test_end = "2025-08-01"
    
    conn = sqlite3.connect("calibration.db")
    query = """
        SELECT 
            symbol,
            COUNT(*) as records,
            MIN(prediction_date) as start_date,
            MAX(prediction_date) as end_date
        FROM predictions 
        WHERE symbol = '600519'
        AND prediction_date >= ?
        AND prediction_date <= ?
        GROUP BY symbol
    """
    
    import pandas as pd
    df = pd.read_sql_query(query, conn, params=[test_start, test_end])
    conn.close()
    
    if len(df) > 0:
        for _, row in df.iterrows():
            print(f"  股票 {row['symbol']}: {row['records']}条记录")
            print(f"    日期范围: {row['start_date']} ~ {row['end_date']}")
        print("✅ 数据生成验证通过")
        return True
    else:
        print("❌ 没有找到生成的数据")
        return False

def test_backtest_after_generation():
    """测试生成数据后的回测"""
    print("\n🚀 测试生成数据后的回测")
    
    from backend.strategy_backtest import StrategyBacktester, StrategyConfig
    
    config = StrategyConfig(
        initial_capital=100000,
        buy_threshold=0.6,
        sell_threshold=0.4
    )
    
    backtester = StrategyBacktester()
    result = backtester.run_backtest(
        config, "2025-07-01", "2025-08-01", ["600519"], use_calibration=False
    )
    
    if 'error' not in result:
        metrics = result['performance_metrics']
        print(f"✅ 回测成功:")
        print(f"  总收益率: {metrics['total_return']:.2%}")
        print(f"  交易次数: {metrics['total_trades']}")
        print(f"  胜率: {metrics['win_rate']:.1%}")
        return True
    else:
        print(f"❌ 回测失败: {result['error']}")
        return False

def main():
    """主测试流程"""
    print("🧪 自动生成历史数据功能测试")
    print("=" * 60)
    
    # 1. 清除测试数据
    clear_test_data()
    
    # 2. 验证没有数据
    no_data = verify_no_data()
    
    if not no_data:
        print("⚠️ 数据清除不完全，但继续测试")
    
    # 3. 测试直接生成
    direct_success = test_direct_generation()
    
    # 4. 验证生成的数据
    if direct_success:
        data_verified = verify_generated_data()
    else:
        print("⚠️ 直接生成失败，跳过数据验证")
        data_verified = False
    
    # 5. 测试生成后的回测
    if data_verified:
        backtest_success = test_backtest_after_generation()
    else:
        print("⚠️ 数据验证失败，跳过回测测试")
        backtest_success = False
    
    # 6. 清除数据后测试API自动生成
    print("\n" + "="*60)
    print("测试API自动生成功能")
    clear_test_data()  # 重新清除数据
    api_success = test_api_auto_generation()
    
    # 7. 总结
    print(f"\n🎯 测试总结:")
    print(f"  直接生成: {'✅' if direct_success else '❌'}")
    print(f"  数据验证: {'✅' if data_verified else '❌'}")
    print(f"  回测功能: {'✅' if backtest_success else '❌'}")
    print(f"  API自动生成: {'✅' if api_success else '❌'}")
    
    if direct_success and api_success:
        print("🎉 自动生成功能正常工作！")
    else:
        print("⚠️ 自动生成功能存在问题，需要进一步调试")
        
        if not direct_success:
            print("  - 直接生成函数可能有问题")
        if not api_success:
            print("  - API自动生成逻辑可能有问题")

if __name__ == "__main__":
    main()












