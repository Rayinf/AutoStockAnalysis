#!/usr/bin/env python3
"""
测试自动生成历史数据功能
"""

import requests
import json
import sqlite3
from datetime import datetime

def test_auto_generate_backtest():
    """测试运行回测时自动生成历史数据"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 测试自动生成历史数据功能")
    print("=" * 60)
    
    # 首先清空数据库中的预测数据（模拟没有历史数据的情况）
    print("🗑️ 清空现有预测数据...")
    try:
        conn = sqlite3.connect("/Volumes/PortableSSD/Azune/stock/Auto-GPT-Stock/calibration.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE symbol IN ('000001', '600519')")
        conn.commit()
        conn.close()
        print("✅ 预测数据已清空")
    except Exception as e:
        print(f"⚠️ 清空数据失败: {e}")
    
    # 测试参数
    test_params = {
        'buy_threshold': 0.6,
        'sell_threshold': 0.4,
        'max_positions': 5,
        'initial_capital': 50000,
        'symbols': '000001,600519',
        'start_date': '2024-01-01',
        'end_date': '2024-03-31'
    }
    
    print(f"📋 测试参数:")
    for key, value in test_params.items():
        print(f"   {key}: {value}")
    print("-" * 50)
    
    try:
        # 发送回测请求
        url = f"{base_url}/api/strategy/run_backtest"
        
        print(f"🚀 发送回测请求: {url}")
        
        response = requests.post(url, params=test_params, timeout=120)
        
        print(f"📊 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 请求成功!")
            
            # 检查是否自动生成了数据
            if data.get('auto_generated_data'):
                print("\n🤖 自动生成历史数据成功!")
                print(f"   生成记录数: {data.get('generated_records', 0)}")
                print(f"   涉及股票: {', '.join(data.get('generated_symbols', []))}")
                print(f"   提示信息: {data.get('message', '无')}")
                
                # 检查回测结果
                if 'performance_metrics' in data:
                    metrics = data['performance_metrics']
                    print(f"\n📈 回测结果:")
                    print(f"   总收益率: {metrics.get('total_return', 0):.4f}")
                    print(f"   夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"   总交易数: {data.get('total_trades', 0)}")
                    print(f"   最终资产: ¥{data.get('final_portfolio_value', 0):,.2f}")
                else:
                    print("❌ 回测结果不完整")
            else:
                print("⚠️ 没有自动生成数据，可能已有历史数据")
                
            # 显示完整响应（调试用）
            print(f"\n📄 完整响应:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
                
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务，请确保后端已启动")
    except Exception as e:
        print(f"❌ 测试异常: {e}")

def check_generated_data():
    """检查生成的历史数据"""
    print("\n🔍 检查生成的历史数据")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect("/Volumes/PortableSSD/Azune/stock/Auto-GPT-Stock/calibration.db")
        cursor = conn.cursor()
        
        # 查询数据统计
        cursor.execute("""
            SELECT symbol, COUNT(*) as count, MIN(prediction_date) as start_date, MAX(prediction_date) as end_date
            FROM predictions 
            WHERE symbol IN ('000001', '600519')
            GROUP BY symbol
        """)
        
        results = cursor.fetchall()
        
        if results:
            print("📊 数据统计:")
            for symbol, count, start_date, end_date in results:
                print(f"   {symbol}: {count} 条记录 ({start_date} ~ {end_date})")
        else:
            print("❌ 没有找到相关数据")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 检查数据失败: {e}")

if __name__ == "__main__":
    print("🔍 开始测试自动生成历史数据功能")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 测试自动生成功能
    test_auto_generate_backtest()
    
    # 检查生成的数据
    check_generated_data()
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")
