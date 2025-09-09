#!/usr/bin/env python3
"""
测试日期范围生成历史数据功能
"""

import requests
import json
from datetime import datetime, timedelta

def test_date_range_generation():
    """测试日期范围数据生成"""
    base_url = "http://127.0.0.1:8000"
    
    # 设置测试参数
    symbols = "000001,600519"
    start_date = "2024-01-01"
    end_date = "2024-06-30"
    
    print(f"🧪 测试日期范围数据生成")
    print(f"股票代码: {symbols}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print("-" * 50)
    
    try:
        # 构建请求URL
        url = f"{base_url}/api/calibration/generate_real_data"
        params = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date
        }
        
        print(f"🚀 发送请求: {url}")
        print(f"📋 参数: {params}")
        
        # 发送POST请求
        response = requests.post(url, params=params, timeout=60)
        
        print(f"📊 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 请求成功!")
            print(f"📈 响应数据:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # 检查是否使用了日期范围
            if 'date_range' in data or 'start_date' in str(data):
                print("✅ 日期范围参数已正确处理")
            else:
                print("❌ 可能仍在使用默认天数模式")
                
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端服务，请确保后端已启动")
    except Exception as e:
        print(f"❌ 测试异常: {e}")

def test_days_mode():
    """测试天数模式数据生成"""
    base_url = "http://127.0.0.1:8000"
    
    symbols = "000001"
    days = 50
    
    print(f"\n🧪 测试天数模式数据生成")
    print(f"股票代码: {symbols}")
    print(f"天数: {days}")
    print("-" * 50)
    
    try:
        url = f"{base_url}/api/calibration/generate_real_data"
        params = {
            'symbols': symbols,
            'days': days
        }
        
        print(f"🚀 发送请求: {url}")
        print(f"📋 参数: {params}")
        
        response = requests.post(url, params=params, timeout=60)
        
        print(f"📊 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 请求成功!")
            print(f"📈 响应数据:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")

if __name__ == "__main__":
    print("🔍 开始测试历史数据生成API")
    print("=" * 60)
    
    # 测试日期范围模式
    test_date_range_generation()
    
    # 测试天数模式
    test_days_mode()
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")
