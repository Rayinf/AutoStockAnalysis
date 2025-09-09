#!/usr/bin/env python3
"""
验证日期范围功能是否正确工作
"""

import requests
import json
from datetime import datetime

def verify_date_range_functionality():
    """验证日期范围功能"""
    base_url = "http://127.0.0.1:8000"
    
    print("🔍 验证日期范围功能是否正确工作")
    print("=" * 60)
    
    # 测试1: 短期日期范围 (1个月)
    print("\n📅 测试1: 短期日期范围 (1个月)")
    short_range_result = test_date_range(
        base_url, "000001", "2024-01-01", "2024-01-31"
    )
    
    # 测试2: 长期日期范围 (6个月)  
    print("\n📅 测试2: 长期日期范围 (6个月)")
    long_range_result = test_date_range(
        base_url, "000001", "2024-01-01", "2024-06-30"
    )
    
    # 测试3: 天数模式 (30天)
    print("\n📅 测试3: 天数模式 (30天)")
    days_result = test_days_mode(base_url, "000001", 30)
    
    # 分析结果
    print("\n" + "=" * 60)
    print("📊 结果分析:")
    
    if short_range_result and long_range_result and days_result:
        short_records = short_range_result.get('total_records', 0)
        long_records = long_range_result.get('total_records', 0)
        days_records = days_result.get('total_records', 0)
        
        print(f"🔸 1个月日期范围: {short_records} 条记录")
        print(f"🔸 6个月日期范围: {long_records} 条记录") 
        print(f"🔸 30天模式: {days_records} 条记录")
        
        # 验证逻辑
        if long_records > short_records:
            print("✅ 日期范围功能正常: 长期范围产生更多记录")
        else:
            print("❌ 日期范围功能异常: 记录数量不符合预期")
            
        if abs(short_records - days_records) <= 5:  # 允许5条记录的差异
            print("✅ 1个月范围与30天模式记录数量接近，符合预期")
        else:
            print(f"⚠️ 1个月范围({short_records})与30天模式({days_records})差异较大")
            
    print("\n🏁 验证完成")

def test_date_range(base_url, symbol, start_date, end_date):
    """测试日期范围模式"""
    try:
        url = f"{base_url}/api/calibration/generate_real_data"
        params = {
            'symbols': symbol,
            'start_date': start_date,
            'end_date': end_date
        }
        
        print(f"📤 请求: {symbol} ({start_date} ~ {end_date})")
        response = requests.post(url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('total_records', 0)
            print(f"📥 成功: {records} 条记录")
            return data
        else:
            print(f"❌ 失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        return None

def test_days_mode(base_url, symbol, days):
    """测试天数模式"""
    try:
        url = f"{base_url}/api/calibration/generate_real_data"
        params = {
            'symbols': symbol,
            'days': days
        }
        
        print(f"📤 请求: {symbol} ({days}天)")
        response = requests.post(url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('total_records', 0)
            print(f"📥 成功: {records} 条记录")
            return data
        else:
            print(f"❌ 失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        return None

if __name__ == "__main__":
    verify_date_range_functionality()
