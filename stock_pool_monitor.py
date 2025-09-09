#!/usr/bin/env python3
"""
股票池变化监控脚本
用于诊断股票池自动变化的问题
"""

import json
import os
import time
import hashlib
from datetime import datetime

CONFIG_FILE = "stock_pools_config.json"
LOG_FILE = "stock_pool_changes.log"

def get_file_hash(filepath):
    """获取文件的MD5哈希值"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_stock_pools():
    """加载股票池配置"""
    if not os.path.exists(CONFIG_FILE):
        return None
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return f"Error: {e}"

def log_change(message):
    """记录变化日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def monitor_stock_pools(interval=10):
    """监控股票池变化"""
    print(f"开始监控股票池配置文件: {CONFIG_FILE}")
    print(f"检查间隔: {interval}秒")
    print(f"日志文件: {LOG_FILE}")
    print("-" * 50)
    
    last_hash = None
    last_pools = None
    
    while True:
        current_hash = get_file_hash(CONFIG_FILE)
        current_pools = load_stock_pools()
        
        if last_hash is None:
            # 首次运行
            log_change("开始监控")
            if current_hash:
                log_change(f"初始文件哈希: {current_hash}")
                if isinstance(current_pools, dict):
                    pool_names = list(current_pools.keys())
                    log_change(f"初始股票池: {pool_names}")
                    if '我的自选' in current_pools:
                        my_stocks = current_pools['我的自选']['symbols']
                        log_change(f"我的自选股票({len(my_stocks)}只): {my_stocks}")
            else:
                log_change("配置文件不存在")
        else:
            # 检查变化
            if current_hash != last_hash:
                log_change(f"❗ 检测到文件变化!")
                log_change(f"旧哈希: {last_hash}")
                log_change(f"新哈希: {current_hash}")
                
                if isinstance(current_pools, dict) and isinstance(last_pools, dict):
                    # 详细对比变化
                    old_names = set(last_pools.keys())
                    new_names = set(current_pools.keys())
                    
                    if old_names != new_names:
                        added = new_names - old_names
                        removed = old_names - new_names
                        if added:
                            log_change(f"新增股票池: {list(added)}")
                        if removed:
                            log_change(f"删除股票池: {list(removed)}")
                    
                    # 检查我的自选变化
                    if '我的自选' in last_pools and '我的自选' in current_pools:
                        old_stocks = set(last_pools['我的自选']['symbols'])
                        new_stocks = set(current_pools['我的自选']['symbols'])
                        
                        if old_stocks != new_stocks:
                            added_stocks = new_stocks - old_stocks
                            removed_stocks = old_stocks - new_stocks
                            log_change(f"我的自选变化:")
                            if added_stocks:
                                log_change(f"  新增: {list(added_stocks)}")
                            if removed_stocks:
                                log_change(f"  删除: {list(removed_stocks)}")
                            log_change(f"  当前: {list(new_stocks)}")
        
        last_hash = current_hash
        last_pools = current_pools
        time.sleep(interval)

if __name__ == "__main__":
    try:
        monitor_stock_pools()
    except KeyboardInterrupt:
        log_change("监控停止")
        print("\n监控已停止")
