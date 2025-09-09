#!/usr/bin/env python3
"""
股票池数据库管理模块
将股票池从JSON文件迁移到SQLite数据库
"""

import sqlite3
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class StockPoolDatabase:
    """股票池数据库管理类"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # 使用绝对路径，确保数据库文件位置固定
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "stock_pools.db")
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以像字典一样访问
        return conn
    
    def init_database(self):
        """初始化数据库表"""
        with self.get_connection() as conn:
            # 创建股票池表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_pools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(50) UNIQUE NOT NULL,
                    display_name VARCHAR(100) NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建股票池股票关联表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_pool_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pool_id INTEGER NOT NULL,
                    stock_symbol VARCHAR(10) NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pool_id) REFERENCES stock_pools(id) ON DELETE CASCADE,
                    UNIQUE(pool_id, stock_symbol)
                )
            ''')
            
            # 创建操作日志表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_pool_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pool_name VARCHAR(50) NOT NULL,
                    operation VARCHAR(20) NOT NULL,  -- CREATE, UPDATE, DELETE, ADD_STOCK, REMOVE_STOCK
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_pool_name ON stock_pools(name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_pool_stocks ON stock_pool_stocks(pool_id, stock_symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_logs_pool ON stock_pool_logs(pool_name, timestamp)')
            
            conn.commit()
            logger.info("数据库表初始化完成")
    
    def log_operation(self, pool_name: str, operation: str, details: str = ""):
        """记录操作日志"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO stock_pool_logs (pool_name, operation, details)
                VALUES (?, ?, ?)
            ''', (pool_name, operation, details))
            conn.commit()
    
    def get_all_pools(self) -> Dict[str, Dict]:
        """获取所有股票池（返回与原JSON格式兼容的结构）"""
        pools = {}
        
        with self.get_connection() as conn:
            # 获取所有股票池
            pool_rows = conn.execute('''
                SELECT id, name, display_name, description, created_at, updated_at
                FROM stock_pools
                ORDER BY name
            ''').fetchall()
            
            for pool in pool_rows:
                # 获取该股票池的所有股票
                stock_rows = conn.execute('''
                    SELECT stock_symbol
                    FROM stock_pool_stocks
                    WHERE pool_id = ?
                    ORDER BY added_at
                ''', (pool['id'],)).fetchall()
                
                stocks = [row['stock_symbol'] for row in stock_rows]
                
                pools[pool['name']] = {
                    'name': pool['display_name'],
                    'symbols': stocks,
                    'description': pool['description'],
                    'created_at': pool['created_at'],
                    'updated_at': pool['updated_at']
                }
        
        logger.info(f"从数据库加载了{len(pools)}个股票池")
        return pools
    
    def save_pool(self, name: str, display_name: str, symbols: List[str], description: str = "") -> bool:
        """保存股票池（创建或更新）"""
        try:
            with self.get_connection() as conn:
                # 检查股票池是否存在
                existing = conn.execute(
                    'SELECT id FROM stock_pools WHERE name = ?', (name,)
                ).fetchone()
                
                if existing:
                    # 更新现有股票池
                    pool_id = existing['id']
                    conn.execute('''
                        UPDATE stock_pools 
                        SET display_name = ?, description = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (display_name, description, pool_id))
                    
                    # 删除现有股票
                    conn.execute('DELETE FROM stock_pool_stocks WHERE pool_id = ?', (pool_id,))
                    operation = "UPDATE"
                else:
                    # 创建新股票池
                    cursor = conn.execute('''
                        INSERT INTO stock_pools (name, display_name, description)
                        VALUES (?, ?, ?)
                    ''', (name, display_name, description))
                    pool_id = cursor.lastrowid
                    operation = "CREATE"
                
                # 添加股票
                for symbol in symbols:
                    conn.execute('''
                        INSERT INTO stock_pool_stocks (pool_id, stock_symbol)
                        VALUES (?, ?)
                    ''', (pool_id, symbol))
                
                conn.commit()
                
                # 记录日志
                self.log_operation(name, operation, f"包含{len(symbols)}只股票: {symbols}")
                
                logger.info(f"成功保存股票池 '{name}'，包含{len(symbols)}只股票")
                return True
                
        except Exception as e:
            logger.error(f"保存股票池失败: {e}")
            return False
    
    def delete_pool(self, name: str) -> bool:
        """删除股票池"""
        try:
            with self.get_connection() as conn:
                # 获取股票池信息用于日志
                pool_info = conn.execute('''
                    SELECT p.id, p.display_name, COUNT(s.stock_symbol) as stock_count
                    FROM stock_pools p
                    LEFT JOIN stock_pool_stocks s ON p.id = s.pool_id
                    WHERE p.name = ?
                    GROUP BY p.id
                ''', (name,)).fetchone()
                
                if not pool_info:
                    logger.warning(f"股票池 '{name}' 不存在")
                    return False
                
                # 删除股票池（外键约束会自动删除相关股票）
                conn.execute('DELETE FROM stock_pools WHERE name = ?', (name,))
                conn.commit()
                
                # 记录日志
                self.log_operation(name, "DELETE", f"删除了包含{pool_info['stock_count']}只股票的股票池")
                
                logger.info(f"成功删除股票池 '{name}'")
                return True
                
        except Exception as e:
            logger.error(f"删除股票池失败: {e}")
            return False
    
    def add_stock_to_pool(self, pool_name: str, symbol: str) -> bool:
        """向股票池添加股票"""
        try:
            with self.get_connection() as conn:
                # 获取股票池ID
                pool = conn.execute(
                    'SELECT id FROM stock_pools WHERE name = ?', (pool_name,)
                ).fetchone()
                
                if not pool:
                    logger.error(f"股票池 '{pool_name}' 不存在")
                    return False
                
                # 添加股票（IGNORE重复）
                conn.execute('''
                    INSERT OR IGNORE INTO stock_pool_stocks (pool_id, stock_symbol)
                    VALUES (?, ?)
                ''', (pool['id'], symbol))
                
                conn.commit()
                self.log_operation(pool_name, "ADD_STOCK", f"添加股票 {symbol}")
                
                logger.info(f"成功向股票池 '{pool_name}' 添加股票 {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"添加股票失败: {e}")
            return False
    
    def remove_stock_from_pool(self, pool_name: str, symbol: str) -> bool:
        """从股票池移除股票"""
        try:
            with self.get_connection() as conn:
                # 获取股票池ID
                pool = conn.execute(
                    'SELECT id FROM stock_pools WHERE name = ?', (pool_name,)
                ).fetchone()
                
                if not pool:
                    logger.error(f"股票池 '{pool_name}' 不存在")
                    return False
                
                # 删除股票
                cursor = conn.execute('''
                    DELETE FROM stock_pool_stocks 
                    WHERE pool_id = ? AND stock_symbol = ?
                ''', (pool['id'], symbol))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self.log_operation(pool_name, "REMOVE_STOCK", f"移除股票 {symbol}")
                    logger.info(f"成功从股票池 '{pool_name}' 移除股票 {symbol}")
                    return True
                else:
                    logger.warning(f"股票池 '{pool_name}' 中不存在股票 {symbol}")
                    return False
                
        except Exception as e:
            logger.error(f"移除股票失败: {e}")
            return False
    
    def get_pool_history(self, pool_name: str, limit: int = 50) -> List[Dict]:
        """获取股票池操作历史"""
        with self.get_connection() as conn:
            rows = conn.execute('''
                SELECT operation, details, timestamp
                FROM stock_pool_logs
                WHERE pool_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (pool_name, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    def migrate_from_json(self, json_file_path: str) -> bool:
        """从JSON文件迁移数据到数据库"""
        try:
            if not os.path.exists(json_file_path):
                logger.error(f"JSON文件不存在: {json_file_path}")
                return False
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            logger.info(f"开始迁移{len(json_data)}个股票池从JSON到数据库")
            
            success_count = 0
            for pool_name, pool_data in json_data.items():
                display_name = pool_data.get('name', pool_name)
                symbols = pool_data.get('symbols', [])
                description = pool_data.get('description', '')
                
                if self.save_pool(pool_name, display_name, symbols, description):
                    success_count += 1
                    logger.info(f"✅ 迁移股票池 '{pool_name}' 成功")
                else:
                    logger.error(f"❌ 迁移股票池 '{pool_name}' 失败")
            
            logger.info(f"迁移完成: {success_count}/{len(json_data)} 个股票池成功迁移")
            return success_count == len(json_data)
            
        except Exception as e:
            logger.error(f"迁移失败: {e}")
            return False
    
    def backup_to_json(self, backup_file_path: str) -> bool:
        """备份数据库到JSON文件"""
        try:
            pools_data = self.get_all_pools()
            
            # 转换为原始JSON格式（移除数据库特有字段）
            json_data = {}
            for name, data in pools_data.items():
                json_data[name] = {
                    'name': data['name'],
                    'symbols': data['symbols'],
                    'description': data['description']
                }
            
            with open(backup_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据库备份到JSON文件完成: {backup_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"备份失败: {e}")
            return False

# 全局数据库实例
_db_instance = None

def get_stock_pool_db() -> StockPoolDatabase:
    """获取股票池数据库实例（单例模式）"""
    global _db_instance
    if _db_instance is None:
        _db_instance = StockPoolDatabase()
    return _db_instance
