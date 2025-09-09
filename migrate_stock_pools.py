#!/usr/bin/env python3
"""
股票池数据迁移脚本
从JSON文件迁移到SQLite数据库
"""

import os
import sys
import logging
import shutil
from datetime import datetime

# 添加backend目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from stock_pool_db import StockPoolDatabase

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主迁移流程"""
    
    # 配置文件路径
    json_file = "stock_pools_config.json"
    backup_file = f"stock_pools_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print("=" * 60)
    print("🔄 股票池数据迁移工具")
    print("从JSON文件迁移到SQLite数据库")
    print("=" * 60)
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_file):
        logger.error(f"❌ JSON配置文件不存在: {json_file}")
        return False
    
    # 创建数据库实例
    logger.info("📊 初始化SQLite数据库...")
    db = StockPoolDatabase()
    
    # 创建JSON文件备份
    logger.info(f"💾 创建JSON文件备份: {backup_file}")
    shutil.copy2(json_file, backup_file)
    
    # 执行迁移
    logger.info(f"🔄 开始迁移数据从 {json_file} 到数据库...")
    success = db.migrate_from_json(json_file)
    
    if success:
        logger.info("✅ 迁移成功完成!")
        
        # 验证迁移结果
        logger.info("🔍 验证迁移结果...")
        db_pools = db.get_all_pools()
        
        print(f"\n📊 迁移结果统计:")
        print(f"   股票池总数: {len(db_pools)}")
        
        for name, data in db_pools.items():
            print(f"   • {name}: {len(data['symbols'])}只股票")
        
        # 创建数据库备份
        db_backup_file = f"stock_pools_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logger.info(f"💾 创建数据库备份: {db_backup_file}")
        db.backup_to_json(db_backup_file)
        
        print(f"\n🎉 迁移成功!")
        print(f"   原JSON文件备份: {backup_file}")
        print(f"   数据库备份文件: {db_backup_file}")
        print(f"   数据库文件: stock_pools.db")
        
        # 询问是否重命名原文件
        print(f"\n⚠️  建议操作:")
        print(f"   1. 原JSON文件已备份到: {backup_file}")
        print(f"   2. 可以将原文件重命名以避免冲突")
        
        response = input(f"\n是否将 {json_file} 重命名为 {json_file}.old? (y/n): ")
        if response.lower() in ['y', 'yes']:
            old_file = f"{json_file}.old"
            os.rename(json_file, old_file)
            logger.info(f"✅ 原文件已重命名为: {old_file}")
        
        return True
        
    else:
        logger.error("❌ 迁移失败!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  迁移被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 迁移过程中发生错误: {e}")
        sys.exit(1)
