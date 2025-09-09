#!/usr/bin/env python3
"""
调试筛选系统问题
"""

import sys
import os
sys.path.append('/Volumes/PortableSSD/Azune/stock/Auto-GPT-Stock')

import asyncio
import logging
from backend.stock_screener import StockScreener, ScreeningCriteria

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

async def debug_screening_issue():
    """调试筛选问题"""
    print("🔍 开始调试筛选系统问题...")
    
    try:
        screener = StockScreener()
        
        # 1. 测试单只股票分析
        print("\n1️⃣ 测试单只股票分析...")
        test_stocks = [
            ('000001', '平安银行'),
            ('000002', '万科A'),
            ('600036', '招商银行')
        ]
        
        for symbol, name in test_stocks:
            print(f"\n分析 {symbol} ({name}):")
            try:
                result = await screener.analyze_single_stock(symbol, name)
                if result:
                    print(f"  ✅ 成功 - 评分: {result.total_score:.2f}")
                    print(f"     技术: {result.technical_score:.1f}, 基本面: {result.fundamental_score:.1f}")
                    print(f"     市场: {result.market_score:.1f}, 风险: {result.risk_score:.1f}")
                    print(f"     市值: {result.market_cap}")
                    print(f"     推荐理由: {result.recommendation_reason[:100]}...")
                else:
                    print("  ❌ 分析失败 - 无结果")
            except Exception as e:
                print(f"  ❌ 分析异常: {e}")
        
        # 2. 测试预筛选
        print("\n2️⃣ 测试预筛选功能...")
        criteria = ScreeningCriteria(top_n=5)
        
        stock_list = await screener.get_all_stock_list()
        print(f"获取股票列表: {len(stock_list)} 只")
        
        prescreened = await screener.get_prescreened_stocks(stock_list, criteria)
        print(f"预筛选结果: {len(prescreened)} 只")
        
        if not prescreened.empty:
            print("前5只预筛选股票:")
            for i, (_, stock) in enumerate(prescreened.head(5).iterrows()):
                print(f"  {i+1}. {stock['code']} ({stock['name']}) - 价格: {stock.get('price', 'N/A')}")
        
        # 3. 测试完整筛选
        print("\n3️⃣ 测试完整筛选...")
        recommendations = await screener.screen_top_stocks(criteria)
        
        print(f"最终推荐: {len(recommendations)} 只")
        
        if recommendations:
            print("推荐详情:")
            for i, rec in enumerate(recommendations):
                print(f"  {i+1}. {rec.symbol} ({rec.name})")
                print(f"     评分: {rec.total_score:.2f} (技术{rec.technical_score:.1f} 基本面{rec.fundamental_score:.1f} 市场{rec.market_score:.1f} 风险{rec.risk_score:.1f})")
                print(f"     价格: ¥{rec.current_price:.2f}, 市值: {rec.market_cap}")
        
        # 4. 检查是否所有评分都相同
        if recommendations:
            scores = [rec.total_score for rec in recommendations]
            unique_scores = set(scores)
            print(f"\n4️⃣ 评分分析:")
            print(f"评分数量: {len(scores)}")
            print(f"唯一评分: {len(unique_scores)}")
            print(f"评分范围: {min(scores):.2f} - {max(scores):.2f}")
            
            if len(unique_scores) == 1:
                print("⚠️  所有股票评分相同，存在问题！")
            else:
                print("✅ 评分正常，有差异化")
        
        return len(recommendations) > 0 and len(set(rec.total_score for rec in recommendations)) > 1
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_screening_issue())
    if success:
        print("\n✅ 筛选系统基本正常")
    else:
        print("\n❌ 筛选系统存在问题，需要修复")

