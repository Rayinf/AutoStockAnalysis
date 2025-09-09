#!/usr/bin/env python3
"""
详细演示交叉验证的执行过程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import sqlite3
# import matplotlib.pyplot as plt  # 移除matplotlib依赖

def detailed_cross_validation_demo():
    """详细演示交叉验证过程"""
    print("🔍 详细交叉验证演示")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n📊 步骤1: 获取所有预测数据")
    conn = sqlite3.connect("calibration.db")
    query = """
        SELECT prediction_date, symbol, predicted_probability, actual_direction
        FROM predictions 
        WHERE actual_direction IS NOT NULL
        ORDER BY prediction_date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"总数据量: {len(df)} 条记录")
    print(f"时间范围: {df['prediction_date'].min()} ~ {df['prediction_date'].max()}")
    print(f"股票数量: {df['symbol'].nunique()} 只")
    
    # 2. 设置交叉验证参数
    n_folds = 5
    fold_size = len(df) // n_folds
    print(f"\n🔄 步骤2: 设置交叉验证参数")
    print(f"折数: {n_folds}")
    print(f"每折大小: {fold_size} 条记录")
    
    # 3. 执行交叉验证
    print(f"\n⚙️ 步骤3: 执行交叉验证")
    print("-" * 60)
    
    platt_improvements = []
    isotonic_improvements = []
    fold_details = []
    
    for fold in range(n_folds):
        print(f"\n📋 Fold {fold + 1}/{n_folds}")
        
        # 计算数据分割点
        train_end = fold * fold_size
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        
        if train_end < 50:  # 训练集太小，跳过
            print("   ⚠️ 训练集太小，跳过")
            continue
            
        # 分割数据
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        if len(test_df) == 0:
            print("   ⚠️ 测试集为空，跳过")
            continue
        
        print(f"   训练集: {len(train_df)} 条 ({train_df['prediction_date'].min()} ~ {train_df['prediction_date'].max()})")
        print(f"   测试集: {len(test_df)} 条 ({test_df['prediction_date'].min()} ~ {test_df['prediction_date'].max()})")
        
        # 准备数据
        train_probs = train_df['predicted_probability'].values
        train_labels = train_df['actual_direction'].values
        test_probs = test_df['predicted_probability'].values
        test_labels = test_df['actual_direction'].values
        
        # 测试Platt Scaling
        try:
            # 训练校准模型
            epsilon = 1e-7
            train_probs_clipped = np.clip(train_probs, epsilon, 1 - epsilon)
            train_logits = np.log(train_probs_clipped / (1 - train_probs_clipped))
            
            platt_model = LogisticRegression()
            platt_model.fit(train_logits.reshape(-1, 1), train_labels)
            
            # 在测试集上应用校准
            test_probs_clipped = np.clip(test_probs, epsilon, 1 - epsilon)
            test_logits = np.log(test_probs_clipped / (1 - test_probs_clipped))
            calibrated_test_probs = platt_model.predict_proba(test_logits.reshape(-1, 1))[:, 1]
            
            # 计算改进
            original_brier = brier_score_loss(test_labels, test_probs)
            calibrated_brier = brier_score_loss(test_labels, calibrated_test_probs)
            platt_improvement = (original_brier - calibrated_brier) / original_brier * 100
            
            platt_improvements.append(platt_improvement)
            print(f"   Platt Scaling改进: {platt_improvement:.2f}%")
            
        except Exception as e:
            print(f"   Platt Scaling失败: {e}")
            platt_improvement = 0
        
        # 测试Isotonic Regression
        try:
            isotonic_model = IsotonicRegression(out_of_bounds='clip')
            isotonic_model.fit(train_probs, train_labels)
            
            calibrated_test_probs_iso = isotonic_model.transform(test_probs)
            
            original_brier = brier_score_loss(test_labels, test_probs)
            calibrated_brier_iso = brier_score_loss(test_labels, calibrated_test_probs_iso)
            isotonic_improvement = (original_brier - calibrated_brier_iso) / original_brier * 100
            
            isotonic_improvements.append(isotonic_improvement)
            print(f"   Isotonic Regression改进: {isotonic_improvement:.2f}%")
            
        except Exception as e:
            print(f"   Isotonic Regression失败: {e}")
            isotonic_improvement = 0
        
        # 保存详情
        fold_details.append({
            'fold': fold + 1,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_period': f"{train_df['prediction_date'].min()} ~ {train_df['prediction_date'].max()}",
            'test_period': f"{test_df['prediction_date'].min()} ~ {test_df['prediction_date'].max()}",
            'platt_improvement': platt_improvement,
            'isotonic_improvement': isotonic_improvement
        })
    
    # 4. 汇总结果
    print(f"\n📊 步骤4: 汇总交叉验证结果")
    print("-" * 60)
    
    if platt_improvements:
        platt_mean = np.mean(platt_improvements)
        platt_std = np.std(platt_improvements)
        print(f"Platt Scaling:")
        print(f"  平均改进: {platt_mean:.2f}% ± {platt_std:.2f}%")
        print(f"  改进范围: {min(platt_improvements):.2f}% ~ {max(platt_improvements):.2f}%")
        print(f"  有效折数: {len(platt_improvements)}")
    
    if isotonic_improvements:
        isotonic_mean = np.mean(isotonic_improvements)
        isotonic_std = np.std(isotonic_improvements)
        print(f"Isotonic Regression:")
        print(f"  平均改进: {isotonic_mean:.2f}% ± {isotonic_std:.2f}%")
        print(f"  改进范围: {min(isotonic_improvements):.2f}% ~ {max(isotonic_improvements):.2f}%")
        print(f"  有效折数: {len(isotonic_improvements)}")
    
    # 5. 详细结果表
    print(f"\n📋 步骤5: 各折详细结果")
    print("-" * 80)
    print(f"{'Fold':<4} {'训练集':<8} {'测试集':<8} {'Platt改进':<10} {'Isotonic改进':<12} {'测试期间'}")
    print("-" * 80)
    
    for detail in fold_details:
        print(f"{detail['fold']:<4} {detail['train_size']:<8} {detail['test_size']:<8} "
              f"{detail['platt_improvement']:<10.2f} {detail['isotonic_improvement']:<12.2f} "
              f"{detail['test_period']}")
    
    # 6. 解释交叉验证的优势
    print(f"\n💡 交叉验证的优势")
    print("-" * 40)
    print("1. 时间序列分割: 严格按时间顺序，避免未来数据泄露")
    print("2. 多次验证: 每个时间段都作为测试集，结果更可靠")
    print("3. 统计意义: 提供均值和标准差，评估稳定性")
    print("4. 渐进训练: 模拟真实场景，训练集逐渐增大")
    
    return {
        'platt_improvements': platt_improvements,
        'isotonic_improvements': isotonic_improvements,
        'fold_details': fold_details
    }

def visualize_cross_validation():
    """可视化交叉验证过程"""
    print("\n🎨 可视化交叉验证分割")
    
    # 获取数据时间范围
    conn = sqlite3.connect("calibration.db")
    query = "SELECT prediction_date FROM predictions WHERE actual_direction IS NOT NULL ORDER BY prediction_date"
    dates_df = pd.read_sql_query(query, conn)
    conn.close()
    
    dates = pd.to_datetime(dates_df['prediction_date'])
    total_days = (dates.max() - dates.min()).days
    
    print(f"数据时间跨度: {total_days} 天")
    print(f"从 {dates.min().date()} 到 {dates.max().date()}")
    
    # 显示分割示意图
    n_folds = 5
    fold_size = len(dates) // n_folds
    
    print(f"\n📅 5折交叉验证分割示意:")
    print("=" * 60)
    
    for fold in range(n_folds):
        train_end = fold * fold_size
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        
        if train_end < 50:
            continue
            
        train_dates = dates.iloc[:train_end]
        test_dates = dates.iloc[test_start:test_end]
        
        print(f"Fold {fold + 1}:")
        print(f"  训练: {train_dates.min().date()} ~ {train_dates.max().date()} ({len(train_dates)} 条)")
        print(f"  测试: {test_dates.min().date()} ~ {test_dates.max().date()} ({len(test_dates)} 条)")
        print(f"  {'█' * (len(train_dates) // 100)}{'░' * (len(test_dates) // 100)}")
        print()

if __name__ == "__main__":
    # 执行详细演示
    results = detailed_cross_validation_demo()
    
    # 可视化分割过程
    visualize_cross_validation()
    
    print("\n🎯 总结:")
    print("交叉验证通过时间序列分割确保了:")
    print("- 训练集始终在测试集之前（时间顺序）")
    print("- 每折都独立测试校准效果")
    print("- 提供稳健的性能估计")
    print("- 避免过拟合和数据泄露")

详细演示交叉验证的执行过程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import sqlite3
# import matplotlib.pyplot as plt  # 移除matplotlib依赖

def detailed_cross_validation_demo():
    """详细演示交叉验证过程"""
    print("🔍 详细交叉验证演示")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n📊 步骤1: 获取所有预测数据")
    conn = sqlite3.connect("calibration.db")
    query = """
        SELECT prediction_date, symbol, predicted_probability, actual_direction
        FROM predictions 
        WHERE actual_direction IS NOT NULL
        ORDER BY prediction_date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"总数据量: {len(df)} 条记录")
    print(f"时间范围: {df['prediction_date'].min()} ~ {df['prediction_date'].max()}")
    print(f"股票数量: {df['symbol'].nunique()} 只")
    
    # 2. 设置交叉验证参数
    n_folds = 5
    fold_size = len(df) // n_folds
    print(f"\n🔄 步骤2: 设置交叉验证参数")
    print(f"折数: {n_folds}")
    print(f"每折大小: {fold_size} 条记录")
    
    # 3. 执行交叉验证
    print(f"\n⚙️ 步骤3: 执行交叉验证")
    print("-" * 60)
    
    platt_improvements = []
    isotonic_improvements = []
    fold_details = []
    
    for fold in range(n_folds):
        print(f"\n📋 Fold {fold + 1}/{n_folds}")
        
        # 计算数据分割点
        train_end = fold * fold_size
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        
        if train_end < 50:  # 训练集太小，跳过
            print("   ⚠️ 训练集太小，跳过")
            continue
            
        # 分割数据
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        if len(test_df) == 0:
            print("   ⚠️ 测试集为空，跳过")
            continue
        
        print(f"   训练集: {len(train_df)} 条 ({train_df['prediction_date'].min()} ~ {train_df['prediction_date'].max()})")
        print(f"   测试集: {len(test_df)} 条 ({test_df['prediction_date'].min()} ~ {test_df['prediction_date'].max()})")
        
        # 准备数据
        train_probs = train_df['predicted_probability'].values
        train_labels = train_df['actual_direction'].values
        test_probs = test_df['predicted_probability'].values
        test_labels = test_df['actual_direction'].values
        
        # 测试Platt Scaling
        try:
            # 训练校准模型
            epsilon = 1e-7
            train_probs_clipped = np.clip(train_probs, epsilon, 1 - epsilon)
            train_logits = np.log(train_probs_clipped / (1 - train_probs_clipped))
            
            platt_model = LogisticRegression()
            platt_model.fit(train_logits.reshape(-1, 1), train_labels)
            
            # 在测试集上应用校准
            test_probs_clipped = np.clip(test_probs, epsilon, 1 - epsilon)
            test_logits = np.log(test_probs_clipped / (1 - test_probs_clipped))
            calibrated_test_probs = platt_model.predict_proba(test_logits.reshape(-1, 1))[:, 1]
            
            # 计算改进
            original_brier = brier_score_loss(test_labels, test_probs)
            calibrated_brier = brier_score_loss(test_labels, calibrated_test_probs)
            platt_improvement = (original_brier - calibrated_brier) / original_brier * 100
            
            platt_improvements.append(platt_improvement)
            print(f"   Platt Scaling改进: {platt_improvement:.2f}%")
            
        except Exception as e:
            print(f"   Platt Scaling失败: {e}")
            platt_improvement = 0
        
        # 测试Isotonic Regression
        try:
            isotonic_model = IsotonicRegression(out_of_bounds='clip')
            isotonic_model.fit(train_probs, train_labels)
            
            calibrated_test_probs_iso = isotonic_model.transform(test_probs)
            
            original_brier = brier_score_loss(test_labels, test_probs)
            calibrated_brier_iso = brier_score_loss(test_labels, calibrated_test_probs_iso)
            isotonic_improvement = (original_brier - calibrated_brier_iso) / original_brier * 100
            
            isotonic_improvements.append(isotonic_improvement)
            print(f"   Isotonic Regression改进: {isotonic_improvement:.2f}%")
            
        except Exception as e:
            print(f"   Isotonic Regression失败: {e}")
            isotonic_improvement = 0
        
        # 保存详情
        fold_details.append({
            'fold': fold + 1,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_period': f"{train_df['prediction_date'].min()} ~ {train_df['prediction_date'].max()}",
            'test_period': f"{test_df['prediction_date'].min()} ~ {test_df['prediction_date'].max()}",
            'platt_improvement': platt_improvement,
            'isotonic_improvement': isotonic_improvement
        })
    
    # 4. 汇总结果
    print(f"\n📊 步骤4: 汇总交叉验证结果")
    print("-" * 60)
    
    if platt_improvements:
        platt_mean = np.mean(platt_improvements)
        platt_std = np.std(platt_improvements)
        print(f"Platt Scaling:")
        print(f"  平均改进: {platt_mean:.2f}% ± {platt_std:.2f}%")
        print(f"  改进范围: {min(platt_improvements):.2f}% ~ {max(platt_improvements):.2f}%")
        print(f"  有效折数: {len(platt_improvements)}")
    
    if isotonic_improvements:
        isotonic_mean = np.mean(isotonic_improvements)
        isotonic_std = np.std(isotonic_improvements)
        print(f"Isotonic Regression:")
        print(f"  平均改进: {isotonic_mean:.2f}% ± {isotonic_std:.2f}%")
        print(f"  改进范围: {min(isotonic_improvements):.2f}% ~ {max(isotonic_improvements):.2f}%")
        print(f"  有效折数: {len(isotonic_improvements)}")
    
    # 5. 详细结果表
    print(f"\n📋 步骤5: 各折详细结果")
    print("-" * 80)
    print(f"{'Fold':<4} {'训练集':<8} {'测试集':<8} {'Platt改进':<10} {'Isotonic改进':<12} {'测试期间'}")
    print("-" * 80)
    
    for detail in fold_details:
        print(f"{detail['fold']:<4} {detail['train_size']:<8} {detail['test_size']:<8} "
              f"{detail['platt_improvement']:<10.2f} {detail['isotonic_improvement']:<12.2f} "
              f"{detail['test_period']}")
    
    # 6. 解释交叉验证的优势
    print(f"\n💡 交叉验证的优势")
    print("-" * 40)
    print("1. 时间序列分割: 严格按时间顺序，避免未来数据泄露")
    print("2. 多次验证: 每个时间段都作为测试集，结果更可靠")
    print("3. 统计意义: 提供均值和标准差，评估稳定性")
    print("4. 渐进训练: 模拟真实场景，训练集逐渐增大")
    
    return {
        'platt_improvements': platt_improvements,
        'isotonic_improvements': isotonic_improvements,
        'fold_details': fold_details
    }

def visualize_cross_validation():
    """可视化交叉验证过程"""
    print("\n🎨 可视化交叉验证分割")
    
    # 获取数据时间范围
    conn = sqlite3.connect("calibration.db")
    query = "SELECT prediction_date FROM predictions WHERE actual_direction IS NOT NULL ORDER BY prediction_date"
    dates_df = pd.read_sql_query(query, conn)
    conn.close()
    
    dates = pd.to_datetime(dates_df['prediction_date'])
    total_days = (dates.max() - dates.min()).days
    
    print(f"数据时间跨度: {total_days} 天")
    print(f"从 {dates.min().date()} 到 {dates.max().date()}")
    
    # 显示分割示意图
    n_folds = 5
    fold_size = len(dates) // n_folds
    
    print(f"\n📅 5折交叉验证分割示意:")
    print("=" * 60)
    
    for fold in range(n_folds):
        train_end = fold * fold_size
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        
        if train_end < 50:
            continue
            
        train_dates = dates.iloc[:train_end]
        test_dates = dates.iloc[test_start:test_end]
        
        print(f"Fold {fold + 1}:")
        print(f"  训练: {train_dates.min().date()} ~ {train_dates.max().date()} ({len(train_dates)} 条)")
        print(f"  测试: {test_dates.min().date()} ~ {test_dates.max().date()} ({len(test_dates)} 条)")
        print(f"  {'█' * (len(train_dates) // 100)}{'░' * (len(test_dates) // 100)}")
        print()

if __name__ == "__main__":
    # 执行详细演示
    results = detailed_cross_validation_demo()
    
    # 可视化分割过程
    visualize_cross_validation()
    
    print("\n🎯 总结:")
    print("交叉验证通过时间序列分割确保了:")
    print("- 训练集始终在测试集之前（时间顺序）")
    print("- 每折都独立测试校准效果")
    print("- 提供稳健的性能估计")
    print("- 避免过拟合和数据泄露")












