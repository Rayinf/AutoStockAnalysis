"""
Qlib集成演示模块
基于微软Qlib平台的量化分析功能演示
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# 模拟Qlib的核心功能
class QlibDataAdapter:
    """Qlib数据适配器 - 演示版本"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def convert_symbol(self, symbol: str) -> str:
        """转换股票代码格式"""
        if symbol.startswith("sh") or symbol.startswith("sz"):
            symbol = symbol[2:]
        
        if symbol.startswith("6"):
            return f"SH{symbol}"
        else:
            return f"SZ{symbol}"
    
    def get_features(self, symbol: str, start_date: str, end_date: str, df: pd.DataFrame) -> Dict:
        """基于历史数据计算Qlib风格的特征"""
        if df.empty:
            return {"error": "数据不足"}
        
        try:
            # 基础特征计算
            features = {}
            
            # 价格相关特征
            features["close"] = df["收盘"].values
            features["high"] = df["最高"].values
            features["low"] = df["最低"].values
            features["volume"] = df["成交量"].values if "成交量" in df.columns else np.zeros(len(df))
            
            # Qlib风格的因子特征
            features.update(self._calculate_alpha_factors(df))
            
            return {
                "symbol": symbol,
                "features": features,
                "feature_names": list(features.keys()),
                "data_length": len(df),
                "date_range": f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            self.logger.error(f"特征计算失败: {e}")
            return {"error": str(e)}
    
    def _calculate_alpha_factors(self, df: pd.DataFrame) -> Dict:
        """计算Alpha因子（量化特征）"""
        factors = {}
        
        try:
            close = df["收盘"].values
            high = df["最高"].values
            low = df["最低"].values
            volume = df["成交量"].values if "成交量" in df.columns else np.ones(len(df))
            
            # 动量因子
            factors["momentum_5"] = self._safe_divide(close[4:] - close[:-4], close[:-4])  # 5日动量
            factors["momentum_20"] = self._safe_divide(close[19:] - close[:-19], close[:-19])  # 20日动量
            
            # 反转因子
            factors["reversal_1"] = -np.diff(close) / close[:-1]  # 1日反转
            factors["reversal_5"] = -self._rolling_return(close, 5)  # 5日反转
            
            # 波动率因子
            returns = np.diff(close) / close[:-1]
            factors["volatility_20"] = self._rolling_std(returns, 20)  # 20日波动率
            factors["volatility_60"] = self._rolling_std(returns, 60)  # 60日波动率
            
            # 成交量因子
            factors["volume_ratio"] = volume[1:] / (np.roll(volume, 1)[1:] + 1e-8)  # 成交量比率
            factors["turnover"] = volume / (close + 1e-8)  # 换手率代理
            
            # 价格位置因子
            factors["price_position"] = (close - low) / (high - low + 1e-8)  # 价格在高低点中的位置
            
            # 技术指标因子
            factors["rsi_14"] = self._calculate_rsi(close, 14)
            factors["ma_ratio_5"] = close / self._moving_average(close, 5) - 1  # 与5日均线的偏离
            factors["ma_ratio_20"] = close / self._moving_average(close, 20) - 1  # 与20日均线的偏离
            
            # 对齐所有因子的长度（取最短长度）
            min_length = min(len(v) for v in factors.values() if len(v) > 0)
            factors = {k: v[-min_length:] if len(v) > min_length else v 
                      for k, v in factors.items() if len(v) > 0}
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Alpha因子计算失败: {e}")
            return {}
    
    def _safe_divide(self, a, b):
        """安全除法"""
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    def _rolling_return(self, prices, window):
        """滚动收益率"""
        if len(prices) < window + 1:
            return np.array([])
        return (prices[window:] - prices[:-window]) / prices[:-window]
    
    def _rolling_std(self, returns, window):
        """滚动标准差"""
        if len(returns) < window:
            return np.array([])
        
        result = []
        for i in range(window - 1, len(returns)):
            result.append(np.std(returns[i-window+1:i+1]))
        return np.array(result)
    
    def _moving_average(self, prices, window):
        """移动平均"""
        if len(prices) < window:
            return prices
        
        result = np.convolve(prices, np.ones(window)/window, mode='valid')
        # 扩展到原长度
        extended = np.full(len(prices), np.nan)
        extended[window-1:] = result
        return extended
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # 扩展到原长度
        extended = np.full(len(prices), 50.0)
        extended[period:] = rsi
        return extended


class QlibPredictor:
    """Qlib预测器 - 演示版本"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_weights = None
        self.feature_importance = None
        
    def train_model(self, features: Dict, target_returns: np.ndarray) -> Dict:
        """训练预测模型（简化版）"""
        try:
            # 准备特征矩阵
            X = self._prepare_feature_matrix(features)
            y = target_returns
            
            # 确保数据长度一致
            min_length = min(len(X), len(y))
            X = X[-min_length:]
            y = y[-min_length:]
            
            if len(X) < 20:  # 最少需要20个样本
                return {"error": "训练数据不足"}
            
            # 简单线性回归（演示用）
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # 使用最小二乘法
            try:
                self.model_weights = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                
                # 计算特征重要性（权重的绝对值）
                self.feature_importance = np.abs(self.model_weights[1:])  # 排除截距项
                
                # 计算训练误差
                y_pred = X_with_intercept @ self.model_weights
                mse = np.mean((y - y_pred) ** 2)
                r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
                
                return {
                    "status": "success",
                    "model_type": "LinearRegression",
                    "n_features": len(self.feature_importance),
                    "training_samples": len(X),
                    "mse": float(mse),
                    "r2_score": float(r2),
                    "feature_importance": self.feature_importance.tolist()
                }
                
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用正则化
                lambda_reg = 0.01
                XTX_reg = X_with_intercept.T @ X_with_intercept + lambda_reg * np.eye(X_with_intercept.shape[1])
                self.model_weights = np.linalg.solve(XTX_reg, X_with_intercept.T @ y)
                
                return {
                    "status": "success",
                    "model_type": "RidgeRegression",
                    "n_features": len(X[0]),
                    "training_samples": len(X),
                    "regularized": True
                }
                
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return {"error": str(e)}
    
    def predict(self, features: Dict) -> Dict:
        """进行预测"""
        if self.model_weights is None:
            return {"error": "模型未训练"}
        
        try:
            # 准备特征
            X = self._prepare_feature_matrix(features)
            if len(X) == 0:
                return {"error": "特征数据为空"}
            
            # 使用最新的特征进行预测
            latest_features = X[-1:] if len(X.shape) > 1 else X.reshape(1, -1)
            X_with_intercept = np.column_stack([np.ones(len(latest_features)), latest_features])
            
            # 预测
            prediction = float(X_with_intercept @ self.model_weights)
            
            # 计算置信度（基于特征的稳定性）
            confidence = self._calculate_confidence(latest_features[0])
            
            # 转换为方向预测
            direction = "上涨" if prediction > 0 else "下跌" if prediction < -0.001 else "震荡"
            
            return {
                "prediction_value": prediction,
                "direction": direction,
                "confidence": confidence,
                "signal_strength": min(abs(prediction) * 100, 1.0),  # 信号强度 0-1
                "model_type": "Qlib-Linear"
            }
            
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return {"error": str(e)}
    
    def _prepare_feature_matrix(self, features: Dict) -> np.ndarray:
        """准备特征矩阵"""
        # 选择数值型特征
        numeric_features = []
        feature_names = []
        
        for name, values in features.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                # 确保是数值型
                try:
                    numeric_values = np.array(values, dtype=float)
                    if not np.all(np.isnan(numeric_values)):  # 不全是NaN
                        numeric_features.append(numeric_values)
                        feature_names.append(name)
                except (ValueError, TypeError):
                    continue
        
        if not numeric_features:
            return np.array([])
        
        # 对齐长度
        min_length = min(len(f) for f in numeric_features)
        aligned_features = [f[-min_length:] for f in numeric_features]
        
        # 转置为 (样本数, 特征数)
        X = np.column_stack(aligned_features)
        
        # 处理NaN值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """计算预测置信度"""
        if self.feature_importance is None:
            return 0.5
        
        # 基于特征重要性和特征值的稳定性
        feature_strength = np.sum(np.abs(features) * self.feature_importance[:len(features)])
        
        # 归一化到0-1范围
        confidence = min(0.9, max(0.1, 0.5 + feature_strength * 0.3))
        
        return float(confidence)


class QlibBacktester:
    """Qlib回测器 - 演示版本"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def simple_backtest(self, predictions: List[float], actual_returns: List[float], 
                       initial_capital: float = 100000) -> Dict:
        """简单回测"""
        if len(predictions) != len(actual_returns) or len(predictions) == 0:
            return {"error": "预测和实际收益数据长度不匹配"}
        
        try:
            portfolio_values = [initial_capital]
            positions = []  # 1: 做多, -1: 做空, 0: 空仓
            trades = 0
            winning_trades = 0
            
            for i, (pred, actual) in enumerate(zip(predictions, actual_returns)):
                current_value = portfolio_values[-1]
                
                # 简单策略：预测>0.01做多，<-0.01做空，否则空仓
                if pred > 0.01:
                    position = 1
                elif pred < -0.01:
                    position = -1
                else:
                    position = 0
                
                positions.append(position)
                
                # 计算收益
                if position != 0:
                    trades += 1
                    pnl = position * actual * current_value
                    if pnl > 0:
                        winning_trades += 1
                else:
                    pnl = 0
                
                new_value = current_value + pnl
                portfolio_values.append(new_value)
            
            # 计算统计指标
            total_return = (portfolio_values[-1] - initial_capital) / initial_capital
            portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # 夏普比率（假设无风险利率为0）
            if np.std(portfolio_returns) > 0:
                sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # 最大回撤
            peak = np.maximum.accumulate(portfolio_values)
            drawdowns = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdowns)
            
            # 胜率
            win_rate = winning_trades / trades if trades > 0 else 0
            
            return {
                "total_return": float(total_return),
                "annualized_return": float(total_return * 252 / len(predictions)),  # 假设日频数据
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "total_trades": trades,
                "winning_trades": winning_trades,
                "portfolio_values": [float(v) for v in portfolio_values],
                "final_value": float(portfolio_values[-1])
            }
            
        except Exception as e:
            self.logger.error(f"回测失败: {e}")
            return {"error": str(e)}


# 演示函数
def demo_qlib_integration(symbol: str, df: pd.DataFrame) -> Dict:
    """Qlib集成演示"""
    
    # 初始化组件
    data_adapter = QlibDataAdapter()
    predictor = QlibPredictor()
    backtester = QlibBacktester()
    
    # 1. 数据适配和特征提取
    start_date = df["日期"].iloc[0] if "日期" in df.columns else "2024-01-01"
    end_date = df["日期"].iloc[-1] if "日期" in df.columns else "2024-12-01"
    
    features_result = data_adapter.get_features(symbol, start_date, end_date, df)
    if "error" in features_result:
        return features_result
    
    # 2. 准备目标变量（未来1日收益率）
    returns = df["收盘"].pct_change().fillna(0).values
    future_returns = np.roll(returns, -1)[:-1]  # 未来1日收益，去掉最后一个
    
    # 3. 模型训练
    train_result = predictor.train_model(features_result["features"], future_returns)
    if "error" in train_result:
        return train_result
    
    # 4. 预测
    prediction_result = predictor.predict(features_result["features"])
    if "error" in prediction_result:
        return prediction_result
    
    # 5. 简单回测（如果数据足够）
    if len(future_returns) > 20:
        # 生成历史预测（简化：使用当前模型对历史数据预测）
        historical_predictions = []
        for i in range(20, len(returns)-1):  # 保留足够的历史数据用于特征计算
            # 模拟历史预测（实际应该是滚动训练预测）
            pred_value = predictor.model_weights[0] if predictor.model_weights is not None else 0
            historical_predictions.append(pred_value)
        
        backtest_result = backtester.simple_backtest(
            historical_predictions, 
            future_returns[-len(historical_predictions):].tolist()
        )
    else:
        backtest_result = {"error": "数据不足，无法进行回测"}
    
    # 6. 整合结果
    return {
        "qlib_analysis": {
            "symbol": symbol,
            "data_info": {
                "total_samples": features_result["data_length"],
                "feature_count": len(features_result["feature_names"]),
                "date_range": features_result["date_range"]
            },
            "model_info": train_result,
            "prediction": prediction_result,
            "backtest": backtest_result,
            "feature_names": features_result["feature_names"][:10],  # 只显示前10个特征名
        }
    }


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("/Volumes/PortableSSD/Azune/stock/Auto-GPT-Stock")
    
    # 模拟数据测试
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    test_df = pd.DataFrame({
        '日期': dates.strftime('%Y-%m-%d'),
        '开盘': prices + np.random.randn(100) * 0.5,
        '最高': prices + np.abs(np.random.randn(100) * 0.8),
        '最低': prices - np.abs(np.random.randn(100) * 0.8),
        '收盘': prices,
        '成交量': np.random.randint(1000000, 10000000, 100)
    })
    
    result = demo_qlib_integration("TEST001", test_df)
    print("Qlib集成演示结果:")
    print(f"数据样本数: {result['qlib_analysis']['data_info']['total_samples']}")
    print(f"特征数量: {result['qlib_analysis']['data_info']['feature_count']}")
    print(f"预测方向: {result['qlib_analysis']['prediction']['direction']}")
    print(f"置信度: {result['qlib_analysis']['prediction']['confidence']:.3f}")
    
    if "error" not in result['qlib_analysis']['backtest']:
        print(f"回测收益率: {result['qlib_analysis']['backtest']['total_return']:.3%}")
        print(f"夏普比率: {result['qlib_analysis']['backtest']['sharpe_ratio']:.3f}")

