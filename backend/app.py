from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# 生产环境配置
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
PORT = int(os.getenv('PORT', 8000))

from Models.Factory import ChatModelFactory
from Tools.stock_quote import StockAnalyser, StockTool
from backend.qlib_provider import QlibProvider
from backend.feature_store_cache import feature_store
from backend.features import build_factors
from backend.strategy_backtest import backtester, StrategyConfig
import traceback
import logging

# 配置日志
logger = logging.getLogger(__name__)
import sqlite3
from typing import List, Dict, Any, Optional
from fastapi.responses import StreamingResponse

# 导入持仓管理模块
from backend.portfolio_manager import portfolio_manager, Position, Transaction, Portfolio

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except Exception:
    ak = None
    pd = None
    np = None

# Qlib集成
try:
    from backend.qlib_demo import demo_qlib_integration, QlibDataAdapter, QlibPredictor
    QLIB_AVAILABLE = True
except Exception:
    QLIB_AVAILABLE = False

app = FastAPI(title="Auto-GPT-Stock API", version="0.1.0")

# CORS配置 - 允许GitHub Pages和本地开发访问
if ENVIRONMENT == 'production':
    # 生产环境：允许GitHub Pages域名和Railway域名
    cors_origins = [
        "https://rayinf.github.io",  # GitHub Pages域名
        "https://autostockanalysis-production.up.railway.app",  # Railway域名
        "http://localhost:5173",  # 本地开发
        "http://127.0.0.1:5173",  # 本地开发
    ]
    # 如果设置了ALLOWED_ORIGINS环境变量，则合并使用
    if ALLOWED_ORIGINS != ['*']:
        cors_origins.extend(ALLOWED_ORIGINS)
else:
    # 开发环境：允许所有来源
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化QlibProvider（惰性单例，可在 /api/qlib/status 查询）
_qlib = QlibProvider.get()

# 自定义JSON编码器来处理NaN和日期
from fastapi.responses import JSONResponse
import json as _json
from datetime import datetime, date
import numpy as np

class CustomJSONEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        import math
        # 处理Python float NaN/inf
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        # 处理numpy数值类型和NaN值
        if hasattr(obj, 'dtype') and np.isscalar(obj):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        return super().default(obj)

# 预处理函数来清理NaN值
def clean_nan_values(obj):
    """递归清理数据结构中的NaN和Infinity值"""
    import math
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'dtype') and np.isscalar(obj):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    else:
        return obj

# 覆盖默认的JSON响应
def custom_json_response(content, **kwargs):
    cleaned_content = clean_nan_values(content)
    return JSONResponse(
        content=cleaned_content,
        **kwargs
    )

def load_predictions_for_enhanced_strategy(symbol_list, start_date, end_date, days):
    """为增强策略加载预测数据"""
    try:
        # 使用现有的校准器加载预测数据
        calibrator_instance = HistoricalBacktestCalibrator()
        
        # 获取预测数据
        predictions = calibrator_instance.load_predictions()
        
        if predictions.empty:
            return predictions
            
        # 筛选股票
        if symbol_list:
            predictions = predictions[predictions['symbol'].isin(symbol_list)]
        
        # 筛选日期
        if start_date:
            predictions = predictions[predictions['prediction_date'] >= start_date]
        if end_date:
            predictions = predictions[predictions['prediction_date'] <= end_date]
        elif days:
            # 取最近N天的数据
            latest_date = predictions['prediction_date'].max()
            start_date = latest_date - pd.Timedelta(days=days)
            predictions = predictions[predictions['prediction_date'] >= start_date]
        
        return predictions
        
    except Exception as e:
        logger.error(f"加载预测数据失败: {e}")
        return pd.DataFrame()

def load_price_data_for_enhanced_strategy(symbol_list, start_date, end_date, days):
    """为增强策略加载价格数据"""
    try:
        from backend.data_manager import get_stock_data
        from datetime import datetime, timedelta
        
        all_price_data = []
        
        if not symbol_list:
            return pd.DataFrame()
        
        for symbol in symbol_list:
            try:
                # 计算日期范围
                if start_date and end_date:
                    price_start = start_date
                    price_end = end_date
                elif days:
                    end_dt = datetime.now().date()
                    start_dt = end_dt - timedelta(days=days + 30)  # 多取30天确保数据充足
                    price_start = start_dt.strftime('%Y-%m-%d')
                    price_end = end_dt.strftime('%Y-%m-%d')
                else:
                    # 默认取过去3个月数据
                    end_dt = datetime.now().date()
                    start_dt = end_dt - timedelta(days=90)
                    price_start = start_dt.strftime('%Y-%m-%d')
                    price_end = end_dt.strftime('%Y-%m-%d')
                
                # 获取价格数据
                df = get_stock_data(symbol, price_start, price_end)
                
                if df is not None and not df.empty:
                    df['symbol'] = symbol
                    all_price_data.append(df)
                    
            except Exception as e:
                logger.warning(f"获取 {symbol} 价格数据失败: {e}")
                continue
        
        if all_price_data:
            combined_data = pd.concat(all_price_data, ignore_index=True)
            # 确保date列是datetime类型
            combined_data['date'] = pd.to_datetime(combined_data['date'])
            return combined_data
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        return pd.DataFrame()


class QueryRequest(BaseModel):
    query: str
    mode: str | None = None  # quick / llm / auto
    model: Optional[str] = None  # 可选：覆盖使用的LLM模型（例如 "kimi"）


class QueryResponse(BaseModel):
    result: str


def get_analyser(model_name: Optional[str] = None):
    selected_name = (model_name or os.getenv("MODEL_NAME", "deepseek-chat")).strip()
    llm = ChatModelFactory.get_model(selected_name)
    analyser = StockAnalyser(llm=llm, verbose=False)
    # 装载工具（供 llm 模式使用）
    try:
        tools = StockTool.extract_tools_from_file("./data/stock.md.txt")
        analyser.set_tools(tools)
    except Exception:
        pass
    return analyser


@app.post("/api/analyse", response_model=QueryResponse)
def analyse(req: QueryRequest):
    analyser = get_analyser(req.model)
    result = analyser.analyse(req.query, mode=req.mode)
    # 检测查询中的股票代码并附加多维度指标
    try:
        extra = _append_quick_indicators_if_any(req.query)
        if extra:
            result = f"{result}\n\n{extra}"
    except Exception:
        pass
    # 二次总结：输出明确的结论/建议/风险/操作
    try:
        summary = _llm_summarize_conclusion(req.query, result, req.model)
        if summary:
            return QueryResponse(result=summary)
    except Exception:
        pass
    return QueryResponse(result=result)


history: list[dict] = []


@app.post("/api/search", response_model=QueryResponse)
def search(req: QueryRequest):
    analyser = get_analyser(req.model)
    result = analyser.analyse(req.query, mode=req.mode)
    try:
        extra = _append_quick_indicators_if_any(req.query)
        if extra:
            result = f"{result}\n\n{extra}"
    except Exception:
        pass
    try:
        summary = _llm_summarize_conclusion(req.query, result, req.model)
        if summary:
            history.append({"query": req.query, "mode": req.mode or "auto", "result": summary, "model": req.model or os.getenv("MODEL_NAME", "deepseek-chat")})
            return QueryResponse(result=summary)
    except Exception:
        pass
    history.append({"query": req.query, "mode": req.mode or "auto", "result": result, "model": req.model or os.getenv("MODEL_NAME", "deepseek-chat")})
    return QueryResponse(result=result)

def _extract_symbol_from_text(text: str) -> Optional[str]:
    import re
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", str(text))
    return m.group(1) if m else None

def _append_quick_indicators_if_any(text: str) -> Optional[str]:
    symbol = _extract_symbol_from_text(text)
    if not symbol or ak is None or pd is None:
        return None
    # 获取近60天数据，计算关键技术指标
    try:
        # 取近180日，确保有足够数据计算60日等指标
        from datetime import datetime, timedelta
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=180)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="")
        if df is None or df.empty:
            return None
        df = df.tail(120).copy()
        close = pd.to_numeric(df["收盘"], errors="coerce")
        high = pd.to_numeric(df["最高"], errors="coerce")
        low = pd.to_numeric(df["最低"], errors="coerce")
        vol = pd.to_numeric(df.get("成交量"), errors="coerce") if "成交量" in df.columns else None
        valid = close.notna() & high.notna() & low.notna()
        close = close[valid]
        high = high[valid]
        low = low[valid]
        if vol is not None:
            vol = vol[valid]
        if len(close) < 30:
            return None

        import numpy as _np

        ma20 = close.rolling(20, min_periods=20).mean()
        ma60 = close.rolling(60, min_periods=60).mean()
        ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100 if ma20.iloc[-5] and _np.isfinite(ma20.iloc[-5]) else _np.nan

        # 价量趋势与动量
        ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 and _np.isfinite(close.iloc[-6]) else _np.nan
        vol_ratio = None
        if vol is not None and len(vol) >= 60:
            v20 = vol.rolling(20, min_periods=20).mean().iloc[-1]
            v60 = vol.rolling(60, min_periods=60).mean().iloc[-1]
            if _np.isfinite(v20) and _np.isfinite(v60) and v60:
                vol_ratio = float(v20 / v60)

        # RSI14
        delta = close.diff()
        gain = (delta.clip(lower=0)).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / (loss.replace(0, _np.nan))
        rsi14 = 100 - (100 / (1 + rs))
        rsi_val = float(rsi14.iloc[-1]) if _np.isfinite(rsi14.iloc[-1]) else None

        # MACD(12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea).iloc[-1]

        # 布林带位置
        mid = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        bpos = None
        if _np.isfinite(upper.iloc[-1]) and _np.isfinite(lower.iloc[-1]):
            rng = (upper.iloc[-1] - lower.iloc[-1]) or _np.nan
            bpos = float((close.iloc[-1] - lower.iloc[-1]) / rng * 100) if _np.isfinite(rng) else None

        # ATR14 (% of price)
        prev_close = close.shift(1)
        tr = _np.maximum(high - low, _np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        atr = pd.Series(tr).rolling(14, min_periods=14).mean()
        atr_pct = float(atr.iloc[-1] / close.iloc[-1] * 100) if _np.isfinite(atr.iloc[-1]) else None

        # 简单风控：近60日最大回撤
        last60 = close.tail(60)
        mdd = None
        if len(last60) >= 2:
            roll_max = last60.cummax()
            drawdown = last60 / roll_max - 1.0
            mdd = float(drawdown.min() * 100)

        parts = []
        if _np.isfinite(ma20_slope):
            parts.append(f"MA20斜率: {ma20_slope:.2f}%")
        if ma60.iloc[-1] == ma60.iloc[-1]:
            parts.append(f"MA60: {ma60.iloc[-1]:.2f}")
        if _np.isfinite(ret5):
            parts.append(f"5日动量: {ret5:.2f}%")
        if vol_ratio is not None:
            parts.append(f"量能比(20/60): {vol_ratio:.2f}")
        if rsi_val is not None:
            parts.append(f"RSI14: {rsi_val:.1f}")
        if macd_hist == macd_hist:
            parts.append(f"MACD柱: {macd_hist:.3f}")
        if bpos is not None:
            parts.append(f"布林带位置: {bpos:.1f}%")
        if atr_pct is not None:
            parts.append(f"ATR14波动率: {atr_pct:.2f}%")
        if mdd is not None:
            parts.append(f"60日最大回撤: {mdd:.2f}%")

        if parts:
            return "📊 附加指标概览（近120日）\n- " + "\n- ".join(parts)
        return None
    except Exception:
        return None

def _llm_summarize_conclusion(query: str, analysed_text: str, model_name: Optional[str] = None) -> Optional[str]:
    """用所选LLM将数据与指标汇总为明确的结论/建议/风险/操作四段结构。"""
    try:
        llm = ChatModelFactory.get_model((model_name or os.getenv("MODEL_NAME", "deepseek-chat")).strip())
        symbol = _extract_symbol_from_text(query) or ""
        prompt = (
            "你是资深A股交易顾问。请基于以下‘数据要点与指标’输出简洁的中文结论，严格包含四个部分：\n"
            "1) 结论：一句话判定当下趋势与结构性位置（看多/观望/谨慎）。\n"
            "2) 建议：2-4条可执行建议（含仓位/节奏/关注位）。\n"
            "3) 风险：2-3条主要风险与触发条件。\n"
            "4) 操作：今日/本周的操作计划与触发价位。\n"
            "要求：\n- 用事实支撑，不复述表格；\n- 不要请求更多信息；\n- 控制在150-220字；\n- 直接输出四段，无需前后缀。\n\n"
            f"标的: {symbol or '未知'}\n用户问题: {query}\n\n数据要点与指标（原始输出如下）:\n{analysed_text}\n"
        )
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", str(resp)).strip()
        # 简单校验是否含有关键分段词，不满足则返回None以回退原结果
        if any(k in content for k in ["结论", "建议", "风险", "操作"]):
            return content
        return None
    except Exception:
        return None


@app.get("/api/history")
def get_history():
    return list(reversed(history))[:100]


class FixedAnalyseRequest(BaseModel):
    symbol: str
    analyses: List[str]  # ["MA20", "RSI14", "MACD", "BOLL", "REALTIME", "TOPUP", "TOPDOWN"]
    lookback_days: int | None = 120
    with_summary: bool = True


def _fetch_hist(symbol: str, start_date: str, end_date: str) -> Any:
    if ak is None or pd is None:
        raise HTTPException(status_code=500, detail="后端缺少依赖：akshare/pandas")
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="未获取到历史数据")
    return df


def _date_range(lookback_days: int) -> tuple[str, str]:
    from datetime import datetime, timedelta
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=lookback_days)
    return start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")


@app.post("/api/fixed_analyse")
def fixed_analyse(req: FixedAnalyseRequest):
    symbol = req.symbol.lower().replace("sh", "").replace("sz", "")
    start_date, end_date = _date_range(req.lookback_days or 120)
    logs: list[str] = []
    result: Dict[str, Any] = {
        "meta": {"symbol": req.symbol, "start": start_date, "end": end_date, "analyses": req.analyses},
        "series": {},   # 用于图表的序列
        "indicators": {},
        "realtime": None,
        "board": None,
        "summary": None,
        "logs": logs,
    }

    try:
        df = None
        need_hist = any(k in req.analyses for k in ["MA20", "RSI14", "MACD", "BOLL"]) 
        if need_hist:
            logs.append(f"开始获取历史数据: symbol={symbol}, range={start_date}~{end_date}")
            df = _fetch_hist(symbol, start_date, end_date)
            logs.append(f"历史数据获取完成: {len(df)} 行")

        if df is not None:
            # 基础close序列
            result["series"]["close"] = {
                "x": df["日期"].tolist(),
                "y": [float(v) for v in df["收盘"].tolist()],
            }

        if "MA20" in req.analyses and df is not None:
            logs.append("计算MA20…")
            ma = df["收盘"].rolling(window=20).mean()
            result["series"]["MA20"] = {
                "x": df["日期"].tolist(),
                "y": [None if pd.isna(v) else float(v) for v in ma.tolist()],
            }

        if "RSI14" in req.analyses and df is not None:
            logs.append("计算RSI14…")
            delta = df["收盘"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, pd.NA)
            rsi = 100 - (100 / (1 + rs))
            result["series"]["RSI14"] = {
                "x": df["日期"].tolist(),
                "y": [None if pd.isna(v) else float(v) for v in rsi.tolist()],
            }
            last = rsi.dropna().tail(1)
            result["indicators"]["RSI14"] = float(last.iloc[0]) if not last.empty else None

        if "MACD" in req.analyses and df is not None:
            logs.append("计算MACD…")
            ema12 = df["收盘"].ewm(span=12).mean()
            ema26 = df["收盘"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            hist = macd - signal
            result["series"]["MACD"] = {
                "x": df["日期"].tolist(),
                "y": [float(v) for v in macd.tolist()],
            }
            result["series"]["MACD_SIGNAL"] = {
                "x": df["日期"].tolist(),
                "y": [float(v) for v in signal.tolist()],
            }
            result["series"]["MACD_HIST"] = {
                "x": df["日期"].tolist(),
                "y": [float(v) for v in hist.tolist()],
            }

        if "BOLL" in req.analyses and df is not None:
            logs.append("计算布林带(20,2)…")
            ma20 = df["收盘"].rolling(window=20).mean()
            std = df["收盘"].rolling(window=20).std()
            upper = ma20 + 2 * std
            lower = ma20 - 2 * std
            result["series"]["BOLL_UP"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in upper.tolist()]}
            result["series"]["BOLL_MID"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in ma20.tolist()]}
            result["series"]["BOLL_LOW"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in lower.tolist()]}

        if "REALTIME" in req.analyses:
            try:
                logs.append("获取实时行情…")
                spot = ak.stock_zh_a_spot_em()
                row = spot[spot["代码"].astype(str) == symbol]
                if row.empty:
                    row = spot[spot["代码"].astype(str).str.contains(symbol)]
                if not row.empty:
                    # 清理NaN值
                    realtime_data = row.head(1).fillna(None).to_dict(orient="records")[0]
                    result["realtime"] = realtime_data
                else:
                    result["realtime"] = None
            except Exception:
                result["realtime"] = None

        if "TOPUP" in req.analyses or "TOPDOWN" in req.analyses:
            try:
                logs.append("获取涨跌幅榜…")
                board = ak.stock_zh_a_spot_em()
                board["涨跌幅"] = pd.to_numeric(board["涨跌幅"], errors="coerce")
                board["最新价"] = pd.to_numeric(board["最新价"], errors="coerce")
                if "TOPDOWN" in req.analyses:
                    board = board.sort_values("涨跌幅").head(10)
                else:
                    board = board.sort_values("涨跌幅", ascending=False).head(10)
                # 清理NaN值，替换为None
                board_clean = board[["代码", "名称", "最新价", "涨跌幅"]].fillna(None)
                result["board"] = board_clean.to_dict(orient="records")
            except Exception:
                result["board"] = None

        # LLM 总结（可选）
        if req.with_summary:
            try:
                logs.append("调用LLM生成综合结论…")
                llm = ChatModelFactory.get_model(os.getenv("MODEL_NAME", "deepseek-chat"))
                # 基于已选指标构造事实
                facts_lines: list[str] = []
                try:
                    # MA20
                    if "MA20" in req.analyses and df is not None and "MA20" in result["series"]:
                        latest_close = float(df["收盘"].tail(1).iloc[0])
                        latest_ma = result["series"]["MA20"]["y"][-1]
                        rel = None
                        if latest_ma is not None:
                            rel = "上方" if latest_close >= latest_ma else "下方"
                        # 简易趋势
                        ma_vals = [v for v in result["series"]["MA20"]["y"] if v is not None]
                        trend = None
                        if len(ma_vals) >= 5:
                            import statistics
                            diff = ma_vals[-1] - ma_vals[-5]
                            trend = "上升" if diff > 0 else ("下降" if diff < 0 else "震荡")
                        facts_lines.append(f"MA20: 收盘 {latest_close:.2f}, MA20 {latest_ma if latest_ma is not None else 'NA'}, 价格位于MA20{rel or '未知'}, 趋势{trend or '未知'}")

                    # RSI14
                    if "RSI14" in req.analyses and "RSI14" in result["series"]:
                        rsi_last = result["series"]["RSI14"]["y"][-1]
                        state = None
                        if rsi_last is not None:
                            state = "超买" if rsi_last >= 70 else ("超卖" if rsi_last <= 30 else "中性")
                        facts_lines.append(f"RSI14: {rsi_last if rsi_last is not None else 'NA'} ({state or '未知'})")

                    # MACD
                    if "MACD" in req.analyses and ("MACD" in result["series"] or "MACD_SIGNAL" in result["series"]):
                        macd_last = result["series"].get("MACD", {}).get("y", [None])[-1]
                        signal_last = result["series"].get("MACD_SIGNAL", {}).get("y", [None])[-1]
                        cross = None
                        if macd_last is not None and signal_last is not None:
                            cross = "金叉" if macd_last >= signal_last else "死叉"
                        facts_lines.append(f"MACD: {macd_last if macd_last is not None else 'NA'}, Signal: {signal_last if signal_last is not None else 'NA'} ({cross or '未知'})")

                    # BOLL
                    if "BOLL" in req.analyses and all(k in result["series"] for k in ["BOLL_UP","BOLL_MID","BOLL_LOW"]) and df is not None:
                        up = result["series"]["BOLL_UP"]["y"][-1]
                        mid = result["series"]["BOLL_MID"]["y"][-1]
                        low = result["series"]["BOLL_LOW"]["y"][-1]
                        close_last = float(df["收盘"].tail(1).iloc[0])
                        pos = None
                        if None not in [up, mid, low]:
                            if close_last >= up:
                                pos = "上轨外"
                            elif close_last >= mid:
                                pos = "上轨-中轨"
                            elif close_last >= low:
                                pos = "中轨-下轨"
                            else:
                                pos = "下轨外"
                        facts_lines.append(f"BOLL: 上 {up if up is not None else 'NA'}, 中 {mid if mid is not None else 'NA'}, 下 {low if low is not None else 'NA'}, 价格位置 {pos or '未知'}")

                    # 实时
                    if "REALTIME" in req.analyses and result["realtime"]:
                        rt = result["realtime"]
                        facts_lines.append(f"实时: 名称 {rt.get('名称','-')}, 价格 {rt.get('最新价','-')}, 涨跌幅 {rt.get('涨跌幅','-')}%")
                except Exception:
                    pass

                facts_text = "\n".join([f"- {line}" for line in facts_lines]) if facts_lines else "- 无"

                # 简要提示，要求基于所选指标给出总结
                prompt = (
                    "你是一名专业证券分析师。请仅基于我提供的指标事实进行中文总结，约300字，包含：趋势判断、关键信号、风险与注意事项、若干可执行的观察点。不要杜撰未提供的数据。\n"
                    f"标的: {req.symbol}\n"
                    f"区间: {start_date}~{end_date}\n"
                    f"所选指标: {', '.join(req.analyses)}\n"
                    "指标事实如下:\n" + facts_text
                )
                summary = llm.invoke(prompt)
                # 兼容不同返回类型
                result["summary"] = getattr(summary, "content", str(summary))
                logs.append("综合结论生成完成")
            except Exception:
                result["summary"] = None
                logs.append("综合结论生成失败，已忽略")

        return custom_json_response(result)
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="固定分析失败")


def _sse_format(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _generate_default_analysis(analyses: list, result: dict, df) -> dict:
    """基于技术指标生成默认的结构化分析"""
    signals = []
    risk_points = []
    confidence_scores = []
    support_level = None
    resistance_level = None
    
    # 判断趋势信号
    trend_signal = "震荡"  # 默认
    
    if df is not None:
        latest_close = float(df["收盘"].tail(1).iloc[0])
        
        # MA分析
        ma_signals = []
        for ma_period in ["MA5", "MA20", "MA60"]:
            if ma_period in analyses and ma_period in result["series"]:
                ma_values = [v for v in result["series"][ma_period]["y"] if v is not None]
                if ma_values:
                    latest_ma = ma_values[-1]
                    if latest_close > latest_ma:
                        ma_signals.append(f"{ma_period}上方")
                        confidence_scores.append(0.6)
                    else:
                        ma_signals.append(f"{ma_period}下方")
                        confidence_scores.append(0.4)
                    
                    # 设定支撑阻力位
                    if ma_period == "MA20":
                        if latest_close > latest_ma:
                            support_level = round(latest_ma, 2)
                        else:
                            resistance_level = round(latest_ma, 2)
        
        # RSI分析
        if "RSI14" in analyses and "RSI14" in result["series"]:
            rsi_values = [v for v in result["series"]["RSI14"]["y"] if v is not None]
            if rsi_values:
                latest_rsi = rsi_values[-1]
                if latest_rsi >= 70:
                    signals.append("RSI超买信号")
                    risk_points.append("RSI超买，注意回调风险")
                    confidence_scores.append(0.3)
                elif latest_rsi <= 30:
                    signals.append("RSI超卖信号")
                    signals.append("可能出现反弹")
                    confidence_scores.append(0.7)
                else:
                    signals.append("RSI中性区间")
                    confidence_scores.append(0.5)
        
        # MACD分析
        if "MACD" in analyses and "MACD" in result["series"] and "MACD_SIGNAL" in result["series"]:
            macd_values = [v for v in result["series"]["MACD"]["y"] if v is not None]
            signal_values = [v for v in result["series"]["MACD_SIGNAL"]["y"] if v is not None]
            if macd_values and signal_values:
                latest_macd = macd_values[-1]
                latest_signal = signal_values[-1]
                if latest_macd > latest_signal:
                    signals.append("MACD金叉")
                    confidence_scores.append(0.6)
                else:
                    signals.append("MACD死叉")
                    confidence_scores.append(0.4)
        
        # 综合判断趋势
        bullish_signals = sum(1 for s in ma_signals if "上方" in s)
        total_ma_signals = len(ma_signals)
        
        if total_ma_signals > 0:
            bullish_ratio = bullish_signals / total_ma_signals
            if bullish_ratio >= 0.6:
                trend_signal = "多头"
            elif bullish_ratio <= 0.4:
                trend_signal = "空头"
    
    # 生成风险提示
    if not risk_points:
        if trend_signal == "多头":
            risk_points.append("注意回调风险，设置止损")
        elif trend_signal == "空头":
            risk_points.append("下跌风险较大，谨慎操作")
        else:
            risk_points.append("方向不明，等待明确信号")
    
    # 计算置信度
    if confidence_scores:
        confidence = sum(confidence_scores) / len(confidence_scores)
    else:
        confidence = 0.5
    
    # 构建结构化数据
    structured_data = {
        "trend_signal": trend_signal,
        "confidence": round(confidence, 2),
        "key_signals": signals[:3] if signals else ["暂无明确信号"],  # 最多显示3个
        "risk_points": risk_points[:2] if risk_points else ["请注意风险控制"],  # 最多显示2个
        "summary": f"基于{', '.join(analyses)}指标分析，当前趋势为{trend_signal}，置信度{round(confidence*100, 1)}%"
    }
    
    if support_level:
        structured_data["support_level"] = support_level
    if resistance_level:
        structured_data["resistance_level"] = resistance_level
    
    return structured_data


def _predict_price_direction(df, result: dict, analyses: list) -> dict:
    """基于技术指标和增强时间序列模型预测未来5-10日价格方向"""
    if df is None or len(df) < 30:  # 增加最小数据要求
        return {"error": "数据不足，无法预测（至少需要30个交易日）"}
    
    try:
        import numpy as np
        import math
        
        # 准备特征数据
        features = []
        feature_names = []
        prices = df["收盘"].values
        volumes = df["成交量"].values if "成交量" in df.columns else None
        
        # === 增强价格动量特征 ===
        price_changes = np.diff(prices) / prices[:-1]  # 使用百分比变化
        
        # 多时间维度动量
        if len(price_changes) >= 20:
            momentum_3d = np.mean(price_changes[-3:])  # 3日动量
            momentum_5d = np.mean(price_changes[-5:])  # 5日动量
            momentum_10d = np.mean(price_changes[-10:])  # 10日动量
            momentum_20d = np.mean(price_changes[-20:])  # 20日动量
            
            features.extend([momentum_3d, momentum_5d, momentum_10d, momentum_20d])
            feature_names.extend(['momentum_3d', 'momentum_5d', 'momentum_10d', 'momentum_20d'])
        else:
            features.extend([0, 0, 0, 0])
            feature_names.extend(['momentum_3d', 'momentum_5d', 'momentum_10d', 'momentum_20d'])
        
        # === 波动率特征 ===
        if len(price_changes) >= 10:
            vol_short = np.std(price_changes[-5:])   # 短期波动率
            vol_medium = np.std(price_changes[-10:]) # 中期波动率
            vol_ratio = vol_short / vol_medium if vol_medium > 0 else 1  # 波动率比值
            
            features.extend([vol_short, vol_ratio])
            feature_names.extend(['volatility_short', 'volatility_ratio'])
        else:
            features.extend([0, 1])
            feature_names.extend(['volatility_short', 'volatility_ratio'])
        
        # === 均线排列特征 ===
        ma_alignment_score = 0
        ma_trend_strength = 0
        
        # 检查MA5, MA20, MA60的排列
        ma_values = {}
        for ma_period in ["MA5", "MA20", "MA60"]:
            if ma_period in analyses and ma_period in result["series"]:
                ma_vals = [v for v in result["series"][ma_period]["y"] if v is not None]
                if len(ma_vals) >= 3:
                    ma_values[ma_period] = ma_vals[-3:]  # 取最近3个值
        
        if len(ma_values) >= 2:
            current_price = prices[-1]
            
            # 均线多头排列得分
            if "MA5" in ma_values and "MA20" in ma_values:
                if ma_values["MA5"][-1] > ma_values["MA20"][-1] > current_price * 0.95:
                    ma_alignment_score += 0.5
                elif ma_values["MA5"][-1] < ma_values["MA20"][-1] < current_price * 1.05:
                    ma_alignment_score -= 0.5
                    
            # 均线趋势强度（斜率一致性）
            for ma_period, vals in ma_values.items():
                if len(vals) >= 3:
                    slope1 = (vals[-1] - vals[-2]) / vals[-2] if vals[-2] != 0 else 0
                    slope2 = (vals[-2] - vals[-3]) / vals[-3] if vals[-3] != 0 else 0
                    if slope1 * slope2 > 0:  # 同向
                        ma_trend_strength += abs(slope1) * 0.3
        
        features.extend([ma_alignment_score, ma_trend_strength])
        feature_names.extend(['ma_alignment', 'ma_trend_strength'])
        
        # === RSI增强特征 ===
        rsi_score = 0
        rsi_divergence = 0
        
        if "RSI14" in analyses and "RSI14" in result["series"]:
            rsi_values = [v for v in result["series"]["RSI14"]["y"] if v is not None]
            if len(rsi_values) >= 5:
                latest_rsi = rsi_values[-1]
                
                # RSI位置得分（非线性）
                if latest_rsi > 70:
                    rsi_score = -((latest_rsi - 70) / 30) ** 2  # 超买惩罚
                elif latest_rsi < 30:
                    rsi_score = ((30 - latest_rsi) / 30) ** 2   # 超卖奖励
                else:
                    rsi_score = (latest_rsi - 50) / 50 * 0.5    # 中性区域
                
                # RSI背离检测
                if len(rsi_values) >= 5 and len(prices) >= 5:
                    price_trend = (prices[-1] - prices[-5]) / prices[-5]
                    rsi_trend = (rsi_values[-1] - rsi_values[-5]) / 100
                    
                    # 价格上涨但RSI下降（顶背离）
                    if price_trend > 0.02 and rsi_trend < -0.05:
                        rsi_divergence = -0.3
                    # 价格下跌但RSI上升（底背离）
                    elif price_trend < -0.02 and rsi_trend > 0.05:
                        rsi_divergence = 0.3
        
        features.extend([rsi_score, rsi_divergence])
        feature_names.extend(['rsi_score', 'rsi_divergence'])
        
        # === MACD增强特征 ===
        macd_signal = 0
        macd_histogram_trend = 0
        
        if "MACD" in analyses and "MACD" in result["series"]:
            macd_values = [v for v in result["series"]["MACD"]["y"] if v is not None]
            signal_values = [v for v in result["series"]["MACD_SIGNAL"]["y"] if v is not None]
            
            if len(macd_values) >= 3 and len(signal_values) >= 3:
                # MACD金叉死叉
                current_diff = macd_values[-1] - signal_values[-1]
                prev_diff = macd_values[-2] - signal_values[-2]
                
                if current_diff > 0 and prev_diff <= 0:  # 金叉
                    macd_signal = 0.4
                elif current_diff < 0 and prev_diff >= 0:  # 死叉
                    macd_signal = -0.4
                else:
                    macd_signal = current_diff * 0.1  # 持续信号
                
                # MACD柱状图趋势
                if len(macd_values) >= 3:
                    hist_trend = (current_diff - prev_diff)
                    macd_histogram_trend = hist_trend * 0.2
        
        features.extend([macd_signal, macd_histogram_trend])
        feature_names.extend(['macd_signal', 'macd_histogram'])
        
        # === 成交量确认特征 ===
        volume_confirmation = 0
        if volumes is not None and len(volumes) >= 10:
            vol_avg = np.mean(volumes[-10:])
            vol_recent = np.mean(volumes[-3:])
            
            if vol_recent > vol_avg * 1.2:  # 放量
                price_trend_3d = (prices[-1] - prices[-4]) / prices[-4] if len(prices) >= 4 else 0
                volume_confirmation = 0.2 if price_trend_3d > 0 else -0.1  # 放量上涨好，放量下跌略坏
            elif vol_recent < vol_avg * 0.8:  # 缩量
                volume_confirmation = -0.1  # 缩量不利
        
        features.append(volume_confirmation)
        feature_names.append('volume_confirmation')
        
        # === 动态权重分配 ===
        # 根据市场环境调整权重
        volatility_level = np.std(price_changes[-10:]) if len(price_changes) >= 10 else 0.02
        
        # 🚀 统一使用趋势跟踪权重配置，专门针对用户的趋势股
        weights = [0.25, 0.15, 0.10, 0.15,  # 趋势延续65%
                  0.05, 0.05,               # 波动控制10%  
                  0.15, 0.10,               # 价格位置25%
                  0.0, 0.0,                 # RSI=0 (移除反转逻辑)
                  0.0, 0.0,                 # MACD=0 
                  0.0]                      # 成交量=0
        
        # 确保特征和权重数量匹配
        if len(features) != len(weights):
            min_len = min(len(features), len(weights))
            features = features[:min_len]
            weights = weights[:min_len]
        
        # 计算加权得分
        score = sum(f * w for f, w in zip(features, weights))
        
        # === 优化的概率转换 ===
        # 使用校准过的sigmoid函数
        sigmoid_factor = 3  # 大幅降低放大倍数，与历史数据生成保持一致
        raw_probability = 1 / (1 + math.exp(-score * sigmoid_factor))
        
        # 🚀 移除所有校准干扰，直接使用原始预测
        calibrated_probability = raw_probability
        
        # === 增强的价格区间预测 ===
        latest_price = float(prices[-1])
        
        # 使用多种波动率估计
        vol_5d = np.std(price_changes[-5:]) if len(price_changes) >= 5 else 0.02
        vol_20d = np.std(price_changes[-20:]) if len(price_changes) >= 20 else 0.02
        vol_adaptive = vol_5d * 0.6 + vol_20d * 0.4  # 自适应波动率
        
        # 基于概率和波动率的价格区间
        expected_return = (calibrated_probability - 0.5) * 0.08  # 最大期望涨跌幅8%
        
        # 价格区间计算（考虑偏度）
        upside_vol = vol_adaptive * 1.2 if calibrated_probability > 0.6 else vol_adaptive
        downside_vol = vol_adaptive * 1.2 if calibrated_probability < 0.4 else vol_adaptive
        
        price_lower = latest_price * (1 + expected_return - downside_vol * 1.5)
        price_upper = latest_price * (1 + expected_return + upside_vol * 1.5)
        
        # === 动态置信度计算 ===
        # 基于特征一致性和市场环境
        feature_consistency = abs(score) * 2  # 特征一致性
        market_stability = 1 / (1 + volatility_level * 50)  # 市场稳定性
        data_quality = min(1.0, len(prices) / 60)  # 数据质量
        
        base_confidence = 50
        confidence_boost = (feature_consistency * 25 + market_stability * 20 + data_quality * 10)
        final_confidence = min(90, max(35, base_confidence + confidence_boost))
        
        # === 构建预测结果 ===
        prediction = {
            "current_price": round(latest_price, 2),
            "prediction_days": "5-10日",
            "up_probability": round(calibrated_probability * 100, 1),
            "down_probability": round((1 - calibrated_probability) * 100, 1),
            "price_range_lower": round(max(price_lower, latest_price * 0.82), 2),  # 最大跌幅18%
            "price_range_upper": round(min(price_upper, latest_price * 1.18), 2),  # 最大涨幅18%
            "confidence_level": round(final_confidence, 1),
            "key_factors": [],
            "model_details": {  # 新增：模型详情
                "feature_score": round(score, 3),
                "volatility_regime": "高" if volatility_level > 0.03 else "低" if volatility_level < 0.015 else "中",
                "signal_strength": "强" if abs(score) > 0.2 else "弱" if abs(score) < 0.05 else "中"
            }
        }
        
        # === 智能因素分析 ===
        key_factors = []
        
        # 主要趋势判断
        if calibrated_probability > 0.65:
            key_factors.append("多头趋势明显")
        elif calibrated_probability < 0.35:
            key_factors.append("空头趋势明显") 
        else:
            key_factors.append("横盘整理态势")
            
        # 动量分析
        if len(features) >= 4:  # 确保有动量特征
            avg_momentum = np.mean(features[:4])
            if avg_momentum > 0.01:
                key_factors.append("价格动量向上")
            elif avg_momentum < -0.01:
                key_factors.append("价格动量向下")
                
        # 波动率分析
        if volatility_level > 0.03:
            key_factors.append("高波动环境")
        elif volatility_level < 0.015:
            key_factors.append("低波动环境")
            
        # RSI状态
        if "RSI14" in analyses and "RSI14" in result["series"]:
            rsi_values = [v for v in result["series"]["RSI14"]["y"] if v is not None]
            if rsi_values:
                latest_rsi = rsi_values[-1]
                if latest_rsi >= 75:
                    key_factors.append("RSI严重超买")
                elif latest_rsi >= 65:
                    key_factors.append("RSI轻度超买")
                elif latest_rsi <= 25:
                    key_factors.append("RSI严重超卖")
                elif latest_rsi <= 35:
                    key_factors.append("RSI轻度超卖")
                    
        # MACD状态
        if macd_signal > 0.2:
            key_factors.append("MACD金叉信号")
        elif macd_signal < -0.2:
            key_factors.append("MACD死叉信号")
            
        # 成交量状态
        if volume_confirmation > 0.1:
            key_factors.append("放量上涨确认")
        elif volume_confirmation < -0.05:
            key_factors.append("量价背离风险")
            
        prediction["key_factors"] = key_factors[:5]  # 最多显示5个因素
        
        # === 保存预测记录用于后续校准 ===
        try:
            record = PredictionRecord(
                symbol=df.iloc[0].get("股票代码", "unknown") if "股票代码" in df.columns else "unknown",
                prediction_date=datetime.now().strftime("%Y-%m-%d"),
                predicted_probability=calibrated_probability,
                prediction_horizon=5,
                features={
                    "feature_score": score,
                    "volatility_regime": "高" if volatility_level > 0.03 else "低" if volatility_level < 0.015 else "中",
                    "signal_strength": abs(score)
                }
            )
            calibrator.save_prediction(record)
        except Exception as e:
            logger.warning(f"保存预测记录失败: {e}")
        
        return prediction
        
    except Exception as e:
        return {"error": f"预测计算失败: {str(e)}"}


@app.get("/api/fixed_analyse_stream")
def fixed_analyse_stream(symbol: str, analyses: str, lookback_days: int = 120, with_summary: bool = True):
    """
    SSE流式输出：
      - event: log, data: 文本日志
      - event: done, data: 最终JSON（与 /api/fixed_analyse 返回结构一致）
    analyses: 逗号分隔，例如 "MA20,RSI14,MACD"
    """
    analyses_list = [a.strip() for a in analyses.split(',') if a.strip()]

    def gen():
        try:
            yield _sse_format("log", "开始固定分析")
            # 组装请求体并调用现有逻辑（复用函数体以避免重复）
            # 统一处理股票代码格式，与非流式接口保持一致
            processed_symbol = symbol.lower().replace("sh", "").replace("sz", "")
            req = FixedAnalyseRequest(symbol=processed_symbol, analyses=analyses_list, lookback_days=lookback_days, with_summary=with_summary)
            # 我们复制 fixed_analyse 的流程，但按步yield日志
            logs: list[str] = []
            start_date, end_date = _date_range(req.lookback_days or 120)
            result: Dict[str, Any] = {
                "meta": {"symbol": req.symbol, "start": start_date, "end": end_date, "analyses": req.analyses},
                "series": {},
                "indicators": {},
                "realtime": None,
                "board": None,
                "summary": None,
                "logs": logs,
            }
            df = None
            need_hist = any(k in req.analyses for k in ["MA5", "MA20", "MA60", "RSI14", "MACD", "BOLL", "OBV", "ATR"])
            if need_hist:
                # 自动调整时间范围以确保MA线有足够的有效数据点
                adjusted_lookback = req.lookback_days or 120
                if "MA60" in req.analyses:
                    adjusted_lookback = max(adjusted_lookback, 80)  # 至少80天确保MA60有20个有效点
                elif "MA20" in req.analyses:
                    adjusted_lookback = max(adjusted_lookback, 50)  # 至少50天确保MA20有30个有效点
                elif "MA5" in req.analyses:
                    adjusted_lookback = max(adjusted_lookback, 30)  # 至少30天确保MA5有25个有效点
                
                if adjusted_lookback != (req.lookback_days or 120):
                    yield _sse_format("log", f"为确保均线数据完整，自动调整查询期间为{adjusted_lookback}天")
                    adjusted_start, _ = _date_range(adjusted_lookback)
                    start_date = adjusted_start
                
                msg = f"开始获取历史数据: symbol={processed_symbol}, range={start_date}~{end_date}"
                logs.append(msg)
                yield _sse_format("log", msg)
                df = _fetch_hist(processed_symbol, start_date, end_date)
                msg = f"历史数据获取完成: {len(df)} 行"
                logs.append(msg)
                yield _sse_format("log", msg)
                # 基础close
                result["series"]["close"] = {"x": df["日期"].tolist(), "y": [float(v) for v in df["收盘"].tolist()]}

            # MA指标系列
            if "MA5" in req.analyses and df is not None:
                logs.append("计算MA5…")
                yield _sse_format("log", "计算MA5…")
                ma5 = df["收盘"].rolling(window=5).mean()
                result["series"]["MA5"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in ma5.tolist()]}

            if "MA20" in req.analyses and df is not None:
                logs.append("计算MA20…")
                yield _sse_format("log", "计算MA20…")
                ma20 = df["收盘"].rolling(window=20).mean()
                result["series"]["MA20"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in ma20.tolist()]}

            if "MA60" in req.analyses and df is not None:
                logs.append("计算MA60…")
                yield _sse_format("log", "计算MA60…")
                ma60 = df["收盘"].rolling(window=60).mean()
                result["series"]["MA60"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in ma60.tolist()]}

            if "RSI14" in req.analyses and df is not None:
                logs.append("计算RSI14…")
                yield _sse_format("log", "计算RSI14…")
                delta = df["收盘"].diff()
                gain = delta.clip(lower=0).rolling(window=14).mean()
                loss = (-delta.clip(upper=0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, pd.NA)
                rsi = 100 - (100 / (1 + rs))
                result["series"]["RSI14"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in rsi.tolist()]}
                last = rsi.dropna().tail(1)
                result["indicators"]["RSI14"] = float(last.iloc[0]) if not last.empty else None

            if "MACD" in req.analyses and df is not None:
                logs.append("计算MACD…")
                yield _sse_format("log", "计算MACD…")
                ema12 = df["收盘"].ewm(span=12).mean()
                ema26 = df["收盘"].ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                hist = macd - signal
                result["series"]["MACD"] = {"x": df["日期"].tolist(), "y": [float(v) for v in macd.tolist()]}
                result["series"]["MACD_SIGNAL"] = {"x": df["日期"].tolist(), "y": [float(v) for v in signal.tolist()]}
                result["series"]["MACD_HIST"] = {"x": df["日期"].tolist(), "y": [float(v) for v in hist.tolist()]}

            if "BOLL" in req.analyses and df is not None:
                logs.append("计算布林带(20,2)…")
                yield _sse_format("log", "计算布林带(20,2)…")
                ma20 = df["收盘"].rolling(window=20).mean()
                std = df["收盘"].rolling(window=20).std()
                upper = ma20 + 2 * std
                lower = ma20 - 2 * std
                result["series"]["BOLL_UP"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in upper.tolist()]}
                result["series"]["BOLL_MID"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in ma20.tolist()]}
                result["series"]["BOLL_LOW"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in lower.tolist()]}

            if "OBV" in req.analyses and df is not None:
                logs.append("计算OBV能量潮…")
                yield _sse_format("log", "计算OBV能量潮…")
                # OBV = 成交量 * sign(收盘-昨收)的累积
                price_change = df["收盘"].diff()
                volume_direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                obv = (df["成交量"] * volume_direction).cumsum()
                result["series"]["OBV"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in obv.tolist()]}

            if "ATR" in req.analyses and df is not None:
                logs.append("计算ATR平均真实波幅…")
                yield _sse_format("log", "计算ATR平均真实波幅…")
                # ATR计算：TR = max(H-L, |H-昨C|, |L-昨C|)
                high_low = df["最高"] - df["最低"]
                high_close = (df["最高"] - df["收盘"].shift(1)).abs()
                low_close = (df["最低"] - df["收盘"].shift(1)).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                result["series"]["ATR"] = {"x": df["日期"].tolist(), "y": [None if pd.isna(v) else float(v) for v in atr.tolist()]}

            if "REALTIME" in req.analyses:
                yield _sse_format("log", "获取实时行情…")
                try:
                    spot = ak.stock_zh_a_spot_em()
                    row = spot[spot["代码"].astype(str) == processed_symbol]
                    if row.empty:
                        row = spot[spot["代码"].astype(str).str.contains(processed_symbol)]
                    if not row.empty:
                        # 清理NaN值
                        realtime_data = row.head(1).fillna(None).to_dict(orient="records")[0]
                        result["realtime"] = realtime_data
                    else:
                        result["realtime"] = None
                    yield _sse_format("log", f"实时行情获取完成: {'1条' if result['realtime'] else '无数据'}")
                except Exception:
                    result["realtime"] = None
                    yield _sse_format("log", "实时行情获取失败，已忽略")

            if "TOPUP" in req.analyses or "TOPDOWN" in req.analyses:
                yield _sse_format("log", "获取涨跌幅榜…")
                try:
                    board = ak.stock_zh_a_spot_em()
                    board["涨跌幅"] = pd.to_numeric(board["涨跌幅"], errors="coerce")
                    board["最新价"] = pd.to_numeric(board["最新价"], errors="coerce")
                    if "TOPDOWN" in req.analyses:
                        board = board.sort_values("涨跌幅").head(10)
                    else:
                        board = board.sort_values("涨跌幅", ascending=False).head(10)
                    # 清理NaN值，替换为None
                    board_clean = board[["代码", "名称", "最新价", "涨跌幅"]].fillna(None)
                    result["board"] = board_clean.to_dict(orient="records")
                    yield _sse_format("log", f"榜单获取完成: {len(result['board']) if result['board'] else 0} 条")
                except Exception:
                    result["board"] = None
                    yield _sse_format("log", "榜单获取失败，已忽略")

            if with_summary:
                yield _sse_format("log", "调用LLM生成综合结论…")
                try:
                    llm = ChatModelFactory.get_model(os.getenv("MODEL_NAME", "deepseek-chat"))
                    # 提炼事实（与非流式一致）
                    facts_lines: list[str] = []
                    
                    # 多均线分析
                    if df is not None:
                        latest_close = float(df["收盘"].tail(1).iloc[0])
                        ma_info = []
                        for ma_period in ["MA5", "MA20", "MA60"]:
                            if ma_period in req.analyses and ma_period in result["series"]:
                                latest_ma = result["series"][ma_period]["y"][-1]
                                if latest_ma is not None:
                                    rel = "上方" if latest_close >= latest_ma else "下方"
                                    ma_vals = [v for v in result["series"][ma_period]["y"] if v is not None]
                                    trend = None
                                    if len(ma_vals) >= 5:
                                        diff = ma_vals[-1] - ma_vals[-5]
                                        trend = "上升" if diff > 0 else ("下降" if diff < 0 else "震荡")
                                    ma_info.append(f"{ma_period}: {latest_ma:.2f} (价格在{rel}, 趋势{trend or '未知'})")
                        if ma_info:
                            facts_lines.append(f"均线分析: 收盘 {latest_close:.2f}, " + ", ".join(ma_info))
                    if "RSI14" in req.analyses and "RSI14" in result["series"]:
                        rsi_last = result["series"]["RSI14"]["y"][-1]
                        state = None
                        if rsi_last is not None:
                            state = "超买" if rsi_last >= 70 else ("超卖" if rsi_last <= 30 else "中性")
                        facts_lines.append(f"RSI14: {rsi_last if rsi_last is not None else 'NA'} ({state or '未知'})")
                    if "MACD" in req.analyses and ("MACD" in result["series"] or "MACD_SIGNAL" in result["series"]):
                        macd_last = result["series"].get("MACD", {}).get("y", [None])[-1]
                        signal_last = result["series"].get("MACD_SIGNAL", {}).get("y", [None])[-1]
                        cross = None
                        if macd_last is not None and signal_last is not None:
                            cross = "金叉" if macd_last >= signal_last else "死叉"
                        facts_lines.append(f"MACD: {macd_last if macd_last is not None else 'NA'}, Signal: {signal_last if signal_last is not None else 'NA'} ({cross or '未知'})")
                    if "BOLL" in req.analyses and all(k in result["series"] for k in ["BOLL_UP","BOLL_MID","BOLL_LOW"]) and df is not None:
                        up = result["series"]["BOLL_UP"]["y"][-1]
                        mid = result["series"]["BOLL_MID"]["y"][-1]
                        low = result["series"]["BOLL_LOW"]["y"][-1]
                        close_last = float(df["收盘"].tail(1).iloc[0])
                        pos = None
                        if None not in [up, mid, low]:
                            if close_last >= up:
                                pos = "上轨外"
                            elif close_last >= mid:
                                pos = "上轨-中轨"
                            elif close_last >= low:
                                pos = "中轨-下轨"
                            else:
                                pos = "下轨外"
                        facts_lines.append(f"BOLL: 上 {up if up is not None else 'NA'}, 中 {mid if mid is not None else 'NA'}, 下 {low if low is not None else 'NA'}, 价格位置 {pos or '未知'}")
                    
                    # 新增指标分析
                    if "OBV" in req.analyses and "OBV" in result["series"]:
                        obv_last = result["series"]["OBV"]["y"][-1]
                        obv_values = [v for v in result["series"]["OBV"]["y"] if v is not None]
                        obv_trend = None
                        if len(obv_values) >= 10:
                            diff = obv_values[-1] - obv_values[-10]
                            obv_trend = "资金流入" if diff > 0 else ("资金流出" if diff < 0 else "震荡")
                        facts_lines.append(f"OBV能量潮: {obv_last if obv_last is not None else 'NA'} (趋势: {obv_trend or '未知'})")
                    
                    if "ATR" in req.analyses and "ATR" in result["series"]:
                        atr_last = result["series"]["ATR"]["y"][-1]
                        if atr_last is not None and df is not None:
                            close_price = float(df["收盘"].tail(1).iloc[0])
                            volatility = "高" if atr_last > close_price * 0.02 else ("中" if atr_last > close_price * 0.01 else "低")
                            facts_lines.append(f"ATR波动率: {atr_last:.2f} (波动性: {volatility})")
                    
                    if "REALTIME" in req.analyses and result["realtime"]:
                        rt = result["realtime"]
                        facts_lines.append(f"实时: 名称 {rt.get('名称','-')}, 价格 {rt.get('最新价','-')}, 涨跌幅 {rt.get('涨跌幅','-')}%")

                    facts_text = "\n".join([f"- {line}" for line in facts_lines]) if facts_lines else "- 无"
                    prompt = (
                        "你是股票技术分析专家。请分析以下指标数据，严格按照JSON格式输出：\n\n"
                        "**输出要求**：\n"
                        "1. 必须是有效的JSON格式\n"
                        "2. 不要有任何解释文字，只输出JSON\n"
                        "3. 包含以下字段（必须完整）：\n\n"
                        "```json\n"
                        "{\n"
                        '  "trend_signal": "多头",\n'
                        '  "confidence": 0.75,\n'
                        '  "key_signals": ["MA20上升", "RSI中性"],\n'
                        '  "risk_points": ["价格在MA20下方", "成交量不足"],\n'
                        '  "support_level": 10.8,\n'
                        '  "resistance_level": 11.5,\n'
                        '  "summary": "基于当前指标，股价处于技术回调阶段，但MA20向上趋势为主导..."\n'
                        "}\n"
                        "```\n\n"
                        f"**分析标的**: {req.symbol}\n"
                        f"**时间区间**: {start_date}至{end_date}\n"
                        f"**技术指标**: {', '.join(req.analyses)}\n\n"
                        "**指标事实**:\n" + facts_text + "\n\n"
                        "请根据上述指标事实，输出JSON格式的分析结果："
                    )
                    # 显示LLM输入内容
                    yield _sse_format("llm_input", prompt)
                    yield _sse_format("log", f"LLM提示长度: {len(prompt)}")

                    summary = llm.invoke(prompt)
                    content = getattr(summary, "content", str(summary))
                    
                    # 尝试解析结构化响应，如果失败则生成默认结构化数据
                    try:
                        import json
                        import re
                        # 提取JSON部分
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            structured_data = json.loads(json_match.group(0))
                            result["structured_analysis"] = structured_data
                            result["summary"] = structured_data.get("summary", content)
                            yield _sse_format("log", f"结构化解析成功: 置信度 {structured_data.get('confidence', 'N/A')}")
                        else:
                            # JSON解析失败，基于指标生成默认结构化数据
                            yield _sse_format("log", "JSON解析失败，生成默认结构化分析")
                            structured_data = _generate_default_analysis(req.analyses, result, df)
                            result["structured_analysis"] = structured_data
                            result["summary"] = content
                            yield _sse_format("log", f"默认结构化数据生成完成: 置信度 {structured_data.get('confidence', 'N/A')}")
                    except Exception as parse_err:
                        # 异常情况下也生成默认结构化数据
                        yield _sse_format("log", f"结构化解析异常: {str(parse_err)}")
                        structured_data = _generate_default_analysis(req.analyses, result, df)
                        result["structured_analysis"] = structured_data
                        result["summary"] = content
                        yield _sse_format("log", f"默认结构化数据生成完成: 置信度 {structured_data.get('confidence', 'N/A')}")
                    
                    yield _sse_format("log", f"LLM输出预览: {content[:120]}…")
                except Exception as e:
                    result["summary"] = None
                    yield _sse_format("log", f"LLM生成失败: {str(e)}")

            # 价格预测
            if need_hist and df is not None:
                yield _sse_format("log", "生成价格预测…")
                try:
                    prediction = _predict_price_direction(df, result, req.analyses)
                    result["price_prediction"] = prediction
                    if "error" not in prediction:
                        yield _sse_format("log", f"预测完成: 上涨概率 {prediction.get('up_probability', 'N/A')}%")
                    else:
                        yield _sse_format("log", f"预测失败: {prediction['error']}")
                except Exception as e:
                    result["price_prediction"] = {"error": f"预测异常: {str(e)}"}
                    yield _sse_format("log", f"价格预测异常: {str(e)}")

            # 最终返回
            import json as _json
            from datetime import datetime, date

            # 自定义JSON编码器处理日期对象
            class DateTimeEncoder(_json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    # 处理numpy数值类型和NaN值
                    import numpy as np
                    import math
                    # 处理Python float NaN/inf
                    if isinstance(obj, float):
                        if math.isnan(obj) or math.isinf(obj):
                            return None
                        return obj
                    # 处理numpy数值类型和NaN值
                    if hasattr(obj, 'dtype') and np.isscalar(obj):
                        if np.isnan(obj) or np.isinf(obj):
                            return None
                        return float(obj)
                    return super().default(obj)

            # 清理NaN值并序列化
            cleaned_result = clean_nan_values(result)
            yield _sse_format("done", _json.dumps(cleaned_result, ensure_ascii=False, cls=DateTimeEncoder))
        except HTTPException as he:
            yield _sse_format("log", f"错误: {he.detail}")
            import json as _json
            yield _sse_format("done", _json.dumps({"error": he.detail}, ensure_ascii=False))
        except Exception as e:
            yield _sse_format("log", f"异常: {str(e)}")
            import json as _json
            yield _sse_format("done", _json.dumps({"error": "固定分析失败"}, ensure_ascii=False))

    return StreamingResponse(gen(), media_type="text/event-stream")


# 兼容带尾斜杠的路径
@app.get("/api/fixed_analyse_stream/")
def fixed_analyse_stream_alias(symbol: str, analyses: str, lookback_days: int = 120, with_summary: bool = True):
    return fixed_analyse_stream(symbol=symbol, analyses=analyses, lookback_days=lookback_days, with_summary=with_summary)


# Qlib相关API端点
class QlibAnalysisRequest(BaseModel):
    symbol: str
    lookback_days: int = 120

@app.post("/api/qlib/analysis")
async def qlib_analysis(request: QlibAnalysisRequest):
    """Qlib量化分析"""
    if not QLIB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Qlib功能不可用")
    
    try:
        symbol = request.symbol.lower().replace("sh", "").replace("sz", "")
        start_date, end_date = _date_range(request.lookback_days)
        
        # 获取历史数据
        df = _fetch_hist(symbol, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="无法获取股票数据")
        
        # 运行Qlib分析
        result = demo_qlib_integration(request.symbol, df)
        
        return custom_json_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qlib分析失败: {str(e)}")

@app.get("/api/qlib/status")
async def qlib_status():
    """检查Qlib状态"""
    return {
        "qlib_available": _qlib.available,
        "enabled": _qlib.enabled,
        "provider_uri": _qlib.provider_uri,
        "region": _qlib.region,
        "error": _qlib.error,
        "version": "m1-demo",
        "features": [
            "特征工程",
            "量化预测",
            "简单回测",
            "Alpha因子计算",
            "FeatureStore缓存",
        ] if _qlib.available else []
    }


class QlibFeaturesRequest(BaseModel):
    symbol: str
    analyses: List[str]
    lookback_days: int = 120
    freq: str = "day"


@app.post("/api/qlib/features", summary="Qlib Features")
async def qlib_features(req: QlibFeaturesRequest):
    try:
        symbol = req.symbol.lower().replace("sh", "").replace("sz", "")
        start_date, end_date = _date_range(req.lookback_days or 120)
        factor_set = ",".join(sorted(req.analyses))

        # 先查缓存
        cached = feature_store.get(symbol, start_date, end_date, req.freq, factor_set)
        if cached is not None:
            return {"series": cached, "cached": True}

        # 获取历史数据
        df = _fetch_hist(symbol, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="无法获取股票数据")

        # 计算因子（当前用Pandas实现；未来可切换Qlib表达式）
        series = build_factors(df, req.analyses)

        # 写入缓存
        feature_store.set(symbol, start_date, end_date, req.freq, factor_set, series, ttl_seconds=1800)

        return {"series": series, "cached": False}
    except HTTPException:
        raise
    except Exception as e:
        # 返回详细错误，便于定位
        raise HTTPException(status_code=500, detail=f"qlib_features失败: {str(e)}")


@app.get("/api/qlib/features_stream")
def qlib_features_stream(symbol: str, analyses: str, lookback_days: int = 120, freq: str = "day"):
    import time
    import json as _json

    def gen():
        t0 = time.time()
        try:
            yield _sse_format("status", "准备构建因子…")
            symbol_clean = symbol.lower().replace("sh", "").replace("sz", "")
            start_date, end_date = _date_range(lookback_days or 120)
            factor_set = ",".join(sorted(analyses.split(",")))
            yield _sse_format("meta", _json.dumps({
                "symbol": symbol, "start": start_date, "end": end_date, "analyses": factor_set
            }, ensure_ascii=False))

            # 缓存命中
            cached = feature_store.get(symbol_clean, start_date, end_date, freq, factor_set)
            if cached is not None:
                dt = (time.time() - t0) * 1000
                yield _sse_format("cache_hit", _json.dumps({"cached": True, "elapsed_ms": round(dt, 1)}))
                yield _sse_format("done", _json.dumps({"series": cached, "cached": True}, ensure_ascii=False))
                return

            # 获取历史数据
            yield _sse_format("status", "拉取历史数据…")
            df = _fetch_hist(symbol_clean, start_date, end_date)
            if df is None or df.empty:
                yield _sse_format("error", "无法获取股票数据")
                yield _sse_format("done", _json.dumps({"error": "无法获取股票数据"}, ensure_ascii=False))
                return

            # 计算因子
            yield _sse_format("status", "计算因子…")
            t1 = time.time()
            series = build_factors(df, analyses.split(","))
            dt_build = (time.time() - t1) * 1000
            yield _sse_format("built", _json.dumps({
                "factors": list(series.keys()),
                "elapsed_ms": round(dt_build, 1)
            }, ensure_ascii=False))

            # 写入缓存
            feature_store.set(symbol_clean, start_date, end_date, freq, factor_set, series, ttl_seconds=1800)
            total_ms = (time.time() - t0) * 1000
            yield _sse_format("done", _json.dumps({
                "series": series, "cached": False, "elapsed_ms": round(total_ms, 1)
            }, ensure_ascii=False))
        except Exception as e:
            yield _sse_format("error", str(e))
            yield _sse_format("done", _json.dumps({"error": str(e)}, ensure_ascii=False))

    return StreamingResponse(gen(), media_type="text/event-stream")


class QlibAnalyseRequest(BaseModel):
    symbol: str
    analyses: List[str]
    lookback_days: int = 120
    horizon_days: int = 5


@app.post("/api/qlib/analyse")
async def qlib_analyse(req: QlibAnalyseRequest):
    """计算因子+价格预测+结构化结论+基础风险（与前端结构对齐）。"""
    try:
        symbol_clean = req.symbol.lower().replace("sh", "").replace("sz", "")
        start_date, end_date = _date_range(req.lookback_days or 120)

        # 历史数据
        df = _fetch_hist(symbol_clean, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="无法获取股票数据")

        # 因子/指标序列
        series = build_factors(df, req.analyses)

        # 价格预测（复用现有优化算法）
        pred = _predict_price_direction(df, {"series": series}, req.analyses)

        # 结构化结论（优先调用已有的默认生成逻辑）
        try:
            structured = _generate_default_analysis(req.analyses, {"series": series, "indicators": {}}, df)
        except Exception:
            # 回退：根据简单规则生成
            structured = {
                "trend_signal": "震荡",
                "confidence": 0.5,
                "key_signals": [],
                "risk_points": []
            }
            try:
                close_last = float(df["收盘"].tail(1).iloc[0])
                ma20_last = series.get("MA20", {}).get("y", [None])[-1]
                if ma20_last is not None:
                    structured["trend_signal"] = "多头" if close_last >= ma20_last else "空头"
            except Exception:
                pass

        # 基础风险指标
        import numpy as _np
        ret = df["收盘"].pct_change().dropna().values
        if ret.size > 0:
            mdd = 0.0
            equity = (1 + ret).cumprod()
            peak = _np.maximum.accumulate(equity)
            mdd = float(_np.max((peak - equity) / peak)) if equity.size > 0 else 0.0
            vol = float(_np.std(ret) * (252 ** 0.5))
            var_95 = float(_np.percentile(ret, 5))
            win_rate = float((_np.sum(ret > 0) / ret.size))
        else:
            mdd = vol = var_95 = win_rate = 0.0

        risk = {
            "max_drawdown": mdd,
            "volatility": vol,
            "var_95": var_95,
            "win_rate": win_rate,
        }

        return custom_json_response({
            "meta": {"symbol": req.symbol, "start": start_date, "end": end_date, "analyses": req.analyses},
            "series": series,
            "structured_analysis": structured,
            "price_prediction": pred,
            "risk": risk,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qlib_analyse失败: {str(e)}")




class GenerateDataRequest(BaseModel):
    symbols: str = "600519"
    days: int = 100
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# 持仓管理相关数据模型
class TransactionRequest(BaseModel):
    symbol: str
    name: str
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    fee: float = 0
    note: str = ""

class PositionEditRequest(BaseModel):
    symbol: str
    quantity: int
    avg_cost: float
    note: str = ""

class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]

class PositionResponse(BaseModel):
    symbol: str
    name: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    profit_loss: float
    profit_loss_pct: float
    last_updated: str

class PortfolioResponse(BaseModel):
    total_assets: float
    available_cash: float
    market_value: float
    total_profit_loss: float
    total_profit_loss_pct: float
    positions: List[PositionResponse]
    position_count: int



# ==================== 策略回测API ====================

@app.post("/api/strategy/run_backtest")
def run_strategy_backtest(
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
    max_positions: int = 10,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.002,
    position_size: float = 0.1,
    start_date: str = None,
    end_date: str = None,
    symbols: str = None,  # 逗号分隔的股票代码
    selection_mode: str = "threshold",  # 🔥 新增：交易选择模式（threshold/topk）
    top_k: int = 3,  # 🔥 新增：top-k模式下的每日选择数
    strategy_mode: str = "standard",  # 🚀 新增：策略模式（standard/high_return）
    days: int = None,  # 🚀 高收益模式的回测天数
    force_refresh: bool = False  # 🔄 新增：强制刷新历史数据
):
    """运行策略回测"""
    try:
        # 🚀 检查是否使用高收益模式
        if strategy_mode == "high_return":
            # 使用增强策略（基于文档完整算法）
            from backend.enhanced_strategy import EnhancedStrategy, EnhancedStrategyConfig
            from backend.opening_prediction import OpeningPredictor, format_prediction_result
            
            # 解析股票代码
            symbol_list = None
            if symbols:
                symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
            
            # 使用调整后的参数配置（适合短期回测）
            enhanced_config = EnhancedStrategyConfig(
                buy_threshold=0.55,      # 降低买入阈值以增加交易机会
                sell_threshold=0.45,     # 提高卖出阈值以减少过早卖出
                position_size=0.12,      # 保持文档最优值
                max_positions=10,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost
            )
            
            # 初始化增强策略
            strategy = EnhancedStrategy(enhanced_config)
            
            try:
                # 加载预测数据和价格数据
                predictions = load_predictions_for_enhanced_strategy(symbol_list, start_date, end_date, days)
                price_data = load_price_data_for_enhanced_strategy(symbol_list, start_date, end_date, days)
                
                if predictions.empty or price_data.empty:
                    logger.warning("增强策略数据不足，降级到基础高收益策略")
                    # 降级到原高收益策略
                    from backend.high_return_strategy import HighReturnStrategy
                    fallback_strategy = HighReturnStrategy()
                    results = fallback_strategy.run_backtest(symbols=symbol_list, days=days or 365, force_refresh=force_refresh)
                    
                    return custom_json_response({
                        "success": True,
                        "strategy_name": "🚀 高收益模式（降级版）",
                        "results": results,
                        "performance_rating": _get_performance_rating(results),
                        "timestamp": _json.dumps(datetime.now(), default=str),
                        "warning": "增强策略数据不足，已自动降级到基础高收益策略"
                    })
                
                # 运行增强策略回测
                results = strategy.run_backtest(predictions, price_data, start_date, end_date)
                
                return custom_json_response({
                    "success": True,
                    "strategy_name": "📈 增强策略（文档完整版）",
                    "results": results,
                    "performance_rating": _get_performance_rating(results.get('performance_metrics', {})),
                    "timestamp": _json.dumps(datetime.now(), default=str)
                })
                
            except Exception as e:
                logger.error(f"增强策略执行失败: {e}")
                # 降级到原高收益策略
                from backend.high_return_strategy import HighReturnStrategy
                fallback_strategy = HighReturnStrategy()
                results = fallback_strategy.run_backtest(symbols=symbol_list, days=days or 365, force_refresh=force_refresh)
                
                return custom_json_response({
                    "success": True,
                    "strategy_name": "🚀 高收益模式（降级版）",
                    "results": results,
                    "performance_rating": _get_performance_rating(results),
                    "timestamp": _json.dumps(datetime.now(), default=str),
                    "warning": "增强策略数据不足，已降级到基础高收益策略"
                })
        
        # 标准模式：使用原有逻辑
        config = StrategyConfig(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            max_positions=max_positions,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            position_size=position_size,
            selection_mode=selection_mode,
            top_k=top_k
        )
        
        # 解析股票代码
        symbol_list = None
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        
        # 首次尝试运行回测
        result = backtester.run_backtest(config, start_date, end_date, symbol_list, force_refresh=force_refresh)
        
        # 如果没有历史数据，自动生成然后重新运行回测
        if isinstance(result, dict) and result.get("error") == "没有可用的预测数据":
            coverage_details = result.get("details", {})
            coverage_reason = coverage_details.get("reason", "数据不足")
            print(f"⚠️ {coverage_reason}，开始自动生成...")
            
            # 确定要生成数据的股票和时间范围
            target_symbols = symbols if symbols else "600519,000001,000002"  # 默认股票
            
            try:
                # 如果指定了日期范围，使用日期范围生成
                if start_date and end_date:
                    print(f"🔄 自动生成历史数据: {target_symbols} ({start_date} ~ {end_date})")
                    generate_result = calibrator.generate_historical_backtest_data_by_date_range(
                        target_symbols, start_date, end_date
                    )
                else:
                    # 否则使用默认天数生成
                    days = 365  # 默认1年，支持更长期回测
                    print(f"🔄 自动生成历史数据: {target_symbols} ({days}天)")
                    generate_result = calibrator.generate_historical_backtest_data(target_symbols, days)
                
                if generate_result and isinstance(generate_result, dict):
                    success_count = len(generate_result.get("success_symbols", []))
                    if success_count > 0:
                        print(f"✅ 自动生成成功，获得 {generate_result.get('total_records', 0)} 条历史数据")
                        
                        # 重新运行回测
                        print("🔄 重新运行回测...")
                        result = backtester.run_backtest(config, start_date, end_date, symbol_list, force_refresh=force_refresh)
                        
                        # 在结果中添加自动生成的标记
                        if isinstance(result, dict) and "error" not in result:
                            result["auto_generated_data"] = True
                            result["generated_records"] = generate_result.get('total_records', 0)
                            result["generated_symbols"] = generate_result.get('success_symbols', [])
                            result["message"] = f"自动生成了 {generate_result.get('total_records', 0)} 条历史数据并完成回测"
                    else:
                        return custom_json_response({
                            "error": "自动生成历史数据失败，无法获取股票数据，请检查股票代码是否正确"
                        })
                else:
                    return custom_json_response({
                        "error": "自动生成历史数据失败，请手动生成历史数据后重试"
                    })
                    
            except Exception as gen_error:
                print(f"❌ 自动生成历史数据异常: {gen_error}")
                return custom_json_response({
                    "error": f"自动生成历史数据失败: {str(gen_error)}，请手动生成历史数据后重试"
                })
        
        # 为标准模式添加success字段以保持API一致性
        if isinstance(result, dict) and "error" not in result:
            result["success"] = True
        return custom_json_response(result)
        
    except Exception as e:
        return custom_json_response({"error": f"策略回测失败: {str(e)}"})

@app.get("/api/strategy/performance/{start_date}/{end_date}")
def get_strategy_performance(start_date: str, end_date: str):
    """获取指定时间段的策略表现"""
    try:
        config = StrategyConfig()  # 使用默认配置
        result = backtester.run_backtest(config, start_date, end_date, force_refresh=False)
        
        if "error" in result:
            return custom_json_response(result)
            
        # 只返回性能指标
        return custom_json_response({
            "performance_metrics": result["performance_metrics"],
            "total_trades": result["total_trades"],
            "final_portfolio_value": result["final_portfolio_value"]
        })
        
    except Exception as e:
        return custom_json_response({"error": f"获取策略表现失败: {str(e)}"})

@app.post("/api/strategy/optimize")
def optimize_strategy(
    start_date: str = None,
    end_date: str = None,
    symbols: str = None,
):
    """策略参数优化（简化版）"""
    from datetime import datetime
    print(f"🔧 策略优化API被调用 - 时间: {datetime.now()}")
    
    # 添加简单的防重复调用机制
    import time
    if not hasattr(optimize_strategy, 'last_call_time'):
        optimize_strategy.last_call_time = 0
    
    current_time = time.time()
    if current_time - optimize_strategy.last_call_time < 5:  # 5秒内不允许重复调用
        print(f"⚠️ 优化API调用过于频繁，忽略重复请求")
        return custom_json_response({"error": "请勿频繁调用优化功能，请等待5秒后再试"})
    
    optimize_strategy.last_call_time = current_time
    try:
        best_config = None
        best_return = -float('inf')
        best_sharpe = -float('inf')
        optimization_results = []
        
        # 解析股票代码
        symbol_list = None
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        
        print(f"🎯 优化参数: start_date={start_date}, end_date={end_date}, symbols={symbols}")

        # 若提供范围与股票，预生成历史预测数据（AKShare补充）
        try:
            if start_date and end_date and symbols:
                print(f"🧰 预生成历史预测数据: {symbols} ({start_date} ~ {end_date})")
                _pre = calibrator.generate_historical_backtest_data_by_date_range(symbols, start_date, end_date)
                print(f"🧰 预生成完成: {_pre}")
        except Exception as _e:
            print(f"⚠️ 预生成失败，继续优化: {_e}")

        # 参数空间（全参数）：
        buy_thresholds = [0.5, 0.55, 0.6, 0.65]
        sell_thresholds = [0.25, 0.3, 0.35, 0.4]
        gamma_list = [0.8, 1.0, 1.2, 1.5, 2.0]
        selection_modes = ["threshold", "topk", "aggressive"]
        topk_list = [2, 3, 5]
        # 激进模式参数
        momentum_weights = [0.2, 0.3, 0.5]
        profit_targets = [0.03, 0.05, 0.08]

        for mode in selection_modes:
            for buy_th in buy_thresholds:
                for sell_th in sell_thresholds:
                    if mode == "threshold" and buy_th <= sell_th:
                        continue
                    for gamma in gamma_list:
                        if mode == "aggressive":
                            for mw in momentum_weights:
                                for pt in profit_targets:
                                    config = StrategyConfig(
                                        buy_threshold=buy_th,
                                        sell_threshold=sell_th,
                                        selection_mode=mode,
                                        momentum_weight=mw,
                                        profit_target=pt
                                    )
                                    backtester.verbose = False
                                    result = backtester.run_backtest(config, start_date, end_date, symbol_list)
                                    if "error" in result:
                                        continue
                                    total_return = result["performance_metrics"]["total_return"]
                                    sharpe_ratio = result["performance_metrics"]["sharpe_ratio"]
                                    score = total_return  # 以最高收益为目标
                                    optimization_results.append({
                                        "selection_mode": mode,
                                        "buy_threshold": buy_th,
                                        "sell_threshold": sell_th,
                                        "momentum_weight": mw,
                                        "profit_target": pt,
                                        "total_return": total_return,
                                        "sharpe_ratio": sharpe_ratio,
                                        "score": score
                                    })
                                    if (score > best_return) or (score == best_return and sharpe_ratio > best_sharpe):
                                        best_return = score
                                        best_sharpe = sharpe_ratio
                                        best_config = config
                        elif mode == "topk":
                            for tk in topk_list:
                                config = StrategyConfig(
                                    buy_threshold=buy_th,
                                    sell_threshold=sell_th,
                                    selection_mode=mode,
                                    top_k=tk
                                )
                                # 静默模式运行（减少日志刷屏）
                                backtester.verbose = False
                                result = backtester.run_backtest(config, start_date, end_date, symbol_list)
                                if "error" in result:
                                    continue
                                total_return = result["performance_metrics"]["total_return"]
                                sharpe_ratio = result["performance_metrics"]["sharpe_ratio"]
                                score = total_return  # 以最高收益为目标
                                optimization_results.append({
                                    "selection_mode": mode,
                                    "top_k": tk,
                                    "buy_threshold": buy_th,
                                    "sell_threshold": sell_th,
                                    "total_return": total_return,
                                    "sharpe_ratio": sharpe_ratio,
                                    "score": score
                                })
                                if (score > best_return) or (score == best_return and sharpe_ratio > best_sharpe):
                                    best_return = score
                                    best_sharpe = sharpe_ratio
                                    best_config = config
                        else:
                            config = StrategyConfig(
                                buy_threshold=buy_th,
                                sell_threshold=sell_th,
                                selection_mode=mode
                            )
                            backtester.verbose = False
                            result = backtester.run_backtest(config, start_date, end_date, symbol_list)
                            if "error" in result:
                                continue
                            total_return = result["performance_metrics"]["total_return"]
                            sharpe_ratio = result["performance_metrics"]["sharpe_ratio"]
                            score = total_return
                            optimization_results.append({
                                "selection_mode": mode,
                                "buy_threshold": buy_th,
                                "sell_threshold": sell_th,
                                "total_return": total_return,
                                "sharpe_ratio": sharpe_ratio,
                                "score": score
                            })
                            if (score > best_return) or (score == best_return and sharpe_ratio > best_sharpe):
                                best_return = score
                                best_sharpe = sharpe_ratio
                                best_config = config
        
        # 使用最优参数运行完整回测
        if best_config:
            best_result = backtester.run_backtest(best_config, start_date, end_date, symbol_list)
            return custom_json_response({
                "best_config": best_config.__dict__,
                "best_performance": best_result["performance_metrics"],
                "optimization_results": optimization_results
            })
        else:
            return custom_json_response({"error": "优化失败，没有找到有效的参数组合"})
            
    except Exception as e:
        return custom_json_response({"error": f"策略优化失败: {str(e)}"})

# ==================== 股票池管理API ====================

@app.get("/api/strategy/available_stocks")
def get_available_stocks():
    """获取可用的股票列表"""
    try:
        conn = sqlite3.connect("calibration.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT symbol, COUNT(*) as prediction_count,
                   MIN(prediction_date) as first_date,
                   MAX(prediction_date) as last_date
            FROM predictions 
            WHERE actual_direction IS NOT NULL
            GROUP BY symbol
            ORDER BY prediction_count DESC
        """)
        
        stocks = []
        for row in cursor.fetchall():
            symbol, count, first_date, last_date = row
            stocks.append({
                "symbol": symbol,
                "prediction_count": count,
                "first_date": first_date,
                "last_date": last_date,
                "name": get_stock_name(symbol)  # 获取股票名称
            })
        
        conn.close()
        return custom_json_response({"available_stocks": stocks})
        
    except Exception as e:
        return custom_json_response({"error": f"获取股票列表失败: {str(e)}"})

def get_stock_name(symbol: str) -> str:
    """获取股票名称（缓存优先 → Akshare 单股信息）"""
    try:
        # 优先使用 PortfolioManager 的全局缓存
        try:
            from backend.portfolio_manager import stock_name_cache  # type: ignore
        except Exception:
            stock_name_cache = None  # noqa: F841

        if 'stock_name_cache' in locals() and stock_name_cache:
            cached = stock_name_cache.get(symbol)
            if cached:
                return cached

        # 直接调用 Akshare 单只股票信息接口
        if ak is None:
            return symbol
        info = ak.stock_individual_info_em(symbol=symbol)
        if info is not None and not info.empty:
            row = info[info['item'] == '股票简称']
            if not row.empty:
                name = row['value'].iloc[0]
                if name and name != symbol:
                    if 'stock_name_cache' in locals() and stock_name_cache:
                        stock_name_cache.set(symbol, name)
                    return name
        return symbol
    except Exception:
        return symbol

# 公开接口：获取单个股票名称
@app.get("/api/stock_name/{symbol}")
def api_get_stock_name(symbol: str):
    try:
        return custom_json_response({"symbol": symbol, "name": get_stock_name(symbol)})
    except Exception as e:
        return custom_json_response({"symbol": symbol, "name": symbol, "error": str(e)})

# 股票池配置文件路径（保留用于兼容性）
STOCK_POOLS_CONFIG_FILE = "stock_pools_config.json"

# 导入数据库管理模块
USE_DATABASE = False
try:
    import sys
    import os
    # 添加backend目录到Python路径
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    
    from stock_pool_db import get_stock_pool_db
    USE_DATABASE = True
    logger.info("✅ 股票池数据库模块加载成功")
except ImportError as e:
    logger.warning(f"⚠️ 股票池数据库模块加载失败，使用JSON文件: {e}")
    USE_DATABASE = False
except Exception as e:
    logger.error(f"❌ 股票池数据库模块加载异常: {e}")
    USE_DATABASE = False

def load_stock_pools():
    """加载股票池配置（数据库优先）"""
    
    # 优先使用数据库
    if USE_DATABASE:
        try:
            db = get_stock_pool_db()
            data = db.get_all_pools()
            if data:  # 如果数据库有数据
                logger.info(f"✅ 从数据库加载股票池配置，包含{len(data)}个股票池: {list(data.keys())}")
                return data
            else:
                logger.info("📊 数据库为空，尝试从JSON文件迁移数据")
                # 如果数据库为空，尝试从JSON迁移一次性数据
                if os.path.exists(STOCK_POOLS_CONFIG_FILE):
                    try:
                        if db.migrate_from_json(STOCK_POOLS_CONFIG_FILE):
                            logger.info("✅ JSON数据迁移成功，重新从数据库加载")
                            data = db.get_all_pools()
                            if data:
                                return data
                        else:
                            logger.error("❌ JSON数据迁移失败")
                    except Exception as e:
                        logger.error(f"JSON数据迁移异常: {e}")
        except Exception as e:
            logger.error(f"❌ 数据库操作失败: {e}")
    
    # 如果数据库不可用，才使用JSON文件作为备用
    if not USE_DATABASE:
        logger.warning("⚠️ 数据库不可用，使用JSON文件")
        try:
            if os.path.exists(STOCK_POOLS_CONFIG_FILE):
                with open(STOCK_POOLS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                    logger.info(f"📁 从JSON文件加载股票池配置，包含{len(data)}个股票池: {list(data.keys())}")
                    return data
            else:
                logger.warning(f"股票池配置文件不存在: {STOCK_POOLS_CONFIG_FILE}")
        except _json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}，文件可能损坏")
        except PermissionError as e:
            logger.error(f"文件权限错误: {e}")
        except Exception as e:
            logger.error(f"加载股票池配置失败: {e}")
    
    # 只有在完全失败时才返回空字典，不再使用默认配置
    logger.error("❌ 无法加载任何股票池数据！请检查数据库或配置文件")
    return {}

def save_stock_pools(pools):
    """保存股票池配置（数据库优先）"""
    
    # 优先使用数据库
    if USE_DATABASE:
        try:
            db = get_stock_pool_db()
            success_count = 0
            
            for pool_name, pool_data in pools.items():
                display_name = pool_data.get('name', pool_name)
                symbols = pool_data.get('symbols', [])
                description = pool_data.get('description', '')
                
                if db.save_pool(pool_name, display_name, symbols, description):
                    success_count += 1
            
            if success_count == len(pools):
                logger.info(f"✅ 成功保存{success_count}个股票池到数据库")
                
                # 同时备份到JSON文件
                try:
                    backup_file = f"{STOCK_POOLS_CONFIG_FILE}.db_backup"
                    db.backup_to_json(backup_file)
                    logger.info(f"📁 已创建JSON备份: {backup_file}")
                except Exception as e:
                    logger.warning(f"创建JSON备份失败: {e}")
                
                return True
            else:
                logger.error(f"❌ 只有{success_count}/{len(pools)}个股票池保存成功")
                return False
                
        except Exception as e:
            logger.error(f"❌ 数据库保存失败: {e}，回退到JSON文件")
    
    # 回退到JSON文件保存
    import shutil
    
    try:
        # 创建备份
        if os.path.exists(STOCK_POOLS_CONFIG_FILE):
            backup_file = f"{STOCK_POOLS_CONFIG_FILE}.backup"
            shutil.copy2(STOCK_POOLS_CONFIG_FILE, backup_file)
            logger.info(f"📁 已创建JSON备份: {backup_file}")
        
        # 使用临时文件进行原子写入
        temp_file = f"{STOCK_POOLS_CONFIG_FILE}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            _json.dump(pools, f, ensure_ascii=False, indent=2)
        
        # 原子替换
        os.replace(temp_file, STOCK_POOLS_CONFIG_FILE)
        
        # 验证写入结果
        with open(STOCK_POOLS_CONFIG_FILE, 'r', encoding='utf-8') as f:
            saved_data = _json.load(f)
            pool_count = len(saved_data)
        
        logger.info(f"📁 成功保存股票池配置到JSON文件，包含{pool_count}个股票池")
        return True
        
    except Exception as e:
        logger.error(f"JSON文件保存失败: {e}")
        # 如果有临时文件，清理它
        temp_file = f"{STOCK_POOLS_CONFIG_FILE}.tmp"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False

@app.get("/api/strategy/stock_pools")
def get_predefined_stock_pools():
    """获取预定义的股票池"""
    pools = load_stock_pools()
    return custom_json_response({"stock_pools": pools})


@app.post("/api/strategy/stock_pools")
def update_stock_pools(pools: Dict):
    """更新股票池配置"""
    try:
        if save_stock_pools(pools.get("stock_pools", {})):
            return custom_json_response({"success": True, "message": "股票池配置已更新"})
        else:
            return custom_json_response({"error": "保存配置失败"})
    except Exception as e:
        return custom_json_response({"error": f"更新股票池失败: {str(e)}"})

@app.delete("/api/strategy/stock_pools/{pool_name}")
def delete_stock_pool(pool_name: str):
    """删除指定的股票池"""
    try:
        pools = load_stock_pools()
        if pool_name in pools:
            del pools[pool_name]
            if save_stock_pools(pools):
                return custom_json_response({"success": True, "message": f"股票池 {pool_name} 已删除"})
        return custom_json_response({"error": "股票池不存在或删除失败"})
    except Exception as e:
        return custom_json_response({"error": f"删除股票池失败: {str(e)}"})

@app.get("/api/strategy/stock_pools_old")
def get_predefined_stock_pools_old():
    """获取预定义的股票池（旧版本，保持兼容）"""
    pools = {
        "蓝筹股": {
            "name": "蓝筹股组合",
            "symbols": ["600519", "600000", "600036", "601398"],
            "description": "大盘蓝筹股，稳健投资"
        },
        "银行股": {
            "name": "银行板块",
            "symbols": ["600000", "600036", "601398", "601818", "601939"],
            "description": "银行业龙头股票"
        },
        "科技股": {
            "name": "科技成长",
            "symbols": ["000002", "002594", "300750"],
            "description": "科技创新概念股"
        },
        "全市场": {
            "name": "全市场",
            "symbols": ["600519", "600000", "600036", "601398", "601818", 
                       "000002", "002594", "300750", "000501"],
            "description": "全部可用股票"
        }
    }
    
    return custom_json_response({"stock_pools": pools})

@app.post("/api/strategy/test_stock_pool")
def test_stock_pool(symbols: str):
    """测试股票池可用性"""
    try:
        symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        
        conn = sqlite3.connect("calibration.db")
        cursor = conn.cursor()
        
        results = []
        for symbol in symbol_list:
            cursor.execute("""
                SELECT COUNT(*) as count,
                       MIN(prediction_date) as first_date,
                       MAX(prediction_date) as last_date
                FROM predictions 
                WHERE symbol = ? AND actual_direction IS NOT NULL
            """, (symbol,))
            
            row = cursor.fetchone()
            count, first_date, last_date = row
            
            results.append({
                "symbol": symbol,
                "name": get_stock_name(symbol),
                "available": count > 0,
                "prediction_count": count,
                "date_range": f"{first_date} ~ {last_date}" if first_date else "无数据"
            })
        
        conn.close()
        
        total_available = sum(1 for r in results if r["available"])
        
        return custom_json_response({
            "test_results": results,
            "summary": {
                "total_symbols": len(symbol_list),
                "available_symbols": total_available,
                "success_rate": f"{total_available/len(symbol_list)*100:.1f}%" if symbol_list else "0%"
            }
        })
        
    except Exception as e:
        return custom_json_response({"error": f"测试股票池失败: {str(e)}"})


# ==================== 高收益策略API ====================

@app.post("/api/strategy/high_return/run")
def run_high_return_strategy(
    symbols: str = None,  # 逗号分隔的股票代码
    days: int = 365,      # 回测天数
    config_override: dict = None,  # 配置覆盖
    force_refresh: bool = False  # 🔄 新增：强制刷新历史数据
):
    """运行高收益策略"""
    try:
        from backend.high_return_strategy import HighReturnStrategy
        
        # 创建策略实例
        strategy = HighReturnStrategy()
        
        # 解析股票代码
        symbol_list = None
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        
        # 配置覆盖
        if config_override:
            for key, value in config_override.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
        
        # 运行回测
        results = strategy.run_backtest(symbols=symbol_list, days=days, force_refresh=force_refresh)
        
        return custom_json_response({
            "success": True,
            "strategy_name": "高收益策略",
            "results": results,
            "performance_rating": _get_performance_rating(results),
            "timestamp": _json.dumps(datetime.now(), default=str)
        })
        
    except Exception as e:
        import traceback
        return custom_json_response({
            "success": False,
            "error": f"高收益策略执行失败: {str(e)}",
            "traceback": traceback.format_exc()
        })

@app.get("/api/strategy/high_return/config")
def get_high_return_config():
    """获取高收益策略配置"""
    try:
        from backend.high_return_strategy import HighReturnStrategy
        
        strategy = HighReturnStrategy()
        config = strategy.config
        
        return custom_json_response({
            "success": True,
            "config": config,
            "strategy_description": config.get('strategy_description', ''),
            "performance_targets": config.get('performance_targets', {})
        })
        
    except Exception as e:
        return custom_json_response({
            "success": False,
            "error": f"获取配置失败: {str(e)}"
        })

@app.post("/api/strategy/high_return/update_config")
def update_high_return_config(config_updates: dict):
    """更新高收益策略配置"""
    try:
        import yaml
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), "..", "high_return_strategy_config.yaml")
        
        # 读取现有配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 更新配置
        def update_nested_dict(d, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                    update_nested_dict(d[key], value)
                else:
                    d[key] = value
        
        update_nested_dict(config, config_updates)
        
        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, ensure_ascii=False, indent=2)
        
        return custom_json_response({
            "success": True,
            "message": "配置更新成功",
            "updated_config": config
        })
        
    except Exception as e:
        return custom_json_response({
            "success": False,
            "error": f"配置更新失败: {str(e)}"
        })

@app.get("/api/strategy/high_return/performance_analysis")
def get_high_return_performance_analysis(
    symbols: str = None,
    days: int = 365,
    benchmark: str = "000300"  # 沪深300作为基准
):
    """获取高收益策略性能分析"""
    try:
        from backend.high_return_strategy import HighReturnStrategy
        
        # 运行策略
        strategy = HighReturnStrategy()
        symbol_list = [s.strip() for s in symbols.split(',') if s.strip()] if symbols else None
        results = strategy.run_backtest(symbols=symbol_list, days=days)
        
        # 计算额外分析指标
        analysis = {
            "basic_metrics": {
                "total_return": results.get('total_return', 0),
                "annualized_return": results.get('annualized_return', 0),
                "sharpe_ratio": results.get('sharpe_ratio', 0),
                "max_drawdown": results.get('max_drawdown', 0),
                "win_rate": results.get('win_rate', 0),
                "trade_count": results.get('trade_count', 0)
            },
            "advanced_metrics": _calculate_advanced_metrics(results),
            "risk_analysis": _analyze_risk_metrics(results),
            "performance_rating": _get_performance_rating(results),
            "comparison_vs_targets": _compare_vs_targets(results),
            "recommendations": _generate_recommendations(results)
        }
        
        return custom_json_response({
            "success": True,
            "analysis": analysis,
            "timestamp": _json.dumps(datetime.now(), default=str)
        })
        
    except Exception as e:
        return custom_json_response({
            "success": False,
            "error": f"性能分析失败: {str(e)}"
        })

def _get_performance_rating(results: dict) -> dict:
    """获取策略性能评级"""
    annual_return = results.get('annualized_return', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    max_drawdown = results.get('max_drawdown', 1)
    win_rate = results.get('win_rate', 0)
    
    # 收益率评级
    if annual_return > 0.30:
        return_rating = "🌟🌟🌟🌟🌟 卓越"
        return_score = 5
    elif annual_return > 0.20:
        return_rating = "🌟🌟🌟🌟 优秀"
        return_score = 4
    elif annual_return > 0.10:
        return_rating = "🌟🌟🌟 良好"
        return_score = 3
    elif annual_return > 0.05:
        return_rating = "🌟🌟 一般"
        return_score = 2
    elif annual_return > 0:
        return_rating = "🌟 及格"
        return_score = 1
    else:
        return_rating = "❌ 需要改进"
        return_score = 0
    
    # 风险评级
    if sharpe_ratio > 2.0:
        risk_rating = "🛡️🛡️🛡️ 优秀"
        risk_score = 3
    elif sharpe_ratio > 1.0:
        risk_rating = "🛡️🛡️ 良好"
        risk_score = 2
    elif sharpe_ratio > 0.5:
        risk_rating = "🛡️ 一般"
        risk_score = 1
    else:
        risk_rating = "⚠️ 较差"
        risk_score = 0
    
    # 综合评级
    total_score = (return_score * 0.6 + risk_score * 0.4)
    if total_score >= 4.0:
        overall_rating = "🏆 卓越策略"
    elif total_score >= 3.0:
        overall_rating = "🥇 优秀策略"
    elif total_score >= 2.0:
        overall_rating = "🥈 良好策略"
    elif total_score >= 1.0:
        overall_rating = "🥉 一般策略"
    else:
        overall_rating = "❌ 需要改进"
    
    return {
        "return_rating": return_rating,
        "risk_rating": risk_rating,
        "overall_rating": overall_rating,
        "total_score": total_score,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }

def _calculate_advanced_metrics(results: dict) -> dict:
    """计算高级指标"""
    daily_returns = results.get('daily_returns', [])
    trades = results.get('trades', [])
    
    if not daily_returns:
        return {}
    
    returns_array = np.array(daily_returns)
    
    # 计算高级指标
    downside_returns = returns_array[returns_array < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    # Sortino比率
    if downside_deviation > 0:
        sortino_ratio = np.mean(returns_array) / downside_deviation * np.sqrt(252)
    else:
        sortino_ratio = 0
    
    # Calmar比率
    max_drawdown = results.get('max_drawdown', 0.001)
    annualized_return = results.get('annualized_return', 0)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # VaR (95%置信度)
    var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
    
    # 最大连续亏损
    max_consecutive_losses = 0
    current_losses = 0
    for trade in trades:
        if trade.get('return', 0) < 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0
    
    return {
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "var_95": var_95,
        "downside_deviation": downside_deviation,
        "max_consecutive_losses": max_consecutive_losses,
        "profit_factor": _calculate_profit_factor(trades)
    }

def _calculate_profit_factor(trades: List[dict]) -> float:
    """计算盈亏比"""
    if not trades:
        return 0
    
    gross_profit = sum(t['return'] for t in trades if t['return'] > 0)
    gross_loss = abs(sum(t['return'] for t in trades if t['return'] < 0))
    
    return gross_profit / gross_loss if gross_loss > 0 else 0

def _analyze_risk_metrics(results: dict) -> dict:
    """分析风险指标"""
    max_drawdown = results.get('max_drawdown', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    win_rate = results.get('win_rate', 0)
    
    risk_level = "低"
    if max_drawdown > 0.15:
        risk_level = "高"
    elif max_drawdown > 0.10:
        risk_level = "中"
    
    return {
        "risk_level": risk_level,
        "max_drawdown": max_drawdown,
        "volatility_estimate": abs(sharpe_ratio) * 0.16 if sharpe_ratio != 0 else 0,
        "win_rate": win_rate,
        "risk_assessment": "可接受" if max_drawdown < 0.12 else "需要关注"
    }

def _compare_vs_targets(results: dict) -> dict:
    """与目标对比"""
    targets = {
        "annual_return": 0.30,
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.10,
        "win_rate": 0.40
    }
    
    comparison = {}
    for metric, target in targets.items():
        actual = results.get(metric, 0)
        achievement = actual / target if target > 0 else 0
        status = "✅ 达标" if achievement >= 1.0 else "⚠️ 未达标"
        
        comparison[metric] = {
            "target": target,
            "actual": actual,
            "achievement_rate": achievement,
            "status": status
        }
    
    return comparison

def _generate_recommendations(results: dict) -> List[str]:
    """生成优化建议"""
    recommendations = []
    
    annual_return = results.get('annualized_return', 0)
    sharpe_ratio = results.get('sharpe_ratio', 0)
    max_drawdown = results.get('max_drawdown', 0)
    win_rate = results.get('win_rate', 0)
    
    if annual_return < 0.20:
        recommendations.append("📈 建议降低信号阈值，增加交易机会")
        recommendations.append("🔧 考虑优化仓位管理，提高资金利用率")
    
    if sharpe_ratio < 1.5:
        recommendations.append("⚡ 建议加强风险管理，提高夏普比率")
        recommendations.append("🎯 优化止盈止损策略")
    
    if max_drawdown > 0.12:
        recommendations.append("🛡️ 建议收紧止损，控制最大回撤")
        recommendations.append("📊 考虑降低单笔仓位大小")
    
    if win_rate < 0.35:
        recommendations.append("🎲 建议增加信号确认机制，提高胜率")
        recommendations.append("🔍 优化入场时机选择")
    
    if not recommendations:
        recommendations.append("🎉 策略表现优秀，建议保持当前配置")
        recommendations.append("📈 可考虑适度增加仓位或扩展股票池")
    
    return recommendations

@app.post("/api/opening_prediction")
async def get_opening_prediction(request: Dict[str, Any]):
    """获取开盘预测策略"""
    try:
        from backend.opening_prediction import OpeningPredictor, format_prediction_result
        
        # 获取股票池
        stock_pool_input = request.get('stock_pool', [])
        
        # 获取强制刷新参数
        force_refresh = request.get('force_refresh', False)
        
        # 处理不同类型的输入
        if isinstance(stock_pool_input, str):
            # 如果是字符串，尝试从股票池配置中获取
            logger.info(f"收到股票池名称: {stock_pool_input}")
            try:
                # 使用统一的股票池加载函数
                pools_config = load_stock_pools()
                
                # 查找匹配的股票池
                stock_pool = None
                logger.info(f"在配置中查找股票池: '{stock_pool_input}'")
                for pool_name, pool_data in pools_config.items():
                    logger.info(f"检查股票池: '{pool_name}' (匹配: {pool_name == stock_pool_input})")
                    if pool_name == stock_pool_input or pool_data.get('name') == stock_pool_input:
                        stock_pool = pool_data.get('symbols', [])
                        logger.info(f"找到匹配的股票池: {stock_pool}")
                        break
                
                if not stock_pool:
                    # 如果没找到，尝试使用"我的自选"股票池
                    if '我的自选' in pools_config:
                        stock_pool = pools_config['我的自选'].get('symbols', [])
                        logger.info(f"使用默认'我的自选'股票池: {stock_pool}")
                    else:
                        # 如果连"我的自选"都没有，使用第一个可用的股票池
                        if pools_config:
                            first_pool_name = list(pools_config.keys())[0]
                            stock_pool = pools_config[first_pool_name].get('symbols', [])
                            logger.info(f"使用第一个可用股票池'{first_pool_name}': {stock_pool}")
                        else:
                            stock_pool = []
                            logger.error("没有可用的股票池数据")
                        
            except Exception as e:
                logger.error(f"加载股票池配置失败: {e}")
                stock_pool = []
                
        elif isinstance(stock_pool_input, list):
            # 如果是列表，直接使用
            stock_pool = stock_pool_input
        else:
            # 其他情况使用默认
            stock_pool = ["000501", "000519", "002182", "600176", "600585", "002436", "600710"]
        
        logger.info(f"开始生成开盘预测，股票池: {stock_pool}")
        logger.info(f"股票池类型: {type(stock_pool)}, 长度: {len(stock_pool)}")
        
        # 确保stock_pool是列表且不为空
        if not isinstance(stock_pool, list) or not stock_pool:
            logger.error(f"股票池格式错误: {stock_pool}")
            raise ValueError("股票池必须是非空的股票代码列表")
        
        # 获取当前持仓信息
        current_positions = []
        try:
            portfolio = portfolio_manager.get_portfolio()
            current_positions = [
                {
                    'symbol': pos.symbol,
                    'name': pos.name,
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'profit_loss': pos.profit_loss,
                    'profit_loss_pct': pos.profit_loss_pct
                }
                for pos in portfolio.positions
            ]
            logger.info(f"获取到当前持仓 {len(current_positions)} 只股票")
        except Exception as e:
            logger.warning(f"获取持仓信息失败，将不考虑持仓: {e}")
        
        # 创建预测器
        predictor = OpeningPredictor()
        
        # 生成预测结果（集成持仓信息）
        prediction_result = predictor.predict_opening_strategy(stock_pool, current_positions, force_refresh_history=force_refresh)

        # 自动联动：根据预测与回测风格建议，生成“待执行”的交易清单（不直接下单）
        try:
            planned_trades = []
            for p in prediction_result.stock_predictions:
                # 只做多逻辑：当recommendation==buy时，建议买入；当should_sell服务端评估为真时，建议卖出
                symbol = p.symbol
                name = p.name
                # 买入建议：按Kelly建议数量估算（若前面已置零，则不会产生极端仓位）
                if p.recommendation == 'buy' and p.current_price > 0:
                    # 计算最大可买（结合现金管理）
                    max_qty = portfolio_manager.calculate_max_buy_quantity(symbol, p.current_price)
                    # 建议买入数量：不超过最大可买，且不超过2000
                    buy_qty = min(2000, max(0, max_qty))
                    if buy_qty >= 100:
                        planned_trades.append({
                            'action': 'buy',
                            'symbol': symbol,
                            'name': name,
                            'price': p.current_price,
                            'quantity': buy_qty,
                            'note': 'opening_prediction_planned'
                        })
                # 卖出建议：调用现有风控判定（仅持仓时）
                holding = predictor._get_holding_info(symbol, current_positions)
                if holding.get('is_holding') and p.current_price > 0:
                    if predictor._should_sell_backtest_style(p, holding):
                        sell_qty = holding.get('quantity', 0)
                        if sell_qty > 0:
                            planned_trades.append({
                                'action': 'sell',
                                'symbol': symbol,
                                'name': name,
                                'price': p.current_price,
                                'quantity': sell_qty,
                                'note': 'opening_prediction_planned'
                            })
        except Exception as _e:
            planned_trades = []
        
        # 格式化结果
        formatted_result = format_prediction_result(prediction_result)
        
        return {
            "success": True,
            "data": formatted_result,
            "planned_trades": planned_trades
        }
        
    except Exception as e:
        logger.error(f"生成开盘预测失败: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "details": traceback.format_exc()
        }


# ==================== 持仓管理API ====================

@app.get("/api/portfolio")
async def get_portfolio():
    """获取投资组合概况"""
    try:
        portfolio = portfolio_manager.get_portfolio()
        
        # 转换为响应模型
        positions_response = [
            {
                "symbol": pos.symbol,
                "name": pos.name,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "profit_loss": pos.profit_loss,
                "profit_loss_pct": pos.profit_loss_pct,
                "last_updated": pos.last_updated
            }
            for pos in portfolio.positions
        ]
        
        return custom_json_response({
            "success": True,
            "data": {
                "total_assets": portfolio.total_assets,
                "available_cash": portfolio.available_cash,
                "market_value": portfolio.market_value,
                "total_profit_loss": portfolio.total_profit_loss,
                "total_profit_loss_pct": portfolio.total_profit_loss_pct,
                "positions": positions_response,
                "position_count": portfolio.position_count
            }
        })
        
    except Exception as e:
        logger.error(f"获取投资组合失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"获取投资组合失败: {str(e)}"
        })

@app.post("/api/portfolio/transaction")
async def add_transaction(request: TransactionRequest):
    """添加交易记录"""
    try:
        success = portfolio_manager.add_transaction(
            symbol=request.symbol,
            name=request.name,
            action=request.action,
            quantity=request.quantity,
            price=request.price,
            fee=request.fee,
            note=request.note
        )
        
        if success:
            return custom_json_response({
                "success": True,
                "message": f"交易记录添加成功: {request.action} {request.quantity}股 {request.symbol}"
            })
        else:
            return custom_json_response({
                "success": False,
                "error": "交易记录添加失败"
            })
            
    except Exception as e:
        logger.error(f"添加交易记录失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"添加交易记录失败: {str(e)}"
        })

@app.get("/api/portfolio/positions")
async def get_positions():
    """获取所有持仓"""
    try:
        positions = portfolio_manager.get_positions()
        
        positions_response = [
            {
                "symbol": pos.symbol,
                "name": pos.name,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "profit_loss": pos.profit_loss,
                "profit_loss_pct": pos.profit_loss_pct,
                "last_updated": pos.last_updated
            }
            for pos in positions
        ]
        
        return custom_json_response({
            "success": True,
            "data": positions_response
        })
        
    except Exception as e:
        logger.error(f"获取持仓失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"获取持仓失败: {str(e)}"
        })

@app.get("/api/portfolio/transactions")
async def get_transactions(limit: int = Query(50), symbol: str = Query(None)):
    """获取交易记录"""
    try:
        transactions = portfolio_manager.get_transactions(limit=limit, symbol=symbol)
        
        return custom_json_response({
            "success": True,
            "data": [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "name": t.name,
                    "action": t.action,
                    "quantity": t.quantity,
                    "price": t.price,
                    "amount": t.amount,
                    "fee": t.fee,
                    "timestamp": t.timestamp,
                    "note": t.note
                }
                for t in transactions
            ]
        })
        
    except Exception as e:
        logger.error(f"获取交易记录失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"获取交易记录失败: {str(e)}"
        })

@app.get("/api/portfolio/cash")
async def get_available_cash():
    """获取可用资金"""
    try:
        cash = portfolio_manager.get_available_cash()
        return custom_json_response({
            "success": True,
            "data": {"available_cash": cash}
        })
        
    except Exception as e:
        logger.error(f"获取可用资金失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"获取可用资金失败: {str(e)}"
        })

@app.post("/api/portfolio/cash")
async def update_cash(request: Dict[str, Any]):
    """更新可用资金"""
    try:
        cash = request.get("amount", 0)
        portfolio_manager.update_cash(cash)
        return custom_json_response({
            "success": True,
            "message": f"资金更新成功: ¥{cash:.2f}"
        })
        
    except Exception as e:
        logger.error(f"更新资金失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"更新资金失败: {str(e)}"
        })

@app.get("/api/portfolio/max_buy")
async def get_max_buy_quantity(symbol: str = Query(...), price: float = Query(...)):
    """计算最大可买数量"""
    try:
        max_quantity = portfolio_manager.calculate_max_buy_quantity(symbol, price)
        return custom_json_response({
            "success": True,
            "data": {
                "symbol": symbol,
                "price": price,
                "max_quantity": max_quantity,
                "max_amount": max_quantity * price
            }
        })
        
    except Exception as e:
        logger.error(f"计算最大买入数量失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"计算最大买入数量失败: {str(e)}"
        })

@app.put("/api/portfolio/position")
async def edit_position(request: PositionEditRequest):
    """编辑持仓信息"""
    try:
        success = portfolio_manager.edit_position(
            symbol=request.symbol,
            new_quantity=request.quantity,
            new_avg_cost=request.avg_cost,
            note=request.note
        )
        
        if success:
            return custom_json_response({
                "success": True,
                "message": f"持仓编辑成功: {request.symbol}"
            })
        else:
            return custom_json_response({
                "success": False,
                "error": "持仓编辑失败"
            })
            
    except Exception as e:
        logger.error(f"编辑持仓失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"编辑持仓失败: {str(e)}"
        })

@app.delete("/api/portfolio/position/{symbol}")
async def delete_position(symbol: str):
    """删除持仓"""
    try:
        success = portfolio_manager.delete_position(symbol)
        
        if success:
            return custom_json_response({
                "success": True,
                "message": f"持仓删除成功: {symbol}"
            })
        else:
            return custom_json_response({
                "success": False,
                "error": "持仓删除失败"
            })
            
    except Exception as e:
        logger.error(f"删除持仓失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"删除持仓失败: {str(e)}"
        })

@app.post("/api/portfolio/batch_transactions")
async def batch_add_transactions(request: BatchTransactionRequest):
    """批量添加交易记录"""
    try:
        transactions_data = [
            {
                'symbol': t.symbol,
                'name': t.name,
                'action': t.action,
                'quantity': t.quantity,
                'price': t.price,
                'fee': t.fee,
                'note': t.note
            }
            for t in request.transactions
        ]
        
        results = portfolio_manager.batch_add_transactions(transactions_data)
        
        return custom_json_response({
            "success": True,
            "data": results
        })
        
    except Exception as e:
        logger.error(f"批量添加交易记录失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"批量添加交易记录失败: {str(e)}"
        })

@app.get("/api/portfolio/analysis")
async def get_portfolio_analysis():
    """获取持仓分析"""
    try:
        analysis = portfolio_manager.get_position_analysis()
        return custom_json_response({
            "success": True,
            "data": analysis
        })
        
    except Exception as e:
        logger.error(f"获取持仓分析失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"获取持仓分析失败: {str(e)}"
        })

@app.post("/api/portfolio/sync_stock_pool")
async def sync_with_stock_pool(stock_pool_symbols: List[str]):
    """与股票池同步分析"""
    try:
        sync_result = portfolio_manager.sync_with_stock_pool(stock_pool_symbols)
        return custom_json_response({
            "success": True,
            "data": sync_result
        })
        
    except Exception as e:
        logger.error(f"股票池同步分析失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"股票池同步分析失败: {str(e)}"
        })

@app.post("/api/portfolio/fix_stock_names")
async def fix_stock_names():
    """修复股票名称"""
    try:
        result = portfolio_manager.fix_stock_names()
        return custom_json_response(result)
    except Exception as e:
        logger.error(f"修复股票名称失败: {e}")
        return custom_json_response({
            "success": False,
            "error": f"修复股票名称失败: {str(e)}"
        })

# 健康检查端点
@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "message": "Auto-GPT-Stock API is running",
        "version": "0.1.0",
        "environment": ENVIRONMENT,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": ENVIRONMENT
    }

# 启动配置
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=ENVIRONMENT == "development"
    )


