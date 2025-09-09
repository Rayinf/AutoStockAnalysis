"""
统一的指标/因子流水线。
优先使用Qlib（若可用），否则回退到Pandas实现。
输出 series schema: { name, x: 日期[], y: 数值[] }，与前端ECharts兼容。
"""

from __future__ import annotations

from typing import Any, Dict, List
from datetime import date, datetime

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # noqa: N816


def _to_date_str(v: Any) -> str:
    try:
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        if pd is not None and isinstance(v, getattr(pd, 'Timestamp', ())) :  # pandas Timestamp
            return v.isoformat()
    except Exception:
        pass
    return str(v)


def _series(name: str, dates: List[Any], values: List[Any]) -> Dict[str, Any]:
    # 统一将日期转为字符串，避免JSON序列化报错
    x = [_to_date_str(d) for d in dates]
    return {"name": name, "x": x, "y": values}


def calc_ma(df, window: int) -> Dict[str, Any]:
    name = f"MA{window}"
    ma = df["收盘"].rolling(window=window).mean()
    return _series(name, df["日期"].tolist(), [None if v != v else float(v) for v in ma.tolist()])


def calc_rsi14(df) -> Dict[str, Any]:
    import pandas as pd

    delta = df["收盘"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return _series("RSI14", df["日期"].tolist(), [None if pd.isna(v) else float(v) for v in rsi.tolist()])


def calc_macd(df) -> Dict[str, Any]:
    ema12 = df["收盘"].ewm(span=12).mean()
    ema26 = df["收盘"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return {
        "MACD": _series("MACD", df["日期"].tolist(), [float(v) for v in macd.tolist()]),
        "MACD_SIGNAL": _series("MACD_SIGNAL", df["日期"].tolist(), [float(v) for v in signal.tolist()]),
        "MACD_HIST": _series("MACD_HIST", df["日期"].tolist(), [float(v) for v in hist.tolist()]),
    }


def calc_boll(df, window: int = 20, n_std: float = 2.0) -> Dict[str, Any]:
    import pandas as pd

    ma = df["收盘"].rolling(window=window).mean()
    std = df["收盘"].rolling(window=window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return {
        "BOLL_UP": _series("BOLL_UP", df["日期"].tolist(), [None if pd.isna(v) else float(v) for v in upper.tolist()]),
        "BOLL_MID": _series("BOLL_MID", df["日期"].tolist(), [None if pd.isna(v) else float(v) for v in ma.tolist()]),
        "BOLL_LOW": _series("BOLL_LOW", df["日期"].tolist(), [None if pd.isna(v) else float(v) for v in lower.tolist()]),
    }


def calc_obv(df) -> Dict[str, Any]:
    import numpy as np

    close = df["收盘"].values
    vol = df["成交量"].values if "成交量" in df.columns else None
    if vol is None:
        return _series("OBV", df["日期"].tolist(), [None] * len(df))
    obv = [0.0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + float(vol[i]))
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - float(vol[i]))
        else:
            obv.append(obv[-1])
    return _series("OBV", df["日期"].tolist(), [float(v) for v in obv])


def calc_atr(df, window: int = 14) -> Dict[str, Any]:
    import numpy as np

    high = df["最高"].values
    low = df["最低"].values
    close = df["收盘"].values
    tr = [0.0]
    for i in range(1, len(df)):
        true_range = max(
            float(high[i] - low[i]),
            float(abs(high[i] - close[i - 1])),
            float(abs(low[i] - close[i - 1])),
        )
        tr.append(true_range)
    import pandas as pd

    atr = pd.Series(tr).rolling(window=window).mean().tolist()
    return _series("ATR", df["日期"].tolist(), [None if v != v else float(v) for v in atr])


def build_factors(df, analyses: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    # 基础收盘价序列，便于前端绘图
    if "收盘" in df.columns:
        result["close"] = _series("close", df["日期"].tolist(), [float(v) for v in df["收盘"].tolist()])
    if "MA5" in analyses:
        result["MA5"] = calc_ma(df, 5)
    if "MA20" in analyses:
        result["MA20"] = calc_ma(df, 20)
    if "MA60" in analyses:
        result["MA60"] = calc_ma(df, 60)
    if "RSI14" in analyses:
        result["RSI14"] = calc_rsi14(df)
    if "MACD" in analyses:
        result.update(calc_macd(df))
    if "BOLL" in analyses:
        result.update(calc_boll(df))
    if "OBV" in analyses:
        result["OBV"] = calc_obv(df)
    if "ATR" in analyses:
        result["ATR"] = calc_atr(df)
    return result


