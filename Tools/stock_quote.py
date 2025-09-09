"""
代码说明：
    从文件 stock.md.txt 中提取接口名称，描述，输入参数，和示例代码，以及数据示例，以便于 AI 理解每一个工具的用法

添加数据示例提取：
    - 增加正则表达式 data_sample_pattern，用于匹配 数据示例 部分
    - 捕获 数据示例 中的内容并添加到 Tool 对象中

输出结果：
运行脚本后，每个工具都会完整记录以下信息：
    - 接口名称
    - 描述
    - 示例代码
    - 数据示例
"""

import re
from typing import Union, List
from langchain.tools import Tool, StructuredTool
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_experimental.utilities import PythonREPL

class PythonCodeParser(BaseOutputParser):
    """从OpenAI返回的文本中提取Python代码。"""

    @staticmethod
    def __remove_marked_lines(input_str: str) -> str:
        lines = input_str.strip().split('\n')
        if lines and lines[0].strip().startswith('```'):
            del lines[0]
        if lines and lines[-1].strip().startswith('```'):
            del lines[-1]
        return '\n'.join(lines)

    def parse(self, text: str) -> str:
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if len(python_code_blocks) > 0:
            return self.__remove_marked_lines(python_code_blocks[0])
        return None

class StockTool:
    """单个股票工具的数据类"""
    def __init__(self, name: str, description: str, example: str, data_sample: str, output_params: str = None):
        self.name = name
        self.description = description
        self.example = example
        self.data_sample = data_sample
        self.output_params = output_params

    @classmethod
    def extract_tools_from_file(cls, file_path: str) -> List[BaseTool]:
        """从文档中提取工具定义"""
        tools = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 分割每个接口部分
            sections = re.split(r'#{3,5}\s+', content)[1:]
            
            for section in sections:
                # 提取接口信息
                name_match = re.search(r"接口:\s*([\w_]+)", section)
                desc_match = re.search(r"描述:\s*(.*?)(?=\n\n|输入参数)", section, re.DOTALL)
                example_match = re.search(r"接口示例\n\n```python\n(.*?)\n```", section, re.DOTALL)
                data_match = re.search(r"数据示例\n\n```\n(.*?)\n```", section, re.DOTALL)
                output_match = re.search(r"输出参数.*?\n\|(.*?)(?=\n\n)", section, re.DOTALL)

                if name_match and desc_match:  # 只需要名称和描述，示例和数据样本可以缺失
                    name = name_match.group(1).strip()
                    description = desc_match.group(1).strip()

                    # 处理示例代码，如果没有则使用默认的
                    if example_match:
                        example = example_match.group(1).strip()
                    else:
                        example = f"import akshare as ak\nresult = ak.{name}()\nprint(result)"

                    # 处理数据样本，如果没有则使用默认的
                    if data_match:
                        data_sample = data_match.group(1).strip()
                    else:
                        data_sample = "返回数据格式请参考AKShare官方文档"
                    
                    # 处理输出参数
                    output_params = ""
                    if output_match:
                        # 解析输出参数表格内容
                        params_content = output_match.group(1)
                        params_rows = [row.strip() for row in params_content.split('\n') if row.strip()]
                        output_params = "\n输出参数：\n" + "\n".join(
                            f"- {param.strip()}" 
                            for param in params_rows 
                            if not all(cell.strip() == '-' for cell in param.split('|'))
                        )
                        # 将输出参数添加到描述中
                        description = f"{description}\n{output_params}"

                    # 创建工具实例
                    tool_instance = cls(
                        name=name,
                        description=description,
                        example=example,
                        data_sample=data_sample,
                        output_params=output_params
                    )

                    # 创建 Langchain Tool
                    tool = Tool(
                        name=name,
                        description=description,
                        func=lambda x="", tool=tool_instance: tool_instance.run(x)
                    )
                    
                    # 将完整信息存储在 tool 对象的 metadata 中
                    tool.metadata = {
                        "example": example,
                        "data_sample": data_sample,
                        "output_params": output_params
                    }
                    
                    tools.append(tool)

        return tools

    def run(self, args: str = "") -> str:
        """执行工具函数"""
        code = f"import akshare as ak\nresult = ak.{self.name}()"
        print(f"\n执行代码:\n```python\n{code}\n```\n")
        return code

class StockAnalyser:
    """
    使用 akshare 接口分析股票数据。
    输入中必须包含具体的分析需求，例如：查询某只股票的行情，或者获取某个指数的成分股等。
    """

    def __init__(
            self,
            llm: Union[BaseLanguageModel, BaseChatModel],
            prompt_file: str = "./prompts/tools/stock_quote.txt",
            verbose: bool = False
    ):
        self.llm = llm
        # 修改 prompt 模板，增加 example 变量
        self.code_prompt = PromptTemplate.from_file(
            template_file=prompt_file,
            input_variables=["name", "description", "example", "data_sample", "query"],
            encoding="utf-8"
        )
        # 简化工具选择的 prompt，只使用名称和描述
        self.tool_select_prompt = PromptTemplate.from_template(
            """基于用户的查询，从以下工具中选择最合适的一个：

{tools}

用户查询: {query}

请只返回工具名称，不需要其他解释。
""")
        self.verbose = verbose
        self.tools = []

    def set_tools(self, tools: List[BaseTool]) -> None:
        """设置可用的工具列表"""
        self.tools = tools

    def select_tool(self, query: str) -> BaseTool:
        """根据查询选择合适的工具"""
        # 只使用工具名称和描述来构建工具列表文本
        tools_text = "\n\n".join([
            f"工具名称: {tool.name}\n描述: {tool.description}"
            for tool in self.tools
        ])

        # 使用 LLM 选择工具
        chain = self.tool_select_prompt | self.llm | StrOutputParser()
        tool_name = chain.invoke({
            "tools": tools_text,
            "query": query
        }).strip()

        # 清洗工具名，去除可能的引号、反引号或多余文本
        tool_name = tool_name.strip().strip("`\"")
        # 仅取第一段连续的工具名标识（防止模型返回解释性文字）
        tool_name = tool_name.split()[0] if tool_name else tool_name

        return next((t for t in self.tools if t.name == tool_name), None)

    def analyse(self, query: str, mode: str | None = None) -> str:
        """分析股票数据"""
        # mode: quick / llm / None(自动)
        if mode != "llm":
            # 先尝试匹配常见的技术指标与请求的快速路径，尽量不走LLM
            quick_result = self.__try_quick_analysis(query)
            if quick_result is not None:
                return quick_result
            if mode == "quick":
                return "未命中快捷分析，请补充更明确的指标/代码/时间范围。"

        tool = self.select_tool(query)
        if not tool:
            return "无法找到合适的工具来处理您的查询"

        if self.verbose:
            print(f"\n选择工具: {tool.name}")

        try:
            # 改为非流式，避免阻塞与中断
            chain = self.code_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "name": tool.name,
                "description": tool.description,
                "example": tool.metadata.get("example", ""),
                "data_sample": tool.metadata.get("data_sample", "")
            })

            code = PythonCodeParser().parse(response)
            if code:
                return query + "\n" + PythonREPL().run(code)
            else:
                return "没有找到可执行的Python代码"
        except Exception as e:
            # 增加更详细的错误信息
            import traceback
            error_details = traceback.format_exc()
            print(f"详细错误信息:\n{error_details}")
            return f"生成代码时发生错误: {str(e)}"

    # ----------------------
    # 辅助：快速路径（无需LLM）
    # ----------------------
    def __try_quick_analysis(self, query: str) -> str | None:
        """
        汇总快捷路径：按优先级依次尝试命中。
        1) MA/mdX/均线
        2) RSI
        3) MACD
        4) BOLL/布林
        5) 单票实时行情
        6) 涨幅榜/跌幅榜（Top N）
        未命中返回 None。
        """
        handlers = [
            self.__try_quick_ma_analysis,
            self.__try_quick_rsi_analysis,
            self.__try_quick_macd_analysis,
            self.__try_quick_boll_analysis,
            self.__try_quick_realtime_quote,
            self.__try_quick_rank_board,
        ]
        for h in handlers:
            res = h(query)
            if res is not None:
                return res
        return None

    def __try_quick_ma_analysis(self, query: str) -> str | None:
        """
        针对常见的 MA / mdX（如 md20/MA20）分析需求，直接走 AKShare + pandas 的本地分析路径，
        避免参数缺失时LLM反复追问，提升鲁棒性。
        返回 None 表示未命中快速路径。
        """
        import re
        from datetime import datetime, timedelta

        # 命中条件：包含 MA 或 md（不区分大小写）；去除\b以兼容中英文混排
        if not re.search(r"(ma|md)\s*\d+", query, flags=re.IGNORECASE):
            # 也支持“均线”关键词
            if "均线" not in query:
                return None

        # 提取周期，如 MA20/md20，默认20
        ma_match = re.search(r"(?:ma|md)\s*(\d+)", query, flags=re.IGNORECASE)
        ma_period = int(ma_match.group(1)) if ma_match else 20

        # 提取股票代码：支持 sz000001 / sh600000 / 6位数字（放宽边界）
        sym_match = re.search(r"((?:sz|sh)?\d{6})", query, flags=re.IGNORECASE)
        if not sym_match:
            return None  # 缺少明确股票代码则不走快速路径

        raw_symbol = sym_match.group(0)
        # 归一化为 AKShare stock_zh_a_hist 所需的 6位数字
        normalized_symbol = re.sub(r"^(?:sz|sh)", "", raw_symbol, flags=re.IGNORECASE)

        # 提取时间范围：支持“最近X天/交易日”，否则默认最近120天
        days_match = re.search(r"最近\s*(\d+)\s*(?:个)?(?:交易)?天", query)
        lookback_days = int(days_match.group(1)) if days_match else 120

        end_date_dt = datetime.today()
        start_date_dt = end_date_dt - timedelta(days=lookback_days)
        start_date = start_date_dt.strftime("%Y%m%d")
        end_date = end_date_dt.strftime("%Y%m%d")

        # 直接调用 AKShare 获取数据并计算 MA
        try:
            import akshare as ak
            import pandas as pd

            df = ak.stock_zh_a_hist(
                symbol=normalized_symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )

            if df is None or df.empty:
                return "未获取到历史数据，请检查股票代码或时间范围。"

            # 确保列存在
            if "收盘" not in df.columns:
                return "数据中缺少‘收盘’列，无法计算均线。"

            df["MA"] = df["收盘"].rolling(window=ma_period).mean()

            latest = df.tail(1)
            if latest.empty:
                return "历史数据不足，无法给出结论。"

            latest_date = latest["日期"].iloc[0]
            latest_close = float(latest["收盘"].iloc[0])
            latest_ma = float(latest["MA"].iloc[0]) if pd.notna(latest["MA"].iloc[0]) else None

            # 计算最近一段时间的MA趋势（简单：近N天MA差分）
            trend_window = min(5, len(df))
            ma_trend = None
            if trend_window >= 2 and df["MA"].notna().sum() >= 2:
                ma_diff = df["MA"].dropna().tail(trend_window).diff().mean()
                if ma_diff > 0:
                    ma_trend = "上升"
                elif ma_diff < 0:
                    ma_trend = "下降"
                else:
                    ma_trend = "震荡"

            rel_pos = None
            if latest_ma is not None:
                rel_pos = "上方" if latest_close >= latest_ma else "下方"

            lines = [
                f"股票 {raw_symbol}（归一: {normalized_symbol}） {ma_period}日均线分析（{start_date} 至 {end_date}）",
                f"- 最新交易日: {latest_date}",
                f"- 收盘价: {latest_close:.2f}",
                f"- {ma_period}日均线: {latest_ma:.2f}" if latest_ma is not None else f"- {ma_period}日均线: 数据不足",
            ]

            if rel_pos and latest_ma is not None:
                lines.append(f"- 价格位置: 位于MA{ma_period}{rel_pos}")
            if ma_trend:
                lines.append(f"- MA趋势: {ma_trend}")

            return "\n".join(lines)
        except Exception as inner_e:
            # 快速路径失败则回退到原有流程
            if self.verbose:
                print(f"快速MA分析失败，回退至LLM路径: {inner_e}")
            return None

    def __extract_symbol_and_days(self, query: str) -> tuple[str | None, str, str]:
        """
        提取股票代码与时间窗口（开始、结束日期字符串：YYYYMMDD）。
        若未指定“最近N天”，默认120天。
        返回：(normalized_symbol or None, start_date, end_date)
        """
        import re
        from datetime import datetime, timedelta

        sym_match = re.search(r"((?:sz|sh)?\d{6})", query, flags=re.IGNORECASE)
        normalized_symbol = None
        if sym_match:
            raw_symbol = sym_match.group(0)
            normalized_symbol = re.sub(r"^(?:sz|sh)", "", raw_symbol, flags=re.IGNORECASE)

        days_match = re.search(r"最近\s*(\d+)\s*(?:个)?(?:交易)?天", query)
        lookback_days = int(days_match.group(1)) if days_match else 120

        from datetime import datetime, timedelta
        end_date_dt = datetime.today()
        start_date_dt = end_date_dt - timedelta(days=lookback_days)
        start_date = start_date_dt.strftime("%Y%m%d")
        end_date = end_date_dt.strftime("%Y%m%d")
        return normalized_symbol, start_date, end_date

    def __try_quick_rsi_analysis(self, query: str) -> str | None:
        import re
        if not re.search(r"rsi", query, flags=re.IGNORECASE):
            return None
        # 默认14
        m = re.search(r"rsi\s*(\d+)", query, flags=re.IGNORECASE)
        period = int(m.group(1)) if m else 14

        symbol, start_date, end_date = self.__extract_symbol_and_days(query)
        if not symbol:
            return None
        try:
            import akshare as ak
            import pandas as pd
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
            if df is None or df.empty:
                return "未获取到历史数据，请检查股票代码或时间范围。"
            if "收盘" not in df.columns:
                return "数据中缺少‘收盘’列，无法计算RSI。"
            delta = df["收盘"].diff()
            gain = delta.clip(lower=0).rolling(window=period).mean()
            loss = (-delta.clip(upper=0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, pd.NA)
            rsi = 100 - (100 / (1 + rs))
            df["RSI"] = rsi

            latest = df.tail(1)
            if latest.empty:
                return "历史数据不足，无法给出结论。"
            latest_date = latest["日期"].iloc[0]
            latest_close = float(latest["收盘"].iloc[0])
            latest_rsi = float(latest["RSI"].iloc[0]) if pd.notna(latest["RSI"].iloc[0]) else None
            state = None
            if latest_rsi is not None:
                if latest_rsi >= 70:
                    state = "超买"
                elif latest_rsi <= 30:
                    state = "超卖"
                else:
                    state = "中性"
            lines = [
                f"RSI分析（{start_date} 至 {end_date}） - {symbol}",
                f"- 最新交易日: {latest_date}",
                f"- 收盘价: {latest_close:.2f}",
                f"- RSI({period}): {latest_rsi:.2f}" if latest_rsi is not None else f"- RSI({period}): 数据不足",
            ]
            if state:
                lines.append(f"- 状态: {state}")
            return "\n".join(lines)
        except Exception:
            return None

    def __try_quick_macd_analysis(self, query: str) -> str | None:
        import re
        if not re.search(r"macd", query, flags=re.IGNORECASE):
            return None
        symbol, start_date, end_date = self.__extract_symbol_and_days(query)
        if not symbol:
            return None
        try:
            import akshare as ak
            import pandas as pd
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
            if df is None or df.empty:
                return "未获取到历史数据，请检查股票代码或时间范围。"
            if "收盘" not in df.columns:
                return "数据中缺少‘收盘’列，无法计算MACD。"
            ema12 = df["收盘"].ewm(span=12).mean()
            ema26 = df["收盘"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            hist = macd - signal
            df["MACD"], df["Signal"], df["Hist"] = macd, signal, hist
            latest = df.tail(1)
            if latest.empty:
                return "历史数据不足，无法给出结论。"
            latest_date = latest["日期"].iloc[0]
            latest_macd = float(latest["MACD"].iloc[0])
            latest_signal = float(latest["Signal"].iloc[0])
            latest_hist = float(latest["Hist"].iloc[0])
            cross = "金叉" if latest_macd >= latest_signal else "死叉"
            lines = [
                f"MACD分析（{start_date} 至 {end_date}） - {symbol}",
                f"- 最新交易日: {latest_date}",
                f"- MACD: {latest_macd:.4f}",
                f"- Signal: {latest_signal:.4f}",
                f"- 柱状图(Hist): {latest_hist:.4f}",
                f"- 信号: {cross}",
            ]
            return "\n".join(lines)
        except Exception:
            return None

    def __try_quick_boll_analysis(self, query: str) -> str | None:
        import re
        if not ("布林" in query or re.search(r"\bboll\b", query, flags=re.IGNORECASE)):
            return None
        # 默认20
        m = re.search(r"boll\s*(\d+)", query, flags=re.IGNORECASE)
        period = int(m.group(1)) if m else 20
        symbol, start_date, end_date = self.__extract_symbol_and_days(query)
        if not symbol:
            return None
        try:
            import akshare as ak
            import pandas as pd
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
            if df is None or df.empty:
                return "未获取到历史数据，请检查股票代码或时间范围。"
            if "收盘" not in df.columns:
                return "数据中缺少‘收盘’列，无法计算布林带。"
            ma = df["收盘"].rolling(window=period).mean()
            std = df["收盘"].rolling(window=period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            df["BOLL_UP"], df["BOLL_MID"], df["BOLL_LOW"] = upper, ma, lower
            latest = df.tail(1)
            if latest.empty:
                return "历史数据不足，无法给出结论。"
            latest_date = latest["日期"].iloc[0]
            up = float(latest["BOLL_UP"].iloc[0]) if pd.notna(latest["BOLL_UP"].iloc[0]) else None
            mid = float(latest["BOLL_MID"].iloc[0]) if pd.notna(latest["BOLL_MID"].iloc[0]) else None
            low = float(latest["BOLL_LOW"].iloc[0]) if pd.notna(latest["BOLL_LOW"].iloc[0]) else None
            close = float(latest["收盘"].iloc[0])
            pos = None
            if all(v is not None for v in [up, mid, low]):
                if close >= up:
                    pos = "上轨外"
                elif close >= mid:
                    pos = "上轨-中轨区间"
                elif close >= low:
                    pos = "中轨-下轨区间"
                else:
                    pos = "下轨外"
            lines = [
                f"布林带分析（{start_date} 至 {end_date}） - {symbol}",
                f"- 最新交易日: {latest_date}",
                f"- 上轨: {up:.2f}" if up is not None else "- 上轨: 数据不足",
                f"- 中轨: {mid:.2f}" if mid is not None else "- 中轨: 数据不足",
                f"- 下轨: {low:.2f}" if low is not None else "- 下轨: 数据不足",
            ]
            if pos:
                lines.append(f"- 价格位置: {pos}")
            return "\n".join(lines)
        except Exception:
            return None

    def __try_quick_realtime_quote(self, query: str) -> str | None:
        # 关键词：实时/行情/报价/现价
        if not any(k in query for k in ["实时", "行情", "报价", "现价"]):
            return None
        import re
        sym_match = re.search(r"((?:sz|sh)?\d{6})", query, flags=re.IGNORECASE)
        if not sym_match:
            return None
        raw_symbol = sym_match.group(0)
        normalized_symbol = re.sub(r"^(?:sz|sh)", "", raw_symbol, flags=re.IGNORECASE)
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return "获取实时行情失败。"
            # 列“代码”通常是6位数字
            row = df[df["代码"].astype(str) == normalized_symbol]
            if row.empty:
                # 有时为“000001.SZ”，尝试去后缀
                if "代码" in df.columns:
                    row = df[df["代码"].astype(str).str.contains(normalized_symbol)]
            if row.empty:
                return "未在实时行情中找到该股票。"
            row = row.iloc[0]
            name = row.get("名称", "-")
            price = row.get("最新价", "-")
            pct = row.get("涨跌幅", "-")
            vol = row.get("成交量", "-")
            amount = row.get("成交额", "-")
            return (
                f"实时行情 - {raw_symbol}（归一: {normalized_symbol}）\n"
                f"- 名称: {name}\n- 最新价: {price}\n- 涨跌幅: {pct}%\n- 成交量: {vol}\n- 成交额: {amount}"
            )
        except Exception:
            return None

    def __try_quick_rank_board(self, query: str) -> str | None:
        # 关键词：涨幅榜/跌幅榜/涨跌幅 TopN
        kw_up = any(k in query for k in ["涨幅榜", "涨得", "上涨最多", "涨幅前", "top涨", "涨停"])
        kw_down = any(k in query for k in ["跌幅榜", "跌得", "下跌最多", "跌幅前", "top跌"])
        if not (kw_up or kw_down) and "涨跌幅" not in query:
            return None
        import re
        m = re.search(r"(?:前|top)\s*(\d+)", query, flags=re.IGNORECASE)
        topn = int(m.group(1)) if m else 10
        try:
            import akshare as ak
            import pandas as pd
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty or "涨跌幅" not in df.columns:
                return "获取涨跌幅榜失败。"
            # 确保数值
            tmp = df.copy()
            tmp["涨跌幅"] = pd.to_numeric(tmp["涨跌幅"], errors="coerce")
            if kw_down:
                board = tmp.sort_values("涨跌幅").head(topn)
                title = f"跌幅榜 Top{topn}"
            else:
                board = tmp.sort_values("涨跌幅", ascending=False).head(topn)
                title = f"涨幅榜 Top{topn}"
            lines = [title]
            for _, r in board.iterrows():
                code = r.get("代码", "-")
                name = r.get("名称", "-")
                pct = r.get("涨跌幅", "-")
                price = r.get("最新价", "-")
                lines.append(f"- {code} {name} | {pct}% | {price}")
            return "\n".join(lines)
        except Exception:
            return None

    def as_tool(self, name: str = None, description: str = None) -> Tool:
        """将当前实例转换为 Langchain Tool
        
        Args:
            name (str, optional): 工具名称. 默认为 "stock_analysis"
            description (str, optional): 工具描述. 默认使用类文档
            
        Returns:
            Tool: Langchain Tool 实例
        """
        return Tool(
            name=name or "stock_analysis",
            description=description or self.__class__.__doc__.replace("\n", ""),
            func=self.analyse
        )

# 导出工具类
stock_quote_tool = StockTool
