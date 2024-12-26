from datetime import datetime, timedelta
from typing import List, Union
from langchain.tools import Tool, BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_experimental.utilities import PythonREPL
import re

class PythonCodeParser(BaseOutputParser):
    """从LLM返回的文本中提取Python代码"""

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

class DateTool:
    """日期工具的数据类"""
    def __init__(
            self,
            llm: Union[BaseLanguageModel, BaseChatModel],
            prompt_file: str = "./prompts/tools/date_tools.txt",
            verbose: bool = False
    ):
        self.llm = llm
        self.verbose = verbose
        self.prompt = PromptTemplate.from_file(
            template_file=prompt_file,
            input_variables=["query", "example", "description"],
            encoding="utf-8"
        )

    def run(self, query: str) -> str:
        """处理日期查询"""
        try:
            # 构建提示
            chain = self.prompt | self.llm | StrOutputParser()
            
            # 使用流式输出
            response = ""
            for chunk in chain.stream({
                "query": query,
                "example": """
from datetime import datetime, timedelta

today = datetime.now()
date_mapping = {
    "今天": today,
    "昨天": today - timedelta(days=1),
    "明天": today + timedelta(days=1),
    "后天": today + timedelta(days=2),
    "大后天": today + timedelta(days=3),
    "前天": today - timedelta(days=2),
    "大前天": today - timedelta(days=3),
}

query = "今天"  # 示例查询
if query in date_mapping:
    target_date = date_mapping[query]
    print(f"{query}是 {target_date.strftime('%Y-%m-%d')}")
else:
    print(f"今天是 {today.strftime('%Y-%m-%d')}")
""",
                "description": "获取指定日期（今天、昨天、明天等）的具体日期值"
            }):
                response += chunk
                if self.verbose:
                    print(chunk, end="", flush=True)

            # 解析LLM生成的Python代码
            code = PythonCodeParser().parse(response)
            if code:
                # 执行代码并返回结果
                return PythonREPL().run(code)
            else:
                # 如果没有找到可执行的代码，返回默认响应
                today = datetime.now()
                return f"今天是 {today.strftime('%Y-%m-%d')}"

        except Exception as e:
            import traceback
            if self.verbose:
                print(f"Error details:\n{traceback.format_exc()}")
            return f"处理日期查询时发生错误: {str(e)}"

    def as_tool(self) -> BaseTool:
        """将当前实例转换为 Langchain Tool"""
        return Tool(
            name="get_date",
            description="获取指定日期（今天、昨天、明天等）的具体日期值，支持自然语言查询",
            func=self.run
        ) 