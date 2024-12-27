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

                if all([name_match, desc_match, example_match, data_match]):
                    name = name_match.group(1).strip()
                    description = desc_match.group(1).strip()
                    example = example_match.group(1).strip()
                    data_sample = data_match.group(1).strip()
                    
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

        return next((t for t in self.tools if t.name == tool_name), None)

    def analyse(self, query: str) -> str:
        """分析股票数据"""
        tool = self.select_tool(query)
        if not tool:
            return "无法找到合适的工具来处理您的查询"

        if self.verbose:
            print(f"\n选择工具: {tool.name}")

        try:
            # 使用流式输出
            chain = self.code_prompt | self.llm | StrOutputParser()
            response = ""
            for chunk in chain.stream({
                "query": query,
                "name": tool.name,
                "description": tool.description,
                "example": tool.metadata.get("example", ""),
                "data_sample": tool.metadata.get("data_sample", "")
            }):
                response += chunk
                if self.verbose:
                    print(chunk, end="", flush=True)
            
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
