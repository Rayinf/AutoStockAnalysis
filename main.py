# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from Agent.ReAct import ReActAgent
from Models.Factory import ChatModelFactory
from Tools import *
from Tools.PythonTool import ExcelAnalyser
from Tools.date_tools import DateTool  # 添加日期工具导入
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from Tools import TOOL_REGISTRY
from Tools.stock_quote import StockAnalyser


def launch_agent(agent: ReActAgent):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    chat_history = ChatMessageHistory()

    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, chat_history, verbose=True)
        print(f"{ai_icon}：{reply}\n")


def main():
    # 语言模型
    # llm = ChatModelFactory.get_model("deepseek")
    llm = ChatModelFactory.get_model("gpt-4o-2024-08-06")
    

    # 从 stock_quote.py 获取所有股票相关工具
    stock_tools = TOOL_REGISTRY["stock_quote"].extract_tools_from_file("./data/stock.md.txt")
    for tool in stock_tools:
        print(f"{tool.name}: {tool.description}")   

    # 创建股票分析器
    stock_analyser = StockAnalyser(
        llm=llm,
        prompt_file="./prompts/tools/stock_quote.txt",
        verbose=True
    )
    stock_analyser.set_tools(stock_tools)

    # 自定义工具集
    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(
            llm=llm,
            prompt_file="./prompts/tools/excel_analyser.txt",
            verbose=True
        ).as_tool(),
        stock_analyser.as_tool(),
        DateTool(  # 添加日期工具
            llm=llm,
            prompt_file="./prompts/tools/date_tools.txt",
            verbose=True
        ).as_tool(),
    ]
    
    # 定义智能体
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        work_dir="./data",
        main_prompt_file="./prompts/main/main.txt",
        max_thought_steps=20,
    )

    # 运行智能体
    launch_agent(agent)


if __name__ == "__main__":
    main()
