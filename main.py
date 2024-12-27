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
import os


def launch_agent(agent: ReActAgent):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    chat_history = ChatMessageHistory()
    print(f"模型名称：{os.getenv('MODEL_NAME', 'gpt-4-turbo')}")

    while True:
        task = input(f"{ai_icon}：我是智能金融分析助手，可以提供股票、基金、期货的信息，有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, chat_history, verbose=True)
        print(f"{ai_icon}：{reply}\n")


def main():
    # 从环境变量获取模型配置
    llm = ChatModelFactory.get_model(os.getenv("MODEL_NAME", "gpt-4-turbo"))
    

    # 从 stock.md.txt 获取所有股票相关工具
    stock_tools = TOOL_REGISTRY["stock_quote"].extract_tools_from_file("./data/stock.md.txt")
    # for tool in stock_tools:
    #     print(f"{tool.name}: {tool.description}")   

    # 创建股票分析器
    stock_analyser = StockAnalyser(
        llm=llm,
        prompt_file="./prompts/tools/stock_quote.txt",
        verbose=True
    )
    stock_analyser.set_tools(stock_tools)

    # 从 futures.md.txt 获取所有期货基础信息相关工具
    futures_tools = TOOL_REGISTRY["stock_quote"].extract_tools_from_file("./data/futures.md.txt")
    # for tool in futures_tools:
    #     print(f"{tool.name}: {tool.description}")   
    # 创建期货基础信息分析器
    futures_analyser = StockAnalyser(
        llm=llm,
        prompt_file="./prompts/tools/stock_quote.txt",
        verbose=True
    )
    futures_analyser.set_tools(futures_tools)

    # 从 fund_public_data.md.txt 获取所有基金基础信息相关工具
    fund_public_data_tools = TOOL_REGISTRY["stock_quote"].extract_tools_from_file("./data/fund_public.md.txt")
    # for tool in fund_public_data_tools:
    #     print(f"{tool.name}: {tool.description}")   
    # 创建基金基础信息分析器
    fund_public_data_analyser = StockAnalyser(
        llm=llm,
        prompt_file="./prompts/tools/stock_quote.txt",
        verbose=True
    )
    fund_public_data_analyser.set_tools(fund_public_data_tools)

    # 从 fund_private.md.txt 获取所有私募基金基础信息相关工具
    fund_private_data_tools = TOOL_REGISTRY["stock_quote"].extract_tools_from_file("./data/fund_private.md.txt")
    # for tool in fund_private_data_tools:
    #     print(f"{tool.name}: {tool.description}")   
    # 创建私募基金基础信息分析器
    fund_private_data_analyser = StockAnalyser(
        llm=llm,
        prompt_file="./prompts/tools/stock_quote.txt",
        verbose=True
    )
    fund_private_data_analyser.set_tools(fund_private_data_tools)

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
        futures_analyser.as_tool("futures_analysis","主要提供金融期货和商品期货相关的数据"),
        fund_public_data_analyser.as_tool("fund_public_data_analysis","主要提供公募基金基础信息相关的数据"),
        fund_private_data_analyser.as_tool("fund_private_data_analysis","主要提供私募基金基础信息相关的数据"),
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
