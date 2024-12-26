from .Tools import (
    document_qa_tool,
    document_generation_tool,
    email_tool,
    excel_inspection_tool,
    directory_inspection_tool,
    finish_placeholder,
)
from .stock_quote import stock_quote_tool

# 首先定义 TOOL_REGISTRY
TOOL_REGISTRY = {}

# 然后更新注册表
TOOL_REGISTRY.update({
    "stock_quote": stock_quote_tool,
})