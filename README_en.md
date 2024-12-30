# AI Intelligent Analysis Assistant - LLM-based Akshare Data Analysis Tool

## 🎯 Project Vision

Making complex financial data analysis simple! Enable everyone to easily access and analyze financial data through natural language interaction.

---

### **Background**

`Akshare` https://akshare.akfamily.xyz/ is a powerful open-source financial data interface library covering data from multiple sectors including stocks, funds, and futures. Through its interfaces, users can conveniently access real-time market data and historical data. To simplify the query process while enhancing flexibility and intelligence, we aim to utilize LLM's (Large Language Model) ReAct Agent method to automatically parse Akshare's interface documentation and achieve automated data queries and intelligent reasoning.

By introducing the **ReAct (Reasoning and Acting)** method, we can break down users' natural language questions into logical reasoning, interface selection, parameter filling, and data query operations, thereby providing efficient and accurate query results.

---

### **System Objectives**

- Parse Akshare's interface documentation (stocks, funds, futures, etc.) to generate multiple Agents responsible for different domains of data queries.
  - Number of stock tools: 310 (https://akshare.akfamily.xyz/data/stock/stock.html)
  - Number of futures tools: 44 (https://akshare.akfamily.xyz/data/futures/futures.html)
  - Number of public fund tools: 58 (https://akshare.akfamily.xyz/data/fund/fund_public.html)
  - Number of private fund tools: 14 (https://akshare.akfamily.xyz/data/fund/fund_private.html)
- Automatically reason and select appropriate Agents and interface methods based on user's natural language input.
- Automatically generate query code, call Akshare interfaces, and return results.
- Provide easily extensible Python implementation for future feature or interface additions.
- Support inputs in Chinese, English, and other languages.
- Model enhancement: Support for Ollama, Deepseek, Qwen, OpenAI, and other models.

---

### **User Experience Examples**

```
🤖: I'm an intelligent financial analysis assistant, I can provide information about stocks, funds, and futures. How can I help you?
👨: What's the latest price of Bank of China?
🤖: The latest stock price of Bank of China is: 5.47

👨: Which public fund has the largest amount?
🤖: E Fund Management Co., Ltd. is the largest public fund company with assets under management of 1,994.044 billion yuan.

👨: What's the current price of rebar futures?
🤖: The current price of rebar futures is: 3266.0, Latest date: 2024-12-27
```

![gpt-4o-2024-08-06](https://shawnsang.github.io/experience/assets/images/posts/ai/akshare/gpt-4o-2024-08-06.gif)

---

### **Design Architecture**

Here's the modular design of the system:

```
User Input -> Natural Language Understanding (LLM) -> Reasoning Selection (ReAct Agent)
        |
        +-> Interface Parser (Parse Akshare Documentation)
        |
        +-> Dynamic Code Generator (Generate Query Code)
        |
        +-> Query Executor (Call Akshare Interface)
        |
        +-> Result Processor (Format and Return Results)
```

---

### **Feature Extensions**

1. **User Interaction**: Integration with Web or CLI interfaces, using visualization tools (like Matplotlib) for market data visualization.
2. **Docker Deployment**: Support for Docker deployment for easy server deployment.

---

### **Core Components**

1. **Interface Parser**
   
   - Download and save Akshare official documentation locally, automatically extract interface information (including functionality, parameters, examples, etc.).
2. **ReAct Agent Design**
   
   - **Agent Pool**: Create multiple Agents based on interface categories (stocks, funds, futures, etc.) to handle queries from different domains.
   - **Reasoning and Selection**: Understand user questions through LLM and select appropriate Agents and interfaces.
   - **Code Generation and Execution**: Dynamically generate Python query code based on selected interfaces and execute to obtain results.
3. **Query and Feedback**
   
   - Automatically process query results, including formatting, filtering, aggregation, etc.
   - Return query results in table or visualization format to enhance user experience.

---

### **Environment Setup**

**Verified Python Version: 3.11**
Recommended to use conda or venv to create a virtual environment to avoid dependency conflicts.

#### 1. Clone Project

```bash
# Clone project
git clone https://github.com/IndigoBlueInChina/Auto-GPT-Stock.git
cd Auto-GPT-Stock
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

#### 3. Configure Environment Variables - LLM Parameters

First create and configure the `.env` file:

```bash
# Create .env file
cp demo.env .env

# Edit .env file, set your OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here
```

#### 4. Install Dependencies

Using `poetry` for dependency management to effectively resolve dependency conflicts.

```bash
pip install poetry
poetry install
```

#### 5. Run Program

```bash
python main.py

🤖: I'm an intelligent financial analysis assistant, I can provide information about stocks, funds, and futures. How can I help you?
```

## Model Capability Comparison

| Model\Question | Stock Questions | Fund Questions | Futures Questions |
| --- | --- | --- | --- |
| o1-mini-2024-09-12 | Multiple rounds to get result | No result | No result |
| gpt-4o-2024-08-06 | One round to get result | One round to get result | Two rounds to get result |
| claude-3-haiku-20240307 | Multiple rounds, no result | Error | NA |
| qwen2.5-32b-instruct | Multiple rounds to get result | Multiple rounds to get result | Found tool, no result after multiple rounds |
| deepseek-v3 | One round to get result | One round to get result | One round to get result |

[Video demonstrations remain the same as in original README] 