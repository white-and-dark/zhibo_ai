from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. 创建LCEL链
chain = (
    ChatPromptTemplate.from_template("用5句话介绍{topic}")  # 提示词模板
    | ChatTongyi(model="qwen-max", streaming=True)    # 启用流式传输
    | StrOutputParser()                                    # 解析为字符串
)

# 2. 调用.stream()获取流式响应
topic = "量子计算"
print(f"开始生成【{topic}】介绍：")
#  流式输出是一部分一部分输出，而不是一次性输出
for chunk in chain.stream({"topic": topic}):
    print(chunk, end="", flush=True)  # 逐块打印，不换行