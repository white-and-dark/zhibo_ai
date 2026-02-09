import os

import langchain
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi, ChatHunyuan
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_openai import ChatOpenAI

# 链式调用
# 1.创建模型对象，基于OpenAI 规范
# 作用：跟踪链路，跟踪每个节点的输入，输出     调试程序时使用
# langchain.debug=True

model = ChatTongyi()

# 2.构建提示词
# prompt = "你是一个翻译专家，请将输入的句子翻译成英文:我喜欢编程"

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    # ("system", "你是一个翻译专家，请将输入的句子翻译成{language}:"),
    SystemMessagePromptTemplate.from_template("你是一个翻译专家，请将输入的句子翻译成{language}:"),
    HumanMessagePromptTemplate.from_template("{text}"),
    # ("human", "{text}"),
    # Assistent  维持上下文，每次的对话都拼接字符串
    # AIMessagePromptTemplate.from_template("作用？"),
])

# 格式化返回结果,字符串格式化,content内容提取
out = StrOutputParser()

# 构建链   |管道    linux命令中有管道
chain = prompt | model | out
# 链的调用
result = chain.invoke({"language": "英文", "text": "我喜欢编程"})

print(result)