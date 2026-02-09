import os

import langchain
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, AIMessagePromptTemplate

from langchain_openai import ChatOpenAI

langchain.debug=True

# 1.构建模型包装器  聊天模型
# model =  ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), model_name="qwen-max", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 不支持openai的模型 参数1：模型名称，参数2：模型提供者
# model = init_chat_model("deepseek-chat",model_provider="deepseek")
# 采样通义千问的方式访问
model = ChatTongyi(model_name="qwen-max")

# 2.构建提示词
# prompt = "请帮我将一下内容翻译成英文：我喜欢编程。"

# 创建提示词模板
# prompt = PromptTemplate.from_template("请将一下内容翻译成{language}：{text}")
#
# factPrompt = prompt.format(language="英文", text="我喜欢编程")

# 提供角色设置  System   user  人类  assistant大模型回复
chatPrompt = ChatPromptTemplate.from_messages([
    # ("system", "你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    SystemMessagePromptTemplate.from_template("你是一个翻译模型，你需要将输入的句子翻译成{language}"),
    # ("user", "{text}"),
    HumanMessagePromptTemplate.from_template("{text}"),
    # ("assistant", "我非常抱歉，但是这个任务无法完成。"),
    # AIMessagePromptTemplate.from_template("我非常抱歉，但是这个任务无法完成。")
])

# # 3.执行，结果解析
# result = model.invoke(factPrompt)
# # 创建一个字符串结果解析器
parser = StrOutputParser()
# # 提取回复内容的字符串
# invoke = parser.invoke(result)
#
# print(invoke)

# 链的形式调用
chain = chatPrompt|model|parser
# 调用执行链
print(chain.invoke({"language": "英文", "text": "我喜欢编程"}))