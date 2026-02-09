from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, AIMessagePromptTemplate

# 模型客户端
model = ChatTongyi()

# 提示词模板
# prompt = PromptTemplate(
#     template="你是一个翻译助手，请帮我把一下内容翻译成{language}：{text}"
#     , input_variables=["text", "language"]
# )

# 对话模型，角色的设置，多轮对话记忆维持
# 提示词角色：system 全局，统一设置   user 用户每次提问  assistant 大模型回复，平常上下文存储

prompt = ChatPromptTemplate.from_messages([
    # 系统提示词
    SystemMessagePromptTemplate.from_template("你是一个翻译助手，可以翻译任何一种语言"),
    # 用户提示词
    HumanMessagePromptTemplate.from_template("请将一下内容翻译成{language}：{text}"),
    # 助手提示词
    # AIMessagePromptTemplate.from_template("{text}"),
])

# 特别复杂场景用的提示词模版：jinjia2 模板，freemarker模板  定义逻辑 if  for 等逻辑代码
# prompt = PromptTemplate(
#     template="你是一个翻译助手，请帮我把一下内容翻译成{{language}}：{{text}}"
#     , input_variables=["text", "language"]
# )

# 给参数赋值
# fact = prompt.format(text="hello world", language="中文")
# fact2 = prompt.format(text="我爱你", language="日文")

# 执行，拿结果
# result = model.invoke(fact)
# result2 = model.invoke(fact2)
# print(result)
# print(result2)

# 结果解析器,格式化输出
out = StrOutputParser()

# print(out.invoke(result))

# 底层还是函数调令，简化书写，固定的流程的调用，格式化
chain = prompt | model | out

print(chain.invoke({"text": "hello world", "language": "中文"}))
