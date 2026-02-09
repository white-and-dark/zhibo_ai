import os

#  debug模式的包
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
langchain.debug=True

model = ChatTongyi()

# model = ChatOpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     model="qwen-max",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     temperature=0.1,
# )

# 2.构建提示词
# 提示词模板
prompt = PromptTemplate(
    template="你是一个翻译助手，请帮我把一下内容翻译成{language}：{text}"
    , input_variables=["text", "language"]
)


# 格式化返回结果,字符串格式化,content内容提取
out = StrOutputParser()

# 构建链   |管道    linux命令中有管道,langchain框架预制了很多工作链
chain = prompt | model | out
# 链的调用
result = chain.invoke({"language": "英文", "text": "我喜欢编程"})

print(result)