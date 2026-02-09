#学会LangChain应用程序的跟踪调试
import langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.tracers import LangChainTracer, ConsoleCallbackHandler

from models import get_lc_model_client
import os
#开启该参数，会输出调试信息
# langchain.debug = True


# 环境变量添加api_key
# os.environ["LANGCHAIN_API_KEY"] = "你的API_KEY"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

# 添加 LangSmith 跟踪器
tracer = LangChainTracer()
tracer.api_url = "https://api.langsmith.com"  # 强制使用正确域名
tracer.project_name = "langchain-demo"    # 设置项目名
console_handler = ConsoleCallbackHandler()


#获得访问大模型客户端
client = get_lc_model_client()

#使用ChatPromptTemplate，它是个大模版，可以组合各种角色的消息模板
#定义提示模版
'''LangChain提供不同类型的角色的消息模板.
最常用的是AIMessagePromptTemplate、 SystemMessagePromptTemplate和HumanMessagePromptTemplate，
分别创建代表人工智能应答消息、系统消息（提示词中的角色、任务、参考案例、约束项等等）和人类消息（具体的任务）。'''
chat_template = ChatPromptTemplate.from_messages(
    [
        #改为 ('system','请将以下的内容翻译成{language}') 也可以
        SystemMessagePromptTemplate.from_template("请将以下的内容翻译成{language}"),
        # 改为 HumanMessagePromptTemplate.from_template("{text}") 也可以
        ('human', '{text}')
    ]
)

# #原始答复比较复杂，包含了很多额外的信息
# result = client.invoke(chat_template.format(language='意大利文', text='朋友啊再见！'))
# print(result)

# #2、解析返回结果
parser = StrOutputParser()
# print(parser.invoke(result))

chain = chat_template | client | parser
result = chain.invoke({'language': '意大利文', 'text': '朋友啊再见！'}, config={'callbacks': [console_handler, tracer]})
print(result)