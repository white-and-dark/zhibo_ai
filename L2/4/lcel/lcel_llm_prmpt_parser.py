#学会使用链
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from models import get_lc_model_client

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

# #2、解析返回结果
parser = StrOutputParser()
# chain = chat_template|client|parser
chain = RunnableSequence(chat_template,client,parser)
result = chain.invoke({'language': '英文', 'text': '朋友啊再见！'})
print(result)
