#消息历史组件ChatMessageHistory的使用

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from models import get_lc_model_client, get_ali_model_client

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是人工智能助手"),
        #作用就是向提示词中插入一段上下文消息
        MessagesPlaceholder(variable_name="messages"),
        # ("placeholder", "{messages}"),
    ]
)

client = get_ali_model_client()

parser = StrOutputParser()
chain =  chat_template | client | parser

#  创建消息历史记录
chat_history = ChatMessageHistory()

while True:
    user_input = input("用户：")
    if user_input == "exit":
        break
    #添加用户输入
    chat_history.add_user_message(user_input)
    #访问LLM时，chat_history.messages 获取所有的历史消息
    response = chain.invoke({'messages': chat_history.messages})
    print(f"大模型回复》》》：{response}")
    #将大模型的回复加入历史记录
    chat_history.add_ai_message(response)

#第一轮对话： 添加用户的提问消息
chat_history.add_user_message('你好，我是大白')#用户提问加入历史记录
response = chain.invoke({'messages': chat_history.messages})

print(response)
#第一轮对话： 添加模型应答消息
chat_history.add_ai_message(response) #模型应答加入历史记录
print("chat_history:",chat_history.messages)

# 第二轮对话： 添加用户的提问消息
chat_history.add_user_message('你好，我是谁？')#用户提问加入历史记录
print(chain.invoke({'messages': chat_history.messages}))