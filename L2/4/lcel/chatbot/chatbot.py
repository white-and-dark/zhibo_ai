#自动会话历史管理组件RunnableWithMessageHistory
#以及如何流式处理大模型的应答
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tracers import LangChainTracer, ConsoleCallbackHandler
from langsmith import traceable

from models import get_lc_model_client
from langchain_redis import RedisChatMessageHistory

REDIS_URL = "redis://:1234@10.4.140.22:6379"

# 用户输入 + config.session_id → 找到 Redis 中的历史记录 → 和当前输入一起发给模型 → 生成回答 → 把回答存回 Redis。

client = get_lc_model_client()

#定义提示模版
# 没有手动插入上下文
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是一个聊天助手，用中文回答所有的问题"),
        ("human", "{input}"),
    ]
)
parser = StrOutputParser()
# 以链的形式
chain = prompt_template | client | parser

'''
#ttl 当前会话数据的过期时间，300秒表示5分钟过期
chat_message_history = RedisChatMessageHistory(url=REDIS_URL, session_id=session_id, ttl=300)
'''
# history = RedisChatMessageHistory(session_id="my_session_id", redis_url="redis://localhost:6379")


# 工厂函数：获取 Redis 历史记录
# 是否将所有历史消息都保存到 Redis中
# 只存储最近的10条，避免Redis数据库过大#将前面历史消息提交大模型进行摘要生成
# # redis本身的淘汰策略，会自动删除过期的记录

# def get_session_history(session_id):
#     return RedisChatMessageHistory(
#         session_id=session_id,
#         redis_url="redis://localhost:6379",
#         ttl=300
#     )
#
# # 包装模型
# chatbot = RunnableWithMessageHistory(
#     模型对象、chain对象,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="history"
# )

# 串联历史记录（记忆） 和 链
# 将对话历史自动集成到模型调用链中，解决了聊天机器人在上下文连续性和多用户支持中的核心问题
chatbot_with_his = RunnableWithMessageHistory(
    # 执行流程
    chain,
    # 根据session_id获取历史记录
    get_session_history=lambda session_id: RedisChatMessageHistory(
        session_id=session_id,
        redis_url=REDIS_URL
       ),
    input_messages_key="input",
)
resp_user123 = chatbot_with_his.invoke(
    {"input": "我是user123,我喜欢音乐，你能给我推荐一首轻音乐吗？"},
        config={
            "configurable": {"session_id": "user123"},
            # 注册跟踪器和控制台输出
            # "callbacks": [tracer, console_handler]
        },

)
print("用户123的回答：", resp_user123)

resp_jack = chatbot_with_his.invoke(
    {"input": "我是jack,我热爱运动，篮球是我的最爱，你猜我喜欢哪个NBA球星？"},
    config={"configurable": {"session_id": "jack"}}
)
print("用户jack的回答：", resp_jack)


resp1_user123 = chatbot_with_his.invoke(
    {"input": "我喜欢什么？"},
    config={"configurable": {"session_id": "user123"}}
)
print("用户123的回答：", resp1_user123)

resp_jack1 = chatbot_with_his.invoke(
    {"input": "我喜欢什么？"},
    config={"configurable": {"session_id": "jack"}}
)
print("用户jack的回答：", resp_jack1)
