# pip install -qU langchain-redis langchain-openai redis
from gradio.themes.builder_app import history
from langchain_redis import RedisChatMessageHistory

REDIS_URL = "redis://:1234@10.4.140.22:6379"

from models import get_lc_model_client
# session_id 识别用户 redis_url 访问路径
# 方法1：在 URL 中包含密码
history = RedisChatMessageHistory(
    session_id="my_session_id",
    redis_url=REDIS_URL
)

history.add_user_message("你是谁？")
client = get_lc_model_client()

aimessage = client.invoke(history.messages)
history.add_ai_message(aimessage)
print(aimessage)

history.add_user_message("重复一次")
aimessage = client.invoke(history.messages)
history.add_ai_message(aimessage)
print(aimessage)

#RedisChatMessageHistory 和 ChatMessageHistory 都有相同的API
#ChatMessageHistory比较早期的命名，功能实现InMemoryChatMessageHistory




