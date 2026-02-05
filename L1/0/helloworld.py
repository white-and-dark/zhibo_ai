
# api_key:sk-8dc9eec809c94eb4989d81e3a9c307a6
# qwen-plus-2025-07-28

import os
from openai import OpenAI
# 安装openai ： pip install openai

# shift + tab 左移
# 1.创建openai客户端
#   api_key:从虚拟环境种获取DASHSCOPE_API_KEY
#   base_url: 通义千问固定的url
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2.访问大模型
#  model: 模型名称
#  messages: 消息,你发送给大模型的内容
#       system: 系统角色（让大模型充当什么角色），content中一般写角色,要求,约束，示例等, 可选
#       user: 用户角色，我们自己，content一般写我们要问大模型的问题， 必选
completion = client.chat.completions.create(
    model="qwen-plus-2025-07-28",  # qwen-plus-2025-07-28
    messages=[
        {'role': 'system', 'content': '你是一个AI助手'},
        {'role': 'user', 'content': '你是谁？'}
    ]
)
print(completion.choices[0].message.content)
