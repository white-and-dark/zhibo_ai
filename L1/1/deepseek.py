import os
from openai import OpenAI, api_key

#deepseek api_key:sk-adb8e1e03e98461daac9bd9fbb97fb07
DEEPSEEK_API_KEY = "sk-adb8e1e03e98461daac9bd9fbb97fb07"

# * deepseek-chat 和 deepseek-reasoner 都已经升级为 DeepSeek-V3.1
# * deepseek-chat 对应 DeepSeek-V3.1 的非思考模式，deepseek-reasoner 对应 DeepSeek-V3.1 的思考模式.
# client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'),
client = OpenAI(api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你是谁，你是DeepSeek-V3.1还是DeepSeek-V3版本"},
    ],
    stream=False
)

print(response.choices[0].message.content)
