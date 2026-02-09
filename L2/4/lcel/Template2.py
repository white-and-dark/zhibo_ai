from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# 模型客户端
model = ChatTongyi()

# 提示词模板
prompt = PromptTemplate(
    template="你是一个翻译助手，请帮我把一下内容翻译成{language}：{text}"
    , input_variables=["text", "language"]
)

# 结果解析器,格式化输出
out = StrOutputParser()


# 底层还是函数调令，简化书写，固定的流程的调用，格式化
chain = RunnableSequence(prompt,model,out)

print(chain.invoke({"text": "hello world", "language": "中文"}))


