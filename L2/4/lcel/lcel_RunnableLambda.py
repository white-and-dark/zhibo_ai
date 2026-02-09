#RunnableLambda的使用
from operator import itemgetter
import langchain
from langchain_community.chat_models import ChatOllama, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain
from models import get_lc_model_client

#开启该参数，会输出调试信息
# langchain.debug = True
#获得访问大模型客户端
client = get_lc_model_client()

#  langchain调用本地部署的模型
# client = ChatOllama()
# ChatHuggingFace()


#定义提示模版
chat_template = ChatPromptTemplate.from_template(" {a} + {b}是多少？")


output = StrOutputParser()
#获得字符串的长度
def length_function(text):
    return len(text)

#将两个字符串长度的数量相乘
def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

#@chain装饰器是RunnableLambda的另一种写法
@chain
def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

# chain1 = chat_template | client |output
chain1 = chat_template | client

#使用RunnableLambda将函数转换为与LCEL兼容的组件

# RunnableLambda 调用自定义函数，自定义函数的功能可以其他工具进行参数的获取
chain2 = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}| multiple_length_function,
    }
    | chain1
)
print(chain2.invoke({"foo": "abc", "bar": "abcd"}))

#模拟用户的业务，可以从数据库、其他文件中获得数据

# itemgetter 根据key获取字典中的值
chain3 = (
          {
             "a": ( itemgetter("foo") | RunnableLambda(length_function) ),
             "b": ( {"text1": itemgetter("foo"), "text2": itemgetter("bar")}| multiple_length_function )
          }
    | chain1  )| output
print(chain3.invoke({"foo": "abc", "bar": "abcd"}))