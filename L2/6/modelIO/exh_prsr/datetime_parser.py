from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import get_lc_model_client

#获得访问大模型客户端
client = get_lc_model_client()

# 定义模板格式
template = """
回答用户的问题：{question}

{format_instructions}
"""

# 使用日期时间解析器
output_parser = DatetimeOutputParser()

prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)
print(prompt)
print("-"*30)
print(output_parser.get_format_instructions())

# 链式调用
chain = prompt | client | output_parser
output = chain.invoke({"question": "新中国是什么时候成立的？"})
# 打印输出
print(output)  # 1949-10-01

# 执行
# output = model.invoke(prompt.format(question='新中国成立的时间？'))
# datetime_parsed = output_parser.parse(output.content)
# # 打印输出
# print(datetime_parsed)  # 1949-10-01
