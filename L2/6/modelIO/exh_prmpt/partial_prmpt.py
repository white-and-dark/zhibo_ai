#提示模板部分格式化：适用于需要先给某些参数赋值，其余参数后期赋值。
#非常适用于动态获得特定变量值的情况，比如日期和时间，某些系统内部配置。这些值是一般是无需用户在每次提问时输入。
#比如，实现一个学习助手，用户在提问前，先确定要问历史问题，地理问题等等
from langchain.prompts import PromptTemplate
from models import get_lc_model_client

#获得访问大模型客户端
client = get_lc_model_client()

def get_datetime():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 配置一个提示模板
prompt_tmplt_txt = "讲一个关于{date}的{story_type}"

prompt = PromptTemplate(
    template=prompt_tmplt_txt,
    input_variables=["date", "story_type"]
)
half_prompt = prompt.partial(date=get_datetime())
print(half_prompt)

result = client.invoke(half_prompt.format(story_type="笑话"))
print(result.content)
result = client.invoke(half_prompt.format(story_type="悲伤故事"))
print(result.content)
