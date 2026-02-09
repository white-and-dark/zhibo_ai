#RunnablePassthrough的使用
#RunnablePassthrough的两种用法都将在我们后面的课程中看到
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# RunnablePassthrough原样进行数据传递
runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: x["num"] + 1,
)
# {'passed': {'num': 1}, 'modified': 2}
print(runnable.invoke({"num": 1}))

#RunnablePassthrough对数据增强后传递
#RunnablePassthrough().assign它会创建一个新的字典，包含原始的所有字段以及你新指定的字段。
runnable = RunnableParallel(
    passed=RunnablePassthrough().assign(query=lambda x: x["num"] + 2),
    modified=lambda x: x["num"] + 1,
)
print(runnable.invoke({"num": 1}))