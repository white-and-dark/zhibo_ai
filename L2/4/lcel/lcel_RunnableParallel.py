#RunnableParallel的使用
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableMap


def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

def mul_three(x: int) -> int:
    return x * 3

# 测试RunnableParallel, RunnableMap 并行执行
runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)

chain_seq = runnable_1 | runnable_2 | runnable_3
print(chain_seq.invoke(1))

chain = runnable_1 | RunnableParallel(
    mul_two=runnable_2,
    mul_three=runnable_3,
)

print(chain.invoke(1))



