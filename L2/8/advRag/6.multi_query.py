#Multi-Query 多路召回
from operator import itemgetter

from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain, RunnableMap  # 添加 chain 导入

from models import get_ali_clients

# 获得访问大模型和嵌入模型客户端
llm, embeddings_model = get_ali_clients()

# 加载文档
loader = TextLoader("./deepseek百度百科.txt", encoding="utf-8")
docs = loader.load()

# 创建文档分割器，并分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings_model
)
# 创建检索器
retriever = vectorstore.as_retriever()

# 检索测试
relevant_docs = retriever.invoke('deepseek的应用场景')
print(f"检索到的相关文档数量：{len(relevant_docs)}")

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

chain_simple = RunnableMap({
    "context": lambda x: relevant_docs,
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("--------------优化前-------------------")
response = chain_simple.invoke({"question": "deepseek的应用场景"})
print("优化前回答：", response)

print("--------------开始优化-------------------")
# 方法一：使用langchain的MultiQueryRetriever
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

retrieval_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

unique_docs = retrieval_from_llm.invoke({"question": 'deepseek的应用场景'})
print(f"MultiQueryRetriever检索到的文档数量：{len(unique_docs)}")

# 方法二：自定义prompt
prompt_perspectives = ChatPromptTemplate.from_template(
    """你是一个AI语言模型助手。你的任务是生成5个给定用户问题的不同版本，以从向量数据库中检索相关文档。
通过对用户问题产生多种观点，你的目标是帮助用户克服基于距离的相似性搜索的一些限制。
提供这些用换行符隔开的可选问题。原始问题: {question}"""
)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

response = generate_queries.invoke({"question": 'deepseek的应用场景'})
print("生成的不同查询：", response)

# 定义去重函数 - 使用 @chain 装饰器
@chain
def get_unique_union(documents: list[list]):
    """ 获取检索文档的唯一并集 """
    # 将列表中的列表展开，并将每个 Document 转换为字符串
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 文档去重
    unique_docs = list(set(flattened_docs))
    # 返回去重后的文档列表
    return [loads(doc) for doc in unique_docs]

# 创建检索链
question = "deepseek的应用场景"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# 测试检索链
docs = retrieval_chain.invoke({"question": question})
print(f"自定义多查询检索到的文档数量：{len(docs)}")

print("--------------优化后-------------------")
final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

response = final_rag_chain.invoke({"question": "deepseek的应用场景"})
print("优化后回答：", response)