from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models import get_ali_clients

#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 加载文档
loader = TextLoader("./deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
split_docs = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=split_docs, embedding=embeddings_model
)

question = "相关评价"

# 向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
doc_vector_retriever = vector_retriever.invoke(question)
print("-------------------向量检索-------------------------")
pretty_print_docs(doc_vector_retriever)

# 关键词检索
BM25_retriever = BM25Retriever.from_documents(split_docs)
BM25Retriever.k = 3
doc_BM25Retriever = BM25_retriever.invoke(question)
print("-------------------BM25检索-------------------------")
pretty_print_docs(doc_BM25Retriever)

# 混合检索
#EnsembleRetriever（归一化）是Langchain集合多个检索器的检索器。（）参数：retrievers：检索器列表，weights：权重列表
ensembleRetriever = EnsembleRetriever(retrievers=[BM25_retriever, vector_retriever], weights=[0.5, 0.5])
retriever_doc = ensembleRetriever.invoke(question)
print("-------------------混合检索-------------------------")
print(retriever_doc)

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建chain
chain1 = RunnableMap({
    "context": lambda x: ensembleRetriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()
chain2 = RunnableMap({
    "context": lambda x: vector_retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("------------模型回复------------------------")
print("------------向量检索+BM25[0.5, 0.5]------------------------")
print(chain1.invoke({"question":question}))
print("------------向量检索------------------------")
print(chain2.invoke({"question":question}))