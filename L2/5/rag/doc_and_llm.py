import os
# 安装 pip install langchain_chroma
# 加载word文档 安装 pip install docx2txt
# 加载json文档 安装 pip install jq
# 加载pdf文档  安装 pip install pymupdf
# 加载HTML文档 安装 pip install unstructured
# 加载MD文档   安装 pip install markdown +  pip install unstructured
import langchain
# pip install langchain-chroma
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_lc_model_client, ALI_TONGYI_API_KEY_OS_VAR_NAME, ALI_TONGYI_EMBEDDING_MODEL, get_ali_model_client


#获得访问大模型客户端
client = get_ali_model_client()

#直接了解LangChain中的“文档”(Document)的具体内容，这里我们跳过了文档与文档加载，文档切割和文档转换过程
#文档的模拟数据
# documents = [
#     Document(
#         page_content="猫是柔软可爱的动物，但相对独立",
#         metadata={"source": "常见动物宠物文档"},
#     ),
#     Document(
#         page_content="狗是人类很早开始的动物伴侣，具有团队能力",
#         metadata={"source": "常见动物宠物文档"},
#     ),
#     Document(
#         page_content="金鱼是我们常常喂养的观赏动物之一，活泼灵动",
#         metadata={"source": "鱼类宠物文档"},
#     ),
#     Document(
#         page_content="鹦鹉是猛禽，但能够模仿人类的语言",
#         metadata={"source": "飞禽宠物文档"},
#     ),
#     Document(
#         page_content="兔子是小朋友比较喜欢的宠物，但是比较难喂养",
#         metadata={"source": "常见动物宠物文档"},
#     ),
# ]

from langchain_community.document_loaders import UnstructuredWordDocumentLoader, Docx2txtLoader

# 1.指定要加载的Word文档路径
loader = Docx2txtLoader("人事管理流程.docx")

# 加载文档、转换格式化成document
documents = loader.load()
print(len(documents))

# 文档切割 递归切割
# separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,

    # separators=[分隔符]
)
split_documents = text_splitter.split_documents(documents)

# 模型包装器：大模型分成三类：LLM  聊天模型  嵌入模型
# 获得一个阿里通义千问嵌入模型的实例，同样在models.py中被包装为get_ali_embeddings()
from langchain_community.embeddings import DashScopeEmbeddings
# 嵌入模型的模型包装器
llm_embeddings = DashScopeEmbeddings(
    model=ALI_TONGYI_EMBEDDING_MODEL,
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# 实例化向量空间，向量化+向量存储到向量数据库中
vector_store = Chroma.from_documents(documents=split_documents,embedding=llm_embeddings)

#展示相似度查询，实际业务中可以不要
# print(vector_store.similarity_search("狸花猫"))
# print("--"*15)
# #按相似度的分数进行排序，分数值越小，越相似（其实是L2距离）
# print(vector_store.similarity_search_with_score("狸花猫"))


#从向量数据库检索，使用chroma原始API查询   bind(k=1)表示返回相似度最高的第一个
docs_find = RunnableLambda(vector_store.similarity_search).bind(k=1)


# print(docs_find.batch(["狸花猫", "海豚"]))

retriever = vector_store.as_retriever()

message = """ 
仅使用提供的上下文回答下面的问题：
{question}
上下文：
{context}
"""
prompt_template = ChatPromptTemplate.from_messages([('human',message)])
# 定义这个链的时候，还不知道问题是什么，
# 用RunnablePassthrough允许我们将用户的具体问题在实际使用过程中进行动态传入
chain = {"question":RunnablePassthrough(),"context":docs_find} | prompt_template | client

#用大模型生成答案
resp = chain.invoke("晋升")
print(resp.content)
