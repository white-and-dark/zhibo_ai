import os

import langchain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document

from models import get_lc_model_client, ALI_TONGYI_API_KEY_OS_VAR_NAME, ALI_TONGYI_EMBEDDING_MODEL, get_ali_model_client
# langchain.debug = True

#获得访问⼤模型客⼾端
client = get_ali_model_client()

#直接了解LangChain中的“⽂档”(Document)的具体内容，这⾥我们跳过了⽂档与⽂档加载，⽂档分割和⽂档转换过程
#⽂档的模拟数据
documents = [
    Document(
        page_content="猫是柔软可爱的动物，但相对独⽴",
        metadata={"source": "常⻅动物宠物⽂档"},
    ),
    Document(
        page_content="狗是⼈类很早开始的动物伴侣，具有团队能⼒",
        metadata={"source": "常⻅动物宠物⽂档"},
    ),
    Document(
        page_content="⾦⻥是我们常常喂养的观赏动物之⼀，活泼灵动",
        metadata={"source": "⻥类宠物⽂档"},
    ),
    Document(
        page_content="鹦鹉是猛禽，但能够模仿⼈类的语⾔",
        metadata={"source": "⻜禽宠物⽂档"},
    ),
    Document(
        page_content="兔⼦是⼩朋友⽐较喜欢的宠物，但是⽐较难喂养",
        metadata={"source": "常⻅动物宠物⽂档"},
    ),
]

# 获得⼀个阿⾥通义千问嵌⼊模型的实例，同样在models.py中被包装为get_ali_embeddings()
from langchain_community.embeddings import DashScopeEmbeddings
llm_embeddings = DashScopeEmbeddings(
    model=ALI_TONGYI_EMBEDDING_MODEL,
    dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
)

# 实例化向量空间
vector_store = Chroma.from_documents(documents=documents,embedding=llm_embeddings)
#展⽰相似度查询，实际业务中可以不要
print(vector_store.similarity_search("狸花猫"))
print("--"*15)
#按相似度的分数进⾏排序，分数值越⼩，越相似（其实是L2距离）
print(vector_store.similarity_search_with_score("狸花猫"))

#检索器，bind(k=1)表⽰返回相似度最⾼的第⼀个
docs_find = RunnableLambda(vector_store.similarity_search).bind(k=1)
print(docs_find.batch(["狸花猫", "海豚"]))

message = """
仅使⽤提供的上下⽂回答下⾯的问题：
{question}
上下⽂：
{context}
"""
prompt_template = ChatPromptTemplate.from_messages([('human',message)])
# 定义这个链的时候，还不知道问题是什么，
# ⽤RunnablePassthrough允许我们将⽤⼾的具体问题在实际使⽤过程中进⾏动态传⼊
chain = {"question":RunnablePassthrough(),"context":docs_find} | prompt_template | client
# #⽤⼤模型⽣成答案
resp = chain.invoke("请介绍⼀下猫")
print(resp.content)

questiong = "请介绍⼀下猫"
# '''如果不⽤RunnablePassthrough()，那么需要这么写，已经测试通过，直接使⽤docs_find不⾏'''
chain1 = prompt_template | client
resp = chain1.invoke({'question': questiong, "context":vector_store.similarity_search(questiong,k=1)})
resp = chain1.invoke({'question': questiong, "context":docs_find})
print(resp.content)