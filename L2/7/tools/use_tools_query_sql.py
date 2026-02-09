import os
from operator import itemgetter

import langchain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import re
from models import get_lc_model_client, get_ali_model_client, ALI_TONGYI_MAX_MODEL

#langchain.debug = True

#获得访问大模型客户端
client = get_ali_model_client(model=ALI_TONGYI_MAX_MODEL)

#数据库配置
HOSTNAME ='10.4.140.22'
PORT ='3306'
DATABASE = 'world'
USERNAME = 'root'
PASSWORD ='root'
MYSQL_URI ='mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)

'''对于问题："请从国家表中查询出China的所有数据"，要分为几步才能出结果：
1、大模型判断这个问题需要调用工具查询数据库，获得所有的表名和表中的字段名，
目的是看那个表才是国家表，国家表有哪些字段，
2、工具执行后，把工具执行结果交给大模型
3、大模型根据国家表和国家表中的字段，生成SQL语句
4、SQL语句的执行依然需要使用工具
5、工具执行后，把工具执行结果交给大模型，大模型生成最终答案'''

#1、用LangChain内置链create_sql_query_chain将大模型和数据库结合，会产生sql而不会执行sql
# 通过create_sql_query_chain将步骤中的1、2、3合起来一起做了
# sql_make_chain = create_sql_query_chain(client, db)
# resp = sql_make_chain.invoke({"question":"请从国家表中查询出China的所有数据"})
# print("产生的SQL语句：",resp)
# print("**"*15)


#2、因为实际产生的sql是形如```sql....```的，无法直接执行，所以需要清理
#自定义一个输出解析器SQLCleaner
class SQLCleaner(StrOutputParser):
    def parse(self, text: str) -> str:
        pattern = r'```sql(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sql = match.group(1).strip()
            # 某些大模型还会产生类似'SQLQuery:'前缀，必须去除
            sql = re.sub(r'^SQLQuery:', '', sql).strip()
            return sql
        # 某些大模型还会产生类似'SQLQuery:'前缀，必须去除
        text = re.sub(r'^SQLQuery:', '', text).strip()
        return text
sql_make_chain = create_sql_query_chain(client, db)| SQLCleaner()
# resp = sql_make_chain.invoke({"question":"请从国家表中查询出China的相关数据"})
# print("实际可用SQL: ",resp)
# print("**"*15)

#3、将前面的部分组合起来，得到最终结果
answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、可能的SQL语句和SQL执行后的结果，回答用户问题
    Question: {question}
    SQL Query: {query}
    SQL Result:{result}
    回答:"""
)
#创建一个执行SQL的工具
execute_sql_tools = QuerySQLDatabaseTool(db = db)
# runnable = RunnablePassthrough.assign(query=sql_make_chain)
# print("RunnablePassthrough-1：",runnable.invoke({"question":"请从国家表中查询出China的相关数据"}))
# runnable = RunnablePassthrough.assign(query=sql_make_chain)| itemgetter('query')
# print("RunnablePassthrough-2：",runnable.invoke({"question":"请从国家表中查询出China的相关数据"}))
# runnable = RunnablePassthrough.assign(query=sql_make_chain)| itemgetter('query') | execute_sql_tools
# print("RunnablePassthrough-3：",runnable.invoke({"question":"请从国家表中查询出China的相关数据"}))
# runnable = RunnablePassthrough.assign(query=sql_make_chain).assign(result=itemgetter('query')|execute_sql_tools)
# print("RunnablePassthrough-4：",runnable.invoke({"question":"请从国家表中查询出China的相关数据"}))
'''通过上面的步骤，就能搞清楚{question}、{query}、{result}这三个字段是如何通过LCEL链一步步获得的
要注意的是result=itemgetter('query')|execute_sql_tools 中，执行顺序是：
itemgetter('query') -> execute_sql_tools -> result=
所以这段代码实际是：result=(itemgetter('query')|execute_sql_tools)'''
chain = (RunnablePassthrough.assign(query=sql_make_chain).assign(result=itemgetter('query')|execute_sql_tools)
        |answer_prompt| client| StrOutputParser())
result = chain.invoke(input={"question":"请从国家表中查询出China的相关数据"})
#result = chain.invoke(input={"question":"请问国家表中有多少条数据"})
print("最终执行的结果：",result)
'''，如果场景是确定的，并不需要大模型来决定是否使用工具，直接在链中加入工具即可
#但是如果需要大模型来决定是否使用工具，比如场景是动态的或者是以工具组的形式提供工具，那么需要使用Function Call：'''