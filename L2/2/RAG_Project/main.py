# main.py
# 需要安装
#   pip install python-docx
#   pip install pypinyin

from function_tools import *

# --------------   1. 上传文档  -------------- #
# 创建一个向量数据库对象
vector_db = MyVectorDBConnector()

# 上传文档：存入向量库
@to_pinyin
def save_to_db(filepath, collection_name='demo'):
    print('-' * 100)
    print('正在存入文档：filepath:', filepath)
    print('正在存入文档：collection_name:', collection_name)
    documents = ''

    # 判断文件格式，读取文件内容
    if filepath.endswith('.docx') or filepath.endswith('.doc'):
        # 读取Word文件，并对文档分块
        documents = extract_text_from_docx(filepath)

    if not documents:
        return '读取文件内容为空'

    # 存入向量数据库
    vector_db.add_documents(documents, collection_name=collection_name)  # 添加文档


# --------------  2. 聊天    -------------- #

# 检索知识库 => top5相关文档 => LLM => 答案
@to_pinyin
def rag_chat(user_query, collection_name='demo', n_results=5):
    print('正在检索文档：collection_name:', collection_name)

    # 1. 检索知识库
    search_results = vector_db.search(user_query, collection_name=collection_name, n_results=n_results)
    print('search_results:', search_results)
    print('-' * 100)

    # 2. 构建 Prompt-增强
    info = '\n'.join(search_results['documents'][0])
    query = user_query

    prompt = f"""
    你是一个问答机器人。
    你的任务是根据下述给定的已知信息回答用户问题。
    确保你的回复完全依据下述已知信息。不要编造答案。
    如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

    已知信息:
    {info}
    
    ----
    用户问：
    {query}

    请用中文回答用户问题。
    """
    print('prompt:', prompt)
    print('-' * 100)

    # 3. 调用 LLM-生成答案
    response = get_completion(prompt)
    return response


# 测试代码
if __name__ == '__main__':
    # 测试代码：直接运行本文件
    save_to_db(filepath='../Data/人事管理流程.docx', collection_name='人事管理流程.docx')
    print('-' * 100)

    user_query = "视为不符合录用条件的情形有哪些?"
    response = rag_chat(user_query, collection_name='人事管理流程.docx', n_results=5)
    print("response:", response)


