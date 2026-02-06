import uuid

import chromadb
from chromadb.config import Settings
from models import *
from functools import wraps
from pypinyin import pinyin, Style
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ChromaDB 向量数据库类
class MyVectorDBConnector:

    def __init__(self):
        # self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.chroma_client = chromadb.PersistentClient(path = "./chroma")

        # 创建client
        self.client = get_normal_client()

    def get_embeddings(self, texts, model=ALI_TONGYI_EMBEDDING_V4):
        '''使用模型进行向量化'''
        data = self.client.embeddings.create(input=texts, model=model).data
        return [x.embedding for x in data]

    # 因为各个模型对一次能处理的文本条数有限制且每个平台不一致，新增一个batch_size参数用以控制。
    def get_embeddings_batch(self,texts, model=ALI_TONGYI_EMBEDDING_V4, batch_size=10):
        '''get_embeddings函数的变体版'''
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]
            data = self.client.embeddings.create(input=batch_text, model=model).data
            all_embeddings.extend([x.embedding for x in data])
        return all_embeddings


    def add_documents(self, documents, collection_name='demo'):
        '''向 collection 中添加文档与向量'''
        print('add_documents: collection_name:', collection_name)
        # 创建or获取一个 collection
        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        batch_size = 10
        # 批量向量化，添加文档
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i: i + batch_size]
            collection.add(
                embeddings=self.get_embeddings(batch_docs),  # 每个文档的向量
                documents=batch_docs,  # 文档的原文
                ids=[str(uuid.uuid4()) for _ in batch_docs]  # 每个文档的id,uuid确保唯一
            )

    def search(self, query, collection_name='demo', n_results=5):
        '''检索向量数据库'''
        print('search: collection_name:', collection_name)
        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # self.collection.query() 这是 ChromaDB 集合对象的一个方法，用于在集合中执行查询操作。
        results = collection.query(
            query_embeddings=self.get_embeddings([query]),  # query_embeddings是查询文本的向量表示
            n_results=n_results  # 最相似文档的数量
        )
        return results

# 读取Word文档
def extract_text_from_docx(filename):
    '''从 DOCX 文件中提取文字'''
    full_text = ''
    # 打开并读取文档
    doc = Document(filename)
    # 提取全部文本
    for para in doc.paragraphs:
        if para.text.strip():
            full_text += para.text + '\n'

    # 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    documents = splitter.split_text(full_text)

    print("documents:", documents)
    return documents

# 访问大模型
def get_completion(prompt, model=ALI_TONGYI_TURBO_MODEL):
    '''封装 openai 接口'''
    messages = [{"role": "user", "content": prompt}]
    client = get_normal_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content

# 装饰器：中文->英文
def to_pinyin(fn):
    @wraps(fn)
    def chinese_to_pinyin(*args, **kwargs):
        chinese_name = kwargs['collection_name']
        # 把.去掉
        chinese_name = chinese_name.replace('.', '')

        # 中文->拼音
        pinyin_list = pinyin(chinese_name, style=Style.NORMAL, heteronym=False)
        # 将拼音列表转换为字符串
        pinyin_str = ''.join([word[0].lower() for word in pinyin_list])
        kwargs['collection_name'] = pinyin_str
        return fn(*args, **kwargs)
    return chinese_to_pinyin

