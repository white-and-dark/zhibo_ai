#Post-Retrieval后检索-上下文压缩
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from models import get_a_t_mix_clients, get_ali_clients

#获得访问大模型和嵌入模型客户端
llm,embeddings_model = get_ali_clients()

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

documents = TextLoader("./deepseek百度百科.txt",encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,  
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

#使用基础检索器
retriever = Chroma.from_documents(texts, embeddings_model).as_retriever()

docs = retriever.invoke("deepseek的发展历程")
print("-------------------压缩前--------------------------")
pretty_print_docs(docs)

print("-------------------第一种：LLMChainExtractor压缩------------------")
#使用上下文压缩检索器
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "deepseek的发展历程"
)
print("-------------------压缩后--------------------------")
pretty_print_docs(compressed_docs)


print("-------------------第二种：LLMChainFilter压缩后--------------------------")
#LLMChainFilter 是稍微简单但更强大的压缩器
_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "deepseek的发展历程"
)

pretty_print_docs(compressed_docs)

print("-------------------第三种：EmbeddingsFilter压缩后--------------------------")
#对每个检索到的文档进行额外的 LLM 调用既昂贵又缓慢。
#EmbeddingsFilter 通过嵌入文档和查询并仅返回那些与查询具有足够相似嵌入的文档来提供更便宜且更快的选项
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.6)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "deepseek的发展历程"
)

pretty_print_docs(compressed_docs)

print("-------------------第四种：组合压缩后--------------------------")
# DocumentCompressorPipeline轻松地按顺序组合多个压缩器
'''1.首先TextSplitters可以用作文档转换器，将文档分割成更小的块，
2.然后EmbeddingsRedundantFilter 根据文档之间嵌入的相似性来过滤掉冗余文档，
该过滤操作以文本的嵌入向量为依据，也就是借助余弦相似度来衡量文本之间的相似程度，
进而判定是否存在冗余，它会把文本列表转化成对应的嵌入向量，然后计算每对文本之间的余弦相似度。
一旦相似度超出设定的阈值，就会将其中一个文本判定为冗余并过滤掉。
3.最后 EmbeddingsFilter 根据与查询的相关性进行过滤。'''
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
# EmbeddingsRedundantFilter 去除重复的文档块（用同一个嵌入模型为每个文档生成向量，
# 判断每两个文档块之间计算余弦相似度值是否相近，余弦相似度高（大于0.95）且相近，排名第二位的文档块过滤掉）
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)
#EmbeddingsFilter 过滤掉问题和文档快召回余弦相似度小于0.6的文档块
relevant_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.6)
#组合以上多种方式
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "deepseek的发展历程"
)
pretty_print_docs(compressed_docs)