#%%1. RECURSIVE_TEXT_SPLITTER
import urllib.request
from langchain.text_splitter import RecursiveCharacterTextSplitter

urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")

with open("2016-10-20.txt", encoding="utf-8") as f:
    file = f.read()
print('텍스트의 길이:', len(file))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))

texts[1].page_content
texts[2].page_content

print('1번 청크의 길이:', len(texts[1].page_content))
print('2번 청크의 길이:', len(texts[2].page_content))

#%% 2. SemanticChunker
#OpenAI의 Embedding API를 사용하여 각 문장을 임베딩 벡터로 변환하고, 유사도를 구해서 유사한 문장끼리 그룹화하는 방식으로 동작
#어느정도 문맥의 의미가 고려된 청크들로 분할된다
import os
import urllib.request
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

with open('gpt_api_key.text', 'r') as f :
    api_key = f.read()
os.environ['OPENAI_API_KEY'] = api_key

with open("openai-api-tutorial-main/ch06/test.txt", encoding="utf-8") as f:
    file = f.read()
print('텍스트의 길이:', len(file))

text_splitter = SemanticChunker(OpenAIEmbeddings())
texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))

#%% 2-1. SematicChunker가 문서를 분할하는 방법 : 백분위수 방식
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))

#%% 2-2. SematicChunker가 문서를 분할하는 방법 : 표준편차 방식
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3,
)
texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))

#%% 2-3. SematicChunker가 문서를 분할하는 방법 : 사분위수 방식
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=1.5
)
texts = text_splitter.create_documents([file])
print('분할된 청크의 수:', len(texts))

















































































































































































