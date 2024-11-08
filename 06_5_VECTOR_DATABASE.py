#현업에서는 문서의 임베딩을 적재하기 위한 용도로 특별히 만들어진 도구인 벡터 데이터베이스를 사용하는 경우가 많다.
#벡터 데이터베이스로는 Faiss, Chroma 등 다양한 데이터베이스가 있다.
import os
import urllib.request
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

with open('gpt_api_key.text', 'r') as f :
    api_key = f.read()
os.environ['OPENAI_API_KEY'] = api_key

#%% 1. 크로마(Chroma)
loader = PyPDFLoader('openai-api-tutorial-main/ch06/2023_북한인권보고서.pdf')
pages = loader.load_and_split()
print('청크의 수:', len(pages))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#파이썬 문자열을 분할할때는 create_documents()를 사용한다.
#하지만 현재는 PyPDFLoader가 로드한 각각의 청크는 Documnet 형식을 가진다.
#이때는 split_documents()를 사용한다.
splited_docs = text_splitter.split_documents(pages)
print('분할된 청크의 수:', len(splited_docs))

chunks = [splited_doc.page_content for splited_doc in splited_docs]
print('청크의 최대 길이 :',max(len(chunk) for chunk in chunks))

#각 청크를 임베딩과 동시에 크로마 데이터베이스에 적재할 때는 Chroma.from_documents()를 사용한다.
db = Chroma.from_documents(splited_docs, OpenAIEmbeddings())
print('문서의 수:', db._collection.count())

#데이터베이스 객체를 만들고 나서 사용자의 입력과 유사도가 높은 문서들을 찾을 때는 similarity_search(사용자 입력)을 사용한다.
question = '북한의 교육과정'
docs = db.similarity_search(question)
print('문서의 수:', len(docs))

for doc in docs:
  print(doc)
  print('--' * 10)

#크로마 벡터 데이터베이스를 파일로 저장
db_to_file = Chroma.from_documents(splited_docs, OpenAIEmbeddings(), persist_directory = './chroma_test.db')
print('문서의 수:', db_to_file._collection.count())

#로드
db_from_file = Chroma(persist_directory='./chroma_test.db',
		    embedding_function=OpenAIEmbeddings())
print('문서의 수:', db_from_file._collection.count())





































































































































