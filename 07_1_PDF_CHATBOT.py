# Objective : 경제 금융 용어 챗봇 만들기
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import urllib.request
import gradio as gr

with open('GPT_API_KEY.txt', 'r') as f:
    api_key = f.read().strip()
os.environ['OPENAI_API_KEY'] = api_key

#%%1. PDF 파일 불러오기 및 정제
loader = PyPDFLoader("openai-api-tutorial-main/ch07/2020_경제금융용어 700선_게시.pdf")
texts = loader.load_and_split()
print('문서의 수 :', len(texts))

#개발자의 판단에 따라 '경제용어 설명하는 챗봇'에 불필요한 페이지 날리기
texts[12].page_content
print(texts[13].page_content)

texts = texts[13:]
print('줄어든 청크의 개수:', len(texts))

print(texts[-1])
texts = texts[:-1]

print(type(texts))

#%%2. 청크 임베딩 및 벡터 데이터베이스 적재

#####################################################################
#코드 방법 1
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
)

#코드 방법 2
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory='./economic_terms_chroma_embedding.db',
                embedding_function=embedding)
# 각 문서를 개별적으로 추가하고 매번 저장
for i in range(len(texts)):
    print(i)
    text = texts[i]
    vectordb.add_documents(documents=[text])
    vectordb.persist()  # 각 문서가 추가된 후 데이터베이스 저장

#### kernel 오류로 끊어졌을 경우 ####
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import urllib.request
import gradio as gr

with open('GPT_API_KEY.txt', 'r') as f:
    api_key = f.read().strip()
os.environ['OPENAI_API_KEY'] = api_key

#(1). 기존에 몇까지 갔는지 확인하기
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory='./economic_terms_chroma_embedding.db',
                    embedding_function=embedding)
print('완료된 문서의 수:', len(vectordb))

#(2). 기존 문서 청크 다시 만들기
loader = PyPDFLoader("openai-api-tutorial-main/ch07/2020_경제금융용어 700선_게시.pdf")
texts = loader.load_and_split()
texts = texts[13:]
texts = texts[:-1]
print('문서의 수 :', len(texts)) #352개가 맞는지 확인

#(3). 시작지점 확인하기
vectordb._collection.count()
documents = vectordb._collection.get()['documents']
documents[98] == texts[98].page_content

#(4). 시작지점부터 다시 시작하기
for i in range(99, len(texts)):
    print(i)
    text = texts[i]
    vectordb.add_documents(documents=[text])
    vectordb.persist()  # 각 문서가 추가된 후 데이터베이스 저장
################################################################################
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import urllib.request
import gradio as gr

with open('GPT_API_KEY.txt', 'r') as f:
    api_key = f.read().strip()
os.environ['OPENAI_API_KEY'] = api_key

#임베딩을 완료한 Chroma DB를 불러온다.
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory='./economic_terms_chroma_embedding.db',
                    embedding_function=embedding)

#vectordb를 선언하고 나면 ._collection 다음에 온점을 찍고 다양한 함수를 사용할 수 있다.
print('완료된 문서의 수:', vectordb._collection.count())

#_collection.get()은 벡터 데이터베이스 객체인 vectordb에 저장된 값들을 볼 수 있는 기능
for key in vectordb._collection.get() : # vectordb._collection.get()은 딕셔너리구나
    print(key)

#임베딩 벡터의 값은 기본적으로는 제공하지 않는다.
# embedding 호출 시도
result = vectordb._collection.get()['embeddings']
print(result)

#get()을 호출시 내부에서 include를 기재해야한다.
# embedding vetor만 조회하기
embeddings = vectordb._collection.get(include=['embeddings'])['embeddings']
print('임베딩 벡터의 개수 :', len(embeddings))

#metadatas는 각 청크의 출처를 의미한다.
metadatas = vectordb._collection.get()['metadatas']
print("metadatas의 개수:", len(metadatas))
print("0번 청크의 출처:", metadatas[0])

#%% 3. 검색기 객체
#벡터 도구 객체(vectordb)를 선언하고 나면 as_retriever()를 통해 입력된 텍스트로부터 유사한 텍스트를 찾아주는 검색기 객체인 retriever를 선언할 수 있다.
#그 후 get_relevant_documnets(입력)을 통해 입력과 유사한 청크들을 찾아서 반환
retriever = vectordb.as_retriever(search_kwargs = {"k":2}) #유사도를 기준으로 몇번째 순위까지 반환할 것인가
docs = retriever.get_relevant_documents("비트코인이 궁금해")
print('유사 문서 개수 :', len(docs))
print('--' * 20)
print('첫번째 유사 문서 :', docs[0])
print('두번째 유사 문서 :', docs[1])

#%% 4. 프롬프트 템플릿과 llm 객체
# Create Prompt
template = """당신은 한국은행에서 만든 금융 용어를 설명해주는 금융쟁이입니다.
윤정한 개발자가 만들었습니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 반말로 친근하게 답변하세요.
{context}

Question: {question}
Answer:
"""
#프롬프트 템플릿
prompt = PromptTemplate.from_template(template)
#llm 객체
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

#%% 5. 체인
#LLM 객체인 llm, 프롬프트 템플릿 객체인 prompt, 검색기 객체인 retriever를 연결
#이 3개의 객체를 연결하는 도구로 랭체인에서는 RetrievalQA.from_chain_type()을 지원
#각각의 객체를 연결하여 Chain 객체인 qa_chain을 만든다.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever,
    return_source_documents=True)

input_text = "디커플링이란 무엇인가?"
chatbot_response = qa_chain.invoke(input_text)
print(chatbot_response)

input_text = "너는 누구야?"
chatbot_response = qa_chain.invoke(input_text)
print(chatbot_response)

#%% 6. 그 외
# 
def get_chatbot_response(input_text):
    chatbot_response = qa_chain.invoke(input_text)
    return chatbot_response['result'].strip()

input_text = "너는 뭘하는 챗봇이니?"
result = get_chatbot_response(input_text)
print(result)

import gradio as gr

# 인터페이스를 생성.
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="경제금융용어 챗봇") # 경제금융용어 챗봇 레이블을 좌측 상단에 구성
    msg = gr.Textbox(label="질문해주세요!")  # 하단의 채팅창의 레이블
    clear = gr.Button("대화 초기화")  # 대화 초기화 버튼

    # 챗봇의 답변을 처리하는 함수
    def respond(message, chat_history):
      bot_message = get_chatbot_response(message)

      # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가.
      chat_history.append((message, bot_message))
      return "", chat_history

    # 사용자의 입력을 제출(submit)하면 respond 함수가 호출.
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # '초기화' 버튼을 클릭하면 채팅 기록을 초기화.
    clear.click(lambda: None, None, chatbot, queue=False)

# 인터페이스 실행.
demo.launch(debug=True)


























































































































































































