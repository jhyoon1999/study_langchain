import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.callbacks.tracers import ConsoleCallbackHandler

#%% 1. 랭체인을 통해 GPT 호출하기
# 랭체인을 통해 호출하기 위해서는 ChatOpenAI()를 이용하여 llm 객체를 생성해야한다.
# llm 객체는 invoke()를 통해 사용자의 질문을 전달하고 답변을 얻는다.

#(1) api 키 불러오기
with open('gpt_api_key.text', 'r') as f :
    api_key = f.read()
os.environ['OPENAI_API_KEY'] = api_key

# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=2048,  # 최대 토큰수
    model_name="gpt-4o",  # 모델명
)

# 질의내용
question = "세종대왕이 누구인지 설명해주세요"

# 질의
result = llm.invoke(question)
print(result.content)

#%% 2. 랭체인의 프롬프트 템플릿
# 프롬프트 템플릿에서는 일종의 변수를 만들어두고, 정해진 템플릿 내에서 변수만 변경하여 GPT에게 입력을 전달하는 것이 가능
# 질문 템플릿 형식 정의
template = "{who}가 누구인지 설명해주세요"

# 템플릿 완성
prompt = PromptTemplate(
        template=template, input_variables=['who']
    )
print(prompt)
print(prompt.format(who="오바마"))

#프롬프트 템플릿은 llm 객체와 연결할 수 있으며, 이를 연결하는 매개체를 체인이라고 한다.

# 연결된 체인(Chain)객체 생성
llm_chain = prompt | llm
result = llm_chain.invoke({"who":"이순신 장군"})
print(result)

result = llm_chain.invoke({"who":"이순신 장군"},
                            config={'callbacks': [ConsoleCallbackHandler()]})
print(result)

#%%3. 랭체인의 메시지 히스토리
#과거 대화 내역을 반영하여 GPT와 대화할 수 있는 RunnableWithMessageHistory

#(1) history와 input이라는 두개의 변수를 가지는 새로운 프롬프트 템플릿을 선언
# 질문 템플릿 형식 정의
template= """아래는 사람과 AI의 친근한 대화입니다. AI의 이름은 공감봇입니다. 대화 문맥을 바탕으로 친절한 답변을 진행하세요.

Current Conversation: {history}

Human: {input}

AI:"""

# 템플릿 완성
prompt = PromptTemplate(
        template=template, input_variables=['history', 'input']
    )
print(prompt)

#LLM 객체 생성
llm = ChatOpenAI(model_name = 'gpt-4o')
chain = prompt | llm

#(2). 세션 설정
# 세션을 사용하면 여러 사용자가 동시에 챗봇과 대화할 때 각 대화를 독립적으로 관리할 수 있다.
store = {} #세션 저장소
session_id = "test" #세션 아이디

#할당된 세션 아이디가 저장소에 없으면 새로운 대화 기록 객체를 생성
if session_id not in store:
    store[session_id] = ChatMessageHistory()

#현재 세션의 대화 기록을 session_history에 할당
session_history = store[session_id]

#RunnableWithMessageHistory 객체 생성
with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: session_history,
    input_messages_key= "input",
    history_messages_key= "history",
)

# 주어진 메시지와 설정으로 체인을 실행합니다.
result = with_message_history.invoke(
    {"input": "당신은 어디에서 만들었습니까?"},
    config={"configurable": {"session_id": "test"}},
)
print(result.content)

result = with_message_history.invoke(
    {"input": "푸른 바다를 주제로 감성적이고 짧은 시를 하나 지어주세요"},
    config={"configurable": {"session_id": "test"}},
)
print(result.content)

result = with_message_history.invoke(
    {"input": "석양을 주제로도 해줘"},
    config={"configurable": {"session_id": "test"}},
)
print(result.content)

print(store)















































