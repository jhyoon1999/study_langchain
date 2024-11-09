import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import gradio as gr

with open('GPT_API_KEY.txt', 'r') as f:
    api_key = f.read().strip()
os.environ['OPENAI_API_KEY'] = api_key

#임베딩을 완료한 Chroma DB를 불러온다.
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory='./economic_terms_chroma_embedding.db',
                    embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs = {"k":2})

template = """당신은 한국은행에서 만든 금융 용어를 설명해주는 금융쟁이입니다.
윤정한 개발자가 만들었습니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 반말로 친근하게 답변하세요.
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever,
    return_source_documents=True)

def get_chatbot_response(input_text):
    chatbot_response = qa_chain.invoke(input_text)
    return chatbot_response['result'].strip()

# Streamlit UI
st.title("경제금융용어 챗봇")
st.write("질문해주세요!")

# 채팅 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사용자의 입력 받기
user_input = st.text_input("질문", "")

# 응답 생성 함수
def respond():
    if user_input:
        bot_message = get_chatbot_response(user_input)
        # 사용자와 챗봇 응답을 세션 상태에 추가
        st.session_state.chat_history.append((user_input, bot_message))
        # 입력 필드를 비우기
        st.experimental_rerun()

# 질문 제출 버튼
if st.button("질문 제출"):
    respond()

# 대화 초기화 버튼
if st.button("대화 초기화"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# 채팅 기록 표시
for user_msg, bot_msg in st.session_state.chat_history:
    st.write(f"**사용자:** {user_msg}")
    st.write(f"**챗봇:** {bot_msg}")

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















































































































































