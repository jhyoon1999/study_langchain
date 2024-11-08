import openai
import os

#%%1. client 객체 만들기

#(1) api 키 불러오기
with open('gpt_api_key.text', 'r') as f :
    api_key = f.read()
os.environ['OPENAI_API_KEY'] = api_key

#(2) client 객체 만들기
client = openai.OpenAI()

#%%2. 기본 질문하기
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me how to make a pizza"}])

print(response)
print(response.choices[0].message.content)

#%%3. 역할 부여하기
# {"role": "system", "content":"역할 지시문"}
response = client.chat.completions.create(
 model="gpt-4o",
 messages=[
 {"role": "system", "content": "너의 이름은 jhyoon의 AI야. 답변을 시작할때 'jhyoon의 AI로서 대답합니다.'라고 먼저 말해야해."},
 {"role": "user", "content": "2020년 월드시리즈에서는 누가 우승했어?"}
 ]
)

print(response.choices[0].message.content)

#%%4. 이어서 질문하기 <- 랭체인에서 RUNNERABLE이 있어서 그냥 그런갑다 하자.
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."}, #역할부여
    {"role": "user", "content": "Who won the world series in 2020?"}, #선행질문
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}, #선행답변
    {"role": "user", "content": "Where was it played?"} #현재질문
  ]
)
print(response.choices[0].message.content)






















































































































































