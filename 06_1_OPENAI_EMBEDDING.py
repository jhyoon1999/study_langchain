#Objective : Use OpenAI's Embedding API and cosine similarity to create a simple search system

#%% 0. import libraries
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings

#(1) api 키 불러오기
with open('gpt_api_key.text', 'r') as f :
    api_key = f.read()
os.environ['OPENAI_API_KEY'] = api_key

#%%1. OpenAI Embedding API 사용해보기
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
query_result = embeddings.embed_query('저는 배가 고파요')
print(query_result)

#%%2. Embedding 벡터 
data = ['저는 배가 고파요',
        '저기 배가 지나가네요',
        '굶어서 허기가 지네요',
        '허기 워기라는 게임이 있는데 즐거워',
        '스팀에서 재밌는 거 해야지',
        '스팀에어프라이어로 연어구이 해먹을거야']

df = pd.DataFrame(data, columns=['text'])
df

def get_embedding(text):
  return embeddings.embed_query(text)

df['embedding'] = df.apply(lambda row: get_embedding(
        row.text,
    ), axis=1)
df

#%%3. 코사인 유사도 계산을 통한 유사도가 높은 문서 찾기
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(
        query
    )
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x),
                                                            np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity",
                                ascending=False).head(3)
    return top_three_doc

sim_result = return_answer_candidate(df, '아무것도 안 먹었더니 꼬르륵 소리가 나네')
sim_result












































































































































