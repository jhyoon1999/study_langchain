#### kernel 오류로 끊어졌을 경우 ####
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

def main() :
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

    #(4). 시작지점부터 다시 시작하기
    for i in range(99, len(texts)):
        print(i)
        text = texts[i]
        vectordb.add_documents(documents=[text])
        vectordb.persist()  # 각 문서가 추가된 후 데이터베이스 저장

if __name__ == "__main__":
    main()
