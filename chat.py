import os
import sys

import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader

import secret
from recipe import general_template

sys.setrecursionlimit(2000)

langchain.verbose = False
os.environ["OPENAI_API_KEY"] = secret.OPENAI_API_KEY

# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.1, max_tokens = 100, )
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0.0, max_tokens=200)

conversation = ConversationChain(
    llm = llm, memory= ConversationBufferMemory()
)

verbose = False

def init(filepath: str) -> RetrievalQA:
    """initialize chat model

    Args:
        filepath (str) knowldge source file in PDF. AI model reply to your answer based on the given context here.
    """
    filename, extension = os.path.splitext(filepath)
    
    if extension.__contains__("pdf"):
        loader = PyPDFLoader(filepath)
        splitter =  RecursiveCharacterTextSplitter(
                separators=["\n", "\n\n"],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
        embeddings = OpenAIEmbeddings()
        pages = loader.load_and_split()
        texts = splitter.split_documents(pages)
    else:
        loader = UnstructuredFileLoader(filepath)
        splitter =  CharacterTextSplitter(
                separator=" ",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
        embeddings = OpenAIEmbeddings()
        pages = loader.load_and_split()
        texts = splitter.split_documents(pages)
        print(texts)
    
    vectordb = Chroma.from_documents(texts, embeddings)
    prompt_template = general_template
    
    qa_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = LLMChain(
        llm=llm,
        prompt=qa_prompt
    )
    
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=vectordb.as_retriever(),
        )
    return qa

def get_answer(user_message):
    ret = conversation.predict(input=user_message)
    return ret

# while True:
#     with get_openai_callback() as cb:
#         user_messasge = input("You: ")
#         ret = conversation.predict(input=user_messasge)
#         print(f"AI: {ret}")
        
#         if verbose == True:
#             print(cb)
        
#         costs = [f"Total Tokens: {cb.total_tokens}\n",
#                  f"Prompt Tokens: {cb.prompt_tokens}\n", 
#                  f"Completion Tokens: {cb.completion_tokens}\n", 
#                  f"Total Cost (USD): ${cb.total_cost}\n"]
        
#         with open("out.md", "w", ) as f:
#             f.write("# Prompt Cost\n")
#             f.writelines(costs)
#             f.write("\n")
        
#         with open("out.md", "a") as f:
#             f.write(ret)

if __name__ == "__main__":
    _init = init("asset/robodone_leaflet.txt")
    