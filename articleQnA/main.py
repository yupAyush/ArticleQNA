import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings




from dotenv import load_dotenv
load_dotenv() #to load env file


st.title("article reasearch tool")
st.sidebar.title("article urls")

urls=[]

for i in range(3):
   url= st.sidebar.text_input(f"URL {i+1}")
   urls.append(url)

process_url_clicked =st.sidebar.button("process urls")
file_path="faiss_store_openai.pkl"

main_placefolder=st.empty()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


llm = GoogleGenerativeAI(model="gemini-pro")
if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()
    main_placefolder.text("Data loading ......started...")
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['/n/n','/n','.',','],
        chunk_size=1000
    )
    docs= text_splitter.split_documents(data)
   
    vectorStore_openai=FAISS.from_documents(docs,embeddings)
    vectorStore_openai.save_local("faiss_index")
    

    

query= main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore=FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever= vectorstore.as_retriever(search_kwargs={"k": 1})) 
            result=chain({"question":query},return_only_outputs=True)    
            st.header("Answer")
            st.subheader(result["answer"])
