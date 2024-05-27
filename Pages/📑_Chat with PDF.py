import streamlit as st
import os
from langchain import hub
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import RetrievalQA 
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores.chroma import Chroma
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
# Load variables from .env file
load_dotenv()


# Now you can access your variables using os.environ
# variable_value = os.environ.get("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = '' #new key
# print(variable_value)


def groq_client():
    groq = ChatGroq(
    temperature= 0,
    model_name = 'mixtral-8x7b-32768',
    streaming=True,
    callback_manager= CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return groq


# load pdf using PDFMinerLoader and trasnform in into vectorstore for additional knowledge for LLM
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    #create embeddings here
    
    #create vector store here
    db_nomic = Chroma.from_documents(texts, embeddings, persist_directory="./new_db")
    db_nomic.persist()
    
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
    
@st.cache_resource
def qa_llm() :
    llm = groq_client()
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectordb = Chroma(persist_directory="./new_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    
    return chain, vectordb
def process_answer(instruction):
    
    instruction = instruction
    qa,db = qa_llm()
    generated_text = qa.invoke(instruction)
    soruce = db.similarity_search(instruction)
    print("---"+generated_text)
    return generated_text,soruce


def main():
    
    st.set_page_config(
    page_title="PDF Augmentation Wizard 📚✨",
    page_icon="🔮",
    layout="wide"
    )
    
    st.header("Chat with your PDF")

    uploaded_file = st.file_uploader("📤 Upload a PDF document", type=["pdf"])
    

    if "embedding_done" not in st.session_state:
                    st.session_state.embedding_done = False    
                    
    if uploaded_file:
        st.success("File uploaded successfully! 🎉")
        st.sidebar.header("Choose an large language Model")
        ai_model = st.sidebar.selectbox("Select an LLM", ["Select Model", "Open-source"])
  
  
        if ai_model == "Open-source":
            if st.sidebar.button("fire_llm"):
                
                if not st.session_state.embedding_done and uploaded_file is not None:
                    filepath = "docs/"+uploaded_file.name
                    with open(filepath, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    
                    with st.spinner('Embeddings are in process...'):
                        start = time.time()
                        ingested_data = data_ingestion()
                        st.session_state.embedding_done = True
                        st.experimental_set_query_params()
                        st.sidebar.success("Vector store created ✔️")
                        end = time.time()
                        print(f"Time taken to create embeddings: {end-start}")
                    
        if st.session_state.embedding_done and ai_model == "Open-source":
            question  = st.text_area("enter your question: ")
            if st.button("ask"):
        
                st.info("RAG ouptut --- ")
                start = time.time()
                answer,source = process_answer(question)
                st.write(answer)
                st.write(source)
                end = time.time()
                print(f"Time taken to get answer: {end-start}")
                
        if st.button("Quit"):
            st.rerun()
            
            
if __name__ == "__main__":
     main()
