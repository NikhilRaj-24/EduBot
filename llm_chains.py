from prompt_templates import memory_prompt_template
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
import chromadb
import yaml
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_llm():
    llm = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")
    return llm

def create_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key = "history", chat_memory = chat_history, k = 3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm = llm, prompt = chat_prompt, memory = memory)

def load_normal_chain(chat_history):
    return chatChain(chat_history)

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma

class chatChain:

    def __init__(self,chat_history):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input = user_input, history = self.memory.chat_memory.messages, stop = ["Human:"])
        
