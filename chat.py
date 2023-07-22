from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever
import logging
from config import LOG_LEVEL
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

def get_custom_prompt():
    # system_message_prompt = SystemMessagePromptTemplate.from_template(
    #     '''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. In brazilian portuguese: Você é um vendedor de consórcios Servopa, e só responde perguntas referentes ao consórcio da Servopa, com base nas informações abaixo.
        
    #     {context}'''
    #     , input_variables=['context']
    # )

    # human_message_prompt = HumanMessagePromptTemplate.from_template(
    #     '''Question: {question}
    #     Answer in Brazilian Portuguese:'''
    #     , input_variables=['question']
    # )

    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    from langchain.prompts import PromptTemplate
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. The user is a potential customer, and is refered to as CONSORCIADO in the pieces of information below. The company you represent is refered to as ADMINISTRADORA, and is named SERVOPA. 

    {context}

    Question: {question}
    Answer in Brazilian Portuguese:"""
    chat_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return chat_prompt




def get_conversation_chain(vector_store:Pinecone, top_k:int) -> Chain:
    retriever:VectorStoreRetriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':top_k})
    
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)

    chain_type_kwargs = {"prompt": get_custom_prompt()}

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    return chain

def ask_question(llm_vector_chain:Chain, query:str) -> str:
    return llm_vector_chain.run(query)
