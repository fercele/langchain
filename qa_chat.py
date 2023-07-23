# flake8: noqa
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from config import *
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever

#Templates below built from the source code of langchain.chains.conversational_retrieval.prompts
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
MY_CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. The user is a potential customer, and is refered to as CONSORCIADO in the pieces of information below. You are a sales assistant, and answer politely to questions of users. The company you represent is refered to as ADMINISTRADORA, and is named SERVOPA.

{context}

Question: {question}
Helpful Answer in Brazilian Portuguese:"""
MY_QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_conversation_chain(vector_store:Pinecone, top_k:int, streaming=True, callback_handler = StreamingStdOutCallbackHandler()) -> ConversationalRetrievalChain:
    llm_condense_question = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL_SECUNDARIO)
    llm_streaming_qa = ChatOpenAI(temperature=LLM_TEMPERATURE, model=LLM_MODEL, streaming=streaming, callbacks=[callback_handler])

    question_generator = LLMChain(llm=llm_condense_question, prompt=MY_CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm_streaming_qa, chain_type="stuff", prompt=MY_QA_PROMPT)

    retriever:VectorStoreRetriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':top_k})

    qa = ConversationalRetrievalChain(
        retriever=retriever, 
        combine_docs_chain=doc_chain, 
        question_generator=question_generator
    )

    return qa


def ask_question(qa_chain:ConversationalRetrievalChain, query:str, chat_history = []) -> str:
    result = qa_chain({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    chat_history.append((query,answer))
    return answer