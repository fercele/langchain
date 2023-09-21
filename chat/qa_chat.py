# flake8: noqa
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from config.config import *
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever

#Templates below built from the source code of langchain.chains.conversational_retrieval.prompts
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language (Brazilian Portuguese) .

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
MY_CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """You are a smart data analysis assistant. Use the following pieces of context to answer the question at the end. They are from a report called 'SÍNTESE DE INDICADORES SOCIAIS - UMA ANÁLISE DAS CONDIÇÕES DE VIDA DA POPULAÇÃO BRASILEIRA - 2022'. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer EXCLUSIVELY in Brazilian Portuguese. 

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

"""
Só passa na conversa as últimas <config.HISTORY_MAX_LENGTH> mensagens, para manter o 
tamanho do prompt sob controle, e não ter que exigir que o usuário comece uma nova conversa,
Mesmo assim, armazena o histórico completo na list chat_history, possibilitando posterior armazenamento no banco 
de dados para referência ou auditoria futura
"""
def ask_question(qa_chain:ConversationalRetrievalChain, query:str, chat_history = []) -> str:
    history_length = len(chat_history)
    start_index =  (history_length - HISTORY_MAX_LENGTH) if (history_length > HISTORY_MAX_LENGTH) else 0

    #print(f"enviando histórico da msg [{start_index}] até o fim da history")
    result = qa_chain({"question": query, "chat_history": chat_history[start_index : ]})
    answer = result["answer"]
    chat_history.append((query,answer))
    return answer