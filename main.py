from langchain.schema import Document
import os, sys
import logging

import config.config as config

os.environ['APP_ROOT_PATH'] = os.getcwd()
logging.basicConfig(level=config.LOG_LEVEL)

import util
from embeddings.pinecone import *
#from embeddings.chroma import *
from chat import simple_chat, qa_chat
from document_load.document_loaders import load_document
from document_load.chunking import chunk_data

def chat_no_history(vector_store):
    chain = simple_chat.get_conversation_chain(vector_store, config.EMBEDDINGS_TOP_K_RESULTS)

    while True:
        question = input("Enter question: ")
        if question == "quit":
            print("Bye")
            break

        #Necessário forçar essas palavras chave na busca vetorial
        question = util.improve_question(question)

        print("Asking LLM...")
        answer = simple_chat.ask_question(chain, question)
        print('answer below:')
        print(answer)

def chat_qa(vector_store):
    chain = qa_chat.get_conversation_chain(vector_store, config.EMBEDDINGS_TOP_K_RESULTS)
    chat_history = []

    while True:
        question = input("Sua Mensagem > ")
        if question == "quit":
            print("Agradecemos seu contato. Até breve!")
            break

        #Necessário forçar essas palavras chave na busca vetorial
        question = util.improve_question(question)

        print("Asking LLM...")
        qa_chat.ask_question(chain, question, chat_history)
        print()

def main(load_doc_to_vector_store=False, no_history = False):
    vector_store = None

    if load_doc_to_vector_store:
        file_path = os.path.join('data', 'CONTRATO.pdf')
        pages:list[Document] = load_document(file_path)
        chunks:list[Document] = chunk_data(pages) #Using defaults from config

        vector_store = insert_or_fetch_embeddings(config.EMBEDDING_INDEX_NAME, chunks, reset_index=True)
    else:
        vector_store = load_embeddings(config.EMBEDDING_INDEX_NAME)

    if no_history:
        chat_no_history(vector_store)
    else:
        chat_qa(vector_store)  


if __name__ == "__main__":
    
    count_args = len(sys.argv)

    load_to_vector_store = True if count_args > 1 and 'reload_vectors' in sys.argv else False
    no_history = True if count_args > 1 and 'nohistory' in sys.argv else False

    main(load_to_vector_store, no_history)
    

