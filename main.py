import config
from document_loaders import load_document
from chunking import chunk_data
from langchain.schema import Document
from embeddings import *
import simple_chat
import qa_chat
from langchain.callbacks.base import BaseCallbackHandler

import os, re, sys
import logging

def improve_question(question):
    question = re.sub(r'eu', 'eu, o consorciado, ', question, flags=re.IGNORECASE)
    question = re.sub(r'servopa', 'ADMINISTRADORA', question, flags=re.IGNORECASE)
    question = re.sub(r'voc[êe]s', 'ADMINISTRADORA', question, flags=re.IGNORECASE)
    return question

#Callback handler de stream - plugar na GUI do streamlit depois
class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}")


def chat_no_history(vector_store):
    chain = simple_chat.get_conversation_chain(vector_store, 10)

    while True:
        question = input("Enter question: ")
        if question == "quit":
            print("Bye")
            break

        #Necessário forçar essas palavras chave na busca vetorial
        question = improve_question(question)

        print("Asking LLM...")
        answer = simple_chat.ask_question(chain, question)
        print('answer below:')
        print(answer)

def chat_qa(vector_store):
    chain = qa_chat.get_conversation_chain(vector_store, 10)
    chat_history = []

    while True:
        question = input("Sua Mensagem > ")
        if question == "quit":
            print("Agradecemos seu contato. Até breve!")
            break

        #Necessário forçar essas palavras chave na busca vetorial
        question = improve_question(question)

        print("Asking LLM...")
        qa_chat.ask_question(chain, question, chat_history)
        print()

def main(load_doc_to_vector_store=False, no_history = False):
    vector_store = None

    if load_doc_to_vector_store:
        file_path = os.path.join('data', 'CONTRATO.pdf')
        pages:list[Document] = load_document(file_path)
        chunks:list[Document] = chunk_data(pages, chunk_size=256, overlap=24)

        vector_store = insert_or_fetch_embeddings(config.EMBEDDING_INDEX_NAME, chunks, reset_index=False)
    else:
        vector_store = load_embeddings(config.EMBEDDING_INDEX_NAME)

    if no_history:
        chat_no_history(vector_store)
    else:
        chat_qa(vector_store)  


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)

    count_args = len(sys.argv)

    load_to_vector_store =  True if count_args > 1 and sys.argv[1] == 'reload_vectors' else False
    no_history = True if  count_args > 2 and sys.argv[2] == 'nohistory' else False

    main(load_to_vector_store, no_history)
    

