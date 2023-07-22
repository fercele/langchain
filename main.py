import config
from document_loaders import load_document
from chunking import chunk_data
from langchain.schema import Document
from embeddings import insert_or_fetch_embeddings
from chat import get_conversation_chain, ask_question

import os, re
import logging

def improve_question(question):
    question = re.sub(r'eu', 'eu, o consorciado, ', question, flags=re.IGNORECASE)
    question = re.sub(r'servopa', 'ADMINISTRADORA', question, flags=re.IGNORECASE)
    question = re.sub(r'voc[êe]s', 'ADMINISTRADORA', question, flags=re.IGNORECASE)
    return question

if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    file_path = os.path.join('data', 'CONTRATO.pdf')
    pages:list[Document] = load_document(file_path)
    chunks:list[Document] = chunk_data(pages, chunk_size=256, overlap=24)

    vector_store = insert_or_fetch_embeddings(config.EMBEDDING_INDEX_NAME, chunks, reset_index=False)
    
    chain = get_conversation_chain(vector_store, 10)

    while True:
        question = input("Enter question: ")
        if question == "quit":
            print("Bye")
            break

        #Necessário forçar essas palavras chave na busca vetorial
        question = improve_question(question)

        print("Asking LLM...")
        answer = ask_question(chain, question)
        print('answer below:')
        print(answer)
