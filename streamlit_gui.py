import streamlit as st
import os, logging
from config.config import *
from langchain.callbacks.base import BaseCallbackHandler

os.environ['APP_ROOT_PATH'] = os.getcwd()
logging.basicConfig(level=LOG_LEVEL)

ROOT_PATH = os.getenv('APP_ROOT_PATH')
logging.debug('in streamlit root_path is %s', ROOT_PATH)

from embeddings.pinecone import *
from chat import qa_chat
import util

#Primeiro - user, segundo - AI
TEXT_COLORS_DARK = ["#FFFFFF", "#f2bdf0"]
TEXT_COLORS_LIGHT = ["#000000", "#f21feb"]

#Callback handler de stream - plugar na GUI do streamlit depois
class SessionStateCallbackHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        #print(token, end='')
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")
    
    def reset_container(self, container):
        #print(f'container updated to {container}')
        self.container = container
        self.text = ""

class AcessoNegadoException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def check_secret():
    current_secret = os.getenv('TEMP_SECRET')
    user_secret = st.session_state['USER_SECRET']
    if current_secret != user_secret:
        raise AcessoNegadoException("Acesso Negado")

def init_chat():
    try:
        check_secret()
        vector_store = load_embeddings(EMBEDDING_INDEX_NAME)
        st.session_state['vector_store'] = vector_store
        chain = qa_chat.get_conversation_chain(vector_store, EMBEDDINGS_TOP_K_RESULTS, callback_handler=st.session_state['callback_handler'])
        st.session_state['chain'] = chain
        st.session_state['chat_history'] = []
    except AcessoNegadoException as e:
        st.error("Acesso negado. Verifique a Chave Temporária digitada.")

def chat() -> str:
    try:
        check_secret()
        st.session_state['answer'] = ''

        if 'vector_store' not in st.session_state:
            # with st.spinner('Por favor aguarde...'):
            init_chat()

        vector_store = st.session_state['vector_store']
        if vector_store is None:
            raise Exception("Falha inicializando vector store")

        chat_history = st.session_state['chat_history']
        chain = st.session_state['chain']
        question = st.session_state['question']

        #Necessário forçar essas palavras chave na busca vetorial
        question = util.improve_question(question)

        answer = qa_chat.ask_question(chain, question, chat_history)
        
        #Por precaucão. Remover depois.
        st.session_state['chat_history'] = chat_history

        return answer
    
    except AcessoNegadoException as e:
        st.error("Acesso negado. Verifique a Chave Temporária digitada.")

def main():
    try:
        st.image(os.path.join(ROOT_PATH, 'assets', 'header-logo.png'))
        st.subheader('Prova de Conceito VictorIA :books:')

        with st.sidebar:
            api_key = st.text_input('Chave Temporária:', type='password', key='USER_SECRET')
            dark_mode = st.checkbox('Ajustar para modo noturno')
            reiniciar_chat = st.button("Reiniciar o chat")
            if reiniciar_chat:
                st.session_state['chat_history'] = []
                callback_handler = st.session_state['callback_handler']
                callback_handler.reset_container(None)
                st.session_state['question'] = ''

        question = st.text_input(label='Sua Pergunta', key='question')
        ask_button = st.button("Enviar")

        #st.markdown("### streaming box")
        chat_box = st.empty()
        #print(f'chat box is {chat_box}')

        st.divider()

        if 'callback_handler' not in st.session_state:
            callback_handler = SessionStateCallbackHandler(chat_box, display_method='write')
            st.session_state['callback_handler'] = callback_handler
        else:
            callback_handler = st.session_state['callback_handler']
            callback_handler.reset_container(chat_box)

        #st.markdown("### together box")

        if question and ask_button:
            answer = chat()
            #print()

        if 'chat_history' in st.session_state:
            chat_history = st.session_state['chat_history']
            colors = TEXT_COLORS_DARK if dark_mode else TEXT_COLORS_LIGHT

            for message in chat_history:
                st.markdown(unsafe_allow_html = True, 
                            body=f'<span style="color: {colors[0]}"><b>Você: </b>{message[0]}</span>')
                st.markdown(unsafe_allow_html = True, 
                            body=f'<span style="color: {colors[1]}"><b>VictorIA: </b>{message[1]}</span><BR/>')

        
    except AcessoNegadoException as e:
        st.error("Acesso negado. Verifique a Chave Temporária digitada.")

if __name__ == "__main__":
    main()