from redis import Redis
from dotenv import load_dotenv, find_dotenv
import os
from .keys import new_chat_key
from typing import List, Dict, Tuple
import json
from datetime import datetime

load_dotenv(find_dotenv())

def connect() -> Redis:
    return Redis(
        host=os.getenv("REDIS_HOST"), 
        port=os.getenv("REDIS_PORT"), 
        password=os.getenv("REDIS_PW"),
        db=0)



SCORE_MAPPING = {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}

class Chat:
    """
    Ao criar um novo chat (sem passar chat_key), é necessário passar os parâmetros de chat, que serão salvos no banco de dados
    Ao recuperar um chat existente (passando chat_key), não é possível passar parâmetros de chat, pois eles já estão salvos no banco de dados
    """
    def __init__(self, user_id: str, params:Dict = None, chat_key: str = None):
        self.redis = connect()
        self.user_id = user_id
        self.chat_key = chat_key
        
        r = self.redis

        if self.chat_key is not None and len(self.chat_key) > 0:
            #Foi passada chave, tenta ler chat existente do banco de dados
            assert params is None, "Não é possível passar parâmetros de chat ao tentar recuperar um chat existente"

            chat_obj = r.json().get(chat_key)
            if chat_obj is None:
                raise Exception(f"Não foi possível encontrar o chat com a chave {chat_key}")
            
            if self.user_id != chat_obj["user_id"]:
                raise Exception(f"O usuário {self.user_id} não corresponde ao chat existente {chat_key}")
            
            self.messages = chat_obj["messages"]
            if self.messages is None:
                self.messages = []
        
        else:
            #Não foi passada chave, cria um novo chat
            chat_key = self.__nova_chave()
            assert params is not None, "É necessário passar parâmetros de chat ao tentar criar um novo chat"
            
            result = r.json().set(
                name=chat_key, 
                path=".",
                obj={
                    "user_id": self.user_id, 
                    "messages": [],
                    "data_hora": json.dumps(datetime.now().isoformat()),
                    "params": params,
                },
                nx=True
            )
            if result is None:
                raise Exception(f"Não foi possível criar o chat, possivelmente tentou-se sobrescrever um chat já existente, chave: {chat_key}")
            
            self.chat_key = chat_key
            self.messages = []


    def __nova_chave(self):
        """
        Cria a chave para um novo chat usando uma key no redis
        Se a chave obtida corresponder a um objeto que já existe no redis, tenta novamente 10 vezes
        """
        r = self.redis

        chat_key = new_chat_key(r)

        retries = 10
        while r.exists(chat_key) and retries > 0:
            chat_key = new_chat_key(r)
            retries -= 1
        
        if retries == 0:
            raise Exception("Não foi possível obter uma chave de chat valida apos 10 tentativas")
        
        return chat_key
    
    def add_message(self, message: Tuple[str]):
        r = self.redis

        message_dict = {
            "pergunta": message[0],
            "resposta": message[1],
            "feedback":  {"score": None, "score_numeric": None, "text": None}
        }

        r.json().arrappend(self.chat_key, ".messages", message_dict)

        self.messages.append(message_dict)


    def get_messages(self, raw: bool = False):
        """
        Por padrão, se raw = False, retorna uma lista de tuplas (pergunta, resposta), formato apropriado para o langchain, por exemplo.
        Caso se deseje recuperar as mensagens com toda a estrutura, e os feedbacks, passar raw como True
        """
        if not raw:
            return [(message["pergunta"], message["resposta"]) for message in self.messages]
        
        #Retorna uma cópia rasa (shallow) da lista, para evitar modificações indesejadas no objeto
        return list(self.messages)
    

#PROMOVIDA A FUNÇÃO INDEPENDENTE PARA EVITAR TER QUE BUSCAR O CHAT INTEIRO SÓ PARA ATUALIZAR O FEEDBACK
def set_message_feedback(chat_key: str, message_index: int, feedback_score: str, feedback_text: str = None):

    r = connect()

    feedback = {"score": feedback_score, "score_numeric": SCORE_MAPPING.get(feedback_score), "text": feedback_text}
    # print(f"feedback: {feedback}")
    r.json().set(
        name=chat_key, 
        path=f".messages[{message_index}].feedback",
        obj=feedback
    )






if __name__ == "__main__":
    chat = Chat("laura", params={"model": "gpt-3.5-turbo-16k"})
    chave_gerada = chat.chat_key

    existing_chat = Chat("laura", chat_key=chave_gerada)
    print(existing_chat)
    print()

    assert existing_chat.chat_key == chave_gerada

    existing_chat.add_message(("system", "Você é um assistente gentil"))

    print(existing_chat)
    print(existing_chat.messages)
    print()

    existing_chat2 = Chat("laura", chat_key=chave_gerada)
    print(existing_chat2)
    print(existing_chat2.messages)
    print()
