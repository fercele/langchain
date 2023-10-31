from redis import Redis

def chat_key(chat_id) -> str:
    return f"chat#{chat_id}"

def new_chat_key(redis:Redis) -> str:
    #Usa uma chave numérica do redis como contador e gerador de ids
    new_chat_id = redis.incr("chat_id") 
    return f"chat#{new_chat_id}"