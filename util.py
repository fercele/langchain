import tiktoken, re

def estimate_price_for_embedding_generation(words:list[str], current_cost_by_1000tokens_for_model=0.0001) -> tuple[float, int]:
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')

    total_tokens = sum([len(enc.encode(word)) for word in words])
    estimated_cost = total_tokens * (current_cost_by_1000tokens_for_model / 1000)
    
    return (estimated_cost, total_tokens)

"""
'Truque' para ajudar a busca vetorial a associar o interlocutor/usuário com o consorciado no contrato,
e termos como 'empresa', 'vocês', 'servopa', com administradora, pois esses termos são centrais a todos
os conceitos dentro do contrato.

TODO - Fazer a substituição pelo prompt, pedindo para o LLM. (Não fiz ainda pois ele gerará substituições no histórico também)
"""
def improve_question(question):
    question = re.sub(r'eu', 'eu, o consorciado, ', question, flags=re.IGNORECASE)
    question = re.sub(r'bb', 'Administradora BB Consórcios', question, flags=re.IGNORECASE)
    question = re.sub(r'voc[êe]s', 'vocês, Administradora BB Consórcios, ', question, flags=re.IGNORECASE)
    return question