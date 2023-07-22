import tiktoken

def estimate_price_for_embedding_generation(words:list[str], current_cost_by_1000tokens_for_model=0.0001) -> tuple[float, int]:
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')

    total_tokens = sum([len(enc.encode(word)) for word in words])
    estimated_cost = total_tokens * (current_cost_by_1000tokens_for_model / 1000)
    
    return (estimated_cost, total_tokens)