import os, sys

import config.config as config

import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity, get_embedding
import tiktoken
import locale


def generate_embedding_for_text_type01(text:str) -> list[float]:
    result = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002'
    )
    embedding_vector = result['data'][0]['embedding']
    print('generate_embedding_for_test_type01:', len(embedding_vector))
    print('generate_embedding_for_test_type01:', embedding_vector[:10])
    return embedding_vector


def generate_embedding_for_text_type02(text:str) -> list[float]:
    embedding_vector = get_embedding(text, engine='text-embedding-ada-002')
    print('generate_embedding_for_test_type02:', len(embedding_vector))
    print('generate_embedding_for_test_type02:', embedding_vector[:10])
    return embedding_vector



def generate_df_with_embeddings(df:pd.DataFrame, col_name_to_embed:str) -> pd.DataFrame:
    result_df = df.copy()

    result_df['embedding'] = result_df[col_name_to_embed].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
   
    return result_df


def convert_embeddings_to_nparray(df) -> pd.DataFrame:
    #the csv contains string representations of python lists
    #eval takes a string and evaluates it as a python expression, so converts the column in python lists
    #np.array gets a list and returns a numpy array, so the final result is a column of the dataframe containing
    #numpy arrays
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    
    return df


def estimate_price_for_embedding_generation(words:list[str], current_cost_by_1000tokens_for_model=0.0001) -> float:
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')

    total_tokens = sum([len(enc.encode(word)) for word in words])
    estimated_cost = total_tokens * (current_cost_by_1000tokens_for_model / 1000)
 
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    print(f'Estimated cost for {total_tokens} tokens is {locale.currency(estimated_cost)} or {estimated_cost:.10f}')
    
    return estimated_cost


def similarity_search_combining_words_in_df(df, *words) -> pd.DataFrame:
    print(f'words_to_be_combined_as_search_criteria: {words}')
    #gets two vectors
    filtered_df = df.loc[df['text'].isin(words)]['embedding']
    print('filtered df')
    print(filtered_df)

    result_vector = None
    for vector in filtered_df:
        print(f'adding vector with first element {vector[0]}')
        if result_vector is None:
            result_vector = vector
        else:
            result_vector += vector
        print(f'result vector is {result_vector}')

    #Find the words that are similar to milk + capuccino
    df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, result_vector))
    sorted_df = df.sort_values('similarities', ascending=False)

    #returns the first 10 lines
    result = sorted_df.iloc[:10].copy()

    print('similarities found')
    print(result)
    
    return result


def similarity_search(df, word):
    embedding = generate_embedding_for_text_type02(word)
    df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, embedding))
    sorted_df = df.sort_values('similarities', ascending=False)
    result = sorted_df.iloc[:10].copy()

    print('similarities found')
    print(result)
    
    return result

def main():
    #text = 'red'
    # generate_embedding_for_test_type01(text)
    # generate_embedding_for_test_type02(text)

    # print(ROOT_PATH)
    # original_df = pd.read_csv(os.path.join(ROOT_PATH, 'data', 'words.csv'))
    # new_df_with_embeddings = generate_df_with_embeddings(original_df, 'text')
    # new_df_with_embeddings.to_csv(os.path.join(ROOT_PATH, 'data', 'my-words-embeddings.csv'), index=False)
    
    df_with_embeddings = pd.read_csv(os.path.join(ROOT_PATH, 'data', 'my-words-embeddings.csv'))
    df_with_embeddings = convert_embeddings_to_nparray(df_with_embeddings)
    
    #estimate_price_for_embedding_generation()

    #estimate_price_for_embedding_generation(df['text'])
    #result = similarity_search_combining_words_in_df(df, 'coffee', 'sandwich')

    result = similarity_search(df_with_embeddings, 'dog')
    print(result)

if __name__ == '__main__':
    main()