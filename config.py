import os, logging
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

LOG_LEVEL = logging.INFO
EMBEDDING_INDEX_NAME = 'ask-a-document'
EMBEDDING_DIMENSION = 1536
EMBEDDING_METRIC = 'cosine'

LLM_TEMPERATURE = 0.5
LLM_MODEL = 'gpt-3.5-turbo-16k'
LLM_MODEL_SECUNDARIO = 'gpt-3.5-turbo'
LLM_MAX_TOKENS = 16384


if __name__ == "__main__":
    print(os.environ.get('PINECONE_API_KEY'))
    print(os.environ.get('PINECONE_ENV'))