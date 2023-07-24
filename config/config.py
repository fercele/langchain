import os, logging
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

logging.basicConfig()
#Logger para a chave secreta
config_logger = logging.getLogger(__name__)
config_logger.setLevel(logging.INFO)

#Nivel de log para toda a aplicação, inclusive bibliotecas da OpenAI, LangChain, etc.
LOG_LEVEL = logging.INFO

EMBEDDING_INDEX_NAME = 'ask-a-document'
EMBEDDING_DIMENSION = 1536
EMBEDDING_METRIC = 'cosine'
EMBEDDINGS_TOP_K_RESULTS = 10

CHUNK_SIZE = 256
CHUNK_OVERLAP = 20

LLM_TEMPERATURE = 0.5
LLM_MODEL = 'gpt-3.5-turbo-16k'
LLM_MODEL_SECUNDARIO = 'gpt-3.5-turbo'
LLM_MAX_TOKENS = 16384

HISTORY_MAX_LENGTH = 5


import secrets
import string

def generate_api_key(length=64):
    alphabet = string.ascii_letters + string.digits
    api_key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return api_key

#A CADA START DA APLICAÇÃO, GERA UMA NOVA CHAVE TEMPORÁRIA E SETA COMO VARIÁVEL DE AMBIENTE
#A CADA DEMONSTRAÇÃO, SUBIR A APLICAÇÃO, OLHAR A CHAVE NO LOG, E USÁ-LA NOS TESTES OU FORNECER A QUEM FOR TESTAR
if os.getenv('TEMP_SECRET') is None:
    random_api_key = generate_api_key()
    os.environ['TEMP_SECRET'] = random_api_key
    config_logger.info("TEMP_SECRET: %s", random_api_key)


