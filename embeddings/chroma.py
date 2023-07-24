from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
import os, logging, shutil
from langchain.schema import Document
from config.config import EMBEDDING_DIMENSION, EMBEDDING_METRIC, LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

ROOT_PATH = os.getenv('APP_ROOT_PATH')
CHROMA_DB_DIR = os.path.join(ROOT_PATH, 'embeddings', 'chroma_data')

def load_embeddings(index_name:str) -> VectorStore:
    if not os.path.exists(CHROMA_DB_DIR):
        raise Exception("Vector store does not exist at configured persistent directory.")
    
    embeddings = OpenAIEmbeddings()
    
    vector_store = Chroma(
        collection_name=index_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
        
    return vector_store
    

#If index does not exist, create, else load embeddings from existing index
def insert_or_fetch_embeddings(index_name:str, chunks:list[Document], reset_index=False) -> VectorStore:
    embeddings = OpenAIEmbeddings()
    
    if reset_index or not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
        os.makedirs(CHROMA_DB_DIR)

        vector_store = Chroma.from_documents(
            collection_name=index_name,
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )

    else:
        vector_store = Chroma(
            collection_name=index_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR
        )

    return vector_store


def delete_index(index_name='all') -> bool:
    if index_name != 'all':
        raise Exception("Unsupported operation. Can only delete full chroma store")
    
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        return True
    
    return False


if __name__ == "__main__":
    logging.basicConfig()
    print(delete_index('all'))