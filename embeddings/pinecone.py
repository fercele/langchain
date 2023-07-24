import pinecone
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
import os, logging
from langchain.schema import Document
from config.config import EMBEDDING_DIMENSION, EMBEDDING_METRIC, LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))


def load_embeddings(index_name:str) -> VectorStore:
    embeddings = OpenAIEmbeddings()

    if index_name in pinecone.list_indexes():
         return Pinecone.from_existing_index(index_name, embeddings)
    else:
        raise Exception(f"index {index_name} not found in Pinecone")
    

#If index does not exist, create, else load embeddings from existing index
def insert_or_fetch_embeddings(index_name:str, chunks:list[Document], reset_index=False)-> VectorStore:
    embeddings = OpenAIEmbeddings()

    if index_name in pinecone.list_indexes():
        if reset_index:
            logger.debug("Index %s already exists. reset is True. Deleting all vectors...")
            index = pinecone.Index(index_name)
            index.delete(delete_all=True)
            logger.debug("Done")
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        else:
            logger.debug("Index %s already exists. Loading embeddings...", index_name)
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
            logger.debug("Embeddings loaded")
    else:
         logger.debug("Index %s doesn't exist, creating and inserting embeddings...", index_name)
         pinecone.create_index(index_name, dimension=EMBEDDING_DIMENSION, metric=EMBEDDING_METRIC)
         vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
         logger.debug("Index created")

    return vector_store

def delete_index(index_name='all') -> bool:
    indexes = pinecone.list_indexes()
    if len(indexes) == 0:
        return False

    if index_name.lower().strip() == 'all':
        logger.info('Deleting all indexes')
        for index in indexes:
            pinecone.delete_index(index)
        return True
    else:
        if index_name in indexes:
            logger.info('Deleting index %s', index_name)
            pinecone.delete_index(index_name)
            return True
    
    return False

if __name__ == "__main__":
    logging.basicConfig()
    print(delete_index('all'))