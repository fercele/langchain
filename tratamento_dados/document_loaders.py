from langchain.schema import Document
import os
import logging
from config.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

def load_document(file) -> list[Document]:
    name, extension = os.path.splitext(file)
    extension = extension.upper().strip('.')

    logger.debug('extension is %s', extension)

    match extension:
        case 'PDF':
            return load_PDF(file)
        case 'DOCX':
            return load_DOCX(file)
        case 'TXT':
            return load_TXT(file)
        case _ :
            logger.warn('unsupported extension: %s', extension)
            return None


#Loads the pdfs using PyPDF into an array of Documents
#Each document contents the page content and metadata with the page number
def load_PDF(file) -> list[Document]:
    from langchain.document_loaders import PyPDFLoader
    logger.debug(f'Loading PDF file {file}')
    loader = PyPDFLoader(file)
    document_list = loader.load()
    return document_list

def load_DOCX(file)  -> list[Document]:
    logger.debug(f'Loading DOCX file {file}')
    from langchain.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file)
    document_list = loader.load()
    return document_list

def load_TXT(file)  -> list[Document]:
    logger.debug(f'Loading DOCX file {file}')
    from langchain.document_loaders import TextLoader
    loader = TextLoader(file)
    document_list = loader.load()
    return document_list

if __name__ == "__main__":
    logging.basicConfig()
    file_path = os.path.join('data', 'react.pdf')
    pages = load_document(file_path)
    if pages is None:
        print("NO DATA FOR PDF FILE")
    else:
        print(f'pdf Document has {len(pages)} pages')

    file_path = os.path.join('data', 'the_great_gatsby.docx')
    pages = load_document(file_path)
    if pages is None:
        print("NO DATA FOR DOCX FILE")
    else:
        print(f'docx Document has {len(pages)} pages')
        print(f'page of doc is {pages[0].page_content[:100]} - {pages[0].metadata}')

