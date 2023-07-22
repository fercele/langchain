from langchain.schema import Document
#Will split on \n \\n and whitespace
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_data(data: list[Document], chunk_size:int = 256, overlap:int = 0) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    #Split documents should be used when the document has already been splitted into pages
    #Use create_documents instead of split_documents, otherwise
    chunks = text_splitter.split_documents(data)
    return chunks