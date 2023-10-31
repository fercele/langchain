from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document

import pdfbox

from typing import List, Text
import enum
import os
import requests

class LoadMode(enum.Enum):
    SINGLE = "single"
    ELEMENTS = "elements"

def load_pdf_unstructured(path, mode=LoadMode.SINGLE) -> List[Document]:
    loader = UnstructuredPDFLoader(file_path=path, mode=mode.value)
    return loader.load()


def load_pdf_apache(pdf_path:Text) -> Document:
    p = pdfbox.PDFBox()

    #set output path to same folder as the file in path
    pdf_filename = os.path.basename(pdf_path)
    output_path = os.path.join(os.path.dirname(pdf_path), pdf_filename.replace(".pdf", ".txt"))

    p.extract_text(pdf_path, sort=True, output_path=output_path)
    p.extract_images(pdf_path, prefix=pdf_filename.replace(".pdf", ""))
    
    with open(output_path, mode="r", encoding="utf-8") as result:
        text = result.read()
    
    return Document(page_content=text, metadata={"source": pdf_filename})


def load_tika(pdf_path:Text) -> Document:
    tika_server = 'http://localhost:9998/tika'
    headers = {
        "X-Tika-PDFextractInlineImages": "true",
        "X-Tika-OCRLanguage": "por",
        "Accept-Charset": "UTF-8"
    }

    with open(pdf_path, 'rb') as file:
        response = requests.put(tika_server, headers=headers, data=file)

    text = response.content.decode('utf-8', 'ignore')
    pdf_filename = os.path.basename(pdf_path)
    return Document(page_content=text, metadata={"source": pdf_filename})


def write_docs_to_txt(out_file_path:Text, documents:List[Document]):
    with open(out_file_path, "w") as f:
        f.write("\n\n=========================================\n\n"
                .join([str(doc.metadata) + "\n\n" + doc.page_content for doc in documents]))
 

if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "dados")
    file_path = os.path.join(folder_path, "anuario-2023-texto-05-as-novas-configuracoes-dos-crimes-patrimoniais-no-brasil.pdf")

    # docs = load_pdf_unstructured(file_path, mode=LoadMode.SINGLE)
    # write_docs_to_txt(os.path.join(folder_path, "anuario-2023-texto-05-as-novas-configuracoes_SINGLE.txt"), docs)

    # docs = load_pdf_unstructured(file_path, mode=LoadMode.ELEMENTS)
    # write_docs_to_txt(os.path.join(folder_path, "anuario-2023-texto-05-as-novas-configuracoes_ELEMENTS.txt"), docs)

    # docs = [load_pdf_apache(file_path)]
    # write_docs_to_txt(os.path.join(folder_path, "anuario-2023-texto-05-as-novas-configuracoes_APACHE.txt"), docs)

    docs = [load_tika(file_path)]
    write_docs_to_txt(os.path.join(folder_path, "anuario-2023-texto-05-as-novas-configuracoes_TIKA.txt"), docs)

