from typing import Union

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    PyPDFLoader,
)


def get_pdf_loader(
    file_path: str,
    type: str = "pypdf",
) -> Union[PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader, PDFMinerLoader]:
    """
    주어진 파일 경로와 로더 유형에 따라 적절한 PDF 로더를 반환하는 함수.

    Args:
        file_path (str): 로드할 PDF 파일의 경로.
        type (str, optional): 사용할 로더의 유형. ("pypdf", "pymupdf", "pdfplumber", "pdfminer"). 기본값은 "pypdf".

    Returns:
        Union[PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader, PDFMinerLoader]: 지정된 유형의 PDF 로더 객체를 반환.

    Raises:
        ValueError: 지원되지 않는 로더 유형이 입력된 경우 발생합니다.
    """
    if type.lower().startswith("pypdf"):
        loader = PyPDFLoader(file_path)
    elif type.lower().startswith("pymupdf"):
        loader = PyMuPDFLoader(file_path)
    elif type.lower().startswith("pdfplumber"):
        loader = PDFPlumberLoader(file_path)
    elif type.lower().startswith("pdfminer"):
        loader = PDFMinerLoader(file_path)
    elif type.lower().startswith("directory"):
        loader = PyPDFDirectoryLoader(file_path)
    else:
        raise ValueError(f"Unsupported loader type: {type}")

    return loader


def get_docx_loader(
    file_path: str,
    type: str = "docx2txt",
) -> Union[Docx2txtLoader]:
    """
    주어진 파일 경로와 로더 유형에 따라 적절한 docx 로더를 반환하는 함수.

    Args:
        file_path (str): 로드할 docx 파일의 경로.
        type (str, optional): 사용할 로더의 유형. ("docx2txt"). 기본값은 "docx2txt".

    Returns:
        Union[Docx2txtLoader]: 지정된 유형의 docx 로더 객체를 반환.

    Raises:
        ValueError: 지원되지 않는 로더 유형이 입력된 경우 발생.
    """
    if type.lower().startswith("docx2txt"):
        loader = Docx2txtLoader(file_path)
    elif type.lower().startswith("directory"):
        loader = DirectoryLoader(file_path, loader_cls=Docx2txtLoader)
    else:
        raise ValueError(f"Unsupported loader type: {type}")

    return loader
