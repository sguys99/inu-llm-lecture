import time
from typing import Callable, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.vectorstores import FAISS, Chroma


def get_vector_store(
    documents: List[Document],
    embedding: Embeddings,
    type: str = "faiss",
) -> Union[FAISS]:
    """
    주어진 Document 객체 목록과 임베딩을 사용하여 지정된 유형의 벡터 스토어를 반환하는 함수.

    Args:
        documents (List[Document]): 임베딩을 생성할 Document 객체 목록.
        embedding (Embeddings): 사용할 임베딩 모델.
        type (str, optional): 생성할 벡터 스토어의 유형. 기본값은 'faiss'.

    Returns:
        Union[FAISS]: 지정된 유형의 벡터 스토어 객체를 반환.

    Raises:
        ValueError: 지원되지 않는 벡터 스토어 유형이 입력된 경우 발생.
    """
    if type.lower().startswith("faiss"):
        vector_store = FAISS.from_documents(documents=documents, embedding=embedding)
    elif type.lower().startswith("chroma"):
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding)
    else:
        raise ValueError(f"Unsupported vector store type: {type}")

    return vector_store


def build_faiss_throttled(
    documents: List[Document],
    embedding: Embeddings,
    batch: int = 6,
    sleep: int = 60,
    progress: Optional[Callable[[int, int], None]] = None,
) -> FAISS:
    """무료 임베딩 한도(3 RPM / 10K TPM)에 맞춰 청크를 나눠 쉬어가며 FAISS 인덱스를 만든다.

    한 번에 모든 청크를 보내면 RateLimitError 가 나므로, 청크를 작게(batch) 나눠 보내고
    요청 사이에 잠시 대기(sleep)한다. 노트북 2.1_vectordb.ipynb 의 패턴을 그대로 옮긴 것.

    Args:
        documents (List[Document]): 임베딩할 분할 문서(청크) 목록.
        embedding (Embeddings): 사용할 임베딩 모델.
        batch (int, optional): 한 요청에 보낼 청크 수. 기본값 6.
        sleep (int, optional): 요청 간 대기 시간(초). 기본값 60.
        progress (Callable[[int, int], None], optional): (완료 수, 전체 수)를 받는 진행 콜백.

    Returns:
        FAISS: 생성된 FAISS 벡터 스토어.

    Raises:
        RuntimeError: 한도 초과로 반복 재시도에 실패한 경우 발생.
    """
    vector_store = None
    total = len(documents)

    for i in range(0, total, batch):
        batch_docs = documents[i : i + batch]

        for attempt in range(5):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch_docs, embedding)
                else:
                    vector_store.add_documents(batch_docs)
                break
            except Exception as e:
                print(f"  재시도 {attempt + 1}/5 - 30초 대기 ({type(e).__name__})")
                time.sleep(30)
        else:
            raise RuntimeError("임베딩 반복 실패 - batch를 줄이거나 결제수단을 등록하세요.")

        done = min(i + batch, total)
        if progress is not None:
            progress(done, total)
        if done < total:
            time.sleep(sleep)

    return vector_store


def get_splitter(
    chunk_size: int,
    chunk_overlap: int,
    type: str = "recursive",
) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter]:
    """
    청크 크기, 청크 중첩 값 및 분할기 유형에 따라 적절한 텍스트 분할기를 반환하는 함수.

    Args:
        chunk_size (int): 텍스트를 분할할 때 각 청크의 크기.
        chunk_overlap (int): 청크 간의 중첩 크기.
        type (str, optional): 사용할 분할기의 유형 ("recursive", "character", "token"). 기본값은 "recursive".
        model_name (str, optional): 사용할 모델의 이름. 기본값은 "gpt-4o".

    Returns:
        Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter]: 지정된 유형의 텍스트 분할기를 반환.

    Raises:
        ValueError: 지원되지 않는 분할기 유형을 입력한 경우 발생.
    """
    if type.lower().startswith("recursive"):
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif type.lower().startswith("character"):
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif type.lower().startswith("token"):
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Unsupported splitter type: {type}")

    return splitter
