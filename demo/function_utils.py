import base64
import os
import re
import shutil
from io import BytesIO
from time import time
from typing import Any

import mammoth
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

from app.modules.llms import get_embedding
from app.modules.vector_db import build_faiss_throttled, get_splitter
from app.utils.path import CACHE_EMBEDDING_PATH, CACHE_FILE_PATH, DEMO_IMG_PATH

load_dotenv()

human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avartar = DEMO_IMG_PATH / "inu-logo.png"

# 채팅 버블 안에서 마크다운 제목(#)이 거대한 H1/H2로 렌더링되는 것을 막기 위한 패턴.
_HEADING_PATTERN = re.compile(r"^[ \t]*#{1,6}[ \t]+(\S.*?)[ \t]*#*$")
_CODE_FENCE_PATTERN = re.compile(r"^[ \t]*(```|~~~)")


def normalize_chat_markdown(text: str) -> str:
    """
    채팅 메시지의 마크다운 제목(#, ##, ...)을 볼드(**...**)로 강등하는 함수.

    LLM이 답변을 마크다운 제목으로 시작하면 좁은 채팅 버블 안에서 글자가
    지나치게 크게 표시되므로, 제목 문법만 볼드로 바꿔 가독성을 유지.
    단, 코드 블록(```, ~~~) 내부의 `#`(주석 등)은 변환하지 않음.

    Args:
        text (str): 원본 메시지 문자열.

    Returns:
        str: 제목이 볼드로 치환된 문자열.
    """
    in_fence = False
    lines = []
    for line in text.split("\n"):
        if _CODE_FENCE_PATTERN.match(line):
            in_fence = not in_fence
            lines.append(line)
        elif in_fence:
            lines.append(line)
        else:
            lines.append(_HEADING_PATTERN.sub(r"**\1**", line))
    return "\n".join(lines)


def load_image(image_path: str) -> bytes:
    """
    주어진 경로에서 이미지를 읽어와 바이트 형태로 반환하는 함수.

    Args:
        image_path (str): 이미지 파일의 경로.

    Returns:
        bytes: 이미지 파일의 바이트 데이터.
    """
    with open(image_path, "rb") as image_file:
        return image_file.read()


class ChatCallbackHandler(BaseCallbackHandler):
    """
    LLM 모델의 실행 중 상태를 처리하는 콜백 핸들러 클래스.

    이 핸들러는 메시지 박스를 업데이트하고, 새로운 토큰을 수신할 때마다 실시간으로
    사용자 인터페이스에 반영.

    Attributes:
        message (str): 현재까지 생성된 메시지를 저장하는 문자열.
        message_box: Streamlit에서 비어 있는 UI 요소로, 생성된 메시지를 실시간으로 업데이트하는 데 사용.
    """

    message: str = ""

    def on_llm_start(self, *args, **kwargs) -> None:
        """
        LLM 모델이 시작될 때 호출되는 메서드.
        빈 메시지 박스를 생성.

        Args:
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs) -> None:
        """
        LLM 모델이 종료될 때 호출되는 메서드.
        최종 생성된 메시지를 저장.

        Args:
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        """
        새로운 토큰을 수신할 때 호출되는 메서드.
        메시지 박스를 실시간으로 업데이트.

        Args:
            token (str): 새로 생성된 토큰.
            *args: 임의의 인자.
            **kwargs: 임의의 키워드 인자.
        """
        self.message += token
        self.message_box.markdown(normalize_chat_markdown(self.message))


@st.cache_data
def display_pdf(file_data: bytes, scale: float = 1.0, height: int = 700) -> str:
    """
    PDF 파일 데이터를 브라우저에 표시하는 함수.

    주어진 PDF 파일 데이터를 base64로 인코딩하고, 이를 HTML iframe을 통해 PDF로 표시.

    Args:
        file_data (bytes): PDF 파일의 바이트 데이터.
        scale (float, optional): PDF를 브라우저에 표시할 때의 확대/축소 비율. 기본값은 1.0.
        height (int, optional): iframe의 높이 (픽셀 단위). 기본값은 700px.

    Returns:
        pdf_display (str): PDF를 표시하기 위한 HTML iframe 요소가 포함된 문자열
    """
    base64_pdf = base64.b64encode(file_data).decode("utf-8")

    pdf_display = f"""
    <div style="display: flex; justify-content: center;">
        <iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=0"
        width="100%" height="{height}px"
        style="border:none; transform: scale({scale}); transform-origin: top center;"
        type="application/pdf"></iframe>
    </div>
    """
    return pdf_display


def convert_docx_to_html(file_data: bytes) -> str:
    """
    docx 파일 데이터를 HTML로 변환하는 함수.

    주어진 docx 파일의 바이트 데이터를 mammoth 라이브러리를 사용하여 HTML 문자열로 변환.
    변환된 HTML에는 원본 문서의 텍스트 내용과 기본적인 서식이 포함.

    Args:
        file_data (bytes): docx 파일의 바이트 데이터.

    Returns:
        html (str): 변환된 HTML 문자열.
    """
    result = mammoth.convert_to_html(BytesIO(file_data))

    html = result.value

    return html


@st.cache_data
def display_docx(file_data: bytes, scale: float = 1.0, height: int = 700) -> str:
    """
    docx 파일을 스타일이 적용된 HTML iframe으로 변환하여 표시하는 함수.

    HTML로 변환된 docx 파일을, iframe 내에서 표시.

    Args:
        file_data (bytes): html 문자열.
        scale (float, optional): 문서를 브라우저에 표시할 때의 확대/축소 비율. 기본값은 1.0.
        height (int, optional): iframe의 높이 (픽셀 단위). 기본값은 700px.

    Returns:
        iframe_template (str): HTML iframe 템플릿 문자열 또는 오류 메시지.
    """
    try:
        html_content = convert_docx_to_html(file_data)

        styled_html = f"""
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
            .docx-container {{ font-family: 'Noto Sans KR', Arial, sans-serif; line-height: 1.6; font-size: 14px; }}
            .docx-container img {{ max-width: 100%; height: auto; }}
            .docx-container table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
            .docx-container td, .docx-container th {{
                border: 1px solid #ddd;
                padding: 8px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                min-width: 50px;
            }}
        </style>
        <div class="docx-container">
            {html_content}
        </div>
        """

        encoded_html = base64.b64encode(styled_html.encode("utf-8")).decode("utf-8")

        iframe_template = f"""
        <iframe src="data:text/html;base64,{encoded_html}"
        width="100%" height="{height}px"
        style="border:none; transform: scale({scale}); transform-origin: top center;">
        </iframe>
        """

        return iframe_template
    except Exception as e:
        return f"문서를 처리하는 중 오류가 발생했습니다: {str(e)}"


@st.cache_resource(show_spinner="Embedding file...")
def embed_file_with_cache(
    file,
    file_content: bytes,
    file_type: str = "pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "voyage-4-lite",
    max_pages: int = 20,
):
    """
    파일을 임베딩하고 리트리버를 생성하는 함수.

    주어진 파일 데이터를 임베딩하고, 이를 기반으로 검색 가능한 리트리버 객체를 생성.
    PDF 또는 DOCX 파일을 지원하며, 문서를 지정된 크기로 분할한 후, 지정된 임베딩 모델을 사용해 벡터 임베딩을 생성.
    생성된 임베딩은 캐시에 저장되며, 이를 사용하여 FAISS 벡터스토어와 리트리버를 생성.
    Voyage 무료 한도(3 RPM / 10K TPM)에 맞춰 앞 `max_pages` 페이지만 사용하고, 청크를 나눠 쉬어가며 임베딩한다.

    Args:
        file: 업로드된 파일 객체.
        file_content (bytes): 업로드된 파일의 바이트 데이터.
        file_type (str, optional): 파일 유형. "pdf" 또는 "docx"로 지정. 기본값은 "pdf".
        chunk_size (int, optional): 문서를 분할할 때 사용하는 청크의 크기. 기본값은 1000.
        chunk_overlap (int, optional): 문서를 분할할 때 청크 간 중첩되는 문자 수. 기본값은 200.
        embedding_model (str, optional): 사용할 임베딩 모델의 이름. 기본값은 "voyage-4-lite".
        max_pages (int, optional): 임베딩에 사용할 최대 페이지 수. 기본값은 20(무료 한도 대응).

    Returns:
        retriever: 검색 가능한 리트리버 객체.
    """
    file_path = CACHE_FILE_PATH / f"{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(CACHE_EMBEDDING_PATH / f"{file.name}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type in [".docx", ".docc"]:
        loader = Docx2txtLoader(file_path)
    docs = loader.load()[:max_pages]  # 무료 한도 대응: 앞 max_pages 페이지만 사용
    split_docs = splitter.split_documents(docs)
    embeddings = get_embedding(model=embedding_model)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = build_faiss_throttled(split_docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(
    file,
    file_content: bytes,
    file_type: str = "pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "voyage-4-lite",
    max_pages: int = 20,
):
    """
    파일을 임베딩하고 리트리버를 생성하는 함수.

    주어진 파일 데이터를 임베딩하고, 이를 기반으로 검색 가능한 리트리버 객체를 생성.
    PDF 또는 DOCX 파일을 지원하며, 문서를 지정된 크기로 분할한 후, 지정된 임베딩 모델을 사용해 벡터 임베딩을 생성.
    이를 사용하여 FAISS 벡터스토어와 리트리버를 생성.
    Voyage 무료 한도(3 RPM / 10K TPM)에 맞춰 앞 `max_pages` 페이지만 사용하고, 청크를 나눠 쉬어가며 임베딩한다.

    Args:
        file: 업로드된 파일 객체.
        file_content (bytes): 업로드된 파일의 바이트 데이터.
        file_type (str, optional): 파일 유형. "pdf" 또는 "docx"로 지정. 기본값은 "pdf".
        chunk_size (int, optional): 문서를 분할할 때 사용하는 청크의 크기. 기본값은 1000.
        chunk_overlap (int, optional): 문서를 분할할 때 청크 간 중첩되는 문자 수. 기본값은 200.
        embedding_model (str, optional): 사용할 임베딩 모델의 이름. 기본값은 "voyage-4-lite".
        max_pages (int, optional): 임베딩에 사용할 최대 페이지 수. 기본값은 20(무료 한도 대응).

    Returns:
        retriever: 검색 가능한 리트리버 객체.
    """
    os.makedirs(CACHE_FILE_PATH, exist_ok=True)
    file_path = CACHE_FILE_PATH / f"{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, type="recursive")

    if file_type == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type in [".docx", ".docc"]:
        loader = Docx2txtLoader(file_path)

    docs = loader.load()[:max_pages]  # 무료 한도 대응: 앞 max_pages 페이지만 사용
    split_docs = splitter.split_documents(docs)
    embeddings = get_embedding(model=embedding_model)
    vectorstore = build_faiss_throttled(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    if os.path.exists(file_path):
        os.remove(file_path)

    return retriever


@st.cache_resource(show_spinner="Loading vectorstore...")
def load_retriver(
    db_path: str,
    embedding_model="voyage-4-lite",
    retriever_k: int = 4,
):
    """
    로컬에서 저장된 벡터스토어를 불러와 리트리버를 생성하는 함수.

    주어진 경로에 있는 벡터스토어를 로드하고, 지정된 임베딩 모델을 사용해 검색 가능한 리트리버 객체를 생성.
    리트리버는 유사성 검색 방식을 사용하며, 검색 결과로 반환되는 문서의 수는 `retriever_k` 값으로 설정.

    Args:
        db_path (str): 로컬 벡터스토어가 저장된 경로.
        embedding_model (str, optional): 사용할 임베딩 모델의 이름. 기본값은 "voyage-4-lite".
        retriever_k (int, optional): 검색 시 반환할 문서의 최대 개수. 기본값은 4.

    Returns:
        retriever: 검색 가능한 리트리버 객체.
    """
    embeddings = get_embedding(model=embedding_model)
    vectorstore = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": retriever_k})
    return retriever


@st.dialog("로그 삭제")
def delete_log(selected_log_path: str) -> None:
    """
    선택한 로그를 삭제하는 함수.

    사용자는 주어진 경로에 있는 로그 삭제를 확인.
    삭제 버튼을 클릭하면 지정된 경로의 로그가 삭제.
    삭제 과정에서 오류가 발생할 경우, 오류 메시지가 표시.

    Args:
        selected_log_path (str): 삭제할 로그가 위치한 파일 경로.
    """
    st.write(f"`{selected_log_path.as_posix().split('/')[-1]}`")
    st.write("위 경로의 모델을 삭제합니까?")
    if st.button("삭제"):
        try:
            shutil.rmtree(selected_log_path)
            st.success(f"모델 '{selected_log_path.as_posix().split('/')[-1]}'이 삭제되었습니다.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"모델 삭제 중 오류가 발생했습니다: {str(e)}")
        st.rerun()


def save_message(message: str, role: str) -> None:
    """
    메시지를 세션 상태에 저장하는 함수.

    Args:
        message (str): 저장할 메시지 내용.
        role (str): 메시지 작성자의 역할 ("ai" 또는 "human").
    """
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message: str, role: str, save: bool = True) -> None:
    """
    채팅 인터페이스에 메시지를 표시하고 선택적으로 저장하는 함수.

    메시지를 채팅 UI에 표시하고, save 매개변수가 True인 경우 세션 상태에도 저장.
    각 메시지는 역할에 따라 다른 아바타 이미지와 함께 표시.

    Args:
        message (str): 표시할 메시지 내용.
        role (str): 메시지 작성자의 역할 ("ai" 또는 "human").
        save (bool, optional): 메시지를 세션 상태에 저장할지 여부. 기본값은 True.
    """
    avatar_image = load_image(ai_avartar if role == "ai" else human_avatar)

    with st.chat_message(role, avatar=avatar_image):
        st.markdown(normalize_chat_markdown(message))

    if save:
        save_message(message, role)


def pain_history() -> None:
    """
    세션 상태에 저장된 모든 대화 내역을 화면에 표시하는 함수.

    세션 상태의 "messages" 리스트에서 각 메시지를 가져와
    채팅 인터페이스에 순서대로 표시.
    """
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


@st.cache_data(show_spinner=False)
def read_file_data(file: Any) -> pd.DataFrame:
    """
    파일 데이터를 읽어 DataFrame으로 변환하는 함수.

    Args:
        file (Any): 업로드된 파일 객체. 보통 Streamlit의 file_uploader로 받은 파일.

    Returns:
        pd.DataFrame: CSV 데이터를 담은 pandas DataFrame.
    """
    df = pd.read_csv(file)
    return df


def check_korean(text):
    """
    RAG, Agent 설정 이름에 한글 포함 여부를 체크하는 함수.

    Args:
        text (str): 입력한 설정 이름.

    Returns:
        bool: 입력한 이름에 한글이 포함되면 True를 반환하고 한글이 포함되어있지 않으면 False를 반환

    """
    p = re.compile("[ㄱ-힣]")
    r = p.search(text)
    if r is None:
        return False
    else:
        return True
