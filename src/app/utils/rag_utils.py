import os
import re
import shutil
from typing import List, Optional

import pandas as pd
import yaml
from langchain_core.documents import Document


def parse_names(file_list: List[str]) -> List[str]:
    """
    파일 이름 목록에서 한글 단어만 추출하고 전처리하는 함수.

    이 함수는 주어진 파일 이름 목록에서 한글 단어만 추출하고, 불필요한 단어를 제거하며,
    여러 공백을 하나의 공백으로 줄여 정리된 문자열 목록을 반환.

    Args:
        file_list (List[str]): 파일 이름 문자열로 구성된 리스트.

    Returns:
        List[str]: 파일 이름에서 추출한 한글 단어로 구성된 문자열 리스트.
                   지정된 단어가 제거되고 불필요한 공백이 정리된 형태로 반환.
    """
    parsed_names = []
    for file_name in file_list:
        korean_words = re.findall(r"[가-힣]+", file_name)
        parsed_text = " ".join(korean_words)

        for word in ["개정후", "개정", "변경", "수정"]:
            parsed_text = parsed_text.replace(word, "")

        parsed_text = " ".join(parsed_text.split())

        parsed_names.append(parsed_text)

    return parsed_names


def format_docs(docs: List[Document]) -> str:
    """
    문서 목록에서 각 문서의 페이지 콘텐츠를 연결하여 하나의 문자열로 반환하는 함수.

    Args:
        docs (List[Document]): Langchain Document 인스턴스로 구성된 리스트.

    Returns:
        str: 각 문서의 페이지 콘텐츠가 두 줄 간격으로 연결된 하나의 문자열.
    """
    return "\n\n".join(document.page_content for document in docs)


def format_docs_with_source(docs: List[Document]) -> str:
    """
    문서 목록에서 각 문서의 출처와 페이지 콘텐츠를 연결하여 하나의 문자열로 반환하는 함수.

    각 문서의 출처 정보는 metadata에서 가져오며, 출처 정보가 없을 경우 "알 수 없음"으로 표시함.
    출처 정보와 페이지 콘텐츠는 지정된 형식으로 연결하여 반환.

    Args:
        docs (List[Document]): Langchain Document 인스턴스로 구성된 리스트.

    Returns:
        str: 각 문서의 출처와 페이지 콘텐츠가 두 줄 간격으로 연결된 하나의 문자열.
    """
    processed_docs = []

    for document in docs:
        source = document.metadata.get("source", "")
        processed_source = parse_names([source])[0] if source else "알 수 없음"
        formatted_content = f"[출처: {processed_source}]\n\n{document.page_content}"
        processed_docs.append(formatted_content)

    return "\n\n".join(processed_docs)


def save_rag_configs(
    save_path: str,
    document_format: str,
    documents: List[str],
    text_splitter_type: str,
    chunk_size: int,
    chunk_overlap: int,
    loader_type: str = "directory",
    vectorstore_type: str = "FAISS",
    embedding_type: str = "text-embedding-3-large",
) -> None:
    """
    RAG(Retrieval-Augmented Generation) 설정을 YAML 파일로 저장.

    Args:
        save_path (str): 설정 파일을 저장할 경로.
        document_format (str): 문서 형식 (예: PDF, DOCX).
        documents (List[str]): 처리할 문서 목록.
        text_splitter_type (str): 사용할 텍스트 분할기 유형.
        chunk_size (int): 분할할 텍스트 청크 크기.
        chunk_overlap (int): 청크 간 중첩되는 문자 수.
        loader_type (str, optional): 문서 로더 유형 (기본값: "directory").
        vectorstore_type (str, optional): 벡터 스토어 유형 (기본값: "FAISS").
        embedding_type (str, optional): 임베딩 모델 유형 (기본값: "text-embedding-3-large").

    Returns:
        None
    """
    config = {
        "document_format": document_format,
        "documents": documents,
        "loader": {"type": loader_type},
        "text_splitter": {
            "type": text_splitter_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "vectorstore": {"type": vectorstore_type},
        "embedding": embedding_type,
    }

    with open(save_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def save_da_agent_configs(
    save_path: str,
    documents: List[str],
):
    config = {
        "documents": documents,
    }

    with open(save_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def process_bulk_data(
    src_path,
    dst_path=None,
):
    df_bulk = pd.read_csv(src_path)
    df_bulk = df_bulk.loc[:, ~df_bulk.columns.str.startswith("Unnamed")]
    column_order = [
        "DATETIME",
        "DELI_PL_CD",
        "DELI_PL_NM",
        "DELI_PL_TYPE",
        "ZIP_ADDR",
        "SALES",
        "LONGITUDE",
        "LATITUDE",
    ]
    df_bulk_reshape = df_bulk.melt(
        ["DELI_PL_CD", "DELI_PL_NM", "ZIP_ADDR", "배송처코드", "주소", "X", "Y"],
        var_name="DATETIME",
        value_name="SALES",
    )
    df_bulk_processed = df_bulk_reshape.drop(["배송처코드", "주소"], axis=1).rename(
        columns={"X": "LONGITUDE", "Y": "LATITUDE"},
    )
    df_bulk_processed["DELI_PL_TYPE"] = df_bulk_processed["DELI_PL_NM"].str.extract(r"\[(.*?)\]")
    df_bulk_processed["DELI_PL_NM"] = df_bulk_processed["DELI_PL_NM"].str.replace(
        r"\[.*?\]\s*",
        "",
        regex=True,
    )
    df_bulk_processed = df_bulk_processed[column_order]
    if dst_path:
        df_bulk_processed.to_csv(dst_path, index=False, encoding="utf-8-sig")
    return df_bulk_processed


def process_pet_data(
    src_path,
    dst_path=None,
):
    df_pet = pd.read_csv(src_path)
    df_pet = df_pet.loc[:, ~df_pet.columns.str.startswith("Unnamed")]
    column_order = [
        "DELI_PL_CD",
        "DELI_PL_NM",
        "DELI_PL_TYPE",
        "ZIP_ADDR",
        "DATETIME",
        "SALES",
        "LONGITUDE",
        "LATITUDE",
    ]
    df_pet_reshape = df_pet.melt(
        ["DELI_PL_CD", "DELI_PL_NM", "ZIP_ADDR", "배송처코드", "주소", "X", "Y"],
        var_name="DATETIME",
        value_name="SALES",
    )
    df_pet_processed = df_pet_reshape.drop(["배송처코드", "주소"], axis=1).rename(
        columns={"X": "LONGITUDE", "Y": "LATITUDE"},
    )
    df_pet_processed["DELI_PL_TYPE"] = df_pet_processed["DELI_PL_NM"].str.extract(r"\[(.*?)\]")
    df_pet_processed["DELI_PL_NM"] = df_pet_processed["DELI_PL_NM"].str.replace(
        r"\[.*?\]\s*",
        "",
        regex=True,
    )
    df_pet_processed = df_pet_processed[column_order]
    if dst_path:
        df_pet_processed.to_csv(dst_path, index=False, encoding="utf-8-sig")
    return df_pet_processed


def process_traffic_data(
    src_path,
    dst_path=None,
):
    df_traffic = pd.read_csv(src_path)
    df_traffic_processed = df_traffic.drop(["idx"], axis=1).rename(
        columns={"도로명": "ROAD_NM", "교통량": "TRAFFIC", "X": "LONGITUDE", "Y": "LATITUDE"},
    )
    if dst_path:
        df_traffic_processed.to_csv(dst_path, index=False, encoding="utf-8-sig")
    return df_traffic_processed


def process_station_data(
    src_path,
    dst_path=None,
):
    df_station = pd.read_csv(src_path)
    df_station = df_station.loc[:, ~df_station.columns.str.startswith("Unnamed")]
    df_station_processed = df_station.rename(
        columns={
            "고유번호": "CD",
            "지역": "LOC",
            "상호": "COMPANY_NM",
            "상표": "TM",
            "주소": "ADDR",
            "전화번호": "PHONE_NM",
            "셀프구분": "SELF",
            "X": "LONGITUDE",
            "Y": "LATITUDE",
        },
    )
    if dst_path:
        df_station_processed.to_csv(dst_path, index=False, encoding="utf-8-sig")
    return df_station_processed


def langsmith_logging(project_name: Optional[str] = None, set_enable: bool = True) -> None:
    """
    LangSmith 로그 추적을 설정하는 함수.

    LangChain API 키가 환경 변수로 설정된 경우, LangSmith 추적 기능을 활성화하거나 비활성화.
    추적이 활성화되면 LangSmith API 엔드포인트, 추적 설정, 프로젝트 이름을 환경 변수에 지정하여 로그 추적.

    Args:
        project_name (Optional[str]): 추적할 프로젝트의 이름. 기본값은 None으로 설정.
        set_enable (bool): 추적을 활성화하려면 True, 비활성화하려면 False를 설정. 기본값은 True.

    Returns:
        None
    """
    if set_enable:
        result = os.environ.get("LANGCHAIN_API_KEY")
        if result is None or result.strip() == "":
            print("LangChain API Key가 설정되지 않았습니다.")
            return
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        print(f"LangSmith 추적을 시작합니다.\n[프로젝트명]\n{project_name}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("LangSmith 추적을 하지 않습니다.")


def check_incomplete_logs(base_path: str, required_files: List[str]) -> List[str]:
    """
    지정된 경로(base_path) 내에서 필수 파일(required_files)이 모두 존재하지 않는 폴더들을 찾아 반환하는 함수.

    Args:
        base_path (str): 검색할 기본 경로.
        required_files (List[str]): 각 폴더 내에 존재해야 하는 파일의 이름 목록.

    Returns:
        List[str]: 모든 필수 파일이 없는 폴더명들의 리스트.
    """

    return [
        f
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
        and not all(os.path.exists(os.path.join(base_path, f, file)) for file in required_files)
    ]


def delete_incomplete_logs(base_path: str, required_files: List[str]) -> None:
    """
    지정된 경로(base_path) 내에서 필수 파일(required_files)이 모두 존재하지 않는 폴더들을 찾아 삭제하는 함수.

    Args:
        base_path (str): 검색할 기본 경로.
        required_files (List[str]): 각 폴더 내에 존재해야 하는 파일의 이름 목록.

    Returns:
        None
    """
    incomplete_dirs = check_incomplete_logs(base_path, required_files)

    for dir_name in incomplete_dirs:
        dir_path = os.path.join(base_path, dir_name)
        shutil.rmtree(dir_path)
        print(f"Deleted incomplete directory: {dir_path}")
