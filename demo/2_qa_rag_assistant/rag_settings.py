import gc
import os
import time
from datetime import datetime

import streamlit as st
import yaml
from dotenv import load_dotenv
from function_utils import check_korean, delete_log, display_docx, display_pdf
from langchain_core.prompts import load_prompt

from app.modules.documents import get_docx_loader, get_pdf_loader
from app.modules.llms import get_embedding
from app.modules.prompts import build_qa_prompt, save_fewshot_prompt, save_prompt
from app.modules.vector_db import build_faiss_throttled, get_splitter
from app.utils.config_loader import dump_yaml, load_yaml
from app.utils.path import LOG_PATH, PROMPT_CONFIG_PATH
from app.utils.rag_utils import delete_incomplete_logs, save_rag_configs

load_dotenv()

RAG_LOG_PATH = LOG_PATH / "rag"
os.makedirs(RAG_LOG_PATH, exist_ok=True)

delete_incomplete_logs(base_path=RAG_LOG_PATH, required_files=["prompt.yaml", "rag_config.yaml"])

if "rag_name" not in st.session_state:
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    preposition = "QA_RAG_"
    st.session_state["rag_name"] = preposition + current_datetime

if "prompt" not in st.session_state:
    st.session_state["prompt"] = None
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = None
if "doc_format" not in st.session_state:
    st.session_state["doc_format"] = "pdf"
if "selected_system_prompt" not in st.session_state:
    st.session_state["selected_system_prompt"] = "simple_qa_prompt_kor"
if "selected_example" not in st.session_state:
    st.session_state["selected_example"] = "example_template"

with st.sidebar:
    st.write("")
    uploaded_files = st.file_uploader(
        "PDF, DOCX 파일을 업로드하세요.",
        type=["pdf", "docx", "docc"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        extensions = set(os.path.splitext(file.name)[1].lower().lstrip(".") for file in uploaded_files)

        if len(extensions) == 1:
            st.session_state["uploaded_files"] = uploaded_files
            st.session_state["doc_format"] = next(iter(extensions))
        else:
            st.error("한 종류의 파일들만 업로드 해주세요.")
            st.session_state["uploaded_files"] = None
    else:
        st.session_state["uploaded_files"] = None

    logs = [f for f in os.listdir(RAG_LOG_PATH) if os.path.isdir(os.path.join(RAG_LOG_PATH, f))]

    if not logs:
        st.error("저장된 항목이 없습니다.")
        selected_db = None
    else:
        selected_log = st.selectbox("RAG 저장 목록", logs)
        selected_log_path = RAG_LOG_PATH / selected_log

        col1, col2, _ = st.columns((3, 3, 0.5))

        with col1:
            with st.popover("RAG 조회", use_container_width=False):
                selected_prompt = load_yaml(selected_log_path / "prompt.yaml")
                selected_rag_config = load_yaml(selected_log_path / "rag_config.yaml")

                tab1, tab2 = st.tabs(
                    ["prompt", "RAG 설정"],
                )

                with tab1:
                    st.code(dump_yaml(selected_prompt), language="yaml")

                with tab2:
                    st.code(dump_yaml(selected_rag_config), language="yaml")

        with col2:
            delete_log_btn = st.button(":warning: :red[RAG 삭제]")

        if delete_log_btn:
            if selected_log:
                delete_log(selected_log_path)
            else:
                st.warning("삭제할 RAG 설정을 선택해주세요.")


st.title(":gear: RAG Settings")
st.caption("RAG Chain을 위한 구성요소를 설정합니다.")


st.write(" ")

col11, col12, col13 = st.columns([1, 1, 1])

with col11:
    st.write("**:blue[1.Documents]**")
    with st.container(height=700):
        num_tabs = (
            min(len(st.session_state["uploaded_files"]), 5) if st.session_state["uploaded_files"] else 0
        )

        if num_tabs > 0:
            tabs = st.tabs([f"DOC #{i+1}" for i in range(num_tabs)])

            for i, tab in enumerate(tabs):
                with tab:
                    if i < len(st.session_state["uploaded_files"]):
                        file = st.session_state["uploaded_files"][i]
                        st.write(f"`{file.name}`")

                        file_extension = os.path.splitext(file.name)[1].lower()
                        file_data = file.read()

                        if file_extension == ".pdf":
                            pdf_display = display_pdf(file_data, scale=1.0, height=500)
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        elif file_extension in [".docx", ".docc"]:
                            docx_display = display_docx(file_data, scale=1.0, height=500)
                            st.markdown(docx_display, unsafe_allow_html=True)
                        else:
                            st.error("지원되지 않는 파일 형식입니다.")

                    else:
                        st.write("문서가 업로드되지 않았습니다.")
        else:
            st.write("문서를 업로드하면 여기에 표시됩니다.")

with col12:
    st.write("**:blue[2.Prompt settings]**")
    with st.container(height=700):
        tab11, tab12 = st.tabs(["System Prompt", "Example Prompt"])

        with tab11:
            col121, col122 = st.columns([1, 1])
            with col121:
                st.session_state["selected_system_prompt"] = st.selectbox(
                    label="Prompt template 선택",
                    options=[
                        "simple_qa_prompt_kor",
                        "simple_qa_prompt",
                        "hr_qa_prompt_kor",
                        "hr_qa_prompt",
                    ],
                    index=[
                        "simple_qa_prompt_kor",
                        "simple_qa_prompt",
                        "hr_qa_prompt_kor",
                        "hr_qa_prompt",
                    ].index(st.session_state["selected_system_prompt"]),
                    help="사용할 프롬프트를 선택하세요.",
                )

            prompt_template = load_prompt(
                PROMPT_CONFIG_PATH / f"{st.session_state['selected_system_prompt']}.yaml",
                encoding="utf-8",
            )
            system_prompt = prompt_template.template
            if "Context:" in system_prompt:
                system_prompt = system_prompt.split("Context:")[0].strip()

            st.write("")
            edited_message = st.text_area(
                "Prompt",
                system_prompt,
                height=380,
                label_visibility="collapsed",
            )

        with tab12:
            col121, col122 = st.columns([1, 1])
            with col121:
                st.write("")
                use_example_check = st.checkbox("Example 사용", value=False)
            with col122:
                st.session_state["selected_example"] = st.selectbox(
                    label="Example template 선택",
                    options=["example_template", "hr_example_template"],
                    index=["example_template", "hr_example_template"].index(
                        st.session_state["selected_example"],
                    ),
                    disabled=not use_example_check,
                    help="사용할 프롬프트를 선택하세요.",
                )

            example_template = load_yaml(
                PROMPT_CONFIG_PATH / f"{st.session_state['selected_example']}.yaml",
            )
            example_content = example_template.get("answer_examples", "")

            st.write("")
            few_shot_msg = st.text_area(
                "Fewshot prompt",
                dump_yaml(example_content),
                height=420,
                disabled=not use_example_check,
                label_visibility="collapsed",
            )

        with st.form(key="prompt_setting", border=False):
            save_prompt_button = st.form_submit_button(label="Save prompt")
        if save_prompt_button:
            st.session_state["use_example"] = use_example_check
            st.session_state["prompt"] = build_qa_prompt(
                system_message=edited_message,
                examples=yaml.safe_load(few_shot_msg) if st.session_state["use_example"] else None,
            )
            st.success("프롬프트 생성완료!")


with col13:
    st.write("**:blue[3.Vector DB settings]**")
    with st.container(height=700):
        if st.session_state["uploaded_files"]:
            doc_names = [doc.name for doc in st.session_state["uploaded_files"]]
            doc_list = "\n".join(["- " + name for name in doc_names])
            st.text_area(
                "doc_list",
                value=doc_list,
                height=100,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.info("문서를 업로드 하세요.")

        st.write(" ")
        rag_name_input = st.text_input(
            "RAG 이름",
            value=st.session_state["rag_name"],
            help="저장할 RAG 이름을 지정하세요.",
        )

        st.write(" ")
        c131, _, c132 = st.columns([1, 0.05, 1])

        with c131:
            pdf_loader_type = st.selectbox(
                "PDF Loader",
                ["pymupdf", "pypdf", "pdfplumber", "pdfminer"],
                help="사용할 PDF Loader를 선택하세요.",
                disabled=(st.session_state["doc_format"] != "pdf"),
            )
            chunk_size = st.slider("Chunk size", min_value=1000, max_value=2000, value=500, step=100)

        with c132:
            text_splitter_type = st.selectbox(
                "Text Splitter",
                ["RecursiveCT", "CharacterText", "TokenText"],
                help="Text Splitter를 선택하세요.",
            )
            chunk_overlap = st.slider(
                "Chunk overlap",
                min_value=100,
                max_value=200,
                value=50,
                step=10,
            )

        st.write(" ")
        st.write(" ")

        if st.button(label="Save settings"):
            if check_korean(rag_name_input):
                st.warning("영문 이름을 사용하세요.")

            elif "uploaded_files" not in st.session_state or not st.session_state["uploaded_files"]:
                st.warning("문서를 업로드해주세요")

            elif "prompt" not in st.session_state or not st.session_state["prompt"]:
                st.warning("prompt를 생성해주세요.")

            else:
                status = st.status("Saving RAG settings...", expanded=True)
                time.sleep(0.5)
                status.write("**`Preparing the log path`**")
                dir_path = RAG_LOG_PATH / rag_name_input
                prompt_path = dir_path / "prompt.yaml"
                doc_path = dir_path / "docs"
                db_path = dir_path / "db"
                os.makedirs(dir_path, exist_ok=True)
                os.makedirs(doc_path, exist_ok=True)
                time.sleep(0.5)

                status.write("**`Loading the documents...`**")
                for file in st.session_state["uploaded_files"]:
                    file_path = doc_path / f"{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())

                splitter = get_splitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    type=text_splitter_type,
                )
                if st.session_state["doc_format"] == "pdf":
                    loader = get_pdf_loader(file_path=doc_path, type="directory")
                elif st.session_state["doc_format"] in ["docx", "docc"]:
                    loader = get_docx_loader(file_path=doc_path, type="directory")
                else:
                    raise ValueError(f"지원되지 않는 문서 포맷: {st.session_state['doc_format']}")
                docs = loader.load()[:20]  # Voyage 무료 한도 대응: 앞 20페이지만 사용

                status.write("**`Setting the Vector Store....`** (무료 한도로 약 4~5분 소요)")
                splits = splitter.split_documents(docs)
                embedding = get_embedding(
                    model=os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite"),
                )
                vectorstore = build_faiss_throttled(
                    documents=splits,
                    embedding=embedding,
                    progress=lambda done, total: status.write(f"임베딩 진행: {done}/{total}"),
                )
                vectorstore.save_local(folder_path=db_path)

                status.write("**`Saving the configs....`**")

                if st.session_state["use_example"]:
                    save_fewshot_prompt(prompt=st.session_state["prompt"], save_path=prompt_path)
                else:
                    save_prompt(prompt=st.session_state["prompt"], save_path=prompt_path)
                save_rag_configs(
                    save_path=dir_path / "rag_config.yaml",
                    document_format=st.session_state["doc_format"],
                    documents=doc_names,
                    text_splitter_type=text_splitter_type,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    loader_type="directory",
                    vectorstore_type="FAISS",
                    embedding_type=os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite"),
                )
                status.update(
                    label="**:blue[RAG 설정 저장 완료.]**",
                    state="complete",
                    expanded=False,
                )

                time.sleep(2)
                keys_to_clear = ["prompt", "uploaded_files", "rag_name"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                gc.collect()
                st.rerun()
