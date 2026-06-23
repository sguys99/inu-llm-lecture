import os

import streamlit as st
from dotenv import load_dotenv
from function_utils import (
    ChatCallbackHandler,
    display_docx,
    display_pdf,
    load_retriver,
    pain_history,
    send_message,
)
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import load_prompt

from app.chains import build_simple_chain
from app.modules.llms import get_llm
from app.utils.config_loader import dump_yaml, load_yaml
from app.utils.path import DEMO_IMG_PATH, LOG_PATH
from app.utils.rag_utils import delete_incomplete_logs, format_docs_with_source

load_dotenv()

RAG_LOG_PATH = LOG_PATH / "rag"
os.makedirs(RAG_LOG_PATH, exist_ok=True)
delete_incomplete_logs(base_path=RAG_LOG_PATH, required_files=["prompt.yaml", "rag_config.yaml"])

human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avartar = DEMO_IMG_PATH / "inu-logo.png"
ai_avatar_image = str(ai_avartar)

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(
        return_messages=True,
        k=2,
        memory_key="chat_history",
    )

memory = st.session_state["memory"]


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


if "rag_qa_on" not in st.session_state:
    st.session_state["rag_qa_on"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "llm_temp" not in st.session_state:
    st.session_state["llm_temp"] = 0.2
if "retriever_k" not in st.session_state:
    st.session_state["retriever_k"] = 3
if "selected_rag_llm" not in st.session_state:
    st.session_state["selected_rag_llm"] = "claude-haiku-4-5"

with st.sidebar:
    st.write("")

    logs = [f for f in os.listdir(RAG_LOG_PATH) if os.path.isdir(os.path.join(RAG_LOG_PATH, f))]
    if not logs:
        st.error("저장된 항목이 없습니다.")
        st.session_state["build_bot"] = False
        selected_db = None
        selected_log = None
        selected_prompt = None
        selected_rag_config = None
    else:
        selected_log = st.selectbox("RAG 저장 목록", logs)
        selected_log_path = RAG_LOG_PATH / selected_log

        with st.expander(label="✔️ **Settings**", expanded=False):
            llm_temperature = st.slider(
                "LLM Temperature",
                min_value=0.1,
                max_value=1.0,
                value=0.2,
                step=0.1,
                disabled=st.session_state["rag_qa_on"],
            )
            retriever_k = st.slider(
                "Retriever K value",
                min_value=2,
                max_value=8,
                value=3,
                step=1,
                disabled=st.session_state["rag_qa_on"],
            )
            llm_type = st.radio(
                "LLM",
                [
                    "claude-haiku-4-5",
                ],
                horizontal=True,
                disabled=st.session_state["rag_qa_on"],
                help="사용할 LLM을 선택하세요.",
            )

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
            reset_btn = st.button("Reset history", disabled=not logs)
            build_bot_btn = st.toggle("Build bot", key="build_bot", disabled=not logs)

        if build_bot_btn:
            selected_doc_path = selected_log_path / "docs"
            selected_db_path = selected_log_path / "db"
            selected_prompt_path = selected_log_path / "prompt.yaml"
            prompt = load_prompt(selected_log_path / "prompt.yaml", encoding="utf-8")
            st.session_state["llm_temp"] = llm_temperature
            st.session_state["retriever_k"] = retriever_k
            st.session_state["selected_rag_llm"] = llm_type
            st.session_state["rag_qa_on"] = True
            st.success("Setting complete!")
        else:
            st.session_state["rag_qa_on"] = False

        if reset_btn and st.session_state["rag_qa_on"]:
            st.session_state["messages"] = []
            st.session_state["memory"].clear()


st.title("RAG Applications")
st.caption("RAG 설정을 호출하여 문서에 대한 내용을 답변합니다.")
st.write("")

col11, col12 = st.columns([1, 1])

with col11:
    st.write("**:blue[Documents]**")
    with st.container(height=670):
        if st.session_state["rag_qa_on"] and selected_rag_config:
            num_tabs = (
                min(len(selected_rag_config["documents"]), 10) if selected_rag_config["documents"] else 0
            )
            if num_tabs > 0:
                tabs = st.tabs([f"DOC #{i+1}" for i in range(num_tabs)])

                for i, tab in enumerate(tabs):
                    with tab:
                        file_name = selected_rag_config["documents"][i]
                        file_path = selected_doc_path / file_name
                        st.write(f"`{file_name}`")
                        with open(file_path, "rb") as file:
                            file_data = file.read()

                        if selected_rag_config["document_format"] == "pdf":
                            pdf_display = display_pdf(file_data, scale=1.0, height=500)
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        elif selected_rag_config["document_format"] in ["docx", "docc"]:
                            docx_display = display_docx(file_data, scale=1.0, height=500)
                            st.markdown(docx_display, unsafe_allow_html=True)
                        else:
                            st.error("지원되지 않는 파일 형식입니다.")
        else:
            st.write("`Build bot`을 toggle하면 문서가 표시됩니다.")

with col12:
    st.write(
        f"**:blue[RAG name :]** `{selected_log}` , **:blue[model:]** `{st.session_state['selected_rag_llm']}`, **:blue[LLM Temp. :]** `{st.session_state['llm_temp']}`, **:blue[Retriever K :]** `{st.session_state['retriever_k']}`",
    )
    with st.container(height=670):
        if st.session_state["rag_qa_on"] and selected_rag_config:
            llm = get_llm(
                model=st.session_state["selected_rag_llm"],
                temperature=st.session_state["llm_temp"],
                streaming=True,
                callbacks=[ChatCallbackHandler()],
            )

            retriever = load_retriver(
                db_path=selected_db_path,
                embedding_model=selected_rag_config["embedding"],
                retriever_k=st.session_state["retriever_k"],
            )

            send_message("준비되었습니다. 질문하세요!", "ai", save=False)
            pain_history()

            with st.bottom:
                _, col122 = st.columns([1, 1])
                with col122:
                    message = st.chat_input("문서 내용에 대해서 물어보세요...")

            if message:
                send_message(message, "human")
                chain = build_simple_chain(
                    retriever=retriever,
                    prompt=prompt,
                    llm=llm,
                    load_memory_func=load_memory,
                    format_docs_func=format_docs_with_source,
                )

                with st.chat_message("ai", avatar=ai_avatar_image):
                    content = chain.invoke(message).content
                    memory.save_context({"input": message}, {"output": content})
                    st.session_state["memory"] = memory
        else:
            st.session_state["messages"] = []
            st.session_state["rag_qa_on"] = False
            memory.clear()
