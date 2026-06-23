import os

import streamlit as st
from dotenv import load_dotenv
from function_utils import (
    ChatCallbackHandler,
    display_docx,
    display_pdf,
    embed_file,
    pain_history,
    send_message,
)
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import load_prompt

from app.chains import build_simple_chain
from app.modules.llms import get_llm
from app.utils.path import DEMO_IMG_PATH, PROMPT_CONFIG_PATH
from app.utils.rag_utils import format_docs

load_dotenv()


human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avartar = DEMO_IMG_PATH / "inu-logo.png"
ai_avatar_image = str(ai_avartar)

prompt = load_prompt(PROMPT_CONFIG_PATH / "simple_qa_prompt.yaml", encoding="utf-8")

if "simple_qa_on" not in st.session_state:
    st.session_state["simple_qa_on"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []


if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(
        return_messages=True,
        k=2,
        memory_key="chat_history",
    )


memory = st.session_state["memory"]


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


st.title("Simple QA assistant")
st.caption("업로드한 문서에 대한 내용을 답변합니다.")
st.write("")

with st.sidebar:
    st.write("")
    file = st.file_uploader("PDF, DOCX 파일을 업로드하세요.", type=["pdf", "docx", "docc"])

    if file is None:
        st.session_state["build_bot"] = False

    with st.expander(label="✔️ **Settings**", expanded=False):
        llm_temperature = st.slider(
            "LLM Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.1,
            disabled=st.session_state["simple_qa_on"],
        )
        chunk_size = st.slider(
            "Chunk size",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            disabled=st.session_state["simple_qa_on"],
        )
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
            disabled=st.session_state["simple_qa_on"],
        )
        llm_type = st.radio(
            "LLM",
            [
                "claude-haiku-4-5",
            ],
            horizontal=True,
            disabled=st.session_state["simple_qa_on"],
            help="사용할 LLM을 선택하세요.",
        )

    col01, _, col02 = st.columns([1, 0.2, 1])
    with col01:
        build_bot_btn = st.toggle("Build bot", key="build_bot", disabled=file is None)
    with col02:
        reset_btn = st.button("Reset history")

    if build_bot_btn and file is not None:
        st.session_state["llm_temp"] = llm_temperature
        st.session_state["chunk_size"] = chunk_size
        st.session_state["chunk_overlap"] = chunk_overlap
        st.session_state["selected_simple_qa_llm"] = llm_type
        st.session_state["simple_qa_on"] = True
        st.success("Setting complete!")
    else:
        st.session_state["simple_qa_on"] = False
        if build_bot_btn:
            st.error("파일을 업로드 해주세요.")

    if reset_btn and st.session_state["simple_qa_on"] and file is not None:
        st.session_state["messages"] = []
        st.session_state["memory"].clear()

if file:
    col11, col12 = st.columns([1, 1])

    with col11:
        st.write("")
        col111, _ = st.columns([0.9, 0.1])
        with col111:
            file_data = file.read()
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension == ".pdf":
                pdf_display = display_pdf(file_data)
                st.markdown(pdf_display, unsafe_allow_html=True)
            elif file_extension in [".docx", ".docc"]:
                docx_display = display_docx(file_data)
                st.markdown(docx_display, unsafe_allow_html=True)
            else:
                st.error("지원되지 않는 파일 형식입니다.")

    with col12:
        if st.session_state["simple_qa_on"]:
            st.write(
                f"model: `{st.session_state['selected_simple_qa_llm']}` , temp.: `{st.session_state['llm_temp']}` , chunk size: `{st.session_state['chunk_size']}` , chunk overlap: `{st.session_state['chunk_overlap']}`",
            )
            with st.container(border=True, height=670):
                llm = get_llm(
                    model=st.session_state["selected_simple_qa_llm"],
                    temperature=llm_temperature,
                    streaming=True,
                    callbacks=[ChatCallbackHandler()],
                )
                retriever = embed_file(
                    file,
                    file_data,
                    file_extension,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-lite"),
                )

                send_message("준비되었습니다. 질문하세요!", "ai", save=False)
                pain_history()

                with st.bottom:
                    _, col122 = st.columns([1, 1])
                    with col122:
                        message = st.chat_input("업로드한 파일에 대해서 물어보세요...")

                if message:
                    send_message(message, "human")
                    chain = build_simple_chain(
                        retriever=retriever,
                        prompt=prompt,
                        llm=llm,
                        load_memory_func=load_memory,
                        format_docs_func=format_docs,
                    )

                    with st.chat_message("ai", avatar=ai_avatar_image):
                        content = chain.invoke(message).content
                        memory.save_context({"input": message}, {"output": content})
                        st.session_state["memory"] = memory


else:
    st.session_state["messages"] = []
    st.session_state["simple_qa_on"] = False
    memory.clear()
