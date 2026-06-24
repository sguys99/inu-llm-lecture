import os
import sys

import streamlit as st
from dotenv import load_dotenv
from function_utils import (
    display_docx,
    display_pdf,
    load_retriver,
    normalize_chat_markdown,
    pain_history,
    save_message,
    send_message,
)
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.utils.config_loader import dump_yaml, load_yaml
from app.utils.path import DEMO_IMG_PATH, LOG_PATH
from app.utils.rag_utils import delete_incomplete_logs

# agent_utils 는 이 페이지 파일과 같은 폴더(demo/3_agentic_rag/)에 있다.
# Streamlit 멀티페이지는 메인 스크립트(demo/) 경로만 sys.path에 올리므로,
# 사이드(형제) 모듈을 import 하려면 현재 파일 디렉터리를 직접 추가한다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent_utils import build_graph, build_history_messages, _to_text  # noqa: E402

load_dotenv()

RAG_LOG_PATH = LOG_PATH / "rag"
os.makedirs(RAG_LOG_PATH, exist_ok=True)
delete_incomplete_logs(base_path=RAG_LOG_PATH, required_files=["prompt.yaml", "rag_config.yaml"])

human_avatar = DEMO_IMG_PATH / "man-icon.png"
ai_avartar = DEMO_IMG_PATH / "inu-logo.png"
ai_avatar_image = str(ai_avartar)

if "agentic_on" not in st.session_state:
    st.session_state["agentic_on"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agentic_llm_temp" not in st.session_state:
    st.session_state["agentic_llm_temp"] = 0.2
if "agentic_retriever_k" not in st.session_state:
    st.session_state["agentic_retriever_k"] = 3
if "agentic_selected_llm" not in st.session_state:
    st.session_state["agentic_selected_llm"] = "claude-haiku-4-5"

with st.sidebar:
    st.write("")

    logs = [f for f in os.listdir(RAG_LOG_PATH) if os.path.isdir(os.path.join(RAG_LOG_PATH, f))]
    if not logs:
        st.error("저장된 항목이 없습니다. `2. Settings`에서 RAG를 먼저 생성하세요.")
        st.session_state["agentic_build_bot"] = False
        selected_log = None
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
                disabled=st.session_state["agentic_on"],
            )
            retriever_k = st.slider(
                "Retriever K value",
                min_value=2,
                max_value=8,
                value=3,
                step=1,
                disabled=st.session_state["agentic_on"],
            )
            llm_type = st.radio(
                "LLM",
                [
                    "claude-haiku-4-5",
                ],
                horizontal=True,
                disabled=st.session_state["agentic_on"],
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
            build_bot_btn = st.toggle("Build bot", key="agentic_build_bot", disabled=not logs)

        if build_bot_btn:
            selected_doc_path = selected_log_path / "docs"
            selected_db_path = selected_log_path / "db"
            st.session_state["agentic_llm_temp"] = llm_temperature
            st.session_state["agentic_retriever_k"] = retriever_k
            st.session_state["agentic_selected_llm"] = llm_type
            st.session_state["agentic_on"] = True
            st.success("Setting complete!")
        else:
            st.session_state["agentic_on"] = False

        if reset_btn and st.session_state["agentic_on"]:
            st.session_state["messages"] = []


st.title("Agentic RAG Applications")
st.caption("질문에 따라 문서검색·날짜·날씨 도구를 에이전트가 스스로 골라 답변합니다.")
st.write("")

col11, col12 = st.columns([1, 1])

with col11:
    st.write("**:blue[Documents]**")
    with st.container(height=670):
        if st.session_state["agentic_on"] and selected_rag_config:
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
        f"**:blue[RAG name :]** `{selected_log}` , **:blue[model:]** `{st.session_state['agentic_selected_llm']}`, **:blue[LLM Temp. :]** `{st.session_state['agentic_llm_temp']}`, **:blue[Retriever K :]** `{st.session_state['agentic_retriever_k']}`",
    )
    with st.container(height=670):
        if st.session_state["agentic_on"] and selected_rag_config:
            retriever = load_retriver(
                db_path=selected_db_path,
                embedding_model=selected_rag_config["embedding"],
                retriever_k=st.session_state["agentic_retriever_k"],
            )

            app = build_graph(
                model=st.session_state["agentic_selected_llm"],
                temperature=st.session_state["agentic_llm_temp"],
                retriever=retriever,
            )

            send_message("준비되었습니다. 질문하세요!", "ai", save=False)
            pain_history()

            with st.bottom:
                _, col122 = st.columns([1, 1])
                with col122:
                    message = st.chat_input("질문하세요. (산업안전보건법 / 날짜 / 날씨)")

            if message:
                send_message(message, "human")

                with st.chat_message("ai", avatar=ai_avatar_image):
                    status = st.status("🤔 추론 중...", expanded=True)  # 추론 과정(텍스트) 실시간 표시
                    answer_box = st.empty()  # 최종 답변 스트리밍 placeholder
                    answer = ""
                    current_msg_id = None  # 현재 스트리밍 중인 AI 턴 식별자
                    tool_turn_ids = set()  # 도구를 호출한(=추론) 턴 → 답변 버블에서 제외

                    inputs = {
                        "messages": build_history_messages(message, st.session_state["messages"]),
                    }

                    for mode, chunk in app.stream(inputs, stream_mode=["updates", "messages"]):
                        if mode == "updates":
                            # 노드 완료 단위: 추론 텍스트 / 도구 호출 / 도구 결과를 패널에 기록
                            for _node, payload in chunk.items():
                                for msg in payload.get("messages", []):
                                    if isinstance(msg, AIMessage):
                                        text = _to_text(msg.content)
                                        if text.strip():
                                            status.write(f"**[모델의 생각]** {text}")
                                        for call in msg.tool_calls:
                                            status.write(
                                                f"**[도구 호출]** `{call['name']}` ({call['args']})"
                                            )
                                    elif isinstance(msg, ToolMessage):
                                        status.write(
                                            f"**[도구 결과]** `{msg.name}` → {str(msg.content)[:300]}"
                                        )
                        elif mode == "messages":
                            # 토큰 단위: 도구 호출 턴(추론)은 제외하고 '최종 답변' 턴만 버블에 스트리밍.
                            # 한 AI 턴은 텍스트 청크가 먼저, tool_call 청크가 뒤에 올 수 있으므로
                            # 메시지 id 단위로 추적하고, 도구 호출이 감지되면 그 턴의 텍스트는 버린다.
                            msg_chunk, _meta = chunk
                            if not isinstance(msg_chunk, AIMessageChunk):
                                continue
                            if msg_chunk.id is not None and msg_chunk.id != current_msg_id:
                                current_msg_id = msg_chunk.id
                                answer = ""  # 새 AI 턴 시작 → 버퍼 초기화
                            if getattr(msg_chunk, "tool_calls", None) or getattr(
                                msg_chunk, "tool_call_chunks", None
                            ):
                                # 이 턴은 도구 호출(추론) 턴 → 누적 텍스트 폐기, 버블 비우기
                                if current_msg_id not in tool_turn_ids:
                                    tool_turn_ids.add(current_msg_id)
                                    answer = ""
                                    answer_box.empty()
                                continue
                            if current_msg_id in tool_turn_ids:
                                continue
                            token = _to_text(msg_chunk.content)
                            if token:
                                answer += token
                                answer_box.markdown(normalize_chat_markdown(answer))

                    status.update(label="✅ 완료", state="complete", expanded=False)
                    answer_box.markdown(normalize_chat_markdown(answer))
                    save_message(answer, "ai")
        else:
            st.session_state["messages"] = []
            st.session_state["agentic_on"] = False
