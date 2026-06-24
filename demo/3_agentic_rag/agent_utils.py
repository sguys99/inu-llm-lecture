"""Agentic RAG 그래프 구성 로직 (Streamlit 비의존).

`notebooks/3.1_langgraph_rag.ipynb` 의 StateGraph 조립을 데모 앱에서 재사용할 수 있도록
함수로 묶은 모듈. 도구·프롬프트·그래프 빌더와 메시지 변환 헬퍼를 제공한다.
"""

from datetime import date

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.modules.llms import get_llm

# notebooks/3.1_langgraph_rag.ipynb 와 동일한 시스템 프롬프트.
AGENTIC_RAG_PROMPT = """너는 '산업안전보건법'과 생활 정보를 함께 도와주는 한국어 어시스턴트야.

[사용할 수 있는 도구]
- search_osh_act : 산업안전보건법 문서에서 관련 조항을 검색한다.
    → 산업안전·보건, 사업주/근로자 의무, 중대재해, 도급, 작업중지, 안전보건진단 등
      '법령·안전' 관련 질문에 사용한다.
- get_today : 오늘 날짜를 알려준다.
- get_weather : 특정 도시의 오늘 날씨를 알려준다.

[행동 규칙]
1. 먼저 질문에 어떤 정보가 필요한지, 어떤 도구를 써야 할지 생각한다.
2. 산업안전보건법·안전·보건 관련 질문이면 반드시 search_osh_act 로 근거를 찾는다.
   너의 사전지식이나 추측으로 답하지 않는다.
3. search_osh_act 결과로 답할 때는 검색된 문서 내용만 근거로 삼는다.
   문서에서 근거를 찾을 수 없으면 "제공된 문서에서 해당 내용을 찾을 수 없습니다." 라고 솔직히 답한다.
4. 날짜·날씨는 추측하지 말고 반드시 도구를 호출해 확인한다.
5. 인사·잡담처럼 도구가 필요 없는 질문은 도구를 호출하지 말고 바로 답한다.
6. 답변은 한국어로, 핵심을 먼저 간결하게 말한다.
   법령 답변은 가능하면 근거 조항(예: 제5조)이나 페이지를 함께 밝힌다.
"""


def _to_text(content) -> str:
    """메시지 content가 문자열이든 블록 리스트든 텍스트만 뽑아낸다.

    Anthropic 모델은 content를 ``[{"type": "text", "text": ...}, ...]`` 형태의
    블록 리스트로 돌려줄 수 있으므로, 화면에 표시할 텍스트만 추출한다.

    Args:
        content: 메시지의 content (str 또는 블록 리스트).

    Returns:
        str: 추출된 텍스트.
    """
    if isinstance(content, str):
        return content
    chunks = []
    for block in content or []:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            chunks.append(block.get("text", ""))
    return "".join(chunks)


def build_tools(retriever):
    """에이전트가 골라 쓸 도구 3종을 생성해 반환한다.

    `search_osh_act` 는 사용자가 선택한 벡터 DB 기반 retriever를 클로저로 캡처한다.
    (RAG 선택·retriever_k 에 따라 retriever가 달라지므로 모듈 전역이 아닌 팩토리로 구성)

    Args:
        retriever: 벡터 검색에 사용할 LangChain retriever.

    Returns:
        list: `@tool` 로 감싼 도구 함수 목록.
    """

    @tool
    def search_osh_act(query: str) -> str:
        """산업안전보건법 문서에서 질문과 관련된 조항을 검색한다.

        산업안전·보건, 사업주/근로자의 의무, 중대재해, 도급, 작업중지,
        안전보건진단 등 '법령·안전' 관련 질문일 때 사용한다.
        검색된 조항 본문을 (페이지 번호와 함께) 돌려준다.
        """
        docs = retriever.invoke(query)
        return "\n\n".join(f"(p.{doc.metadata.get('page')}) {doc.page_content}" for doc in docs)

    @tool
    def get_today() -> str:
        """오늘 날짜를 'YYYY년 MM월 DD일' 형식으로 알려준다."""
        return date.today().strftime("%Y년 %m월 %d일")

    @tool
    def get_weather(city: str) -> str:
        """도시 이름을 입력받아 오늘 그 도시의 날씨를 알려준다. (실습용 목업 데이터)"""
        weather_db = {
            "서울": "맑음 (최고 24도 / 최저 14도)",
        }
        return weather_db.get(city, f"'{city}'의 날씨 정보가 없습니다.")

    return [search_osh_act, get_today, get_weather]


def build_graph(model: str, temperature: float, retriever):
    """Agentic RAG용 LangGraph StateGraph를 조립·컴파일해 반환한다.

    `agent` 노드(LLM 판단)와 `tools` 노드(ToolNode)를 두고, `tools_condition`으로
    도구 호출 여부를 분기한다. 시스템 프롬프트는 `agent` 노드에서 직접 주입한다.

    Args:
        model (str): 사용할 LLM 모델명 (예: "claude-haiku-4-5").
        temperature (float): LLM temperature.
        retriever: 벡터 검색용 retriever (도구에 캡처됨).

    Returns:
        CompiledStateGraph: `.stream()` 호출 가능한 컴파일된 그래프.
    """
    tools = build_tools(retriever)
    # 스트리밍은 .stream(stream_mode="messages")로 직접 처리하므로 콜백은 넘기지 않는다.
    llm = get_llm(model=model, temperature=temperature, streaming=True)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        """대화 메시지를 LLM에 넘겨 '다음 행동'(도구 호출 또는 최종 답변)을 결정한다."""
        messages = [SystemMessage(content=AGENTIC_RAG_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_history_messages(new_question: str, history: list, window: int = 4) -> list:
    """표시용 대화 히스토리에서 직전 N개를 LangChain 메시지로 복원해 새 질문과 합친다.

    `ConversationBufferWindowMemory` 대신, 화면에 저장된 messages 리스트에서
    직전 `window` 개 메시지를 가져와 멀티턴 맥락을 구성한다.

    Args:
        new_question (str): 이번 사용자 질문.
        history (list): `st.session_state["messages"]` 형태의 표시용 히스토리.
        window (int, optional): 사용할 직전 메시지 개수. 기본값 4 (≈2턴).

    Returns:
        list: LangGraph 입력용 메시지 리스트.
    """
    msgs = []
    for m in history[-window:]:
        cls = HumanMessage if m["role"] == "human" else AIMessage
        msgs.append(cls(content=m["message"]))
    msgs.append(HumanMessage(content=new_question))
    return msgs
