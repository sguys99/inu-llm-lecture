import streamlit as st
from dotenv import load_dotenv
from page_utils import login, logout

load_dotenv()

if "login" not in st.session_state:
    st.session_state["login"] = False


st.set_page_config(
    page_title="LFC RAG",
    page_icon="img/favicon.svg",
    layout="wide",
)

st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

home = st.Page(
    "home/home.py",
    title="Home page",
    icon=":material/home:",
    default=True,
)

qa_application = st.Page(
    "1_single_doc_application/qa_application.py",
    title="Simple QA Application",
    icon=":material/smart_toy:",
)

qa_rag_settings = st.Page(
    "2_qa_rag_assistant/rag_settings.py",
    title="Settings",
    icon=":material/settings:",
)

qa_rag_application = st.Page(
    "2_qa_rag_assistant/rag_application.py",
    title="Application",
    icon=":material/summarize:",
)

if st.session_state["login"]:
    pg = st.navigation(
        {
            "⚙️ 계정": [logout_page],
            "0️⃣ Home": [home],
            "1️⃣ 단일문서 어플리케이션": [qa_application],
            "2️⃣ RAG기반 QA 어시스턴트": [qa_rag_settings, qa_rag_application],
        },
    )
else:
    pg = st.navigation([login_page])


with st.sidebar:
    st.image("img/inu-signature.png", width=290)

pg.run()
