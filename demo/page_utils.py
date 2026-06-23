import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def login():
    """
    로그인 폼을 생성하고 사용자 인증을 처리하는 함수..
    """
    for _ in range(5):
        st.write("")

    col1, _ = st.columns([1, 1.5])
    with col1:
        with st.form("login_form"):
            st.header("Log in")
            st.write("")
            username = st.text_input("User ID")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                correct_username = os.getenv("STREAMLIT_ID")
                correct_password = os.getenv("STREAMLIT_PW")
                if username == correct_username and password == correct_password:
                    st.success("Login successful!")
                    st.session_state["login"] = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password")


def logout():
    """
    로그아웃을 처리하는 함수.
    """
    st.session_state["login"] = None
    st.rerun()
