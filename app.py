"""Streamlit 프론트: 식당 추천 챗봇 UI (플랫 구조)."""

from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st

CHAT_API_URL = "http://127.0.0.1:8000/api/chat"
REQUEST_TIMEOUT_S = 30

_INTRO_ASSISTANT = (
    "안녕하세요! 어떤 맛집을 찾으시나요? 분위기, 지역, 메뉴를 편하게 말씀해 주시면 추천을 도와드릴게요."
)


def _init_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": _INTRO_ASSISTANT}]


st.set_page_config(
    page_title="식당 추천 챗봇",
    page_icon="🍽️",
    layout="centered",
)

_init_messages()

st.title("식당 추천 챗봇")
st.markdown(
    "안녕하세요! 어떤 맛집을 찾으시나요? 분위기, 지역, 메뉴를 편하게 말씀해 주세요."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("어떤 식당을 찾으시나요?"):
    user_text = prompt.strip()
    if not user_text:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    assistant_reply = ""
    with st.chat_message("assistant"):
        with st.spinner("맛집을 검색하는 중입니다..."):
            try:
                response = requests.post(
                    CHAT_API_URL,
                    json={"user_message": user_text},
                    timeout=REQUEST_TIMEOUT_S,
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                assistant_reply = str(data.get("bot_message", "")).strip()
                if not assistant_reply:
                    assistant_reply = "응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요."
            except requests.RequestException:
                assistant_reply = "서버와 연결할 수 없습니다. 잠시 후 다시 시도해 주세요."
            except (json.JSONDecodeError, ValueError, TypeError):
                assistant_reply = "응답 형식이 올바르지 않습니다. 잠시 후 다시 시도해 주세요."

        st.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
