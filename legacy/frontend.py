from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import requests
import streamlit as st


# =========================
# 설정
# =========================


DEFAULT_API_URL = "http://127.0.0.1:8000/chat"


Role = Literal["user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    answer: str
    error_message: str | None = None
    latency_ms: int | None = None
    http_status: int | None = None
    intent: dict[str, Any] | None = None
    top_shops: list[dict[str, Any]] | None = None


# =========================
# API 통신
# =========================


def call_chat_api(
    api_url: str,
    message: str,
    history: list[dict[str, str]],
    timeout_sec: float = 60.0,
) -> ApiResult:
    """
    FastAPI 서버(/chat)로 메시지를 전송하고 답변을 받아온다.
    - 네트워크 오류/서버 다운/응답 포맷 오류 등은 모두 안전하게 처리한다.
    - UX 관점에서 앱이 멈추지 않도록, 예외를 ApiResult로 흡수한다.
    """

    payload = {"message": message, "history": history}

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout_sec)
    except requests.RequestException:
        return ApiResult(
            ok=False,
            answer="",
            error_message="서버와 연결할 수 없습니다. 백엔드 서버가 켜져 있는지 확인해 주세요.",
            http_status=None,
        )

    status = resp.status_code

    def _detail_message() -> str:
        try:
            body: dict[str, Any] = resp.json()
        except (json.JSONDecodeError, ValueError):
            return ""
        detail = body.get("detail")
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict) and first.get("msg"):
                return str(first["msg"])
            return str(first)
        if isinstance(detail, str):
            return detail
        return ""

    if status == 422:
        msg = _detail_message() or "요청 형식이 올바르지 않습니다. (입력 길이·역할 등을 확인해 주세요)"
        return ApiResult(ok=False, answer="", error_message=msg, http_status=status)

    if status == 503:
        msg = _detail_message() or "서버가 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."
        return ApiResult(ok=False, answer="", error_message=msg, http_status=status)

    if status != 200:
        msg = _detail_message() or f"서버 오류가 발생했습니다. (HTTP {status})"
        return ApiResult(ok=False, answer="", error_message=msg, http_status=status)

    # JSON 파싱 및 스키마 방어
    try:
        data: dict[str, Any] = resp.json()
    except json.JSONDecodeError:
        return ApiResult(
            ok=False,
            answer="",
            error_message="서버 응답을 해석할 수 없습니다. 잠시 후 다시 시도해 주세요.",
            http_status=status,
        )

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return ApiResult(
            ok=False,
            answer="",
            error_message="서버 응답 형식이 올바르지 않습니다. 백엔드 로그를 확인해 주세요.",
            http_status=status,
        )

    lat: int | None = None
    raw_lat = data.get("latency_ms")
    if isinstance(raw_lat, int):
        lat = raw_lat
    elif isinstance(raw_lat, float) and raw_lat == int(raw_lat):
        lat = int(raw_lat)

    intent_raw = data.get("intent")
    intent: dict[str, Any] | None = intent_raw if isinstance(intent_raw, dict) else None

    shops_raw = data.get("top_shops")
    top_shops: list[dict[str, Any]] | None = None
    if isinstance(shops_raw, list) and all(isinstance(x, dict) for x in shops_raw):
        top_shops = shops_raw  # type: ignore[assignment]

    return ApiResult(
        ok=True,
        answer=answer.strip(),
        error_message=None,
        latency_ms=lat,
        http_status=status,
        intent=intent,
        top_shops=top_shops,
    )


# =========================
# UI 렌더링
# =========================


def init_session_state() -> None:
    """
    세션 상태 초기화.
    - 리렌더링되어도 대화 기록이 유지되도록 st.session_state에 저장한다.
    """

    if "history" not in st.session_state:
        st.session_state.history = []  # type: ignore[attr-defined]


def render_header() -> None:
    st.title("식당 추천 AI 에이전트")
    st.caption("지역과 메뉴를 알려주면 빠르게 맛집을 추천해 드립니다. 예: 강남역 맛집, 압구정 파스타")


def render_sidebar() -> str:
    """
    UX 편의 기능:
    - 백엔드 URL을 사이드바에서 바꿀 수 있게 제공(로컬/원격 전환 쉬움)
    - 대화 초기화 버튼 제공
    """

    st.sidebar.header("설정")
    api_url = st.sidebar.text_input("백엔드 API URL", value=DEFAULT_API_URL)

    if st.sidebar.button("대화 초기화"):
        st.session_state.history = []  # type: ignore[attr-defined]
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("백엔드가 로컬에서 실행 중인지 확인하세요. (예: uvicorn main:app --reload)")
    return api_url.strip() or DEFAULT_API_URL


def render_history(history: list[ChatMessage]) -> None:
    """
    채팅 히스토리를 ChatGPT 스타일로 렌더링한다.
    - Streamlit 기본 채팅 컴포넌트를 사용한다.
    """

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def append_message(role: Role, content: str) -> None:
    st.session_state.history.append({"role": role, "content": content})  # type: ignore[attr-defined]


def main() -> None:
    st.set_page_config(page_title="식당 추천 챗봇", page_icon="🍽️", layout="centered")

    init_session_state()
    render_header()
    api_url = render_sidebar()

    # 기존 대화 렌더링
    render_history(st.session_state.history)  # type: ignore[attr-defined]

    # 입력 UI
    user_input = st.chat_input("어떤 식당을 찾으시나요?")
    if not user_input:
        return

    user_input = user_input.strip()
    if not user_input:
        return

    # 사용자 메시지 먼저 화면에 반영
    append_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 응답 로딩 UX
    with st.chat_message("assistant"):
        with st.spinner("맛집을 검색하는 중입니다..."):
            prior_history: list[ChatMessage] = st.session_state.history[:-1]  # type: ignore[attr-defined]
            result = call_chat_api(
                api_url=api_url,
                message=user_input,
                history=list(prior_history),
            )

        if not result.ok:
            st.error(result.error_message or "알 수 없는 오류가 발생했습니다.")
            # UX: 실패 메시지도 히스토리에 남겨서 사용자가 상황을 인지할 수 있게 한다.
            append_message(
                "assistant",
                result.error_message
                or "서버와 연결할 수 없습니다. 백엔드 서버가 켜져 있는지 확인해 주세요.",
            )
            return

        st.markdown(result.answer)
        if result.latency_ms is not None:
            st.caption(f"응답 시간: {result.latency_ms} ms")
        append_message("assistant", result.answer)


if __name__ == "__main__":
    main()

