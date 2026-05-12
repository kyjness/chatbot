"""LangChain 기반 의도 분석(조건 분해) 및 추천 결과 기반 자연어 응답 (플랫 구조)."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IntentResult(BaseModel):
    is_restaurant_query: bool = Field(
        description="사용자가 식당·음식·맛집·카페 추천 등과 관련된 질문인지 여부"
    )
    required_location: str = Field(
        default="",
        description="사용자가 명시한 지명·지역(예: 성수역, 강남구, 홍대). 없으면 빈 문자열",
    )
    core_menu: str = Field(
        default="",
        description=(
            "음식/메뉴. 단, 검색 매칭률을 극대화하기 위해 반드시 유의어/상위카테고리 단어를 띄어쓰기로 1~2개 추가해. "
            "(예: '라면' -> '라면 라멘 국수 면요리', '고기' -> '고기 삼겹살 소고기 육류'). 없으면 빈 문자열."
        ),
    )
    ambience: str = Field(
        default="",
        description="분위기·목적·상황(예: 데이트, 조용한, 회식). 없으면 빈 문자열",
    )
    rejection_message: str = Field(
        default="",
        description="식당 질문이 아닐 때 정중한 거절 문구. 식당 질문이면 빈 문자열",
    )


_INTENT_SYSTEM = """너는 식당 검색 조건을 분석하는 전문가야.

사용자의 질문에서 '지역', '메뉴', '분위기'를 각각 분리해서 추출해.
- required_location: 역·구·동·상권 등 지리적 표현만 넣어.
- core_menu: 사용자가 찾는 음식·메뉴·요리의 핵심 표현을 넣되, 메뉴 추출 시 반드시 동의어나 관련 키워드를 추가하여 확장해라. (검색 매칭률을 높이기 위해 유의어·상위 카테고리를 띄어쓰기로 1~2개 덧붙임. 예: 라면→라면 라멘 국수 면요리)
- ambience: 데이트, 조용한, 회식 등 분위기·목적·상황만 넣어.

'맛집', '추천', '근처', '어디', '알려줘' 같은 검색에 도움이 되지 않는 불용어는 철저히 버리고, 위 세 필드에는 넣지 마.
특정 필드가 언급되지 않았다면 해당 필드는 빈 문자열로 둬.

식당·음식과 무관한 질문이면 is_restaurant_query를 False로 하고 rejection_message에 정중한 거절 문구를 적어.
식당 관련 질문이면 is_restaurant_query를 True로 하고 rejection_message는 빈 문자열로 둬."""

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _INTENT_SYSTEM),
        ("human", "{user_message}"),
    ]
)

REPLY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """너는 친절하고 전문적인 맛집 추천 가이드야.
사용자의 질문: {user_message}
추천된 식당 리스트: {top_shops}

fallback_reason: {fallback_reason}

위 식당 리스트의 정보를 바탕으로 사용자에게 친절하게 식당을 추천해주는 답변을 작성해.
[규칙]

반드시 제공된 '추천된 식당 리스트'의 정보(이름, 카테고리, 메뉴, 태그 등)만 사용할 것. 절대 지어내지(환각) 말 것.

식당의 llm_situation_tags나 menus를 활용하여 왜 이 식당이 질문에 어울리는지 가볍게 언급해 줄 것.

가독성을 위해 마크다운(글머리 기호, 볼드체 등)을 적절히 사용할 것.

fallback_reason에 내용이 있다면(빈 문자열이 아니라면), 답변 서두에 그 이유를 자연스럽게 녹여 대안 추천임을 설명해.
(예: "요청하신 성수역 주변에는 라면집이 없어서, 대신 다른 지역의 맛있는 라면집을 추천해 드립니다.")""",
        ),
    ]
)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content).strip()


class LLMRouter:
    """조건 분해(가드레일) 및 추천 결과 기반 자연어 응답 생성."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        intent_temperature: float = 0.0,
        reply_temperature: float = 0.7,
    ) -> None:
        self._intent_llm = ChatOpenAI(
            model=model,
            temperature=intent_temperature,
        )
        self._reply_llm = ChatOpenAI(
            model=model,
            temperature=reply_temperature,
        )
        self._intent_chain = INTENT_PROMPT | self._intent_llm.with_structured_output(
            IntentResult
        )
        self._reply_chain = REPLY_PROMPT | self._reply_llm

    def analyze_intent(self, user_message: str) -> IntentResult:
        """식당 관련 여부, 지역·메뉴·분위기 분해, 비식당 질문 시 거절 문구."""
        text = (user_message or "").strip()
        if not text:
            return IntentResult(
                is_restaurant_query=False,
                required_location="",
                core_menu="",
                ambience="",
                rejection_message="질문을 입력해 주세요. 맛집이나 식당과 관련된 내용을 알려드릴 수 있어요.",
            )
        try:
            result = self._intent_chain.invoke({"user_message": text})
        except Exception:
            logger.exception("analyze_intent 호출 실패")
            raise
        if not isinstance(result, IntentResult):
            raise TypeError(f"structured output 타입이 IntentResult 가 아닙니다: {type(result)}")
        return result

    def generate_reply(
        self,
        user_message: str,
        top_shops: list[dict[str, Any]],
        fallback_reason: str = "",
    ) -> str:
        """추천 식당 리스트만 근거로 마크다운 형태의 최종 답변 생성."""
        um = (user_message or "").strip()
        payload = json.dumps(
            top_shops,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        fr = (fallback_reason or "").strip()
        try:
            out = self._reply_chain.invoke(
                {
                    "user_message": um,
                    "top_shops": payload,
                    "fallback_reason": fr if fr else "(없음)",
                }
            )
        except Exception:
            logger.exception("generate_reply 호출 실패")
            raise

        if isinstance(out, AIMessage):
            return _message_text(out)
        if isinstance(out, str):
            return out.strip()
        return str(out).strip()
