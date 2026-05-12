"""FastAPI 백엔드: ML 추천(recommender) + LLM(llm_router) 다단계 폴백 파이프라인 (플랫 구조)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_router import IntentResult, LLMRouter
from recommender import RestaurantRecommender

_LOG_PREFIX = "[api/chat]"


def _log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}", flush=True)


def _compose_search_query(intent: IntentResult) -> str:
    """검색 엔진용 쿼리: core_menu + ambience (불용어는 LLM 단계에서 제거)."""
    menu = (intent.core_menu or "").strip()
    amb = (intent.ambience or "").strip()
    return " ".join(p for p in (menu, amb) if p).strip()


def _multi_stage_recommend(
    recommender: RestaurantRecommender,
    intent: IntentResult,
) -> tuple[list[dict[str, Any]], str]:
    """1단계(지역+쿼리) → 2단계(지역 해제) → 3단계(메뉴 포기·지역+분위기 또는 지역 인기 폴백)."""
    loc = (intent.required_location or "").strip()
    menu = (intent.core_menu or "").strip()
    amb = (intent.ambience or "").strip()
    search_query = _compose_search_query(intent)

    top = recommender.recommend(search_query, loc, 3)
    if top:
        _log(f"검색 1단계 성공: query={search_query!r}, location={loc!r}, n={len(top)}")
        return top, ""

    _log(f"검색 1단계 결과 없음: query={search_query!r}, location={loc!r}")

    if loc:
        top = recommender.recommend(search_query, "", 3)
        if top:
            fr2 = (
                f"{loc} 주변에는 조건에 맞는 식당이 없어 다른 지역으로 찾아보았습니다."
            )
            _log(f"검색 2단계(지역 해제) 성공: query={search_query!r}, n={len(top)}")
            return top, fr2
        _log("검색 2단계(지역 해제) 결과 없음")

    if loc and menu:
        fr3 = (
            f"해당 지역에 {intent.core_menu} 식당이 없어, "
            f"{intent.required_location}의 다른 훌륭한 맛집들을 찾아보았습니다."
        )
        fallback_query = amb if amb else "맛집 식당 인기"
        top = recommender.recommend(fallback_query, loc, 3)
        if top:
            _log(
                f"검색 3단계(메뉴 포기·지역+분위기/기본쿼리) 성공: "
                f"query={fallback_query!r}, location={loc!r}, n={len(top)}"
            )
            return top, fr3
        _log(
            f"검색 3단계 결과 없음: query={fallback_query!r}, location={loc!r}"
        )

    return [], ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] RestaurantRecommender 로딩 중...")
    app.state.recommender = RestaurantRecommender()
    print("[startup] RestaurantRecommender 로딩 완료.")

    print("[startup] LLMRouter 로딩 중...")
    app.state.llm_router = LLMRouter()
    print("[startup] LLMRouter 로딩 완료.")

    print("[startup] 애플리케이션 준비 완료.")
    yield


app = FastAPI(
    title="Restaurant Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, description="사용자 메시지")


class ChatResponse(BaseModel):
    bot_message: str = Field(..., description="봇 응답")


def _get_recommender(request: Request) -> RestaurantRecommender:
    rec = getattr(request.app.state, "recommender", None)
    if rec is None:
        raise HTTPException(status_code=503, detail="추천 엔진이 초기화되지 않았습니다.")
    return rec


def _get_llm_router(request: Request) -> LLMRouter:
    router = getattr(request.app.state, "llm_router", None)
    if router is None:
        raise HTTPException(status_code=503, detail="LLM 라우터가 초기화되지 않았습니다.")
    return router


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """의도 분해 → 가드레일 → 다단계 ML 검색 → LLM 최종 응답."""
    recommender = _get_recommender(request)
    llm_router = _get_llm_router(request)
    user_message = payload.user_message.strip()

    _log(f"요청 수신: {user_message!r}")

    intent = await asyncio.to_thread(llm_router.analyze_intent, user_message)
    _log(
        "의도 분석 완료: "
        f"is_restaurant_query={intent.is_restaurant_query}, "
        f"required_location={(intent.required_location or '').strip()!r}, "
        f"core_menu={(intent.core_menu or '').strip()!r}, "
        f"ambience={(intent.ambience or '').strip()!r}"
    )

    if not intent.is_restaurant_query:
        _log("가드레일: 식당 질문 아님 → rejection_message 반환")
        return ChatResponse(bot_message=intent.rejection_message)

    top_shops, fallback_reason = await asyncio.to_thread(
        _multi_stage_recommend,
        recommender,
        intent,
    )

    if not top_shops:
        _log("다단계 검색 후에도 결과 없음 → 최종 실패 메시지")
        return ChatResponse(bot_message="조건에 맞는 식당을 찾을 수 없습니다.")

    shop_names = [s.get("shop_name", "") for s in top_shops]
    _log(
        f"LLM 최종 응답 생성: fallback_reason={fallback_reason!r}, "
        f"shops={[str(n) for n in shop_names]}"
    )

    final_answer = await asyncio.to_thread(
        llm_router.generate_reply,
        user_message,
        top_shops,
        fallback_reason,
    )
    _log(f"응답 완료 (bot_message 길이={len(final_answer)}자)")
    return ChatResponse(bot_message=final_answer)
