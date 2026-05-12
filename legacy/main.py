from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import re
import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder, SentenceTransformer

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)


def _sanitize_chat_answer(text: str) -> str:
    """모델이 프롬프트 예시 문구를 그대로 출력하는 경우 제거·공백 정리."""
    t = text or ""
    for pat in (r"\(빈\s*줄\)", r"（빈\s*줄）", r"\(빈줄\)", r"（빈줄）"):
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    t = t.replace("(빈 줄)", "").replace("（빈 줄）", "")
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t.strip()


def _retrieval_ranking_query(intent: "RestaurantIntent", message: str) -> str:
    """
    Bi-Encoder·TF-IDF·CE에 넣을 질의 문자열.
    LLM이 search_query에 메뉴/일부 지명을 빼먹는 경우가 많아 intent.menu·locations를 중복 없이 덧붙인다.
    """
    sq = (intent.search_query or "").strip()
    menu = (intent.menu or "").strip()
    base = sq or menu or (message or "").strip()
    parts = [base]
    blob = base.lower()
    if menu and menu.lower() not in blob:
        parts.append(menu)
        blob = " ".join(parts).lower()
    for loc in intent.locations:
        ls = (loc or "").strip()
        if ls and ls.lower() not in blob:
            parts.append(ls)
            blob = " ".join(parts).lower()
    return " ".join(p for p in parts if p).strip()


def _log_preview(text: str, max_len: int = 220) -> str:
    """터미널 로그용 한 줄 미리보기(개행 제거·길이 제한)."""
    t = (text or "").replace("\r", " ").replace("\n", " ↳ ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _log_df_head_shops(df: pd.DataFrame, label: str, max_rows: int = 3) -> None:
    """후보 DataFrame 상단 몇 곳의 이름·점수 컬럼을 한 블록으로 요약 로그한다."""
    if df.empty:
        logger.info("%s 후보 0건", label)
        return
    score_cols = [c for c in ("final_score", "lgbm_score", "ce_score", "_faiss_score") if c in df.columns]
    _sc_lab = {"final_score": "T", "lgbm_score": "L", "ce_score": "C", "_faiss_score": "F"}
    bits: list[str] = []
    n = min(max_rows, len(df))
    for i in range(n):
        row = df.iloc[i]
        name = ""
        if "shop_name" in df.columns:
            name = str(row.get("shop_name", "") or "")[:36]
        pri = score_cols[0] if score_cols else None
        sc = ""
        if pri:
            try:
                lab = _sc_lab.get(pri, "?")
                sc = f"{lab}={float(row[pri]):.3f}"
            except (TypeError, ValueError):
                sc = "?=?"
        bits.append(f"{i + 1}:{name or '?'}({sc})" if sc else f"{i + 1}:{name or '?'}")
    tail = f" …외{len(df) - n}건" if len(df) > n else ""
    logger.info("%s %d건 | %s%s", label, len(df), " | ".join(bits), tail)


def _configure_intent_logging() -> None:
    """Step1·Step2 관측용: 루트 로거 레벨과 무관하게 INFO가 터미널에 보이도록 한다."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# =========================
# FastAPI App
# =========================

app = FastAPI(title="식당 추천 챗봇 RAG 서버", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /chat 대화 한도 (ChatMessageItem·ChatRequest보다 위에 두어 Field에서 참조 가능하게 함)
CHAT_HISTORY_MAX_ITEMS = int(os.getenv("CHAT_HISTORY_MAX_ITEMS", "40"))
CHAT_MESSAGE_ITEM_MAX_LEN = int(os.getenv("CHAT_MESSAGE_ITEM_MAX_LEN", "8000"))
INTENT_HISTORY_TAIL = int(os.getenv("INTENT_HISTORY_TAIL", "8"))


# =========================
# Request / Response Models
# =========================


class ChatMessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=CHAT_MESSAGE_ITEM_MAX_LEN)

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v: object) -> str:
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("user", "assistant"):
                return t
        raise ValueError("role은 user 또는 assistant여야 합니다.")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=200, description="사용자 질의")
    history: list[ChatMessageItem] = Field(
        default_factory=list, description="이전 대화 기록"
    )

    @model_validator(mode="after")
    def _trim_history(self) -> ChatRequest:
        if len(self.history) > CHAT_HISTORY_MAX_ITEMS:
            object.__setattr__(
                self,
                "history",
                list(self.history[-CHAT_HISTORY_MAX_ITEMS:]),
            )
        return self


class ChatResponse(BaseModel):
    answer: str = Field(..., description="챗봇 답변")
    intent: Optional[dict[str, Any]] = Field(default=None, description="추출된 의도(디버깅용)")
    top_shops: Optional[list[dict[str, Any]]] = Field(default=None, description="추천 후보(디버깅용)")
    latency_ms: int = Field(..., description="서버 처리 지연(밀리초)")


# =========================
# Step 1: Intent Extraction (Structured Output, LLM Router)
# =========================


GUARDRAIL_NON_RESTAURANT_MESSAGE = (
    "저는 식당 추천 봇입니다. 죄송하지만 식당 관련 질의해 주시겠어요?"
)


class RestaurantIntent(BaseModel):
    is_restaurant_query: bool = Field(
        ..., description="식당 탐색, 맛집 추천 관련 질의면 true. 비식당 주제면 false."
    )
    locations: list[str] = Field(
        default_factory=list,
        description="질문에 명시된 지역명을 기반으로, 실제 검색에 유용하도록 인접 지명(동, 역, 구)을 3~4개로 자동 확장한 리스트. (예: ['강남역', '역삼동', '서초동']). 없으면 빈 리스트.",
    )
    menu: str = Field(
        default="",
        description="음식·메뉴 종류가 명확할 때만 채운다 (예: 파스타, 삼겹살). 데이트·맛집·추천 등 상황어만 있으면 빈 문자열.",
    )
    search_query: str = Field(
        default="",
        description="추천 엔진에 넘길 전체 검색 문장 (지역 제외, 분위기/조건 위주).",
    )


_INTENT_SYSTEM_PROMPT = """
역할
- 너는 식당 추천 챗봇의 1단계 라우터다.
- [대화 기록]은 '거기', '아까 말한 곳', '같은 조건으로'처럼 **생략된 말을 보충**할 때만 사용한다.

[현재 입력 우선 — 매우 중요]
- [현재 입력]에 **새 지역**(예: 강남역, 홍대)이나 **새 목적·메뉴**가 분명히 나오면, 이전 턴의 지역·메뉴(예: 압구정, 파스타)는 **폐기**하고 현재 입력만 반영한다.
- 이전 턴과 같은 주제를 이어가려는 말이 없는 한, locations·menu·search_query를 **현재 입력 기준으로 새로** 채운다.

출력 규칙
1) is_restaurant_query: 식당·맛집·카페·술집·배달·예약·영업시간 등 음식점 탐색이면 true.
   주식·투자·환율, 날씨, 뉴스, 정치, 코딩, 일반 상식, 회사 업황 전망 등 식당 추천과 무관한 주제는 전부 false.
2) locations: **현재 입력**의 지역을 우선 파악하고, 검색 성공률을 위해 주변 동/역을 3~4개로 확장한다. (현재 입력에 지역이 없고 대화 기록만 있을 때만 기록에서 보충.)
3) menu: **실제 메뉴·음식 종류**가 있을 때만 짧게 채운다. 데이트·맛집·추천·동반 표현만 있으면 반드시 빈 문자열.
4) search_query: **현재 입력**의 목적·분위기·조건을 한 문장으로 압축한다.
   질문에 음식·메뉴 이름(예: 파스타, 삼겹살, 초밥)이 있으면 **그 단어를 이 문장 안에 반드시 포함**한다(지역명은 locations에만 넣고 여기서는 빼도 된다).
""".strip()


class IntentExtractor:
    """LLM 기반 구조화 의도 추출. Step 1 전용."""

    def __init__(self, model_name: str) -> None:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=160,
            timeout=20.0,
        )
        human_it = HumanMessagePromptTemplate.from_template("{message}")
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=_INTENT_SYSTEM_PROMPT),
                human_it,
            ]
        )
        self._runnable = prompt | llm.with_structured_output(RestaurantIntent)

    @staticmethod
    def normalize(raw: RestaurantIntent) -> RestaurantIntent:
        locs_in = raw.locations if isinstance(raw.locations, list) else []
        locs: list[str] = []
        seen_loc: set[str] = set()
        for item in locs_in:
            if not isinstance(item, str):
                continue
            t = item.strip()
            if not t or t in seen_loc:
                continue
            seen_loc.add(t)
            locs.append(t)
        menu = (raw.menu or "").strip()
        sq = (raw.search_query or "").strip()
        return RestaurantIntent(
            is_restaurant_query=bool(raw.is_restaurant_query),
            locations=locs,
            menu=menu,
            search_query=sq,
        )

    async def ainvoke(self, message: str) -> RestaurantIntent:
        out: RestaurantIntent = await self._runnable.ainvoke({"message": message})
        return self.normalize(out)


# =========================
# Step 2: 1-Stage FAISS Retrieval (Bi-Encoder + 인덱스)
# =========================


_LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("RESTAURANT_DATA_DIR", os.path.join(_LEGACY_DIR, "data"))
FAISS_INDEX_PATH = os.getenv("RESTAURANT_FAISS_INDEX_PATH", os.path.join(DATA_DIR, "faiss.index"))
SHOPS_METADATA_PKL_PATH = os.path.join(DATA_DIR, "shops_metadata.pkl")
LGBM_RANKER_PATH = os.path.join(DATA_DIR, "lgbm_ranker.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
BI_ENCODER_MODEL_ID = os.getenv("RESTAURANT_BI_ENCODER_MODEL", "jhgan/ko-sbert-nli")
# 한국어 검색·리랭킹에 특화된 공개 Cross-Encoder. 비공개 저장소는 HF_TOKEN 과 RESTAURANT_CROSS_ENCODER_MODEL 만으로 교체 가능.
CROSS_ENCODER_MODEL_ID = os.getenv(
    "RESTAURANT_CROSS_ENCODER_MODEL",
    "bongsoo/kpf-cross-encoder-v1",
)
RETRIEVAL_TOP_K = int(os.getenv("RESTAURANT_RETRIEVAL_TOP_K", "100"))
# Step 3 이후 Cross-Encoder에 넘길 최대 행 수 (CPU·지연 완화)
CROSS_ENCODER_MAX_INPUT = int(os.getenv("RESTAURANT_CROSS_ENCODER_MAX_INPUT", "20"))
# intent.menu ↔ 매장 텍스트: Bi-Encoder 코사인(임계값·완화 단계는 env로 조절, 동의어 테이블 없음)
_MENU_MATCH_MAX_CHARS = int(os.getenv("RESTAURANT_MENU_MATCH_MAX_CHARS", "480"))


def _menu_semantic_params() -> tuple[float, float, float]:
    return (
        float(os.getenv("RESTAURANT_MENU_SEM_MIN", "0.25")),
        float(os.getenv("RESTAURANT_MENU_SEM_RELAXED", "0.18")),
        float(os.getenv("RESTAURANT_MENU_SEM_RELAX_MARGIN", "0.10")),
    )


def _resolve_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_faiss_index(path: str) -> faiss.Index:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FAISS 인덱스 파일이 없습니다: {path}")
    return faiss.read_index(path)


def _load_meta_dataframe(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"메타 피클이 없습니다: {path}")
    obj: Any
    errors: list[str] = []
    try:
        obj = joblib.load(path)
    except Exception as e:
        errors.append(f"joblib.load: {e}")
        try:
            obj = pd.read_pickle(path)
        except Exception as e2:
            errors.append(f"pandas.read_pickle: {e2}")
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
            except Exception as e3:
                errors.append(f"pickle.load: {e3}")
                raise RuntimeError(
                    "shops_metadata.pkl을 읽지 못했습니다. joblib으로 저장된 경우가 많습니다. "
                    "생성 스크립트에서 `joblib.dump(df, path)` 또는 `df.to_pickle(path)`를 확인하세요.\n"
                    + "\n".join(errors)
                ) from e3
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("shops_metadata.pkl은 pandas.DataFrame이어야 합니다.")
    return obj.reset_index(drop=True)


def _huggingface_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _load_bi_encoder(model_id: str, device: str) -> SentenceTransformer:
    tok = _huggingface_token()
    if tok:
        return SentenceTransformer(model_id, device=device, token=tok)
    return SentenceTransformer(model_id, device=device)


def _load_cross_encoder(model_id: str, device: str) -> CrossEncoder:
    tok = _huggingface_token()
    if tok:
        return CrossEncoder(model_id, device=device, token=tok)
    return CrossEncoder(model_id, device=device)


def _sentence_transformer_embedding_dim(model: SentenceTransformer) -> int:
    fn = getattr(model, "get_embedding_dimension", None)
    if callable(fn):
        return int(fn())
    return int(model.get_sentence_embedding_dimension())


def faiss_retrieve_top_k(
    *,
    query_text: str,
    bi_encoder: SentenceTransformer,
    faiss_index: faiss.Index,
    meta_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Bi-Encoder 임베딩 후 FAISS search로 상위 후보 행을 meta_df에서 복원한다.
    """
    ntotal = int(faiss_index.ntotal)
    k = min(top_k, ntotal)
    logger.info(
        "[Step2 FAISS] q=%r | k=%d/%d | meta=%d행",
        _log_preview(query_text, 120),
        k,
        ntotal,
        len(meta_df),
    )

    if k <= 0:
        logger.info("[Step2 FAISS] 후보 0건(인덱스 비어 있음)")
        return meta_df.iloc[0:0].copy()

    embedding = bi_encoder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q = embedding.astype("float32", copy=False)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    scores, indices = faiss_index.search(q, k)

    row_positions: list[int] = []
    row_scores: list[float] = []
    for rank in range(k):
        idx = int(indices[0][rank])
        if idx < 0:
            continue
        if idx >= len(meta_df):
            continue
        row_positions.append(idx)
        row_scores.append(float(scores[0][rank]))

    candidates_df = meta_df.iloc[row_positions].copy()
    if row_scores:
        candidates_df["_faiss_score"] = row_scores

    _log_df_head_shops(candidates_df, "[Step2 FAISS]")
    logger.info("[Step2 FAISS] 복원 %d건(top_k≤%d)", len(candidates_df), top_k)
    return candidates_df


def filter_candidates_by_menu_semantic(
    candidates_df: pd.DataFrame,
    menu_phrase: str,
    bi_encoder: SentenceTransformer,
) -> pd.DataFrame:
    """
    LLM이 채운 intent.menu와 각 후보의 요약 텍스트(meta_text 우선) 간 **Bi-Encoder 코사인 유사도**로
    관련도 필터링한다. 언어·표기(한글/영문) 차이는 임베딩이 흡수하고, 동의어 사전은 쓰지 않는다.

    임계값을 넘는 행이 없으면 완화 임계값을 시도한다.
    최대 유사도가 전반적으로 너무 낮으면(노이즈) 필터를 생략하며, 완화 후에도 없으면 **엉뚱한 업종을 top-k로 끌어오지 않고** 필터 생략 후 랭킹(LGBM·CE)에 맡긴다.
    """
    n_before = len(candidates_df)
    phrase = (menu_phrase or "").strip()
    if not phrase or candidates_df.empty:
        return candidates_df

    if "meta_text" in candidates_df.columns:
        doc_src = candidates_df["meta_text"].fillna("").astype(str)
    else:
        cols = [c for c in ("shop_name", "categories", "menus", "search_intents") if c in candidates_df.columns]
        if not cols:
            logger.warning("[Step3 메뉴] 요약 텍스트 컬럼 없음 → 필터 생략")
            return candidates_df
        acc = candidates_df[cols[0]].fillna("").astype(str)
        for c in cols[1:]:
            acc = acc + " " + candidates_df[c].fillna("").astype(str)
        doc_src = acc

    docs = [t[:_MENU_MATCH_MAX_CHARS] for t in doc_src.tolist()]
    if not any(d.strip() for d in docs):
        logger.warning("[Step3 메뉴] 문서 텍스트가 모두 비어 있음 → 필터 생략")
        return candidates_df

    sim_min, sim_relaxed, relax_margin = _menu_semantic_params()
    merged = [phrase] + docs
    emb = bi_encoder.encode(
        merged,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q = emb[0].astype(np.float32, copy=False)
    doc_e = emb[1:].astype(np.float32, copy=False)
    sims = (doc_e @ q).astype(np.float64, copy=False)
    mx = float(np.max(sims)) if len(sims) else 0.0

    # 전 후보가 메뉴와 의미적으로 거의 안 붙으면(노이즈만 있음) 필터를 쓰지 않고 랭킹에 맡긴다.
    too_weak = float(os.getenv("RESTAURANT_MENU_SEM_MAX_TOO_WEAK", "0.20"))
    if mx < too_weak:
        logger.info(
            "[Step3 메뉴·의미] max_sim=%.3f < %.2f → 필터 생략(%d건, 랭킹에 위임)",
            mx,
            too_weak,
            n_before,
        )
        return candidates_df

    def _take_sorted(idxs: np.ndarray) -> pd.DataFrame:
        if len(idxs) == 0:
            return candidates_df.iloc[0:0].copy()
        order = idxs[np.argsort(-sims[idxs])]
        return candidates_df.iloc[order].copy()

    tier = "skip"
    out = candidates_df
    primary = sims >= sim_min
    if bool(np.any(primary)):
        out = _take_sorted(np.flatnonzero(primary))
        tier = f">={sim_min:.2f}"
    else:
        thr2 = max(sim_relaxed, mx - relax_margin)
        relaxed = sims >= thr2
        if bool(np.any(relaxed)):
            out = _take_sorted(np.flatnonzero(relaxed))
            tier = f">={thr2:.2f}(완화)"
        else:
            # 유사도가 전반적으로 낮을 때 상위 N만 강제로 남기면 엉뚱한 업종이 섞이므로 필터 생략
            logger.info(
                "[Step3 메뉴·의미] 임계 미달·폴백 생략 | max_sim=%.3f | %d건 유지(랭킹에 위임)",
                mx,
                n_before,
            )
            return candidates_df

    logger.info(
        "[Step3 메뉴·의미] %r | %d→%d건 | max_sim=%.3f | %s",
        _log_preview(phrase, 80),
        n_before,
        len(out),
        mx,
        tier,
    )
    return out


_BROAD_LOCATION_TOKENS: frozenset[str] = frozenset(
    {
        "서울특별시",
        "서울시",
        "서울",
        "경기도",
        "인천광역시",
        "인천",
        "강남구",
        "서초구",
        "송파구",
        "마포구",
        "용산구",
        "중구",
        "종로구",
        "영등포구",
        "성동구",
        "광진구",
        "동대문구",
        "성북구",
        "은평구",
        "강서구",
        "양천구",
        "구로구",
        "금천구",
        "노원구",
        "도봉구",
        "강동구",
        "관악구",
        "서대문구",
        "부산광역시",
        "부산",
    }
)


def _location_tokens_for_filter(locations: list[str]) -> list[str]:
    """
    구·시 전체 같은 **너무 넓은** 토큰만 있으면 그대로 쓰고,
    압구정·신사동 등 **좁은** 토큰이 하나라도 있으면 넓은 토큰은 필터에서 제외한다.
    (주소에 '강남구'만 걸려 전역이 되는 완화를 막음)
    """
    ordered = [(x or "").strip() for x in locations if (x or "").strip()]
    if not ordered:
        return []
    narrow = [t for t in ordered if t not in _BROAD_LOCATION_TOKENS]
    return narrow if narrow else ordered


def _filter_candidates_by_locations(
    candidates_df: pd.DataFrame,
    locations: list[str],
) -> pd.DataFrame:
    """주소·가게명 중 하나라도 지역 토큰을 포함하면 통과(여러 지역은 OR)."""
    if not locations or candidates_df.empty:
        return candidates_df
    has_addr = "address" in candidates_df.columns
    has_shop = "shop_name" in candidates_df.columns
    if not has_addr and not has_shop:
        return candidates_df
    mask = pd.Series(False, index=candidates_df.index, dtype=bool)
    for loc in locations:
        loc_s = (loc or "").strip()
        if not loc_s:
            continue
        if has_addr:
            mask |= candidates_df["address"].fillna("").astype(str).str.contains(
                loc_s, case=False, regex=False, na=False
            )
        if has_shop:
            mask |= candidates_df["shop_name"].fillna("").astype(str).str.contains(
                loc_s, case=False, regex=False, na=False
            )
    return candidates_df.loc[mask].copy()


def _lgbm_predict_feature_matrix(
    ranker: Any,
    index: pd.Index,
    *,
    shop_popularity: pd.Series,
    semantic_score: np.ndarray,
    pop_x_semantic: pd.Series,
    query_length: int,
    faiss_score: pd.Series,
    meta_text_length: pd.Series,
) -> Any:
    """
    joblib로 로드한 LGBM sklearn 추정기의 predict 입력을 학습 시와 동일한 피처 수·이름에 맞춘다.
    """
    ql = pd.Series(float(query_length), index=index, dtype=float)
    sem = pd.Series(np.asarray(semantic_score, dtype=float).reshape(-1), index=index, dtype=float)

    pool: dict[str, pd.Series] = {
        "shop_popularity": shop_popularity.astype(float),
        "semantic_score": sem,
        "pop_x_semantic": pop_x_semantic.astype(float),
        "query_length": ql,
        "faiss_score": faiss_score.astype(float),
        "_faiss_score": faiss_score.astype(float),
        "meta_text_length": meta_text_length.astype(float),
        "meta_length": meta_text_length.astype(float),
    }

    names_attr = getattr(ranker, "feature_names_in_", None)
    if names_attr is not None and len(names_attr) > 0:
        name_list = [str(c) for c in list(names_attr)]
        cols: dict[str, pd.Series] = {}
        for c in name_list:
            if c in pool:
                cols[c] = pool[c]
            else:
                logger.warning(
                    "LightGBM 학습 피처 '%s'에 대응하는 런타임 피처가 없어 0으로 채웁니다.",
                    c,
                )
                cols[c] = pd.Series(0.0, index=index, dtype=float)
        return pd.DataFrame(cols, index=index)[name_list]

    # 이름 정보 없음(Column_0 등): 학습이 (pop, sem, pop*sem, query_len, faiss) 순이라고 가정
    default_order = ("shop_popularity", "semantic_score", "pop_x_semantic", "query_length", "faiss_score")
    return np.column_stack([pool[k].to_numpy(dtype=np.float64, copy=False) for k in default_order])


def lightgbm_scoring(
    query_text: str,
    candidates_df: pd.DataFrame,
    tfidf: Any,
    ranker: Any,
) -> pd.DataFrame:
    """Step 4: LightGBM 행동 기반 스코어링"""
    if candidates_df.empty:
        logger.info("[Step4 LightGBM] 0건 → 스킵")
        return candidates_df

    logger.info("[Step4 LightGBM] 입력 %d건 | q=%r", len(candidates_df), _log_preview(query_text, 100))

    # 1. TF-IDF를 이용한 시만틱 점수(의미적 유사도) 실시간 계산
    q_vec = tfidf.transform([query_text])
    m_vecs = tfidf.transform(candidates_df["meta_text"].fillna(""))
    semantic_score = cosine_similarity(q_vec, m_vecs).flatten()

    # 2. 피처 생성 (shops_metadata에 shop_popularity가 없다면 기본값 0 처리)
    if "shop_popularity" in candidates_df.columns:
        pop = pd.to_numeric(candidates_df["shop_popularity"], errors="coerce").fillna(0)
    else:
        pop = pd.Series(0.0, index=candidates_df.index, dtype=float)
    pop_x_semantic = pop * semantic_score
    query_length = len(query_text)

    if "_faiss_score" in candidates_df.columns:
        faiss_feat = pd.to_numeric(candidates_df["_faiss_score"], errors="coerce").fillna(0.0)
    else:
        faiss_feat = pd.Series(0.0, index=candidates_df.index, dtype=float)

    if "meta_text" in candidates_df.columns:
        meta_text_length = candidates_df["meta_text"].fillna("").astype(str).str.len().astype(float)
    else:
        meta_text_length = pd.Series(0.0, index=candidates_df.index, dtype=float)

    X = _lgbm_predict_feature_matrix(
        ranker,
        candidates_df.index,
        shop_popularity=pop,
        semantic_score=semantic_score,
        pop_x_semantic=pop_x_semantic,
        query_length=query_length,
        faiss_score=faiss_feat,
        meta_text_length=meta_text_length,
    )

    # 3. LightGBM 순위 예측
    candidates_df["lgbm_score"] = ranker.predict(X)
    lgbm_arr = candidates_df["lgbm_score"].to_numpy(dtype=float, copy=False)
    sem_arr = semantic_score
    logger.info(
        "[Step4 LightGBM] sem[%.3f~%.3f] lgbm[%.3f~%.3f]",
        float(np.min(sem_arr)),
        float(np.max(sem_arr)),
        float(np.min(lgbm_arr)),
        float(np.max(lgbm_arr)),
    )
    _log_df_head_shops(
        candidates_df.sort_values("lgbm_score", ascending=False).head(3),
        "[Step4 LGBM 상위]",
    )
    logger.info("[Step4 LightGBM] 완료 %d건", len(candidates_df))
    return candidates_df


def cross_encoder_rerank_with_fusion(
    query_text: str,
    candidates_df: pd.DataFrame,
    cross_encoder: CrossEncoder,
) -> pd.DataFrame:
    """Step 5: Cross-Encoder 평가 및 Late Fusion (LGBM 0.7 + CE 0.3)"""
    if candidates_df.empty:
        logger.info("[Step5 CE+Fusion] 입력 0건 → 스킵")
        return candidates_df

    logger.info("[Step5 CE+Fusion] 입력 %d건 | q=%r", len(candidates_df), _log_preview(query_text, 100))

    candidates_df = candidates_df.copy()

    pairs = [
        [query_text, str(row["meta_text"]) if pd.notna(row["meta_text"]) else ""]
        for _, row in candidates_df.iterrows()
    ]
    candidates_df["ce_score"] = cross_encoder.predict(pairs, show_progress_bar=False)

    if len(candidates_df) > 1:
        scaler_lgbm = MinMaxScaler()
        scaler_ce = MinMaxScaler()
        lgbm_scaled = scaler_lgbm.fit_transform(candidates_df["lgbm_score"].values.reshape(-1, 1)).flatten()
        ce_scaled = scaler_ce.fit_transform(candidates_df["ce_score"].values.reshape(-1, 1)).flatten()
    else:
        lgbm_scaled = np.array([1.0])
        ce_scaled = np.array([1.0])

    candidates_df["final_score"] = (lgbm_scaled * 0.7) + (ce_scaled * 0.3)
    sorted_df = candidates_df.sort_values("final_score", ascending=False)
    ce_raw = candidates_df["ce_score"].to_numpy(dtype=float, copy=False)
    _log_df_head_shops(sorted_df.head(3), "[Step5 final 상위]")
    logger.info(
        "[Step5 CE+Fusion] 완료 %d건 | ce[%.3f~%.3f] (0.7·lgbm+0.3·ce)",
        len(sorted_df),
        float(np.min(ce_raw)),
        float(np.max(ce_raw)),
    )
    return sorted_df


def _safe_context_value(row: pd.Series, col: str) -> str:
    if col not in row or pd.isna(row[col]):
        return ""
    v = row[col]
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _score_for_display(row: pd.Series) -> float:
    if "final_score" in row.index and pd.notna(row.get("final_score")):
        return float(row.get("final_score", 0.0))
    if "base_popularity" in row.index and pd.notna(row.get("base_popularity")):
        return float(pd.to_numeric(row.get("base_popularity"), errors="coerce") or 0.0)
    return 0.0


def _format_search_focus_for_generation(
    intent: RestaurantIntent,
    query_text: str,
    current_message: str,
) -> str:
    """답변 LLM이 이전 턴 표현에 끌리지 않도록, 이번 검색에 맞춘 의도를 고정해 넣는다."""
    locs = ", ".join(intent.locations) if intent.locations else "(지역 미명시 — 질문·기록에서 해석)"
    menu = (intent.menu or "").strip() or "(특정 메뉴 조건 없음)"
    sq = (intent.search_query or "").strip() or query_text
    return (
        f"- 확장 지역: {locs}\n"
        f"- 메뉴/키워드: {menu}\n"
        f"- 검색·분위기 요약: {sq}\n"
        f"- 사용자 원문(현재 턴): {current_message.strip()}"
    )


def _build_context_from_top(df_top: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, r in df_top.iterrows():
        shop_name = _safe_context_value(r, "shop_name")
        categories = _safe_context_value(r, "categories")
        address = _safe_context_value(r, "address")
        intents = _safe_context_value(r, "search_intents")
        facilities = _safe_context_value(r, "facilities")
        awards = _safe_context_value(r, "awards")
        score = _score_for_display(r)

        menus = _safe_context_value(r, "menus")
        food_line = categories
        if menus:
            food_line = f"{categories}" + (f" / {menus}" if categories else menus)

        traits: list[str] = []
        if intents:
            traits.append(f"검색의도: {intents}")
        if facilities:
            traits.append(f"편의: {facilities}")
        if awards:
            traits.append(f"수상·선정: {awards}")
        trait_str = " | ".join(traits) if traits else "(특징 정보 없음)"

        lines.append(
            "\n".join(
                [
                    f"이름: {shop_name}",
                    f"- 음식: {food_line or '(미기재)'}",
                    f"- 위치: {address or '(미기재)'}",
                    f"- 특징: {trait_str}",
                    f"- 점수(참고): {score}",
                ]
            )
        )
    return "\n\n".join(lines)


# =========================
# Response Generation Chain
# =========================


_RESPONSE_SYSTEM_PROMPT = """
너는 식당 추천 전용 챗봇이다.

[현재 질문이 전부다]
- 답변의 **첫 문장·도입부**는 반드시 **현재 질문**과 아래 **검색 의도(search_focus)**에 맞춘다.
- [대화 기록]은 참고만 한다. **현재 질문**에 강남역이 나왔는데 이전 턴에 압구정·파스타가 있었다고, 압구정·파스타 위주로 말하지 마라.
- 이전 턴 문장을 복사해 반복하지 마라.

[데이터 및 환각 방지]
- 오직 아래 "식당 정보"(컨텍스트) 블록의 내용만 근거로 삼는다.
- 웹 검색·외부 지식·추측으로 식당을 만들지 않는다.

[답변 형식 — 반드시 준수]
- 맨 위에 search_focus에 맞는 **한 줄 도입**(질문·지역 인지)만 쓴다. 긴 줄글 서술은 금지.
- 추천 식당은 **최대 3곳**. 각 식당마다 반드시 이 순서와 라벨을 지킨다 (줄글로 한 덩어리 쓰지 마라).
- 식당 블록이 여러 개면 블록 **사이에 실제 빈 줄**을 한 줄 넣는다.
- **절대 출력 금지**: "(빈 줄)", "빈 줄", 괄호로 된 줄바꿈 설명 등 메타 문구. 줄바꿈만으로 구분한다.

형식 예시(백틱·코드블록 없이 그대로 출력):
1. 식당이름
- 음식: 컨텍스트의 카테고리·메뉴만 요약
- 위치: 컨텍스트의 주소 전체
- 특징: 컨텍스트의 검색의도·편의·수상 등 1~2문장으로만

2. 식당이름
- 음식: ...
- 위치: ...
- 특징: ...

3. 식당이름
...

- 번호는 반드시 1. 2. 3. 처럼 숫자+점+공백으로 시작하는 한 줄에 식당명을 둔다.
- 그 다음 줄부터는 반드시 "- 음식:", "- 위치:", "- 특징:" 세 줄(값이 없으면 "정보 없음").
- 컨텍스트에 없는 메뉴·주소·특징은 쓰지 않는다.

[컨텍스트와 질문의 관계]
- 소개하는 식당은 **오직** "식당 정보" 컨텍스트에 나온 곳만이다.
- 컨텍스트 식당의 카테고리·메뉴가 질문 메뉴와 다르더라도, **먼저** 컨텍스트에 적힌 음식·위치를 그대로 전달한다.
- 도입은 한 줄로 짧게 짚은 뒤 바로 1. 추천으로 넘어간다.
- **금지 문구·톤**: "추천해드리기 어렵다", "양해 부탁", "대신 인근", "다른 맛집을 소개" 등 사과·회피·대체 제안으로 한두 문단 늘어놓지 마라. 컨텍스트에 나온 식당이면 그 사실만으로 소개한다.

[대화 유도]
- 컨텍스트가 부족하면 필요한 조건을 한두 가지 정중히 되묻는다.
""".strip()


def build_response_chain(model_name: str) -> Any:
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.35,
        max_tokens=650,
        timeout=60.0,
    )

    # system 문은 from_template를 거치지 않게 한다. (중괄호·한글 설명이 변수로 오인되지 않도록)
    human_t = HumanMessagePromptTemplate.from_template(
        "검색 의도(이번 답변은 반드시 이에 맞출 것):\n{search_focus}\n\n"
        "대화 기록:\n{history}\n\n"
        "현재 질문: {message}\n\n"
        "식당 정보:\n{context}",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=_RESPONSE_SYSTEM_PROMPT),
            human_t,
        ]
    )

    return prompt | llm


# =========================
# Startup: env, data, chains, retrieval singletons
# =========================


@app.on_event("startup")
def on_startup() -> None:
    load_dotenv()
    _configure_intent_logging()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env를 확인해주세요.")

    model_intent = os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini")
    model_answer = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini")

    device = _resolve_torch_device()
    logger.info("Bi-Encoder device: %s", device)

    faiss_index = _load_faiss_index(FAISS_INDEX_PATH)
    retrieval_meta_df = _load_meta_dataframe(SHOPS_METADATA_PKL_PATH)
    bi_encoder = _load_bi_encoder(BI_ENCODER_MODEL_ID, device=device)
    cross_encoder = _load_cross_encoder(CROSS_ENCODER_MODEL_ID, device=device)

    logger.info("LightGBM 및 TF-IDF 모델 로딩 중...")
    try:
        app.state.lgbm_ranker = joblib.load(LGBM_RANKER_PATH)
        app.state.tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        logger.info("LightGBM 및 TF-IDF 모델 로딩 완료!")
    except Exception as e:
        logger.error(f"LightGBM 모델 로딩 실패: {e}")

    model_dim = _sentence_transformer_embedding_dim(bi_encoder)
    if int(faiss_index.d) != model_dim:
        raise RuntimeError(
            f"FAISS 인덱스 차원({faiss_index.d})과 Bi-Encoder 출력 차원({model_dim})이 일치하지 않습니다."
        )
    if int(faiss_index.ntotal) != len(retrieval_meta_df):
        logger.warning(
            "FAISS ntotal(%d)과 메타 행 수(%d)가 다릅니다. 인덱스·메타 정합성을 확인하세요.",
            int(faiss_index.ntotal),
            len(retrieval_meta_df),
        )

    app.state.faiss_index = faiss_index
    app.state.retrieval_meta_df = retrieval_meta_df
    app.state.bi_encoder = bi_encoder
    app.state.cross_encoder = cross_encoder
    app.state.intent_extractor = IntentExtractor(model_intent)
    app.state.response_chain = build_response_chain(model_answer)


# =========================
# API
# =========================


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message는 비어 있을 수 없습니다.")

    history_lines = [f"{m.role}: {m.content}" for m in req.history[-INTENT_HISTORY_TAIL:]]
    history_str = "\n".join(history_lines) if history_lines else "없음"

    logger.info(
        "[/chat] msg=%r | history=%d턴 | tail=%r",
        _log_preview(message, 160),
        len(req.history),
        _log_preview(history_str, 200),
    )

    extractor: IntentExtractor = app.state.intent_extractor

    # --- Step 1: 의도 추출 및 가드레일 (LLM Router) ---
    intent_input = f"[대화 기록]\n{history_str}\n\n[현재 입력]\n{message}"
    try:
        intent = await extractor.ainvoke(intent_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"의도 추출 실패: {e}") from e

    logger.info(
        "[Step1] 식당질의=%s loc=%s menu=%r sq=%r",
        intent.is_restaurant_query,
        intent.locations,
        intent.menu,
        _log_preview(intent.search_query or "", 80),
    )

    if not intent.is_restaurant_query:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("[Step1] 가드레일 종료 | %dms", latency_ms)
        return ChatResponse(
            answer=GUARDRAIL_NON_RESTAURANT_MESSAGE,
            intent=intent.model_dump(),
            top_shops=[],
            latency_ms=latency_ms,
        )

    # --- Step 1: 검색어 정제 ---
    query_text = (intent.search_query or intent.menu or message or "").strip()
    ranking_query = _retrieval_ranking_query(intent, message)

    menu_for_hard_filter = (intent.menu or "").strip()
    idx_cap = int(app.state.faiss_index.ntotal)
    retrieval_k = RETRIEVAL_TOP_K
    if intent.locations or menu_for_hard_filter:
        # 하드 필터 전에 후보가 너무 적으면(상위 K만 보면) 조건을 동시에 만족하는 행이 0건이 될 수 있음
        retrieval_k = min(idx_cap, max(RETRIEVAL_TOP_K, min(400, idx_cap)))

    logger.info(
        "[검색] q=%r | rank_q=%r | menu필터=%s | k=%d(기본%d) ntotal=%d CE≤%d",
        _log_preview(query_text, 100),
        _log_preview(ranking_query, 140),
        menu_for_hard_filter or "—",
        retrieval_k,
        RETRIEVAL_TOP_K,
        idx_cap,
        CROSS_ENCODER_MAX_INPUT,
    )

    # --- Step 2: 1-Stage 의미 검색 (FAISS) ---
    candidates_df = await asyncio.to_thread(
        faiss_retrieve_top_k,
        query_text=ranking_query,
        bi_encoder=app.state.bi_encoder,
        faiss_index=app.state.faiss_index,
        meta_df=app.state.retrieval_meta_df,
        top_k=retrieval_k,
    )

    # --- Step 3: 필수 조건 하드 필터링 (지역 & 메뉴) ---
    n_after_faiss = len(candidates_df)
    if intent.locations:
        n_before_loc = len(candidates_df)
        loc_tokens = _location_tokens_for_filter(intent.locations)
        candidates_df = _filter_candidates_by_locations(candidates_df, loc_tokens)
        logger.info(
            "[Step3 지역] raw=%s use=%s | %d→%d건",
            intent.locations,
            loc_tokens,
            n_before_loc,
            len(candidates_df),
        )
    else:
        logger.info("[Step3 지역] 생략 | FAISS %d건", n_after_faiss)

    if menu_for_hard_filter:
        n_before_menu = len(candidates_df)
        candidates_df = await asyncio.to_thread(
            filter_candidates_by_menu_semantic,
            candidates_df,
            menu_for_hard_filter,
            app.state.bi_encoder,
        )
        logger.info("[Step3 메뉴] %d→%d건", n_before_menu, len(candidates_df))
    else:
        logger.info("[Step3 메뉴] 생략")

    if candidates_df.empty:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "[조기종료] Step3 후 0건 | loc=%s menu=%r | %dms",
            intent.locations,
            menu_for_hard_filter,
            latency_ms,
        )
        return ChatResponse(
            answer="요청하신 조건(지역/메뉴)에 딱 맞는 식당을 찾지 못했어요. 조건을 조금 넓혀서 다시 검색해 주시겠어요?",
            intent=intent.model_dump(),
            top_shops=[],
            latency_ms=latency_ms,
        )

    # --- Step 4: 2-Stage 행동 기반 랭킹 (LightGBM) ---
    tfidf_vec = getattr(app.state, "tfidf_vectorizer", None)
    ranker_m = getattr(app.state, "lgbm_ranker", None)
    if tfidf_vec is None or ranker_m is None:
        raise HTTPException(
            status_code=503,
            detail="LightGBM 또는 TF-IDF 모델이 로드되지 않았습니다. data 경로와 서버 기동 로그를 확인해 주세요.",
        )

    candidates_df = await asyncio.to_thread(
        lightgbm_scoring,
        ranking_query,
        candidates_df,
        tfidf_vec,
        ranker_m,
    )

    # Cross-Encoder 부하를 줄이기 위해 LightGBM 점수 기준 상위 20개만 컷
    n_before_ce_cut = len(candidates_df)
    candidates_df = candidates_df.nlargest(CROSS_ENCODER_MAX_INPUT, "lgbm_score").copy()
    logger.info("[Step4→5] CE입력 %d→%d건(상위%d)", n_before_ce_cut, len(candidates_df), CROSS_ENCODER_MAX_INPUT)

    # --- Step 5: 3-Stage 최종 검증 및 융합 (Late Fusion) ---
    df_top = await asyncio.to_thread(
        cross_encoder_rerank_with_fusion,
        ranking_query,
        candidates_df,
        app.state.cross_encoder,
    )

    # 최종 Top 3 확정
    df_top = df_top.head(3)

    if df_top.empty:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("[조기종료] Step5 Top3 비움 | %dms", latency_ms)
        return ChatResponse(
            answer="조건에 맞는 식당을 찾지 못했습니다. 다른 지역이나 메뉴로 말씀해 주시겠어요?",
            intent=intent.model_dump(),
            top_shops=[],
            latency_ms=latency_ms,
        )

    rank_parts: list[str] = []
    for rank, (_, row) in enumerate(df_top.iterrows(), start=1):
        shop = _safe_context_value(row, "shop_name") or "(이름 없음)"
        fs = float(row.get("final_score", 0.0))
        rank_parts.append(f"{rank}:{_log_preview(shop, 28)}(T={fs:.3f})")
    logger.info("[Step5 Top3] %s", " ".join(rank_parts) if rank_parts else "(없음)")

    # --- Step 6: 자연어 응답 생성 ---
    context = _build_context_from_top(df_top)
    search_focus = _format_search_focus_for_generation(intent, ranking_query, message)
    logger.info("[Step6] LLM | ctx=%d자 | focus=%r", len(context), _log_preview(search_focus, 120))
    try:
        msg = await app.state.response_chain.ainvoke(
            {
                "search_focus": search_focus,
                "history": history_str,
                "message": message,
                "context": context,
            }
        )
        answer = _sanitize_chat_answer(getattr(msg, "content", None) or str(msg))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {e}") from e

    logger.info("[Step6] 완료 | 답변 %d자 %r", len(answer or ""), _log_preview(answer or "", 120))

    top_shops: list[dict[str, Any]] = []
    for _, r in df_top.iterrows():
        top_shops.append(
            {
                "shop_name": _safe_context_value(r, "shop_name"),
                "categories": _safe_context_value(r, "categories"),
                "address": _safe_context_value(r, "address"),
                "final_score": _score_for_display(r),
            }
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("[/chat 완료] %dms | 추천 %d곳", latency_ms, len(top_shops))
    return ChatResponse(
        answer=(answer or "").strip(),
        intent=intent.model_dump(),
        top_shops=top_shops,
        latency_ms=latency_ms,
    )


@app.get("/health")
def health() -> dict[str, str]:
    ok = hasattr(app.state, "faiss_index") and app.state.faiss_index is not None
    return {"status": "ok" if ok else "not_ready"}
