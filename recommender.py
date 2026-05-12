"""식당 추천 엔진: TF-IDF 의미 유사도 + LightGBM 랭커 (플랫 구조, 루트 배치)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_EVENT_TYPES = ("click", "bookmark", "reservation")
_RANKER_FEATURES: tuple[str, ...] = (
    "shop_popularity",
    "semantic_score",
    "semantic_sq",
    "pop_x_semantic",
    "query_length",
)


def _to_json_safe_float(x: Any) -> float:
    if isinstance(x, (np.floating, float)):
        v = float(x)
    elif isinstance(x, (np.integer, int)):
        v = float(int(x))
    else:
        v = float(x)
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return v


def _to_json_safe_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        if np.isnan(x):
            return ""
        s = str(float(x))
        if s.endswith(".0") and s[:-2].isdigit():
            return s[:-2]
        return s
    return str(x).strip()


class RestaurantRecommender:
    """LightGBM 랭커 + TF-IDF 기반 실시간 식당 추천."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._data_dir = Path(data_dir).expanduser().resolve()
        self._assert_data_dir()

        ranker_path = self._data_dir / "lgbm_ranker_v2.pkl"
        tfidf_path = self._data_dir / "tfidf_vectorizer_v2.pkl"
        shops_path = self._data_dir / "shops_augmented_v2.csv"
        logs_path = self._data_dir / "logs.csv"

        for p in (ranker_path, tfidf_path, shops_path, logs_path):
            if not p.is_file():
                raise FileNotFoundError(f"필수 데이터 파일이 없습니다: {p}")

        self.ranker = joblib.load(ranker_path)
        self.tfidf = joblib.load(tfidf_path)
        shops_df = pd.read_csv(shops_path, dtype={"shop_id": str})
        logs_df = pd.read_csv(logs_path, dtype={"shop_id": str})

        self._validate_frames(shops_df, logs_df)

        shops_df = self._merge_popularity(shops_df, logs_df)
        shops_df["meta_text"] = self._build_meta_text(shops_df)

        self.shops_df = shops_df.reset_index(drop=True)
        self.shop_vecs = self.tfidf.transform(self.shops_df["meta_text"])

        logger.info(
            "RestaurantRecommender 초기화 완료: shops=%d, shop_vecs=%s",
            len(self.shops_df),
            type(self.shop_vecs).__name__,
        )

    def _assert_data_dir(self) -> None:
        if not self._data_dir.is_dir():
            raise NotADirectoryError(f"data_dir 이 디렉터리가 아닙니다: {self._data_dir}")

    @staticmethod
    def _validate_frames(shops_df: pd.DataFrame, logs_df: pd.DataFrame) -> None:
        required_shops = {
            "shop_id",
            "shop_name",
            "categories",
            "menus",
            "llm_situation_tags",
            "address",
        }
        missing_s = required_shops - set(shops_df.columns)
        if missing_s:
            raise ValueError(f"shops CSV 에 필요한 컬럼이 없습니다: {sorted(missing_s)}")

        for col in ("shop_id", "event_type"):
            if col not in logs_df.columns:
                raise ValueError(f"logs CSV 에 '{col}' 컬럼이 필요합니다.")

    @staticmethod
    def _merge_popularity(shops_df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
        mask = logs_df["event_type"].isin(_EVENT_TYPES)
        filtered = logs_df.loc[mask]
        counts = filtered.groupby("shop_id", observed=False).size().astype(np.float64)
        counts = counts.rename("shop_popularity")
        shop_popularity = np.log1p(counts)

        out = shops_df.merge(shop_popularity, on="shop_id", how="left")
        out["shop_popularity"] = out["shop_popularity"].fillna(0.0).astype(np.float64)
        return out

    @staticmethod
    def _build_meta_text(shops_df: pd.DataFrame) -> pd.Series:
        def col(series: pd.Series) -> pd.Series:
            return series.fillna("").astype(str).str.strip()

        shop_name = col(shops_df["shop_name"])
        categories = col(shops_df["categories"])
        menus = col(shops_df["menus"])
        llm_situation_tags = col(shops_df["llm_situation_tags"])

        cat_twice = (categories + " ") + (categories + " ")
        meta = shop_name + " " + cat_twice + menus + " " + llm_situation_tags
        return meta.str.strip()

    def recommend(
        self,
        query: str,
        required_location: str = "",
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """검색어 기준 상위 식당 추천. 의미 점수 하드 필터 후 선택적 지역 하드 필터, 랭커 정렬."""
        if top_n < 1:
            return []

        q = (query or "").strip()
        if not q:
            return []

        q_vec = self.tfidf.transform([q])
        sim = cosine_similarity(q_vec, self.shop_vecs)
        semantic_all = np.asarray(sim, dtype=np.float64).ravel()

        keep = np.flatnonzero(semantic_all >= 0.05)
        if keep.size == 0:
            return []

        cand = self.shops_df.iloc[keep].copy()
        sem = semantic_all[keep].astype(np.float64, copy=False)

        cand["semantic_score"] = sem

        loc_raw = (required_location or "").strip()
        if loc_raw:
            loc_keyword = loc_raw.replace("역", "").replace("동", "").strip()
            if loc_keyword:
                addr = cand["address"].fillna("").astype(str)
                name = cand["shop_name"].fillna("").astype(str)
                loc_mask = addr.str.contains(
                    loc_keyword, na=False, regex=False
                ) | name.str.contains(loc_keyword, na=False, regex=False)
                cand = cand.loc[loc_mask]
                if cand.empty:
                    return []
        cand["semantic_sq"] = cand["semantic_score"].to_numpy(dtype=np.float64) ** 2
        cand["pop_x_semantic"] = (
            cand["shop_popularity"].to_numpy(dtype=np.float64)
            * cand["semantic_score"].to_numpy(dtype=np.float64)
        )
        cand["query_length"] = float(len(q))

        # 학습 시 DataFrame 컬럼명으로 피팅된 LGBMRanker는 ndarray 전달 시 경고가 난다.
        X = cand.loc[:, list(_RANKER_FEATURES)].astype(np.float64, copy=False)
        pred = np.asarray(self.ranker.predict(X), dtype=np.float64).ravel()
        cand["pred_score"] = pred

        cand = cand.sort_values("pred_score", ascending=False).head(top_n)

        rows: list[dict[str, Any]] = []
        for _, row in cand.iterrows():
            rows.append(
                {
                    "shop_id": _to_json_safe_str(row["shop_id"]),
                    "shop_name": _to_json_safe_str(row["shop_name"]),
                    "categories": _to_json_safe_str(row["categories"]),
                    "menus": _to_json_safe_str(row["menus"]),
                    "llm_situation_tags": _to_json_safe_str(row["llm_situation_tags"]),
                    "address": _to_json_safe_str(row["address"]),
                    "semantic_score": _to_json_safe_float(row["semantic_score"]),
                    "pred_score": _to_json_safe_float(row["pred_score"]),
                }
            )
        return rows
