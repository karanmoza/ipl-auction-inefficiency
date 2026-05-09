from __future__ import annotations

from difflib import SequenceMatcher, get_close_matches
import re
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .utils import (
    MANUAL_IS_OVERSEAS_MAP,
    canonicalize_player_name,
    canonicalize_team_name,
    coerce_bool,
    first_existing,
    infer_role_bucket,
    normalize_columns,
    parse_price_to_inr,
    snake_case,
    standardize_season_year,
)


def standardize_auction_data(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    player_col = first_existing(out, ["player_name"])
    if player_col:
        out["player_name"] = out[player_col]
    year_col = first_existing(out, ["auction_year"])
    if year_col:
        out["auction_year"] = out[year_col].map(standardize_season_year)
    elif "source_file" in out.columns:
        out["auction_year"] = out["source_file"].astype(str).str.extract(r"(20\d{2})")[0].map(standardize_season_year)
    else:
        out["auction_year"] = np.nan
    price_col = first_existing(out, ["price"])
    if price_col:
        out["price_in_inr"] = out[price_col].map(parse_price_to_inr)
    else:
        out["price_in_inr"] = np.nan
    out["price_in_crore"] = out["price_in_inr"] / 10_000_000.0
    out["player_name"] = out.get("player_name", pd.Series(index=out.index, dtype=object)).astype(str)
    out["player_name_clean"] = out["player_name"].map(canonicalize_player_name)
    if "role" in out.columns:
        out["role_bucket"] = out["role"].map(infer_role_bucket)
    else:
        out["role_bucket"] = "unknown"
    if "nationality" in out.columns:
        nationality = out["nationality"].astype(str).str.lower()
        out["is_overseas"] = (~nationality.str.contains("india")).astype(int)
    elif "is_overseas" in out.columns:
        out["is_overseas"] = coerce_bool(out["is_overseas"])
    else:
        out["is_overseas"] = 0
    manual_overseas = out["player_name_clean"].map(MANUAL_IS_OVERSEAS_MAP)
    out["is_overseas"] = manual_overseas.fillna(out["is_overseas"]).astype(int)
    if "nationality" in out.columns:
        out.loc[manual_overseas.eq(0), "nationality"] = "Indian"
        out.loc[manual_overseas.eq(1), "nationality"] = "Overseas"
    for flag_col in ["is_wicketkeeper", "is_allrounder"]:
        if flag_col in out.columns:
            out[flag_col] = coerce_bool(out[flag_col])
        else:
            inferred = out["role_bucket"].eq("wicketkeeper" if flag_col == "is_wicketkeeper" else "all_rounder")
            out[flag_col] = inferred.astype(int)
    if "team" not in out.columns:
        out["team"] = np.nan
    out["team"] = out["team"].map(canonicalize_team_name)
    return out


def standardize_matches_data(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    if "season" in out.columns and "match_season" not in out.columns:
        out["match_season"] = out["season"].map(standardize_season_year)
    elif "auction_year" in out.columns:
        out["match_season"] = out["auction_year"].map(standardize_season_year)
    else:
        season_like = first_existing(out, ["match_year", "year"])
        out["match_season"] = out[season_like].map(standardize_season_year) if season_like else np.nan
    if "match_id" not in out.columns:
        out["match_id"] = np.arange(1, len(out) + 1)
    for column in ["team1", "team2", "winner", "toss_winner"]:
        if column in out.columns:
            out[column] = out[column].map(canonicalize_team_name)
    return out


def standardize_deliveries_data(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    rename_fallbacks = {
        "batter_name": ["batsman", "batter"],
        "bowler_name": ["bowler"],
        "non_striker_name": ["non_striker"],
        "batter_runs": ["batsman_runs"],
        "extra_runs": ["extras"],
    }
    for canonical, aliases in rename_fallbacks.items():
        if canonical not in out.columns:
            for alias in aliases:
                if alias in out.columns:
                    out[canonical] = out[alias]
                    break
    for column in ["is_wide", "is_no_ball", "byes", "legbyes", "penalty"]:
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
    if "extra_runs" not in out.columns:
        out["extra_runs"] = out[["is_wide", "is_no_ball", "byes", "legbyes", "penalty"]].sum(axis=1)
    if "batter_runs" not in out.columns:
        out["batter_runs"] = 0
    if "total_runs" not in out.columns:
        out["total_runs"] = out["batter_runs"] + out["extra_runs"]
    if "is_wicket" not in out.columns:
        dismissal_source = out["dismissal_kind"].astype(str) if "dismissal_kind" in out.columns else ""
        out["is_wicket"] = dismissal_source.ne("").astype(int)
    if "extra_type" not in out.columns:
        out["extra_type"] = ""
        out.loc[out["is_wide"].gt(0), "extra_type"] = "wides"
        out.loc[out["is_no_ball"].gt(0), "extra_type"] = "noballs"
        out.loc[out["byes"].gt(0), "extra_type"] = "byes"
        out.loc[out["legbyes"].gt(0), "extra_type"] = "legbyes"
    numeric_defaults = ["over", "ball", "inning", "batter_runs", "extra_runs", "total_runs", "is_wicket"]
    for column in numeric_defaults:
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
    if "match_id" not in out.columns:
        out["match_id"] = np.nan
    for column in ["batter_name", "bowler_name", "non_striker_name"]:
        if column not in out.columns:
            out[column] = ""
        out[f"{column}_clean"] = out[column].map(canonicalize_player_name)
    if "dismissal_kind" not in out.columns:
        out["dismissal_kind"] = ""
    return out


def combine_auction_sources(auction_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    standardized = [standardize_auction_data(frame) for frame in auction_frames if frame is not None and not frame.empty]
    if not standardized:
        return pd.DataFrame()
    out = pd.concat(standardized, ignore_index=True, sort=False)
    out["source_priority"] = out["source_file"].astype(str).str.contains("2013|2026|expanded", case=False, na=False).map(
        {True: 0, False: 1}
    )
    out = out.sort_values(["source_priority", "player_name_clean", "auction_year", "team"]).drop_duplicates(
        subset=["player_name_clean", "auction_year", "team"], keep="first"
    )
    return out.drop(columns=["source_priority"])


def fuzzy_match_name(name: str, candidate_names: list[str]) -> Tuple[str, float]:
    if not name or not candidate_names:
        return "", 0.0
    close = get_close_matches(name, candidate_names, n=1, cutoff=0.0)
    if not close:
        return "", 0.0
    candidate = close[0]
    confidence = SequenceMatcher(None, name, candidate).ratio()
    return candidate, float(confidence)


def _name_parts(name: str) -> list[str]:
    return [part for part in str(name).split() if part]


def _surname(name: str) -> str:
    parts = _name_parts(name)
    return parts[-1] if parts else ""


def _first_initial(name: str) -> str:
    parts = _name_parts(name)
    return parts[0][0] if parts else ""


def initials_signature_match(name: str, candidate_names: list[str]) -> Tuple[str, float]:
    if not name:
        return "", 0.0
    target_parts = _name_parts(name)
    if not target_parts:
        return "", 0.0
    target_surname = _surname(name)
    target_first_initial = _first_initial(name)
    candidates = []
    for candidate in candidate_names:
        candidate_parts = _name_parts(candidate)
        if not candidate_parts:
            continue
        if _surname(candidate) != target_surname:
            continue
        if _first_initial(candidate) != target_first_initial:
            continue
        candidates.append(candidate)
    if not candidates:
        return "", 0.0
    scored = sorted(
        ((candidate, SequenceMatcher(None, name, candidate).ratio()) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    best_candidate, fuzzy_conf = scored[0]
    # Boost confidence when surname and initial pattern are both aligned.
    confidence = max(float(fuzzy_conf), 0.93)
    return best_candidate, confidence


def harmonize_auction_to_performance_names(
    auction_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    threshold: float = 0.88,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = auction_df.copy()
    out["matched_player_name_clean"] = out["player_name_clean"]
    out["match_confidence"] = np.where(out["player_name_clean"].ne(""), 1.0, 0.0)

    performance_names = sorted(performance_df["player_name_clean"].dropna().unique().tolist())
    missing_mask = ~out["player_name_clean"].isin(performance_names)
    reviews = []
    for idx in out.index[missing_mask]:
        candidate, confidence = initials_signature_match(out.at[idx, "player_name_clean"], performance_names)
        if not candidate:
            candidate, confidence = fuzzy_match_name(out.at[idx, "player_name_clean"], performance_names)
        if confidence >= threshold:
            out.at[idx, "matched_player_name_clean"] = candidate
            out.at[idx, "match_confidence"] = confidence
        reviews.append(
            {
                "player_name": out.at[idx, "player_name"],
                "player_name_clean": out.at[idx, "player_name_clean"],
                "suggested_match": candidate,
                "confidence": confidence,
                "accepted_automatically": confidence >= threshold,
            }
        )
    review_df = pd.DataFrame(reviews).sort_values("confidence", ascending=False) if reviews else pd.DataFrame()
    return out, review_df


def coverage_step(step: str, before_rows: int, after_rows: int) -> dict:
    return {
        "step": step,
        "rows_before": before_rows,
        "rows_after": after_rows,
        "retention_rate": after_rows / before_rows if before_rows else np.nan,
    }
