from __future__ import annotations

import math
import re
import string
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "player": "player_name",
    "player_name": "player_name",
    "name": "player_name",
    "playername": "player_name",
    "season": "auction_year",
    "year": "auction_year",
    "auction_season": "auction_year",
    "auction_year": "auction_year",
    "price": "price",
    "amount": "price",
    "sold_price": "price",
    "final_price": "price",
    "price_in_rs": "price",
    "price_in_": "price",
    "pricein_crore_indian_rupees": "price",
    "winning_bid_in_rs": "price",
    "baseprices_in_rs": "base_price",
    "base_price_in_rs": "base_price",
    "team": "team",
    "franchise": "team",
    "squad": "team",
    "teamname": "team",
    "role": "role",
    "player_role": "role",
    "type": "role",
    "batting_style": "batting_style",
    "bowling_style": "bowling_style",
    "nationality": "nationality",
    "country": "nationality",
    "overseas": "is_overseas",
    "wicket_keeper": "is_wicketkeeper",
    "wicketkeeper": "is_wicketkeeper",
    "keeper": "is_wicketkeeper",
    "all_rounder": "is_allrounder",
    "allrounder": "is_allrounder",
    "match_id": "match_id",
    "id": "match_id",
    "matchid": "match_id",
    "batsman": "batter_name",
    "batter": "batter_name",
    "striker": "batter_name",
    "bowler": "bowler_name",
    "non_striker": "non_striker_name",
    "total_runs": "total_runs",
    "batsman_runs": "batter_runs",
    "batter_runs": "batter_runs",
    "extra_runs": "extra_runs",
    "extras": "extra_runs",
    "is_wicket": "is_wicket",
    "wicket_type": "dismissal_kind",
    "dismissal_kind": "dismissal_kind",
    "inning": "inning",
    "innings": "inning",
    "ball": "ball",
    "over": "over",
    "batting_team": "batting_team",
    "bowling_team": "bowling_team",
    "runs_off_bat": "batter_runs",
    "iswide": "is_wide",
    "isnoball": "is_no_ball",
    "byes": "byes",
    "legbyes": "legbyes",
    "penalty": "penalty",
}

MANUAL_NAME_MAP = {
    "ravindra jadeja": "ravindra jadeja",
    "r jadeja": "ravindra jadeja",
    "yuvraj singh": "yuvraj singh",
    "a b de villiers": "ab de villiers",
    "ab de villiers": "ab de villiers",
    "d j bravo": "dwayne bravo",
    "dj bravo": "dwayne bravo",
    "m s dhoni": "ms dhoni",
    "m.s. dhoni": "ms dhoni",
    "r ashwin": "ravichandran ashwin",
    "ravichandran ashwin": "ravichandran ashwin",
    "v kohli": "virat kohli",
    "s dhawan": "shikhar dhawan",
    "j bumrah": "jasprit bumrah",
    "h pandya": "hardik pandya",
    "k pandya": "krunal pandya",
    "k l rahul": "kl rahul",
    "k.l. rahul": "kl rahul",
    "s iyer": "shreyas iyer",
    "ss iyer": "shreyas iyer",
    "rr pant": "rishabh pant",
    "jc buttler": "jos buttler",
    "jr hazlewood": "josh hazlewood",
    "jc archer": "jofra archer",
    "ta boult": "trent boult",
    "ma starc": "mitchell starc",
    "pj cummins": "pat cummins",
    "vr iyer": "venkatesh iyer",
    "wp saha": "wriddhiman saha",
    "jm sharma": "jitesh sharma",
    "q de kock": "quinton de kock",
    "qhdk": "quinton de kock",
}

MANUAL_IS_OVERSEAS_MAP = {
    "arshdeep singh": 0,
}

TEAM_NAME_MAP = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "rcb": "Royal Challengers Bengaluru",
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "dd": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "kxip": "Punjab Kings",
    "rising pune supergiants": "Rising Pune Supergiant",
    "rising pune supergiant": "Rising Pune Supergiant",
    "gujarat lions": "Gujarat Lions",
    "gujarat titans": "Gujarat Titans",
    "mumbai indians": "Mumbai Indians",
    "chennai super kings": "Chennai Super Kings",
    "kolkata knight riders": "Kolkata Knight Riders",
    "rajasthan royals": "Rajasthan Royals",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "deccan chargers": "Deccan Chargers",
    "kochi tuskers kerala": "Kochi Tuskers Kerala",
    "pune warriors": "Pune Warriors India",
    "pune warriors india": "Pune Warriors India",
    "lucknow super giants": "Lucknow Super Giants",
}

DISMISSALS_CREDITED_TO_BOWLER = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

NON_BOWLER_EXTRAS = {"byes", "legbyes", "leg_byes", "byes_runs"}

LEADERSHIP_PROXY_NAMES = {
    "ajinkya rahane",
    "axar patel",
    "hardik pandya",
    "kl rahul",
    "nitish rana",
    "pat cummins",
    "ravichandran ashwin",
    "rishabh pant",
    "ruturaj gaikwad",
    "sanju samson",
    "shikhar dhawan",
    "shreyas iyer",
    "shubman gill",
    "surya kumar yadav",
    "suryakumar yadav",
    "virat kohli",
    "ms dhoni",
    "faf du plessis",
}

CAPTAINCY_PROXY_NAMES = LEADERSHIP_PROXY_NAMES | {
    "david warner",
    "rohit sharma",
    "gautam gambhir",
    "steve smith",
    "kane williamson",
}

TITLE_WINNING_CAPTAIN_AUCTION_MAP = {
    (2018, "rohit sharma"),
    (2019, "ms dhoni"),
    (2020, "rohit sharma"),
    (2021, "rohit sharma"),
    (2022, "ms dhoni"),
    (2023, "hardik pandya"),
    (2024, "ms dhoni"),
    (2025, "shreyas iyer"),
}


def project_root_from(start: Path | str) -> Path:
    start_path = Path(start).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    return start_path


def ensure_directories(paths: Iterable[Path | str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def snake_case(text: object) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"[\/\-\s]+", "_", value)
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: COLUMN_ALIASES.get(snake_case(col), snake_case(col)) for col in df.columns}
    out = df.rename(columns=renamed).copy()
    out.columns = [snake_case(col) for col in out.columns]
    return out


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    existing = set(df.columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return series
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def standardize_season_year(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    matches = re.findall(r"(20\d{2})", text)
    if matches:
        return float(matches[0])
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) == 4 and digits.startswith("20"):
        return float(digits)
    return np.nan


def canonicalize_player_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    parts = text.split()
    if len(parts) == 2 and len(parts[0]) == 1:
        text = f"{parts[0]} {parts[1]}"
    text = MANUAL_NAME_MAP.get(text, text)
    return text


def canonicalize_team_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return TEAM_NAME_MAP.get(text, text.title())


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    den = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    result = num / den.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan)


def parse_price_to_inr(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        numeric = float(value)
        if numeric <= 0:
            return np.nan
        if numeric < 1_000:
            if numeric <= 30:
                return numeric * 10_000_000.0
            return numeric * 100_000.0
        return numeric
    text = str(value).strip().lower()
    if not text:
        return np.nan
    cleaned = text.replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "").strip()
    multiplier = 1.0
    if any(token in cleaned for token in ["crore", "cr", "crores"]):
        multiplier = 10_000_000.0
    elif any(token in cleaned for token in ["lakh", "lakhs", "lac", "l", "lk"]):
        multiplier = 100_000.0
    number_match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not number_match:
        return np.nan
    number = float(number_match.group(1))
    if multiplier == 1.0 and number < 5_000:
        return np.nan
    return number * multiplier


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "overseas": 1,
        "wk": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)


def infer_role_bucket(role: object) -> str:
    text = snake_case(role)
    if any(token in text for token in ["keeper", "wicket"]):
        return "wicketkeeper"
    if "all" in text and "round" in text:
        return "all_rounder"
    if any(token in text for token in ["bowl", "spinner", "pacer", "fast"]):
        return "bowler"
    if any(token in text for token in ["bat", "opener", "middle"]):
        return "batter"
    return "unknown"


def format_inr_crore(series: pd.Series) -> pd.Series:
    return series / 10_000_000.0


def describe_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": [name],
            "rows": [len(df)],
            "columns": [len(df.columns)],
            "null_share": [float(df.isna().mean().mean()) if not df.empty else np.nan],
            "sample_columns": [", ".join(map(str, df.columns[:8]))],
        }
    )


def quantile_labels(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(["fair"] * len(series), index=series.index)
    q20, q40, q60, q80 = valid.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    bins = [-math.inf, q20, q40, q60, q80, math.inf]
    labels = [
        "strongly_overvalued",
        "moderately_overvalued",
        "fair",
        "moderately_undervalued",
        "strongly_undervalued",
    ]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str)
