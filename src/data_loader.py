from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import describe_frame, normalize_columns, project_root_from


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def discover_raw_files(data_root: Path | str | None = None) -> pd.DataFrame:
    root = Path(data_root) if data_root else project_root_from(Path.cwd()) / "data" / "raw"
    records: List[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        records.append(
            {
                "path": str(path),
                "file_name": path.name,
                "suffix": path.suffix.lower(),
                "size_kb": round(path.stat().st_size / 1024, 1),
                "concept_guess": classify_file_name(path.name),
            }
        )
    return pd.DataFrame(records)


def classify_file_name(name: str) -> str:
    lowered = name.lower()
    if "auction" in lowered:
        return "auction"
    if any(token in lowered for token in ["deliver", "ball", "bpb"]):
        return "deliveries"
    if "match" in lowered:
        return "matches"
    return "unknown"


def load_table(path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, low_memory=False, nrows=nrows)
    else:
        df = pd.read_excel(file_path, nrows=nrows)
    return normalize_columns(df)


def profile_raw_files(files_df: pd.DataFrame) -> pd.DataFrame:
    profiles = []
    for row in files_df.to_dict("records"):
        try:
            sample = load_table(row["path"], nrows=50)
            summary = describe_frame(sample, row["file_name"]).assign(
                path=row["path"], concept_guess=guess_concept_from_columns(sample, row["concept_guess"])
            )
            profiles.append(summary)
        except Exception as exc:  # pragma: no cover - defensive profiling
            profiles.append(
                pd.DataFrame(
                    {
                        "dataset": [row["file_name"]],
                        "rows": [None],
                        "columns": [None],
                        "null_share": [None],
                        "sample_columns": [f"Failed to read: {exc}"],
                        "path": [row["path"]],
                        "concept_guess": [row["concept_guess"]],
                    }
                )
            )
    return pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()


def guess_concept_from_columns(df: pd.DataFrame, fallback: str = "unknown") -> str:
    cols = set(df.columns)
    if {"player_name", "price"}.issubset(cols) or {"player_name", "auction_year"}.issubset(cols):
        return "auction"
    if {"match_id", "batter_name", "bowler_name"}.issubset(cols):
        return "deliveries"
    if "match_id" in cols and any(col in cols for col in ["season", "city", "winner", "venue", "auction_year"]):
        return "matches"
    return fallback


def identify_concept_files(files_df: pd.DataFrame) -> Dict[str, list[str]]:
    concepts = {"auction": [], "matches": [], "deliveries": []}
    if files_df.empty:
        return concepts
    for row in files_df.to_dict("records"):
        try:
            sample = load_table(row["path"], nrows=30)
            concept = guess_concept_from_columns(sample, row["concept_guess"])
        except Exception:
            concept = row["concept_guess"]
        if concept in concepts:
            concepts[concept].append(row["path"])
    return concepts


def load_concept_tables(files_df: pd.DataFrame) -> Dict[str, list[pd.DataFrame]]:
    discovered = identify_concept_files(files_df)
    loaded: Dict[str, list[pd.DataFrame]] = {}
    for concept, paths in discovered.items():
        loaded[concept] = []
        for path in paths:
            df = load_table(path)
            df["source_file"] = Path(path).name
            loaded[concept].append(df)
    return loaded

