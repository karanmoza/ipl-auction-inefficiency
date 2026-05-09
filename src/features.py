from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    DISMISSALS_CREDITED_TO_BOWLER,
    NON_BOWLER_EXTRAS,
    canonicalize_player_name,
    canonicalize_team_name,
    safe_divide,
)


def build_match_season_lookup(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame(columns=["match_id", "match_season"])
    cols = [col for col in ["match_id", "match_season"] if col in matches_df.columns]
    return matches_df[cols].drop_duplicates()


def attach_season_to_deliveries(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    if deliveries_df.empty:
        return deliveries_df.copy()
    season_lookup = build_match_season_lookup(matches_df)
    out = deliveries_df.merge(season_lookup, on="match_id", how="left")
    return out


def _batting_base(deliveries: pd.DataFrame) -> pd.DataFrame:
    out = deliveries.copy()
    out["is_ball_faced"] = (~out["extra_type"].astype(str).str.lower().eq("wides")).astype(int)
    out["is_boundary_four"] = out["batter_runs"].eq(4).astype(int)
    out["is_boundary_six"] = out["batter_runs"].eq(6).astype(int)
    out["is_dot_ball_batting"] = out["batter_runs"].eq(0).astype(int) * out["is_ball_faced"]
    out["phase"] = np.select(
        [out["over"].between(1, 6), out["over"].between(7, 15), out["over"] >= 16],
        ["powerplay", "middle", "death"],
        default="other",
    )
    dismissal_text = out["player_dismissed"].astype(str) if "player_dismissed" in out.columns else pd.Series("", index=out.index)
    out["batter_dismissed"] = dismissal_text.map(canonicalize_player_name).eq(out["batter_name_clean"]).astype(int)
    out["inning_key"] = out["match_id"].astype(str) + "_" + out["inning"].astype(int).astype(str)
    return out


def engineer_batting_features(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    if deliveries_df.empty:
        return pd.DataFrame()
    batting = _batting_base(deliveries_df)
    grouped = batting.groupby(["match_season", "batter_name_clean"], dropna=False)
    summary = grouped.agg(
        matches_played=("match_id", "nunique"),
        innings_batted=("inning_key", "nunique"),
        runs_scored=("batter_runs", "sum"),
        balls_faced=("is_ball_faced", "sum"),
        boundary_runs=("batter_runs", lambda s: s[s.isin([4, 6])].sum()),
        fours=("is_boundary_four", "sum"),
        sixes=("is_boundary_six", "sum"),
        dismissals=("batter_dismissed", "sum"),
        dot_balls_batting=("is_dot_ball_batting", "sum"),
    ).reset_index()
    summary["batting_strike_rate"] = 100 * safe_divide(summary["runs_scored"], summary["balls_faced"])
    summary["batting_average"] = safe_divide(summary["runs_scored"], summary["dismissals"])
    summary["boundary_ball_pct"] = 100 * safe_divide(summary["fours"] + summary["sixes"], summary["balls_faced"])
    summary["dot_ball_pct_batting"] = 100 * safe_divide(summary["dot_balls_batting"], summary["balls_faced"])
    summary["runs_per_innings"] = safe_divide(summary["runs_scored"], summary["innings_batted"])

    phase_summary = (
        batting.groupby(["match_season", "batter_name_clean", "phase"], dropna=False)
        .agg(phase_runs=("batter_runs", "sum"), phase_balls=("is_ball_faced", "sum"))
        .reset_index()
    )
    if not phase_summary.empty:
        phase_pivot = phase_summary.pivot_table(
            index=["match_season", "batter_name_clean"], columns="phase", values=["phase_runs", "phase_balls"], fill_value=0
        )
        phase_pivot.columns = ["_".join(col).strip("_") for col in phase_pivot.columns]
        phase_pivot = phase_pivot.reset_index()
        summary = summary.merge(phase_pivot, on=["match_season", "batter_name_clean"], how="left")
    for phase in ["powerplay", "middle", "death"]:
        runs_col = f"phase_runs_{phase}"
        balls_col = f"phase_balls_{phase}"
        if runs_col not in summary.columns:
            summary[runs_col] = 0
            summary[balls_col] = 0
        summary[f"{phase}_strike_rate"] = 100 * safe_divide(summary[runs_col], summary[balls_col])
    return summary.rename(columns={"batter_name_clean": "player_name_clean"})


def _bowling_base(deliveries: pd.DataFrame) -> pd.DataFrame:
    out = deliveries.copy()
    extra_type = out["extra_type"].astype(str).str.lower()
    out["is_legal_ball"] = (~extra_type.isin(["wides", "noballs", "no_ball", "wide"])).astype(int)
    out["is_dot_ball_bowling"] = out["total_runs"].eq(0).astype(int) * out["is_legal_ball"]
    bowler_extra = np.where(extra_type.isin(NON_BOWLER_EXTRAS), 0, out["extra_runs"])
    out["runs_conceded_bowler"] = out["batter_runs"] + bowler_extra
    out["bowler_wicket"] = (
        out["is_wicket"].eq(1) & out["dismissal_kind"].astype(str).str.lower().isin(DISMISSALS_CREDITED_TO_BOWLER)
    ).astype(int)
    out["phase"] = np.select(
        [out["over"].between(1, 6), out["over"].between(7, 15), out["over"] >= 16],
        ["powerplay", "middle", "death"],
        default="other",
    )
    out["inning_key"] = out["match_id"].astype(str) + "_" + out["inning"].astype(int).astype(str)
    return out


def engineer_bowling_features(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    if deliveries_df.empty:
        return pd.DataFrame()
    bowling = _bowling_base(deliveries_df)
    grouped = bowling.groupby(["match_season", "bowler_name_clean"], dropna=False)
    summary = grouped.agg(
        matches_played=("match_id", "nunique"),
        innings_bowled=("inning_key", "nunique"),
        balls_bowled=("ball", "count"),
        legal_balls_bowled=("is_legal_ball", "sum"),
        runs_conceded=("runs_conceded_bowler", "sum"),
        wickets=("bowler_wicket", "sum"),
        dot_balls_bowling=("is_dot_ball_bowling", "sum"),
    ).reset_index()
    summary["bowling_strike_rate"] = safe_divide(summary["legal_balls_bowled"], summary["wickets"])
    summary["economy_rate"] = safe_divide(summary["runs_conceded"], summary["legal_balls_bowled"] / 6.0)
    summary["dot_ball_pct_bowling"] = 100 * safe_divide(summary["dot_balls_bowling"], summary["legal_balls_bowled"])

    phase_summary = (
        bowling.groupby(["match_season", "bowler_name_clean", "phase"], dropna=False)
        .agg(phase_runs=("runs_conceded_bowler", "sum"), phase_balls=("is_legal_ball", "sum"), phase_wickets=("bowler_wicket", "sum"))
        .reset_index()
    )
    if not phase_summary.empty:
        phase_pivot = phase_summary.pivot_table(
            index=["match_season", "bowler_name_clean"],
            columns="phase",
            values=["phase_runs", "phase_balls", "phase_wickets"],
            fill_value=0,
        )
        phase_pivot.columns = ["_".join(col).strip("_") for col in phase_pivot.columns]
        phase_pivot = phase_pivot.reset_index()
        summary = summary.merge(phase_pivot, on=["match_season", "bowler_name_clean"], how="left")
    for phase in ["powerplay", "death"]:
        runs_col = f"phase_runs_{phase}"
        balls_col = f"phase_balls_{phase}"
        wickets_col = f"phase_wickets_{phase}"
        for col in [runs_col, balls_col, wickets_col]:
            if col not in summary.columns:
                summary[col] = 0
        summary[f"{phase}_wickets"] = summary[wickets_col]
        summary[f"{phase}_economy"] = safe_divide(summary[runs_col], summary[balls_col] / 6.0)
    return summary.rename(columns={"bowler_name_clean": "player_name_clean"})


def build_player_season_features(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    if deliveries_df.empty:
        return pd.DataFrame()
    deliveries = attach_season_to_deliveries(deliveries_df, matches_df)
    batting = engineer_batting_features(deliveries)
    bowling = engineer_bowling_features(deliveries)
    if batting.empty and bowling.empty:
        return pd.DataFrame()
    merged = batting.merge(bowling, on=["match_season", "player_name_clean"], how="outer", suffixes=("", "_bowl"))
    merged["matches_played"] = merged[["matches_played", "matches_played_bowl"]].max(axis=1)
    merged = merged.sort_values(["player_name_clean", "match_season"]).copy()
    grouped = merged.groupby("player_name_clean", dropna=False)
    merged["experience_proxy"] = grouped.cumcount()

    for feature in ["runs_scored", "wickets", "batting_strike_rate", "economy_rate"]:
        if feature not in merged.columns:
            merged[feature] = np.nan
        merged[f"recent_{feature}"] = grouped[feature].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())

    merged["prior_seasons_played"] = grouped["match_season"].transform(lambda s: s.shift(1).notna().cumsum())
    merged["career_runs_before_auction"] = grouped["runs_scored"].transform(lambda s: s.shift(1).cumsum())
    merged["career_wickets_before_auction"] = grouped["wickets"].transform(lambda s: s.shift(1).cumsum())
    merged["career_matches_before_auction"] = grouped["matches_played"].transform(lambda s: s.shift(1).cumsum())
    merged["three_year_runs_before_auction"] = grouped["runs_scored"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    merged["three_year_wickets_before_auction"] = grouped["wickets"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    merged["three_year_matches_before_auction"] = grouped["matches_played"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    merged["three_year_batting_sr_before_auction"] = grouped["batting_strike_rate"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    merged["three_year_economy_before_auction"] = grouped["economy_rate"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    merged["career_best_runs_season_before_auction"] = grouped["runs_scored"].transform(lambda s: s.shift(1).cummax())
    merged["career_best_wickets_season_before_auction"] = grouped["wickets"].transform(lambda s: s.shift(1).cummax())
    merged["career_best_sr_season_before_auction"] = grouped["batting_strike_rate"].transform(lambda s: s.shift(1).cummax())
    merged["career_best_economy_season_before_auction"] = grouped["economy_rate"].transform(lambda s: s.shift(1).cummin())
    merged["prior_500_run_seasons"] = grouped["runs_scored"].transform(lambda s: s.shift(1).ge(500).cumsum())
    merged["prior_400_run_seasons"] = grouped["runs_scored"].transform(lambda s: s.shift(1).ge(400).cumsum())
    merged["prior_20_wicket_seasons"] = grouped["wickets"].transform(lambda s: s.shift(1).ge(20).cumsum())
    merged["prior_15_wicket_seasons"] = grouped["wickets"].transform(lambda s: s.shift(1).ge(15).cumsum())
    merged["pedigree_batter_index"] = (
        0.30 * merged["career_runs_before_auction"].fillna(0).rank(pct=True)
        + 0.25 * merged["three_year_runs_before_auction"].fillna(0).rank(pct=True)
        + 0.20 * merged["three_year_batting_sr_before_auction"].fillna(merged["three_year_batting_sr_before_auction"].median()).rank(pct=True)
        + 0.15 * merged["prior_500_run_seasons"].fillna(0).rank(pct=True)
        + 0.10 * merged["career_best_runs_season_before_auction"].fillna(0).rank(pct=True)
    )
    merged["pedigree_bowler_index"] = (
        0.30 * merged["career_wickets_before_auction"].fillna(0).rank(pct=True)
        + 0.25 * merged["three_year_wickets_before_auction"].fillna(0).rank(pct=True)
        + 0.20 * (1 - merged["three_year_economy_before_auction"].fillna(merged["three_year_economy_before_auction"].median()).rank(pct=True))
        + 0.15 * merged["prior_20_wicket_seasons"].fillna(0).rank(pct=True)
        + 0.10 * merged["career_best_wickets_season_before_auction"].fillna(0).rank(pct=True)
    )
    return merged


def build_pre_auction_dataset(auction_df: pd.DataFrame, player_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if auction_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    coverage = []
    merged = auction_df.copy()
    coverage.append({"step": "auction_rows", "rows_before": len(merged), "rows_after": len(merged), "retention_rate": 1.0})
    merged["performance_season"] = merged["auction_year"] - 1
    merged = merged.loc[merged["performance_season"].notna()].copy()
    coverage.append(
        {
            "step": "valid_auction_year",
            "rows_before": coverage[-1]["rows_after"],
            "rows_after": len(merged),
            "retention_rate": len(merged) / coverage[-1]["rows_after"] if coverage[-1]["rows_after"] else np.nan,
        }
    )
    if player_features.empty:
        return merged, pd.DataFrame(coverage)
    performance = player_features.rename(columns={"match_season": "performance_season"})
    merged = merged.merge(
        performance,
        left_on=["matched_player_name_clean", "performance_season"],
        right_on=["player_name_clean", "performance_season"],
        how="left",
        suffixes=("", "_perf"),
    )
    coverage.append(
        {
            "step": "merged_pre_auction_performance",
            "rows_before": coverage[-1]["rows_after"],
            "rows_after": int(merged["player_name_clean_perf"].notna().sum()) if "player_name_clean_perf" in merged.columns else len(merged),
            "retention_rate": (
                int(merged["player_name_clean_perf"].notna().sum()) / coverage[-1]["rows_after"]
                if coverage[-1]["rows_after"] and "player_name_clean_perf" in merged.columns
                else np.nan
            ),
        }
    )
    return merged, pd.DataFrame(coverage)


def build_team_season_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame()

    matches = matches_df.copy()
    required = {"match_season", "team1", "team2"}
    if not required.issubset(matches.columns):
        return pd.DataFrame()

    for col in ["team1", "team2", "winner"]:
        if col in matches.columns:
            matches[col] = matches[col].map(canonicalize_team_name)
    matches = matches.loc[matches["match_season"].notna()].copy()
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for _, row in matches.iterrows():
        team1 = row.get("team1", "")
        team2 = row.get("team2", "")
        winner = row.get("winner", "")
        outcome_text = " ".join(
            [
                str(row.get("outcome", "")),
                str(row.get("result", "")),
                str(row.get("method", "")),
            ]
        ).lower()
        no_result = winner in ("", "No Result") and any(token in outcome_text for token in ["tie", "no result", "abandoned"])
        for team in [team1, team2]:
            if not team:
                continue
            points = 0
            wins = 0
            if winner and winner == team:
                points = 2
                wins = 1
            elif no_result:
                points = 1
            rows.append(
                {
                    "match_season": row["match_season"],
                    "team": team,
                    "played": 1,
                    "wins": wins,
                    "points": points,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame()

    summary = (
        summary.groupby(["match_season", "team"], dropna=False)
        .agg(played=("played", "sum"), wins=("wins", "sum"), points=("points", "sum"))
        .reset_index()
    )
    summary["finish_rank"] = (
        summary.groupby("match_season")[["points", "wins"]]
        .apply(lambda frame: frame.rank(method="dense", ascending=False))
        .reset_index(level=0, drop=True)["points"]
    )
    summary["made_playoffs"] = summary["finish_rank"].le(4).astype(int)
    return summary


def build_realized_season_review(
    final_model_dataset: pd.DataFrame,
    player_features: pd.DataFrame,
    matches_df: pd.DataFrame | None = None,
    auction_year: int = 2025,
    realized_season: int = 2025,
    top_n: int = 5,
    mispricing_col: str = "article_mispricing_in_inr",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if final_model_dataset.empty or player_features.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = final_model_dataset.copy()
    base["player_name_display"] = base["player_name"].astype(str).str.strip()
    base["player_name_join"] = (
        base.get("matched_player_name_clean", base.get("player_name_clean", base["player_name_display"]))
        .fillna(base["player_name_display"])
        .astype(str)
        .str.strip()
    )
    season_rows = base.loc[base["auction_year"].eq(auction_year)].copy()
    if season_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    realized = player_features.loc[player_features["match_season"].eq(realized_season)].copy()
    if realized.empty:
        return pd.DataFrame(), pd.DataFrame()

    realized["player_name_join"] = realized["player_name_clean"].astype(str).str.strip()
    realized["role_bucket_realized"] = np.select(
        [
            realized["wickets"].fillna(0).gt(0) & realized["runs_scored"].fillna(0).gt(150),
            realized["wickets"].fillna(0).gt(0),
            realized["runs_scored"].fillna(0).gt(0),
        ],
        ["all_rounder", "bowler", "batter"],
        default="unknown",
    )
    def _zscore(series: pd.Series) -> pd.Series:
        values = series.astype(float)
        std = values.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return (values - values.mean()) / std

    realized["batter_component"] = (
        0.5 * _zscore(realized["runs_scored"].fillna(0))
        + 0.35 * _zscore(realized["batting_strike_rate"].fillna(realized["batting_strike_rate"].median()))
        + 0.15 * _zscore(realized["runs_per_innings"].fillna(0))
    )
    realized["bowler_component"] = (
        0.45 * _zscore(realized["wickets"].fillna(0))
        - 0.30 * _zscore(realized["economy_rate"].fillna(realized["economy_rate"].median()))
        - 0.25 * _zscore(realized["bowling_strike_rate"].fillna(realized["bowling_strike_rate"].median()))
    )
    realized["all_rounder_component"] = 0.5 * realized["batter_component"] + 0.5 * realized["bowler_component"]
    realized["realized_value_index"] = np.select(
        [
            realized["role_bucket_realized"].eq("batter"),
            realized["role_bucket_realized"].eq("bowler"),
            realized["role_bucket_realized"].eq("all_rounder"),
        ],
        [
            realized["batter_component"],
            realized["bowler_component"],
            realized["all_rounder_component"],
        ],
        default=0.5 * realized["batter_component"] + 0.5 * realized["bowler_component"],
    )
    team_summary = build_team_season_summary(matches_df if matches_df is not None else pd.DataFrame())
    team_delta = pd.DataFrame()
    if not team_summary.empty:
        prev_summary = team_summary.loc[team_summary["match_season"].eq(realized_season - 1), ["team", "finish_rank", "made_playoffs"]].rename(
            columns={"finish_rank": "finish_rank_prev", "made_playoffs": "made_playoffs_prev"}
        )
        curr_summary = team_summary.loc[team_summary["match_season"].eq(realized_season), ["team", "finish_rank", "made_playoffs"]].rename(
            columns={"finish_rank": "finish_rank_current", "made_playoffs": "made_playoffs_current"}
        )
        team_delta = curr_summary.merge(prev_summary, on="team", how="left")
        team_delta["finish_rank_delta"] = team_delta["finish_rank_prev"] - team_delta["finish_rank_current"]
        team_delta["playoff_delta"] = team_delta["made_playoffs_current"].fillna(0) - team_delta["made_playoffs_prev"].fillna(0)

    review_cols = [
        "player_name_join",
        "role_bucket_realized",
        "matches_played",
        "runs_scored",
        "batting_strike_rate",
        "runs_per_innings",
        "wickets",
        "economy_rate",
        "bowling_strike_rate",
        "realized_value_index",
    ]
    season_rows = season_rows.merge(realized[review_cols], on="player_name_join", how="left")
    season_rows["player_name"] = season_rows["player_name_display"]
    season_rows["team"] = season_rows["team"].map(canonicalize_team_name)
    if not team_delta.empty:
        season_rows = season_rows.merge(team_delta, on="team", how="left")
    else:
        season_rows["finish_rank_prev"] = np.nan
        season_rows["finish_rank_current"] = np.nan
        season_rows["finish_rank_delta"] = np.nan
        season_rows["playoff_delta"] = np.nan

    captain_adjustment = (
        0.6 * season_rows["finish_rank_delta"].fillna(0)
        + 0.8 * season_rows["playoff_delta"].fillna(0)
    )
    season_rows["captaincy_outcome_adjustment"] = np.where(
        season_rows["captaincy_proxy"].fillna(0).eq(1),
        captain_adjustment,
        0.0,
    )
    season_rows["realized_news_index"] = season_rows["realized_value_index"].fillna(0) + season_rows["captaincy_outcome_adjustment"].fillna(0)
    season_rows["realized_percentile_within_role"] = (
        season_rows.groupby("role_bucket_realized")["realized_news_index"].rank(pct=True).fillna(0.5)
    )
    season_rows["realized_outcome_label"] = pd.cut(
        season_rows["realized_percentile_within_role"],
        bins=[-np.inf, 0.40, 0.75, np.inf],
        labels=["below_expectation", "solid", "impact"],
    ).astype(str)

    sort_col = mispricing_col if mispricing_col in season_rows.columns else "mispricing_in_inr"
    top_undervalued = (
        season_rows.sort_values(sort_col, ascending=False)
        .head(top_n)
        .copy()
        .sort_values(sort_col, ascending=False)
    )
    top_overvalued = (
        season_rows.sort_values(sort_col, ascending=True)
        .head(top_n)
        .copy()
        .sort_values(sort_col, ascending=True)
    )
    return top_undervalued, top_overvalued


def build_selected_player_followup(
    selected_players: pd.DataFrame,
    final_model_dataset: pd.DataFrame,
    player_features: pd.DataFrame,
    matches_df: pd.DataFrame | None = None,
    realized_season: int = 2025,
) -> pd.DataFrame:
    if selected_players.empty or final_model_dataset.empty or player_features.empty:
        return pd.DataFrame()

    base = final_model_dataset.copy()
    base["player_name_display"] = base["player_name"].astype(str).str.strip()
    base["player_name_join"] = (
        base.get("matched_player_name_clean", base.get("player_name_clean", base["player_name_display"]))
        .fillna(base["player_name_display"])
        .astype(str)
        .str.strip()
    )
    selected = selected_players.copy()
    selected["player_name_display"] = selected["player_name"].astype(str).str.strip()
    selected["player_name_join"] = (
        selected.get("matched_player_name_clean", selected.get("player_name_clean", selected["player_name_display"]))
        .fillna(selected["player_name_display"])
        .astype(str)
        .str.strip()
    )
    stale_review_cols = [
        "matches_played_2025",
        "runs_scored_2025",
        "batting_strike_rate_2025",
        "runs_per_innings_2025",
        "wickets_2025",
        "economy_rate_2025",
        "bowling_strike_rate_2025",
        "realized_value_index_2025",
        "realized_news_index",
        "realized_percentile_within_role",
        "realized_outcome_label",
        "article_rating",
        "realized_statline",
        "role_bucket_realized",
    ]
    selected = selected.drop(columns=[col for col in stale_review_cols if col in selected.columns], errors="ignore")
    selected = selected.drop_duplicates(subset=["player_name_display", "auction_year"])

    realized = player_features.loc[player_features["match_season"].eq(realized_season)].copy()
    if realized.empty:
        return pd.DataFrame()

    realized["player_name_join"] = realized["player_name_clean"].astype(str).str.strip()
    realized["role_bucket_realized"] = np.select(
        [
            realized["wickets"].fillna(0).gt(0) & realized["runs_scored"].fillna(0).gt(150),
            realized["wickets"].fillna(0).gt(0),
            realized["runs_scored"].fillna(0).gt(0),
        ],
        ["all_rounder", "bowler", "batter"],
        default="unknown",
    )

    def _zscore(series: pd.Series) -> pd.Series:
        values = series.astype(float)
        std = values.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return (values - values.mean()) / std

    realized["batter_component"] = (
        0.5 * _zscore(realized["runs_scored"].fillna(0))
        + 0.35 * _zscore(realized["batting_strike_rate"].fillna(realized["batting_strike_rate"].median()))
        + 0.15 * _zscore(realized["runs_per_innings"].fillna(0))
    )
    realized["bowler_component"] = (
        0.45 * _zscore(realized["wickets"].fillna(0))
        - 0.30 * _zscore(realized["economy_rate"].fillna(realized["economy_rate"].median()))
        - 0.25 * _zscore(realized["bowling_strike_rate"].fillna(realized["bowling_strike_rate"].median()))
    )
    realized["all_rounder_component"] = 0.5 * realized["batter_component"] + 0.5 * realized["bowler_component"]
    realized["realized_value_index"] = np.select(
        [
            realized["role_bucket_realized"].eq("batter"),
            realized["role_bucket_realized"].eq("bowler"),
            realized["role_bucket_realized"].eq("all_rounder"),
        ],
        [
            realized["batter_component"],
            realized["bowler_component"],
            realized["all_rounder_component"],
        ],
        default=0.5 * realized["batter_component"] + 0.5 * realized["bowler_component"],
    )

    team_summary = build_team_season_summary(matches_df if matches_df is not None else pd.DataFrame())
    team_delta = pd.DataFrame()
    if not team_summary.empty:
        prev_summary = team_summary.loc[team_summary["match_season"].eq(realized_season - 1), ["team", "finish_rank", "made_playoffs"]].rename(
            columns={"finish_rank": "finish_rank_prev", "made_playoffs": "made_playoffs_prev"}
        )
        curr_summary = team_summary.loc[team_summary["match_season"].eq(realized_season), ["team", "finish_rank", "made_playoffs"]].rename(
            columns={"finish_rank": "finish_rank_current", "made_playoffs": "made_playoffs_current"}
        )
        team_delta = curr_summary.merge(prev_summary, on="team", how="left")
        team_delta["finish_rank_delta"] = team_delta["finish_rank_prev"] - team_delta["finish_rank_current"]
        team_delta["playoff_delta"] = team_delta["made_playoffs_current"].fillna(0) - team_delta["made_playoffs_prev"].fillna(0)

    review_cols = [
        "player_name_join",
        "role_bucket_realized",
        "matches_played",
        "runs_scored",
        "batting_strike_rate",
        "runs_per_innings",
        "wickets",
        "economy_rate",
        "bowling_strike_rate",
        "realized_value_index",
    ]
    realized_review = realized[review_cols].rename(
        columns={
            "matches_played": "matches_played_2025",
            "runs_scored": "runs_scored_2025",
            "batting_strike_rate": "batting_strike_rate_2025",
            "runs_per_innings": "runs_per_innings_2025",
            "wickets": "wickets_2025",
            "economy_rate": "economy_rate_2025",
            "bowling_strike_rate": "bowling_strike_rate_2025",
            "realized_value_index": "realized_value_index_2025",
        }
    )
    followup = selected.merge(
        realized_review,
        on="player_name_join",
        how="left",
    )
    followup["team"] = followup["team"].map(canonicalize_team_name)
    if not team_delta.empty:
        followup = followup.merge(team_delta, on="team", how="left")
    else:
        followup["finish_rank_prev"] = np.nan
        followup["finish_rank_current"] = np.nan
        followup["finish_rank_delta"] = np.nan
        followup["playoff_delta"] = np.nan

    captain_adjustment = 0.6 * followup["finish_rank_delta"].fillna(0) + 0.8 * followup["playoff_delta"].fillna(0)
    followup["captaincy_outcome_adjustment"] = np.where(
        followup["captaincy_proxy"].fillna(0).eq(1),
        captain_adjustment,
        0.0,
    )
    followup["realized_news_index"] = followup["realized_value_index_2025"].fillna(0) + followup["captaincy_outcome_adjustment"].fillna(0)
    followup["realized_percentile_within_role"] = (
        followup.groupby("role_bucket_realized")["realized_news_index"].rank(pct=True).fillna(0.5)
    )
    followup["realized_outcome_label"] = pd.cut(
        followup["realized_percentile_within_role"],
        bins=[-np.inf, 0.40, 0.75, np.inf],
        labels=["below_expectation", "solid", "impact"],
    ).astype(str)
    followup["article_rating"] = followup["realized_outcome_label"].map(
        {"impact": "hit", "solid": "mid", "below_expectation": "flop"}
    ).fillna("mid")

    batter_mask = followup["role_bucket_realized"].isin(["batter", "wicketkeeper"])
    bowler_mask = followup["role_bucket_realized"].eq("bowler")
    all_rounder_mask = followup["role_bucket_realized"].eq("all_rounder")
    followup["realized_statline"] = np.select(
        [
            all_rounder_mask,
            bowler_mask,
            batter_mask,
        ],
        [
            followup["runs_scored_2025"].fillna(0).round().astype(int).astype(str)
            + " runs | "
            + followup["batting_strike_rate_2025"].fillna(0).round(1).astype(str)
            + " SR | "
            + followup["wickets_2025"].fillna(0).round().astype(int).astype(str)
            + " wkts | "
            + followup["economy_rate_2025"].fillna(0).round(2).astype(str)
            + " econ",
            followup["wickets_2025"].fillna(0).round().astype(int).astype(str)
            + " wkts | "
            + followup["economy_rate_2025"].fillna(0).round(2).astype(str)
            + " econ | "
            + followup["bowling_strike_rate_2025"].fillna(0).round(1).astype(str)
            + " bsr",
            followup["runs_scored_2025"].fillna(0).round().astype(int).astype(str)
            + " runs | "
            + followup["batting_strike_rate_2025"].fillna(0).round(1).astype(str)
            + " SR",
        ],
        default="Limited 2025 sample",
    )
    return followup
