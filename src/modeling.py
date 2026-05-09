from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import CAPTAINCY_PROXY_NAMES, LEADERSHIP_PROXY_NAMES, TITLE_WINNING_CAPTAIN_AUCTION_MAP, quantile_labels

WICKETKEEPER_PROXY_NAMES = {
    "rishabh pant",
    "sanju samson",
    "kl rahul",
    "jitesh sharma",
    "ms dhoni",
    "quinton de kock",
    "rahmanullah gurbaz",
    "jonny bairstow",
    "ishan kishan",
    "phil salt",
    "jos buttler",
    "jc buttler",
    "nicholas pooran",
}

BOWLER_PROXY_NAMES = {
    "arshdeep singh",
    "avesh khan",
    "bhuvneshwar kumar",
    "harshal patel",
    "hv patel",
    "jasprit bumrah",
    "jofra archer",
    "jc archer",
    "josh hazlewood",
    "jr hazlewood",
    "kagiso rabada",
    "marco jansen",
    "mitchell starc",
    "mohammed siraj",
    "mohammad siraj",
    "pat cummins",
    "trent boult",
    "ta boult",
}

LEFT_ARM_BOWLER_PROXY_NAMES = {
    "arshdeep singh",
    "khaleel ahmed",
    "marco jansen",
    "mitchell starc",
    "r sai kishore",
    "r sai kishore",
    "trent boult",
    "ta boult",
    "yash dayal",
}

ELITE_INDIAN_PACER_PROXY_NAMES = {
    "arshdeep singh",
    "bhuvneshwar kumar",
    "jasprit bumrah",
    "mohammed shami",
    "mohammed siraj",
    "mohammad siraj",
}

BASE_FEATURE_COLUMNS = [
    "auction_year",
    "performance_season",
    "year_median_price_in_inr",
    "year_mean_price_in_inr",
    "year_total_spend_in_inr",
    "year_players_sold",
    "inflation_index",
    "spend_index",
    "role_year_median_price_in_inr",
    "role_year_mean_price_in_inr",
    "is_mega_auction_proxy",
    "leadership_proxy",
    "captaincy_proxy",
    "title_winning_captain_proxy",
    "indian_batter_premium_proxy",
    "sena_overseas_premium_proxy",
    "wicketkeeper_scarcity_proxy",
    "death_bowler_premium_proxy",
    "control_bowler_premium_proxy",
    "left_arm_bowler_proxy",
    "premium_bowler_quality_index",
    "elite_indian_pacer_scarcity_proxy",
    "marquee_player_proxy",
    "previous_auction_price_in_inr",
    "auction_history_mean_price_in_inr",
    "career_peak_auction_price_in_inr",
    "prior_auction_count",
    "years_since_last_auction",
    "runs_scored",
    "balls_faced",
    "batting_strike_rate",
    "batting_average",
    "boundary_ball_pct",
    "dot_ball_pct_batting",
    "runs_per_innings",
    "powerplay_strike_rate",
    "middle_strike_rate",
    "death_strike_rate",
    "legal_balls_bowled",
    "wickets",
    "bowling_strike_rate",
    "economy_rate",
    "dot_ball_pct_bowling",
    "powerplay_wickets",
    "death_wickets",
    "powerplay_economy",
    "death_economy",
    "matches_played",
    "experience_proxy",
    "recent_runs_scored",
    "recent_wickets",
    "recent_batting_strike_rate",
    "recent_economy_rate",
    "prior_seasons_played",
    "career_runs_before_auction",
    "career_wickets_before_auction",
    "career_matches_before_auction",
    "three_year_runs_before_auction",
    "three_year_wickets_before_auction",
    "three_year_matches_before_auction",
    "three_year_batting_sr_before_auction",
    "three_year_economy_before_auction",
    "career_best_runs_season_before_auction",
    "career_best_wickets_season_before_auction",
    "career_best_sr_season_before_auction",
    "career_best_economy_season_before_auction",
    "prior_500_run_seasons",
    "prior_400_run_seasons",
    "prior_20_wicket_seasons",
    "prior_15_wicket_seasons",
    "pedigree_batter_index",
    "pedigree_bowler_index",
    "is_overseas",
    "is_wicketkeeper",
    "is_allrounder",
]

CATEGORICAL_FEATURES = ["role_bucket", "team"]


@dataclass
class ModelBundle:
    name: str
    estimator: object
    feature_columns: List[str]
    metrics: Dict[str, float]
    predictions: pd.DataFrame


def prepare_model_dataset(df: pd.DataFrame, min_year: int = 2018) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out = out.loc[out["auction_year"].ge(min_year)].copy()
    out = out.loc[out["price_in_inr"].notna() & out["price_in_inr"].gt(0)].copy()
    out["target_log_price"] = np.log1p(out["price_in_inr"])
    out["player_name_clean_model"] = (
        out.get("matched_player_name_clean", out.get("player_name_clean", out["player_name"].astype(str)))
        .fillna(out["player_name"].astype(str))
        .astype(str)
    )
    out["is_wicketkeeper"] = np.where(
        out["player_name_clean_model"].isin(WICKETKEEPER_PROXY_NAMES),
        1,
        out["is_wicketkeeper"].fillna(0),
    )
    role_override_mask = (
        out["wickets"].fillna(0).ge(8)
        & out["runs_scored"].fillna(0).le(150)
        & (
            out["economy_rate"].notna()
            | out["bowling_strike_rate"].notna()
            | out["death_wickets"].fillna(0).gt(0)
            | out["powerplay_wickets"].fillna(0).gt(0)
        )
    )
    all_round_override_mask = out["wickets"].fillna(0).ge(8) & out["runs_scored"].fillna(0).gt(150)
    out.loc[out["is_wicketkeeper"].eq(1), "role_bucket"] = "wicketkeeper"
    out.loc[out["is_allrounder"].eq(1) | all_round_override_mask, "role_bucket"] = "all_rounder"
    out.loc[role_override_mask, "role_bucket"] = "bowler"
    out.loc[
        out["player_name_clean_model"].isin(BOWLER_PROXY_NAMES) & out["is_wicketkeeper"].ne(1) & out["is_allrounder"].ne(1),
        "role_bucket",
    ] = "bowler"
    unknown_role_mask = out["role_bucket"].fillna("unknown").eq("unknown")
    out.loc[unknown_role_mask & out["runs_scored"].fillna(0).gt(0), "role_bucket"] = "batter"
    year_summary = (
        out.groupby("auction_year", dropna=False)
        .agg(
            year_median_price_in_inr=("price_in_inr", "median"),
            year_mean_price_in_inr=("price_in_inr", "mean"),
            year_total_spend_in_inr=("price_in_inr", "sum"),
            year_players_sold=("player_name", "count"),
        )
        .reset_index()
    )
    overall_median = out["price_in_inr"].median()
    overall_mean = out["price_in_inr"].mean()
    players_threshold = year_summary["year_players_sold"].quantile(0.75)
    spend_threshold = year_summary["year_total_spend_in_inr"].quantile(0.75)
    year_summary["inflation_index"] = year_summary["year_median_price_in_inr"] / overall_median if overall_median else 1.0
    year_summary["spend_index"] = year_summary["year_mean_price_in_inr"] / overall_mean if overall_mean else 1.0
    year_summary["is_mega_auction_proxy"] = (
        year_summary["year_players_sold"].ge(players_threshold) & year_summary["year_total_spend_in_inr"].ge(spend_threshold)
    ).astype(int)
    out = out.merge(year_summary, on="auction_year", how="left")

    role_year_summary = (
        out.groupby(["auction_year", "role_bucket"], dropna=False)
        .agg(
            role_year_median_price_in_inr=("price_in_inr", "median"),
            role_year_mean_price_in_inr=("price_in_inr", "mean"),
        )
        .reset_index()
    )
    out = out.merge(role_year_summary, on=["auction_year", "role_bucket"], how="left")
    out["leadership_proxy"] = out["player_name_clean_model"].isin(LEADERSHIP_PROXY_NAMES).astype(int)
    out["captaincy_proxy"] = out["player_name_clean_model"].isin(CAPTAINCY_PROXY_NAMES).astype(int)
    out["title_winning_captain_proxy"] = (
        out.apply(
            lambda row: (int(row["auction_year"]), row["player_name_clean_model"]) in TITLE_WINNING_CAPTAIN_AUCTION_MAP
            if pd.notna(row["auction_year"])
            else False,
            axis=1,
        )
        .astype(int)
    )
    out["indian_batter_premium_proxy"] = (
        out["is_overseas"].fillna(0).eq(0) & out["role_bucket"].isin(["batter", "wicketkeeper"])
    ).astype(int)
    nationality_text = out.get("nationality", pd.Series("", index=out.index)).astype(str).str.lower()
    out["sena_overseas_premium_proxy"] = (
        out["is_overseas"].fillna(0).eq(1)
        & nationality_text.str.contains("australia|south africa|england|new zealand", regex=True, na=False)
    ).astype(int)
    bowler_mask = out["role_bucket"].isin(["bowler", "all_rounder"])
    death_wickets_threshold = out.loc[bowler_mask, "death_wickets"].fillna(0).quantile(0.7) if bowler_mask.any() else 0
    powerplay_wickets_threshold = out.loc[bowler_mask, "powerplay_wickets"].fillna(0).quantile(0.7) if bowler_mask.any() else 0
    control_wickets_threshold = out.loc[bowler_mask, "wickets"].fillna(0).quantile(0.6) if bowler_mask.any() else 0
    control_economy_threshold = out.loc[bowler_mask, "economy_rate"].fillna(out["economy_rate"].median()).quantile(0.45) if bowler_mask.any() else np.nan
    control_dot_threshold = out.loc[bowler_mask, "dot_ball_pct_bowling"].fillna(0).quantile(0.6) if bowler_mask.any() else 0
    out["wicketkeeper_scarcity_proxy"] = out["is_wicketkeeper"].fillna(0).astype(int)
    out["death_bowler_premium_proxy"] = (
        bowler_mask
        & (
            out["death_wickets"].fillna(0).ge(death_wickets_threshold)
            | out["powerplay_wickets"].fillna(0).ge(powerplay_wickets_threshold)
        )
    ).astype(int)
    out["control_bowler_premium_proxy"] = (
        bowler_mask
        & out["wickets"].fillna(0).ge(control_wickets_threshold)
        & out["dot_ball_pct_bowling"].fillna(0).ge(control_dot_threshold)
        & out["economy_rate"].fillna(out["economy_rate"].median()).le(control_economy_threshold if pd.notna(control_economy_threshold) else out["economy_rate"].median())
    ).astype(int)
    out["left_arm_bowler_proxy"] = (
        bowler_mask & out["player_name_clean_model"].isin(LEFT_ARM_BOWLER_PROXY_NAMES)
    ).astype(int)
    wicket_quality = out["wickets"].fillna(0).rank(pct=True)
    death_quality = out["death_wickets"].fillna(0).rank(pct=True)
    powerplay_quality = out["powerplay_wickets"].fillna(0).rank(pct=True)
    economy_quality = 1 - out["economy_rate"].fillna(out["economy_rate"].median()).rank(pct=True)
    strike_quality = 1 - out["bowling_strike_rate"].fillna(out["bowling_strike_rate"].median()).rank(pct=True)
    three_year_wickets_quality = out["three_year_wickets_before_auction"].fillna(0).rank(pct=True)
    prior_20_quality = out["prior_20_wicket_seasons"].fillna(0).rank(pct=True)
    out["premium_bowler_quality_index"] = np.where(
        bowler_mask,
        (
            0.22 * wicket_quality
            + 0.14 * death_quality
            + 0.10 * powerplay_quality
            + 0.12 * economy_quality
            + 0.08 * strike_quality
            + 0.16 * out["pedigree_bowler_index"].fillna(0)
            + 0.10 * three_year_wickets_quality
            + 0.05 * prior_20_quality
            + 0.03 * out["left_arm_bowler_proxy"].fillna(0)
        ),
        0.0,
    )
    out["elite_indian_pacer_scarcity_proxy"] = (
        out["is_overseas"].fillna(0).eq(0)
        & bowler_mask
        & out["player_name_clean_model"].isin(ELITE_INDIAN_PACER_PROXY_NAMES)
    ).astype(int)
    out["leadership_proxy"] = np.where(
        out["experience_proxy"].fillna(0).ge(5) & out["player_name_clean_model"].str.split().str[-1].notna(),
        np.maximum(out["leadership_proxy"], out["experience_proxy"].fillna(0).ge(7).astype(int)),
        out["leadership_proxy"],
    )
    out = out.sort_values(["player_name_clean_model", "auction_year", "price_in_inr"]).copy()
    grouped = out.groupby("player_name_clean_model", dropna=False)
    out["previous_auction_price_in_inr"] = grouped["price_in_inr"].shift(1)
    out["prior_auction_count"] = grouped.cumcount()
    previous_auction_year = grouped["auction_year"].shift(1)
    out["years_since_last_auction"] = out["auction_year"] - previous_auction_year
    out["career_peak_auction_price_in_inr"] = grouped["price_in_inr"].transform(lambda s: s.shift(1).cummax())
    out["auction_history_mean_price_in_inr"] = grouped["price_in_inr"].transform(lambda s: s.shift(1).expanding().mean())
    previous_price_threshold = (
        out["previous_auction_price_in_inr"].dropna().quantile(0.85)
        if out["previous_auction_price_in_inr"].notna().any()
        else np.nan
    )
    peak_price_threshold = (
        out["career_peak_auction_price_in_inr"].dropna().quantile(0.85)
        if out["career_peak_auction_price_in_inr"].notna().any()
        else np.nan
    )
    out["marquee_player_proxy"] = (
        out["captaincy_proxy"].eq(1)
        | out["title_winning_captain_proxy"].eq(1)
        | out["previous_auction_price_in_inr"].fillna(0).ge(0 if np.isnan(previous_price_threshold) else previous_price_threshold)
        | out["career_peak_auction_price_in_inr"].fillna(0).ge(0 if np.isnan(peak_price_threshold) else peak_price_threshold)
    ).astype(int)
    for feature in BASE_FEATURE_COLUMNS:
        if feature not in out.columns:
            out[feature] = np.nan
    for feature in CATEGORICAL_FEATURES:
        if feature not in out.columns:
            out[feature] = "unknown"
        out[feature] = out[feature].fillna("unknown").astype(str)
    return out


def scorecard_fair_price(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    role_scores = {
        "batter": {
            "runs_scored": 0.20,
            "batting_strike_rate": 0.25,
            "boundary_ball_pct": 0.15,
            "runs_per_innings": 0.10,
            "recent_batting_strike_rate": 0.10,
            "pedigree_batter_index": 0.20,
        },
        "bowler": {
            "wickets": 0.22,
            "economy_rate": -0.20,
            "dot_ball_pct_bowling": 0.15,
            "death_wickets": 0.13,
            "recent_wickets": 0.10,
            "pedigree_bowler_index": 0.20,
        },
        "all_rounder": {
            "runs_scored": 0.15,
            "batting_strike_rate": 0.15,
            "wickets": 0.20,
            "economy_rate": -0.15,
            "recent_runs_scored": 0.15,
            "recent_wickets": 0.10,
            "pedigree_batter_index": 0.10,
        },
        "wicketkeeper": {
            "runs_scored": 0.18,
            "batting_strike_rate": 0.22,
            "boundary_ball_pct": 0.15,
            "runs_per_innings": 0.15,
            "recent_runs_scored": 0.10,
            "pedigree_batter_index": 0.20,
        },
        "unknown": {"runs_scored": 0.3, "wickets": 0.2, "batting_strike_rate": 0.25, "economy_rate": -0.25},
    }
    out["scorecard_score"] = 0.0
    for role, weights in role_scores.items():
        mask = out["role_bucket"].eq(role)
        if not mask.any():
            continue
        role_df = out.loc[mask].copy()
        score = pd.Series(0.0, index=role_df.index)
        for feature, weight in weights.items():
            values = role_df[feature].fillna(role_df[feature].median())
            std = values.std()
            normalized = (values - values.mean()) / std if std and not np.isnan(std) else values * 0
            score = score.add(weight * normalized, fill_value=0.0)
        out.loc[mask, "scorecard_score"] = score
    scaled = out["scorecard_score"].rank(pct=True).fillna(0.5)
    low_price, high_price = out["price_in_inr"].quantile([0.1, 0.9]).tolist()
    out["scorecard_predicted_price_in_inr"] = low_price + scaled * (high_price - low_price)
    out["scorecard_predicted_log_price"] = np.log1p(out["scorecard_predicted_price_in_inr"])
    return out


def year_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(df["auction_year"].dropna().unique().tolist())
    if len(years) <= 2:
        cutoff_years = years[-1:]
    else:
        cutoff_years = years[-2:]
    train = df.loc[~df["auction_year"].isin(cutoff_years)].copy()
    test = df.loc[df["auction_year"].isin(cutoff_years)].copy()
    if train.empty:
        split_year = years[-1] if years else None
        train = df.loc[df["auction_year"].ne(split_year)].copy()
        test = df.loc[df["auction_year"].eq(split_year)].copy()
    return train, test


def _build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_columns = [col for col in BASE_FEATURE_COLUMNS + CATEGORICAL_FEATURES if col in df.columns]
    numeric_features = [col for col in feature_columns if col not in CATEGORICAL_FEATURES]
    categorical_features = [col for col in feature_columns if col in CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return preprocessor, feature_columns, numeric_features


def evaluate_predictions(actual: pd.Series, predicted_log: np.ndarray) -> Dict[str, float]:
    actual_log = np.log1p(actual)
    mae = mean_absolute_error(actual_log, predicted_log)
    rmse = float(np.sqrt(mean_squared_error(actual_log, predicted_log)))
    r2 = r2_score(actual_log, predicted_log) if len(actual) > 1 else np.nan
    return {"mae_log": mae, "rmse_log": rmse, "r2_log": r2}


def train_models(model_df: pd.DataFrame) -> Dict[str, ModelBundle]:
    if model_df.empty:
        return {}
    scored_df = scorecard_fair_price(model_df)
    intrinsic_floor_log = (
        np.log1p(scored_df["intrinsic_predicted_price_in_inr"])
        if "intrinsic_predicted_price_in_inr" in scored_df.columns
        else scored_df["scorecard_predicted_log_price"]
    )
    scored_df["market_premium_target_log"] = scored_df["target_log_price"] - intrinsic_floor_log
    train_df, test_df = year_holdout_split(scored_df)
    bundles: Dict[str, ModelBundle] = {}

    scorecard_metrics = evaluate_predictions(test_df["price_in_inr"], test_df["scorecard_predicted_log_price"].to_numpy())
    scorecard_predictions = scored_df[
        ["player_name", "auction_year", "price_in_inr", "scorecard_predicted_log_price", "scorecard_predicted_price_in_inr"]
    ].rename(
        columns={
            "scorecard_predicted_log_price": "predicted_log_price",
            "scorecard_predicted_price_in_inr": "predicted_price_in_inr",
        }
    )
    bundles["scorecard"] = ModelBundle(
        name="scorecard",
        estimator=None,
        feature_columns=[],
        metrics=scorecard_metrics,
        predictions=scorecard_predictions,
    )

    preprocessor, feature_columns, _ = _build_preprocessor(scored_df)
    estimators = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    x_train = train_df[feature_columns]
    y_train = train_df["target_log_price"]
    x_test = test_df[feature_columns]

    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        test_pred_log = pipeline.predict(x_test)
        metrics = evaluate_predictions(test_df["price_in_inr"], test_pred_log)
        full_pred_log = pipeline.predict(scored_df[feature_columns])
        full_pred_inr = np.expm1(full_pred_log)
        predictions = scored_df[["player_name", "auction_year", "price_in_inr"]].copy()
        predictions["predicted_log_price"] = full_pred_log
        predictions["predicted_price_in_inr"] = full_pred_inr
        bundles[name] = ModelBundle(
            name=name,
            estimator=pipeline,
            feature_columns=feature_columns,
            metrics=metrics,
            predictions=predictions,
        )

    premium_features = [
        "auction_year",
        "performance_season",
        "inflation_index",
        "spend_index",
        "year_median_price_in_inr",
        "role_year_median_price_in_inr",
        "is_mega_auction_proxy",
        "leadership_proxy",
        "captaincy_proxy",
        "title_winning_captain_proxy",
        "indian_batter_premium_proxy",
        "sena_overseas_premium_proxy",
        "wicketkeeper_scarcity_proxy",
        "death_bowler_premium_proxy",
        "control_bowler_premium_proxy",
        "left_arm_bowler_proxy",
        "premium_bowler_quality_index",
        "elite_indian_pacer_scarcity_proxy",
        "marquee_player_proxy",
        "previous_auction_price_in_inr",
        "auction_history_mean_price_in_inr",
        "career_peak_auction_price_in_inr",
        "prior_auction_count",
        "years_since_last_auction",
        "experience_proxy",
        "prior_seasons_played",
        "career_runs_before_auction",
        "career_wickets_before_auction",
        "three_year_runs_before_auction",
        "three_year_wickets_before_auction",
        "three_year_batting_sr_before_auction",
        "three_year_economy_before_auction",
        "prior_500_run_seasons",
        "prior_20_wicket_seasons",
        "pedigree_batter_index",
        "pedigree_bowler_index",
        "role_bucket",
        "team",
        "is_overseas",
        "is_wicketkeeper",
        "is_allrounder",
    ]
    premium_features = [col for col in premium_features if col in scored_df.columns]
    premium_numeric = [col for col in premium_features if col not in CATEGORICAL_FEATURES]
    premium_categorical = [col for col in premium_features if col in CATEGORICAL_FEATURES]
    premium_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                premium_numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                premium_categorical,
            ),
        ]
    )
    premium_model = Pipeline(
        [
            ("preprocessor", premium_preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )
    premium_model.fit(train_df[premium_features], train_df["market_premium_target_log"])
    test_premium_pred = premium_model.predict(test_df[premium_features])
    test_two_stage_pred = test_df["scorecard_predicted_log_price"].to_numpy() + test_premium_pred
    two_stage_metrics = evaluate_predictions(test_df["price_in_inr"], test_two_stage_pred)
    full_premium_pred = premium_model.predict(scored_df[premium_features])
    full_two_stage_pred_log = scored_df["scorecard_predicted_log_price"].to_numpy() + full_premium_pred
    full_two_stage_pred_inr = np.expm1(full_two_stage_pred_log)
    two_stage_predictions = scored_df[["player_name", "auction_year", "price_in_inr"]].copy()
    two_stage_predictions["predicted_log_price"] = full_two_stage_pred_log
    two_stage_predictions["predicted_price_in_inr"] = full_two_stage_pred_inr
    two_stage_predictions["market_premium_log"] = full_premium_pred
    bundles["two_stage_market"] = ModelBundle(
        name="two_stage_market",
        estimator=premium_model,
        feature_columns=premium_features,
        metrics=two_stage_metrics,
        predictions=two_stage_predictions,
    )
    return bundles


def select_primary_model(bundles: Dict[str, ModelBundle]) -> ModelBundle:
    ranking = sorted(
        bundles.values(),
        key=lambda bundle: (
            np.inf if np.isnan(bundle.metrics.get("mae_log", np.nan)) else bundle.metrics.get("mae_log", np.inf),
            np.inf if np.isnan(bundle.metrics.get("rmse_log", np.nan)) else bundle.metrics.get("rmse_log", np.inf),
        ),
    )
    return ranking[0]


def add_mispricing_columns(base_df: pd.DataFrame, prediction_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = base_df.merge(prediction_df, on=["player_name", "auction_year", "price_in_inr"], how="left")
    intrinsic_df = scorecard_fair_price(base_df.copy())[
        ["player_name", "auction_year", "price_in_inr", "scorecard_predicted_log_price", "scorecard_predicted_price_in_inr"]
    ].rename(
        columns={
            "scorecard_predicted_log_price": "intrinsic_predicted_log_price",
            "scorecard_predicted_price_in_inr": "intrinsic_predicted_price_in_inr",
        }
    )
    out = out.merge(intrinsic_df, on=["player_name", "auction_year", "price_in_inr"], how="left")
    out["selected_model"] = model_name
    out["predicted_price_in_inr"] = out["predicted_price_in_inr"].clip(lower=0)
    out["predicted_price_in_crore"] = out["predicted_price_in_inr"] / 10_000_000.0
    out["intrinsic_predicted_price_in_crore"] = out["intrinsic_predicted_price_in_inr"] / 10_000_000.0
    role_price_anchor = (
        out["role_year_median_price_in_inr"]
        .fillna(out["year_median_price_in_inr"])
        .fillna(out["price_in_inr"].median())
    )
    history_anchor = (
        0.50 * out["previous_auction_price_in_inr"].fillna(0)
        + 0.30 * out["career_peak_auction_price_in_inr"].fillna(0)
        + 0.20 * out["auction_history_mean_price_in_inr"].fillna(0)
    )
    history_anchor = history_anchor.where(history_anchor.gt(0), role_price_anchor)
    predicted_anchor = out["predicted_price_in_inr"].fillna(0).to_numpy()
    intrinsic_anchor = out["intrinsic_predicted_price_in_inr"].fillna(0).to_numpy()
    model_anchor = np.maximum(predicted_anchor, intrinsic_anchor)
    blended_anchor = np.select(
        [
            out["role_bucket"].eq("bowler"),
            out["role_bucket"].eq("all_rounder"),
            out["role_bucket"].isin(["batter", "wicketkeeper"]),
        ],
        [
            0.35 * predicted_anchor + 0.45 * intrinsic_anchor + 0.20 * history_anchor.to_numpy(),
            0.40 * predicted_anchor + 0.35 * intrinsic_anchor + 0.25 * history_anchor.to_numpy(),
            0.40 * predicted_anchor + 0.15 * intrinsic_anchor + 0.45 * history_anchor.to_numpy(),
        ],
        default=0.50 * predicted_anchor + 0.20 * intrinsic_anchor + 0.30 * history_anchor.to_numpy(),
    )
    marquee_batter_mask = (
        out["role_bucket"].isin(["batter", "wicketkeeper"])
        & (
            out["marquee_player_proxy"].fillna(0).eq(1)
            | out["captaincy_proxy"].fillna(0).eq(1)
            | out["title_winning_captain_proxy"].fillna(0).eq(1)
            | out["pedigree_batter_index"].fillna(0).ge(0.88)
        )
    )
    elite_bowler_mask = (
        out["role_bucket"].eq("bowler")
        & (
            out["elite_indian_pacer_scarcity_proxy"].fillna(0).eq(1)
            | out["premium_bowler_quality_index"].fillna(0).ge(0.82)
            | (
                out["pedigree_bowler_index"].fillna(0).ge(0.88)
                & (
                    out["death_bowler_premium_proxy"].fillna(0).eq(1)
                    | out["control_bowler_premium_proxy"].fillna(0).eq(1)
                )
            )
        )
    )
    strong_bowler_mask = (
        out["role_bucket"].eq("bowler")
        & ~elite_bowler_mask
        & out["premium_bowler_quality_index"].fillna(0).ge(0.72)
    )
    market_reference_in_inr = np.select(
        [
            out["role_bucket"].eq("bowler"),
            out["role_bucket"].eq("all_rounder"),
            out["role_bucket"].isin(["batter", "wicketkeeper"]),
        ],
        [
            np.maximum(
                0.92 * blended_anchor,
                (0.45 * predicted_anchor + 0.25 * history_anchor.to_numpy() + 0.30 * role_price_anchor.to_numpy()),
            ),
            np.maximum(
                0.90 * blended_anchor,
                (0.45 * blended_anchor + 0.25 * history_anchor.to_numpy() + 0.30 * role_price_anchor.to_numpy()),
            ),
            np.maximum(
                0.95 * blended_anchor,
                (0.35 * predicted_anchor + 0.45 * history_anchor.to_numpy() + 0.20 * role_price_anchor.to_numpy()),
            ),
        ],
        default=np.maximum(blended_anchor, (0.55 * history_anchor + 0.45 * role_price_anchor).to_numpy()),
    )
    marquee_batter_reference = np.maximum(
        blended_anchor,
        (1.00 * history_anchor.to_numpy()),
    )
    market_reference_in_inr = np.where(
        marquee_batter_mask,
        np.maximum(market_reference_in_inr, marquee_batter_reference),
        market_reference_in_inr,
    )
    elite_pacer_reference = np.maximum(
        blended_anchor,
        (1.30 * role_price_anchor.to_numpy()),
    )
    market_reference_in_inr = np.where(
        elite_bowler_mask,
        np.maximum(
            market_reference_in_inr,
            np.maximum(
                1.10 * role_price_anchor.to_numpy(),
                0.45 * predicted_anchor + 0.20 * intrinsic_anchor + 0.20 * history_anchor.to_numpy() + 0.15 * role_price_anchor.to_numpy(),
            ),
        ),
        market_reference_in_inr,
    )
    market_reference_in_inr = np.where(
        strong_bowler_mask,
        np.maximum(
            market_reference_in_inr,
            0.50 * predicted_anchor + 0.20 * intrinsic_anchor + 0.20 * history_anchor.to_numpy() + 0.10 * role_price_anchor.to_numpy(),
        ),
        market_reference_in_inr,
    )
    market_reference_in_inr = np.where(
        out["elite_indian_pacer_scarcity_proxy"].fillna(0).eq(1),
        np.maximum(market_reference_in_inr, elite_pacer_reference),
        market_reference_in_inr,
    )
    pedigree_heat = np.select(
        [
            out["role_bucket"].isin(["batter", "wicketkeeper"]),
            out["role_bucket"].eq("bowler"),
            out["role_bucket"].eq("all_rounder"),
        ],
        [
            (
                0.12 * out["pedigree_batter_index"].fillna(0)
                + 0.06 * out["prior_500_run_seasons"].fillna(0).rank(pct=True)
                + 0.04 * out["career_best_runs_season_before_auction"].fillna(0).rank(pct=True)
            ),
            (
                0.12 * out["pedigree_bowler_index"].fillna(0)
                + 0.05 * out["prior_20_wicket_seasons"].fillna(0).rank(pct=True)
                + 0.04 * out["career_best_wickets_season_before_auction"].fillna(0).rank(pct=True)
            ),
            (
                0.08 * out["pedigree_batter_index"].fillna(0)
                + 0.08 * out["pedigree_bowler_index"].fillna(0)
            ),
        ],
        default=0.0,
    )
    performance_heat = np.select(
        [
            out["role_bucket"].isin(["batter", "wicketkeeper"]),
            out["role_bucket"].eq("bowler"),
            out["role_bucket"].eq("all_rounder"),
        ],
        [
            (
                0.10 * out["runs_scored"].fillna(0).rank(pct=True)
                + 0.12 * out["batting_strike_rate"].fillna(out["batting_strike_rate"].median()).rank(pct=True)
            ),
            (
                0.12 * out["wickets"].fillna(0).rank(pct=True)
                + 0.08 * (1 - out["economy_rate"].fillna(out["economy_rate"].median()).rank(pct=True))
                + 0.05 * (1 - out["bowling_strike_rate"].fillna(out["bowling_strike_rate"].median()).rank(pct=True))
            ),
            (
                0.08 * out["runs_scored"].fillna(0).rank(pct=True)
                + 0.08 * out["batting_strike_rate"].fillna(out["batting_strike_rate"].median()).rank(pct=True)
                + 0.08 * out["wickets"].fillna(0).rank(pct=True)
            ),
        ],
        default=0.05,
    )
    premium_multiplier_raw = (
        0.14 * out["marquee_player_proxy"].fillna(0)
        + 0.10 * out["captaincy_proxy"].fillna(0)
        + 0.08 * out["title_winning_captain_proxy"].fillna(0)
        + 0.18 * out["wicketkeeper_scarcity_proxy"].fillna(0)
        + 0.07 * out["indian_batter_premium_proxy"].fillna(0)
        + 0.07 * out["sena_overseas_premium_proxy"].fillna(0)
        + 0.12 * out["death_bowler_premium_proxy"].fillna(0)
        + 0.10 * out["control_bowler_premium_proxy"].fillna(0)
        + 0.14 * out["premium_bowler_quality_index"].fillna(0)
        + 0.05 * out["left_arm_bowler_proxy"].fillna(0)
        + 0.12 * out["elite_indian_pacer_scarcity_proxy"].fillna(0)
        + 0.05 * out["leadership_proxy"].fillna(0)
        + pedigree_heat
        + performance_heat
    ).clip(lower=0)
    premium_multiplier = np.select(
        [
            out["role_bucket"].eq("bowler"),
            out["role_bucket"].eq("all_rounder"),
            out["role_bucket"].eq("wicketkeeper"),
        ],
        [
            np.where(
                elite_bowler_mask,
                (0.65 * premium_multiplier_raw + 0.06 * out["premium_bowler_quality_index"].fillna(0)).clip(upper=0.20),
                np.where(
                    strong_bowler_mask,
                    (0.45 * premium_multiplier_raw + 0.04 * out["premium_bowler_quality_index"].fillna(0)).clip(upper=0.12),
                    (0.30 * premium_multiplier_raw).clip(upper=0.06),
                ),
            ),
            (0.75 * premium_multiplier_raw).clip(upper=0.32),
            np.where(
                marquee_batter_mask,
                premium_multiplier_raw.clip(upper=0.42),
                premium_multiplier_raw.clip(upper=0.40),
            ),
        ],
        default=np.where(marquee_batter_mask, premium_multiplier_raw.clip(upper=0.45), premium_multiplier_raw.clip(upper=0.25)),
    )
    out["article_fair_price_in_inr"] = market_reference_in_inr * (1 + premium_multiplier)
    out["article_market_premium_in_inr"] = out["article_fair_price_in_inr"] - model_anchor
    out["article_fair_price_in_crore"] = out["article_fair_price_in_inr"] / 10_000_000.0
    out["actual_price_in_crore"] = out["price_in_inr"] / 10_000_000.0
    out["mispricing_in_inr"] = out["predicted_price_in_inr"] - out["price_in_inr"]
    out["mispricing_pct"] = out["mispricing_in_inr"] / out["price_in_inr"].replace({0: np.nan})
    out["mispricing_label"] = quantile_labels(out["mispricing_in_inr"])
    out["article_mispricing_in_inr"] = out["article_fair_price_in_inr"] - out["price_in_inr"]
    out["article_mispricing_pct"] = out["article_mispricing_in_inr"] / out["price_in_inr"].replace({0: np.nan})
    out["article_mispricing_label"] = quantile_labels(out["article_mispricing_in_inr"])
    out["intrinsic_mispricing_in_inr"] = out["intrinsic_predicted_price_in_inr"] - out["price_in_inr"]
    out["intrinsic_mispricing_pct"] = out["intrinsic_mispricing_in_inr"] / out["price_in_inr"].replace({0: np.nan})
    return out


def model_comparison_table(bundles: Dict[str, ModelBundle]) -> pd.DataFrame:
    records = []
    for name, bundle in bundles.items():
        records.append({"model": name, **bundle.metrics})
    return pd.DataFrame(records).sort_values(["mae_log", "rmse_log"])
