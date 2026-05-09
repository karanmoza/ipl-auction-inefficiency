from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BCG_NAVY = "#16324F"
BCG_BLUE = "#5B6573"
BCG_TEAL = "#2E8B57"
BCG_GREEN = "#2E8B57"
BCG_RED = "#9F1D20"
BCG_GOLD = "#B88A1B"
BCG_GRID = "#D9DEE5"
BCG_TEXT = "#1F2933"
BCG_LIGHT = "#F5F1EB"


def set_matplotlib_theme() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 6.3),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": BCG_GRID,
            "axes.linewidth": 0.8,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.titlecolor": BCG_NAVY,
            "axes.labelsize": 11,
            "axes.labelcolor": BCG_TEXT,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": BCG_TEXT,
            "ytick.color": BCG_TEXT,
            "grid.color": BCG_GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _add_subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=10, color=BCG_BLUE)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BCG_GRID)
    ax.spines["bottom"].set_color(BCG_GRID)
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _add_value_labels_barh(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
    for patch, value in zip(ax.patches, values):
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        offset = 0.18 if x >= 0 else -0.18
        ha = "left" if x >= 0 else "right"
        ax.text(x + offset, y, f"{value:.1f}", va="center", ha=ha, fontsize=9, color=BCG_TEXT)


def _format_crore_ticks(values: pd.Series | np.ndarray) -> list[str]:
    return [f"{value:.0f}" for value in values]


def _highlight_player(ax: plt.Axes, df: pd.DataFrame, player_name: str = "Rishabh Pant") -> None:
    focus = df.loc[df["player_name"].astype(str).str.lower().eq(player_name.lower())].copy()
    if focus.empty:
        return
    latest = focus.sort_values("auction_year").tail(1)
    row = latest.iloc[0]
    y_value = (
        row["article_fair_price_in_crore"]
        if "article_fair_price_in_crore" in latest.columns
        else row["predicted_price_in_crore"]
    )
    ax.scatter(
        [row["actual_price_in_crore"]],
        [y_value],
        s=120,
        color=BCG_GOLD,
        edgecolor=BCG_NAVY,
        linewidth=1.2,
        zorder=4,
    )
    ax.annotate(
        f"{row['player_name']} ({int(row['auction_year'])})",
        (row["actual_price_in_crore"], y_value),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color=BCG_NAVY,
        weight="bold",
    )


def plot_price_vs_predicted(
    df: pd.DataFrame, output_path: Path | str, focus_player: str = "Rishabh Pant"
) -> None:
    fig, ax = plt.subplots()
    ax.set_facecolor("#FCFBF8")
    ax.scatter(
        df["actual_price_in_crore"],
        df.get("article_fair_price_in_crore", df["predicted_price_in_crore"]),
        alpha=0.55,
        s=42,
        color=BCG_BLUE,
        edgecolor="white",
        linewidth=0.3,
    )
    y_series = df.get("article_fair_price_in_crore", df["predicted_price_in_crore"])
    max_val = np.nanmax([df["actual_price_in_crore"].max(), y_series.max(), 0])
    ax.plot([0, max_val], [0, max_val], linestyle="--", color=BCG_RED, linewidth=1.5)
    _highlight_player(ax, df, focus_player)
    _style_axes(ax)
    ax.grid(axis="x", alpha=0.20)
    ax.set_title("Actual Price vs Article Fair Price", loc="left", pad=28)
    _add_subtitle(
        ax, "Fair price here means intrinsic cricket value plus an estimated market premium."
    )
    ax.set_xlabel("Observed auction price (crore INR)")
    ax.set_ylabel("Article fair price (crore INR)")
    _save(fig, output_path)


def plot_top_mispricing(
    df: pd.DataFrame,
    output_path: Path | str,
    kind: str = "undervalued",
    top_n: int = 15,
    focus_player: str = "Rishabh Pant",
    mispricing_col: str = "mispricing_in_inr",
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    ascending = kind == "overvalued"
    top = (
        df.sort_values(mispricing_col, ascending=ascending).head(top_n).sort_values(mispricing_col)
    )
    fig, ax = plt.subplots()
    ax.set_facecolor("#FCFBF8")
    values = top[mispricing_col] / 10_000_000.0
    base_color = BCG_TEAL if kind == "undervalued" else BCG_RED
    colors = [base_color] * len(top)
    if focus_player and focus_player in top["player_name"].values:
        focus_idx = list(top["player_name"]).index(focus_player)
        colors[focus_idx] = BCG_GOLD
    ax.barh(top["player_name"], values, color=colors, alpha=0.92)
    _style_axes(ax)
    ax.grid(axis="x", alpha=0.8)
    ax.set_title(
        title or f"Top {top_n} {kind.replace('_', ' ').title()} IPL Auction Calls",
        loc="left",
        pad=28,
    )
    _add_subtitle(
        ax,
        subtitle
        or "Positive bars suggest bargains; negative bars suggest the market likely paid above fair value.",
    )
    ax.set_xlabel("Mispricing (crore INR)")
    ax.set_ylabel("Player")
    if kind == "overvalued":
        ax.axvline(0, color=BCG_NAVY, linewidth=1.0, alpha=0.8)
    else:
        ax.axvline(0, color=BCG_NAVY, linewidth=1.0, alpha=0.45)
    _add_value_labels_barh(ax, values)
    _save(fig, output_path)


def plot_role_boxplot(df: pd.DataFrame, output_path: Path | str) -> None:
    roles = [
        role
        for role in ["batter", "bowler", "all_rounder", "wicketkeeper", "unknown"]
        if role in df["role_bucket"].unique()
    ]
    data = [
        df.loc[df["role_bucket"].eq(role), "mispricing_in_inr"] / 10_000_000.0 for role in roles
    ]
    fig, ax = plt.subplots()
    ax.set_facecolor("#FCFBF8")
    box = ax.boxplot(
        data,
        labels=roles,
        patch_artist=True,
        medianprops={"color": BCG_NAVY, "linewidth": 1.8},
        whiskerprops={"color": BCG_BLUE},
        capprops={"color": BCG_BLUE},
    )
    palette = [BCG_BLUE, BCG_TEAL, BCG_GREEN, "#8BB8D8", "#B8C7D9"]
    for patch, color in zip(box["boxes"], palette[: len(box["boxes"])]):
        patch.set(facecolor=color, alpha=0.5, edgecolor=BCG_NAVY)
    _style_axes(ax)
    ax.grid(axis="x", alpha=0.0)
    ax.axhline(0, color=BCG_NAVY, linewidth=1.0, alpha=0.8)
    ax.set_title("Role-Level Distribution of Auction Mispricing", loc="left", pad=28)
    _add_subtitle(ax, "This shows which archetypes tend to attract persistent premiums.")
    ax.set_ylabel("Mispricing (crore INR)")
    ax.set_xlabel("Role bucket")
    _save(fig, output_path)


def plot_franchise_efficiency(df: pd.DataFrame, output_path: Path | str) -> None:
    summary = (
        df.groupby("team", dropna=False)
        .agg(avg_mispricing_captured=("mispricing_in_inr", "mean"))
        .sort_values("avg_mispricing_captured")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#FCFBF8")
    values = summary["avg_mispricing_captured"] / 10_000_000.0
    colors = [BCG_TEAL if value > 0 else BCG_BLUE for value in values]
    ax.bar(summary["team"].fillna("Unknown"), values, color=colors, alpha=0.92)
    _style_axes(ax)
    ax.grid(axis="x", alpha=0.0)
    ax.axhline(0, color=BCG_NAVY, linewidth=1.0, alpha=0.8)
    ax.set_title("Franchise Value Capture by Auction Spend", loc="left", pad=28)
    _add_subtitle(
        ax, "Read this as a directional buying-discipline view, not a final front-office ranking."
    )
    ax.set_ylabel("Average mispricing captured (crore INR)")
    ax.set_xlabel("Franchise")
    ax.tick_params(axis="x", rotation=45)
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + (0.12 if value >= 0 else -0.18),
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
            color=BCG_TEXT,
        )
    _save(fig, output_path)


def plot_quadrant(
    df: pd.DataFrame,
    output_path: Path | str,
    annotate_n: int = 8,
    focus_player: str = "Rishabh Pant",
) -> None:
    fig, ax = plt.subplots()
    ax.set_facecolor("#FCFBF8")
    ax.scatter(
        df["actual_price_in_crore"],
        df.get("article_fair_price_in_crore", df["predicted_price_in_crore"]),
        alpha=0.5,
        s=45,
        color=BCG_BLUE,
        edgecolor="white",
        linewidth=0.3,
    )
    x_mid = df["actual_price_in_crore"].median()
    y_series = df.get("article_fair_price_in_crore", df["predicted_price_in_crore"])
    y_mid = y_series.median()
    ax.axvspan(0, x_mid, ymin=0, ymax=1, color="#EEF4F8", alpha=0.45, zorder=0)
    ax.axvspan(
        x_mid,
        df["actual_price_in_crore"].max(),
        ymin=0,
        ymax=1,
        color="#FAF1EE",
        alpha=0.35,
        zorder=0,
    )
    ax.axhspan(y_mid, y_series.max(), xmin=0, xmax=1, color="#EDF7F0", alpha=0.20, zorder=0)
    ax.axvline(x_mid, linestyle="--", color=BCG_NAVY, linewidth=1.2)
    ax.axhline(y_mid, linestyle="--", color=BCG_NAVY, linewidth=1.2)
    notable = df.reindex(
        df["mispricing_in_inr"].abs().sort_values(ascending=False).head(annotate_n).index
    )
    for _, row in notable.iterrows():
        fontweight = "bold" if str(row["player_name"]).lower() == focus_player.lower() else "normal"
        color = BCG_GOLD if str(row["player_name"]).lower() == focus_player.lower() else BCG_NAVY
        ax.annotate(
            row["player_name"],
            (
                row["actual_price_in_crore"],
                row.get("article_fair_price_in_crore", row["predicted_price_in_crore"]),
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight=fontweight,
        )
    _highlight_player(ax, df, focus_player)
    _style_axes(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Value Matrix: Market Price vs Fair Price", loc="left", pad=28)
    _add_subtitle(
        ax, "Top-left is where teams find bargains; bottom-right is where hype outruns value."
    )
    ax.set_xlabel("Observed auction price (crore INR)")
    ax.set_ylabel("Article fair price (crore INR)")
    ax.text(
        x_mid * 0.35, y_mid * 1.12, "CHEAP / HIGH VALUE", fontsize=8, color=BCG_TEAL, weight="bold"
    )
    ax.text(
        x_mid * 1.05,
        y_mid * 1.12,
        "EXPENSIVE / HIGH VALUE",
        fontsize=8,
        color=BCG_NAVY,
        weight="bold",
    )
    ax.text(
        x_mid * 0.35, y_mid * 0.72, "CHEAP / LOWER VALUE", fontsize=8, color=BCG_BLUE, weight="bold"
    )
    ax.text(
        x_mid * 1.03,
        y_mid * 0.72,
        "EXPENSIVE / LOWER VALUE",
        fontsize=8,
        color=BCG_RED,
        weight="bold",
    )
    _save(fig, output_path)
