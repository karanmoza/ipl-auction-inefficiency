# IPL Auction Inefficiency

This project analyzes inefficiencies in the IPL auction by comparing player prices to a data-driven estimate of their fair value.

The analysis builds a pre-season fair-price estimate for each player using only information available before the auction. It combines prior IPL performance, multi-year pedigree, role scarcity, leadership signals, and auction history, then compares the model-implied fair price with the actual auction clearing price.

The main finding is that the IPL auction is not irrational, but it does tend to overpay for scarcity. Expensive players are often genuinely valuable, yet wicketkeepers, Indian core batters, and high-leverage bowlers can still command premiums that run ahead of modeled fair value. The cleaner bargains often sit one tier below the glamour names.

The best holdout model in this repo is a gradient-boosted regressor with:

- `MAE (log price): 0.68`
- `RMSE (log price): 0.93`
- `R^2: 0.50`

For a narrative version of the project, see `docs/substack_article.html`.

## Problem

The project is designed to answer four questions:

1. Do IPL auction prices align with measurable cricketing value?
2. Which players were undervalued and overvalued?
3. Which player archetypes appear systematically mispriced?
4. Which franchises seem to buy value more efficiently than others?

## Method

1. Discover raw CSV/XLSX files recursively and classify them as auction, matches, or deliveries data.
2. Normalize schemas, standardize player names, and resolve common aliases conservatively.
3. Build player-season batting and bowling features from historical IPL performance.
4. Map auction year `Y` to information available before season `Y` begins.
5. Train multiple pricing models on historical auction outcomes.
6. Select the strongest holdout model and compute:
   - `mispricing = fair_price - actual_price`
   - positive values imply undervaluation
   - negative values imply overvaluation
7. Summarize mispricing by player, role bucket, domestic versus overseas status, and franchise.

## Data Sources

Place Kaggle downloads inside `data/raw/`:

- Auction + IPL dataset: `https://www.kaggle.com/datasets/nkitgupta/ipl-auction-and-ipl-dataset`
- Ball-by-ball dataset: `https://www.kaggle.com/datasets/dgsports/ipl-ball-by-ball-2008-to-2022`
- Expanded auction years dataset: `https://www.kaggle.com/datasets/sunnyyadav754/ipl-auction-dataset-20132026`

The pipeline is built to handle messy Kaggle downloads by discovering files recursively instead of relying on exact filenames.

## Data Coverage

Due to name-matching constraints and incomplete historical records, the final modeling dataset covers `577` of `1381` auction entries, or `41.8%` of the raw auction rows.

This likely biases the analysis toward players with more consistent IPL participation and cleaner historical data. The results should therefore be interpreted as a benchmark on this subset rather than a full-market estimate.

## What Drives The Model

At a high level, the pricing system is designed to respond most strongly to:

- recent batting output and strike-rate quality
- wicket-taking ability and bowling efficiency
- role scarcity proxies such as wicketkeeping and high-leverage bowling
- multi-year IPL pedigree and prior auction reputation
- market-context features such as auction-year inflation and role-level pricing

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/ipl_auction_inefficiency_end_to_end.ipynb
```

## Reproduce

1. Download the three Kaggle datasets and extract them under `data/raw/`.
2. Create a virtual environment and install `requirements.txt`.
3. Run `notebooks/ipl_auction_inefficiency_end_to_end.ipynb` top to bottom.
4. Review exported artifacts in:
   - `outputs/tables/`
   - `outputs/charts/`
   - `data/processed/`

## Key Outputs

Tables:

- `outputs/tables/final_model_dataset.csv`
- `outputs/tables/player_mispricing_table.csv`
- `outputs/tables/top_undervalued_players.csv`
- `outputs/tables/top_overvalued_players.csv`
- `outputs/tables/franchise_value_summary.csv`
- `outputs/tables/role_mispricing_summary.csv`

Charts:

- `outputs/charts/price_vs_predicted_price.png`
- `outputs/charts/top_15_undervalued.png`
- `outputs/charts/top_15_overvalued.png`
- `outputs/charts/role_mispricing_boxplot.png`
- `outputs/charts/franchise_value_efficiency.png`
- `outputs/charts/quadrant_value_vs_price.png`

Article artifacts:

- `article/substack_draft.md`
- `docs/substack_article.html`

## Limitations

- T20 performance is noisy, and individual seasons can be small-sample events.
- Auction prices reflect roster fit, scarcity, leadership, and bidding dynamics, not just pure ability.
- Overseas-slot constraints distort direct cross-player comparisons.
- Retained-player economics sit outside the auction and are intentionally excluded.
- This model is not designed to estimate a single "true" player price, but to provide a consistent benchmark for comparing auction outcomes.
- The model is a disciplined benchmark, not a claim to a single true player price.

## Project Structure

```text
ipl-auction-inefficiency/
├── article/
│   └── substack_draft.md
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── substack_article.html
├── notebooks/
│   └── ipl_auction_inefficiency_end_to_end.ipynb
├── outputs/
│   ├── charts/
│   └── tables/
├── src/
│   ├── cleaning.py
│   ├── data_loader.py
│   ├── features.py
│   ├── modeling.py
│   ├── utils.py
│   └── visuals.py
├── requirements.txt
└── README.md
```
