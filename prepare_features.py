import pandas as pd
import numpy as np
import sys
import os


def prepare_features(input_filepath: str) -> str:
    df = pd.read_csv(input_filepath, parse_dates=True)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # ------------------------------------------------------------------ #
    # 1. RETURN-BASED REPLACEMENTS FOR RAW PRICES                         #
    # ------------------------------------------------------------------ #

    # Core bar returns
    df["Return"]          = df["Close(t)"].pct_change()
    df["Open_Return"]     = (df["Open"] - df["Close(t)"].shift(1)) / df["Close(t)"].shift(1)
    df["High_Return"]     = (df["High"] - df["Close(t)"].shift(1)) / df["Close(t)"].shift(1)
    df["Low_Return"]      = (df["Low"]  - df["Close(t)"].shift(1)) / df["Close(t)"].shift(1)

    # Lagged returns (replace raw lagged closes)
    for lag in [1, 2, 3, 5]:
        df[f"Return_t-{lag}"] = df["Return"].shift(lag)

    # Price relative to its own moving averages (ratio, not raw)
    for ma in [5, 10, 20, 50, 200]:
        col = f"MA{ma}"
        if col in df.columns:
            df[f"Close_vs_MA{ma}"] = df["Close(t)"] / df[col] - 1

    for ema in [10, 20, 50, 100, 200]:
        col = f"EMA{ema}"
        if col in df.columns:
            df[f"Close_vs_EMA{ema}"] = df["Close(t)"] / df[col] - 1

    # Bollinger band position (normalised, already relative)
    if "Upper_Band" in df.columns and "Lower_Band" in df.columns:
        band_width = df["Upper_Band"] - df["Lower_Band"]
        df["BB_Position"] = np.where(
            band_width != 0,
            (df["Close(t)"] - df["Lower_Band"]) / band_width,
            0.5
        )
        df["BB_Width"] = np.where(
            df["Close(t)"] != 0,
            band_width / df["Close(t)"],
            np.nan
        )

    # ------------------------------------------------------------------ #
    # 2. MICROSTRUCTURE / WITHIN-BAR FEATURES                             #
    # ------------------------------------------------------------------ #

    bar_range = df["High"] - df["Low"]
    df["Close_Position"] = np.where(
        bar_range != 0,
        (df["Close(t)"] - df["Low"]) / bar_range,
        0.5
    )
    df["Bar_Range_Pct"] = np.where(
        df["Close(t)"].shift(1) != 0,
        bar_range / df["Close(t)"].shift(1),
        np.nan
    )

    # ------------------------------------------------------------------ #
    # 3. VOLUME NORMALISATION                                              #
    # ------------------------------------------------------------------ #

    if "Volume" in df.columns:
        df["Volume_MA20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = np.where(
            df["Volume_MA20"] != 0,
            df["Volume"] / df["Volume_MA20"],
            np.nan
        )

    # ------------------------------------------------------------------ #
    # 4. TREND REGIME BINARY                                              #
    # ------------------------------------------------------------------ #

    if "MA200" in df.columns:
        df["Above_MA200"] = (df["Close(t)"] > df["MA200"]).astype(int)

    # ------------------------------------------------------------------ #
    # 5. RELATIVE PERFORMANCE VS INDEX                                    #
    # ------------------------------------------------------------------ #

    for index_col, return_col in [
        ("QQQ_Close",  "QQQ_Return"),
        ("SnP_Close",  "SnP_Return"),
        ("DJIA_Close", "DJIA_Return"),
    ]:
        if index_col in df.columns:
            df[return_col] = df[index_col].pct_change()
            df[f"Rel_{return_col}"] = df["Return"] - df[return_col]

    # Lagged index returns
    for index_close, prefix in [
        ("QQQ_Close",  "QQQ"),
        ("SnP_Close",  "SnP"),
        ("DJIA_Close", "DJIA"),
    ]:
        if index_close in df.columns:
            base_ret = df[index_close].pct_change()
            for lag in [1, 2, 5]:
                df[f"{prefix}_Return_t-{lag}"] = base_ret.shift(lag)

    # ------------------------------------------------------------------ #
    # 6. DROP UNWANTED COLUMNS                                            #
    # ------------------------------------------------------------------ #

    cols_to_drop = [
        # Raw non-stationary prices
        "Open", "High", "Low", "Close(t)",
        "Upper_Band", "Lower_Band",
        "MA5", "MA10", "MA20", "MA50", "MA200",
        "EMA10", "EMA20", "EMA50", "EMA100", "EMA200",
        # Raw lagged closes (replaced by return lags)
        "S_Close(t-1)", "S_Close(t-2)", "S_Close(t-3)", "S_Close(t-5)",
        "S_Open(t-1)",
        # Raw index prices and their raw lags
        "QQQ_Close", "QQQ(t-1)", "QQQ(t-2)", "QQQ(t-5)", "QQQ_MA10", "QQQ_MA20", "QQQ_MA50",
        "SnP_Close",  "SnP(t-1)", "SnP(t-5)",
        "DJIA_Close", "DJIA(t-1)", "DJIA(t-5)",
        # Raw volume (replaced by ratio)
        "Volume", "Volume_MA20",
        # Noisy date features
        "Day", "DayofWeek", "DayofYear", "Week",
        "Is_leap_year", "Year", "Month",
    ]

    # Only drop columns that actually exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Calendar anomaly columns worth keeping (comment out if unwanted)
    # Is_month_end, Is_month_start, Is_quarter_end, Is_quarter_start,
    # Is_year_end, Is_year_start are kept as they have real rebalancing basis

    # ------------------------------------------------------------------ #
    # 7. DROP NaN ROWS CREATED BY ROLLING / SHIFTING                     #
    # ------------------------------------------------------------------ #

    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    print(f"Dropped {rows_before - rows_after} NaN rows. {rows_after} rows remaining.")

    # ------------------------------------------------------------------ #
    # 8. WRITE OUTPUT                                                     #
    # ------------------------------------------------------------------ #

    base, ext = os.path.splitext(input_filepath)
    output_filepath = base + "_PREPARED.csv"
    df.to_csv(output_filepath, index=False)
    print(f"Saved prepared file to: {output_filepath}")
    print(f"Final shape: {df.shape}")
    print(f"\nFinal columns ({len(df.columns)}):\n{list(df.columns)}")

    return output_filepath


prepare_features("data/generic/CLEAN_AAPL.csv")