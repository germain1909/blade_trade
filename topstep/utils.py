import pandas as pd
from typing import List, Dict, Any


def minute_bars_to_df(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert Topstep minute bars into a Pandas DataFrame with datetime index.
    Bars MUST contain keys: t, o, h, l, c, v.
    """
    if not bars:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"])

    df = pd.DataFrame(bars)

    # Convert timestamp string → datetime
    df["t"] = pd.to_datetime(df["t"])

    # Index + sort for consistent time ordering
    df = df.set_index("t")
    df = df.sort_index()

    return df


def save_minute_bars_to_csv(bars: List[Dict[str, Any]], filename: str) -> None:
    """
    Save list of minute bars to CSV.
    """
    df = minute_bars_to_df(bars)
    df.to_csv(filename)
    print(f"[Topstep] Saved {len(df)} bars → {filename}")